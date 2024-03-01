# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import os
import pickle
import re
import tempfile
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
import yaml
from diffusers import StableDiffusionPipeline

from peft import (
    AdaLoraConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    PeftModel,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptLearningConfig,
    PromptTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from peft.tuners.lora import LoraLayer
from peft.utils import _get_submodules, infer_device

from .testing_utils import get_state_dict


CONFIG_TESTING_KWARGS = (
    # IA³
    {
        "target_modules": None,
        "feedforward_modules": None,
    },
    # LoRA
    {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": None,
        "lora_dropout": 0.05,
        "bias": "none",
    },
    # prefix tuning
    {
        "num_virtual_tokens": 10,
    },
    # prompt encoder
    {
        "num_virtual_tokens": 10,
        "encoder_hidden_size": 32,
    },
    # prompt tuning
    {
        "num_virtual_tokens": 10,
    },
    # AdaLoRA
    {
        "target_modules": None,
    },
)

CLASSES_MAPPING = {
    "ia3": (IA3Config, CONFIG_TESTING_KWARGS[0]),
    "lora": (LoraConfig, CONFIG_TESTING_KWARGS[1]),
    "prefix_tuning": (PrefixTuningConfig, CONFIG_TESTING_KWARGS[2]),
    "prompt_encoder": (PromptEncoderConfig, CONFIG_TESTING_KWARGS[3]),
    "prompt_tuning": (PromptTuningConfig, CONFIG_TESTING_KWARGS[4]),
    "adalora": (AdaLoraConfig, CONFIG_TESTING_KWARGS[5]),
}


# Adapted from https://github.com/huggingface/transformers/blob/48327c57182fdade7f7797d1eaad2d166de5c55b/src/transformers/activations.py#LL166C7-L166C22
class ClassInstantier(OrderedDict):
    def __getitem__(self, key, *args, **kwargs):
        # check if any of the kwargs is inside the config class kwargs
        if any(kwarg in self[key][1] for kwarg in kwargs):
            new_config_kwargs = self[key][1].copy()
            new_config_kwargs.update(kwargs)
            return (self[key][0], new_config_kwargs)

        return super().__getitem__(key, *args, **kwargs)

    def get_grid_parameters(self, grid_parameters, filter_params_func=None):
        r"""
        Returns a list of all possible combinations of the parameters in the config classes.

        Args:
            grid_parameters (`dict`):
                A dictionary containing the parameters to be tested. There should be at least the key "model_ids" which
                contains a list of model ids to be tested. The other keys should be the name of the config class
                post-fixed with "_kwargs" and the value should be a dictionary containing the parameters to be tested
                for that config class.
            filter_params_func (`callable`, `optional`):
                A function that takes a list of tuples and returns a list of tuples. This function is used to filter
                out the tests that needs for example to be skipped.

        Returns:
            generated_tests (`list`):
                A list of tuples containing the name of the test, the model id, the config class and the config class
                kwargs.
        """
        generated_tests = []
        model_list = grid_parameters["model_ids"]
        task_type = grid_parameters["task_type"] if "task_type" in grid_parameters else None

        for model_id in model_list:
            for key, value in self.items():
                if f"{key}_kwargs" in grid_parameters:
                    peft_configs = []
                    current_peft_config = value[1].copy()
                    for current_key, current_value in grid_parameters[f"{key}_kwargs"].items():
                        for kwarg in current_value:
                            current_peft_config.update({current_key: kwarg})

                            if task_type is not None:
                                current_peft_config.update({"task_type": task_type})

                            peft_configs.append(current_peft_config.copy())
                else:
                    current_peft_config = value[1].copy()
                    if task_type is not None:
                        current_peft_config.update({"task_type": task_type})
                    peft_configs = [current_peft_config]

                for peft_config in peft_configs:
                    generated_tests.append((f"test_{model_id}_{key}", model_id, value[0], peft_config))

        if filter_params_func is not None:
            generated_tests = filter_params_func(generated_tests)

        return generated_tests


PeftTestConfigManager = ClassInstantier(CLASSES_MAPPING)


class PeftCommonTester:
    r"""
    A large testing suite for testing common functionality of the PEFT models.

    Attributes:
        torch_device (`torch.device`):
            The device on which the tests will be run.
        transformers_class (`transformers.PreTrainedModel`):
            The transformers class that is being tested.
    """

    torch_device = infer_device()
    transformers_class = None

    def prepare_inputs_for_common(self):
        raise NotImplementedError

    def check_modelcard(self, tmp_dirname, model):
        # check the generated README.md
        filename = os.path.join(tmp_dirname, "README.md")
        assert os.path.exists(filename)
        with open(filename, encoding="utf-8") as f:
            readme = f.read()
        metainfo = re.search(r"---\n(.*?)\n---", readme, re.DOTALL).group(1)
        dct = yaml.safe_load(metainfo)
        assert dct["library_name"] == "peft"

        if hasattr(model, "config"):
            assert dct["base_model"] == model.config.to_dict()["_name_or_path"]
        else:  # a custom model
            assert "base_model" not in dct

    def check_config_json(self, tmp_dirname, model):
        # check the generated config.json
        filename = os.path.join(tmp_dirname, "adapter_config.json")
        assert os.path.exists(filename)
        with open(filename, encoding="utf-8") as f:
            config = json.load(f)

        if hasattr(model, "config"):  # custom models don't have a config attribute
            assert config["base_model_name_or_path"] == model.config.to_dict()["_name_or_path"]

    def _test_model_attr(self, model_id, config_cls, config_kwargs):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)

        assert hasattr(model, "save_pretrained")
        assert hasattr(model, "from_pretrained")
        assert hasattr(model, "push_to_hub")

    def _test_adapter_name(self, model_id, config_cls, config_kwargs):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config, adapter_name="test-adapter")
        correctly_converted = False
        for n, _ in model.named_parameters():
            if "test-adapter" in n:
                correctly_converted = True
                break

        assert correctly_converted

    def _test_prepare_for_training(self, model_id, config_cls, config_kwargs):
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)

        dummy_input = self.prepare_inputs_for_testing()
        dummy_output = model.get_input_embeddings()(dummy_input["input_ids"])

        assert not dummy_output.requires_grad

        # load with `prepare_model_for_int8_training`
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        model = prepare_model_for_int8_training(model)

        for param in model.parameters():
            assert not param.requires_grad

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)

        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        dummy_input = self.prepare_inputs_for_testing()
        dummy_output = model.get_input_embeddings()(dummy_input["input_ids"])

        assert dummy_output.requires_grad

    def _test_save_pretrained(self, model_id, config_cls, config_kwargs, safe_serialization=True):
        # ensure that the weights are randomly initialized
        if issubclass(config_cls, LoraConfig):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_lora_weights"] = False
        if issubclass(config_cls, IA3Config):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_ia3_weights"] = False

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            if safe_serialization:
                model.save_pretrained(tmp_dirname)
            else:
                model.save_pretrained(tmp_dirname, safe_serialization=False)

            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

            # check if the state dicts are equal
            if issubclass(config_cls, PromptEncoderConfig):
                # For prompt encoding, when loading the whole state_dict, there are differences, therefore, only load
                # adapter-specific weights for comparison.
                # TODO: is this expected?
                state_dict = get_peft_model_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained, unwrap_compiled=True)
            else:
                state_dict = get_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_state_dict(model_from_pretrained, unwrap_compiled=True)

            # check if tensors equal
            for key in state_dict.keys():
                assert torch.allclose(
                    state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)
                )

            target_adapter_filename = "adapter_model.safetensors" if safe_serialization else "adapter_model.bin"

            # check if `adapter_model.safetensors` is present
            assert os.path.exists(os.path.join(tmp_dirname, target_adapter_filename))

            # check if `adapter_config.json` is present
            assert os.path.exists(os.path.join(tmp_dirname, "adapter_config.json"))

            # check if `model.safetensors` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "model.safetensors"))

            # check if `config.json` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "config.json"))

            self.check_modelcard(tmp_dirname, model)
            self.check_config_json(tmp_dirname, model)

    def _test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs, safe_serialization=True):
        if issubclass(config_cls, AdaLoraConfig):
            # AdaLora does not support adding more than 1 adapter
            return pytest.skip(f"Test not applicable for {config_cls}")

        # ensure that the weights are randomly initialized
        if issubclass(config_cls, LoraConfig):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_lora_weights"] = False
        if issubclass(config_cls, IA3Config):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_ia3_weights"] = False

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        new_adapter_config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        model.add_adapter("new_adapter", new_adapter_config)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            if safe_serialization:
                model.save_pretrained(tmp_dirname)
            else:
                model.save_pretrained(tmp_dirname, safe_serialization=False)

            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

            new_adapter_dir = os.path.join(tmp_dirname, "new_adapter")
            model_from_pretrained.load_adapter(new_adapter_dir, "new_adapter")

            # check if the state dicts are equal
            if issubclass(config_cls, PromptEncoderConfig):
                # For prompt encoding, when loading the whole state_dict, there are differences, therefore, only load
                # adapter-specific weights for comparison.
                # TODO: is this expected?
                state_dict = get_peft_model_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained, unwrap_compiled=True)
            else:
                state_dict = get_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_state_dict(model_from_pretrained, unwrap_compiled=True)

            # check if same keys
            assert state_dict.keys() == state_dict_from_pretrained.keys()

            # check if tensors equal
            for key in state_dict.keys():
                assert torch.allclose(
                    state_dict[key].to(self.torch_device), state_dict_from_pretrained[key].to(self.torch_device)
                )

            target_adapter_filename = "adapter_model.safetensors" if safe_serialization else "adapter_model.bin"

            # check if `adapter_model.safetensors` is present
            assert os.path.exists(os.path.join(tmp_dirname, target_adapter_filename))
            assert os.path.exists(os.path.join(new_adapter_dir, target_adapter_filename))

            # check if `adapter_config.json` is present
            assert os.path.exists(os.path.join(tmp_dirname, "adapter_config.json"))
            assert os.path.exists(os.path.join(new_adapter_dir, "adapter_config.json"))

            # check if `model.safetensors` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "model.safetensors"))
            assert not os.path.exists(os.path.join(new_adapter_dir, "model.safetensors"))

            # check if `config.json` is not present
            assert not os.path.exists(os.path.join(tmp_dirname, "config.json"))
            assert not os.path.exists(os.path.join(new_adapter_dir, "config.json"))

            self.check_modelcard(tmp_dirname, model)
            self.check_config_json(tmp_dirname, model)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname, selected_adapters=["default"])

            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

            assert "default" in model_from_pretrained.peft_config.keys()
            assert "new_adapter" not in model_from_pretrained.peft_config.keys()

    def _test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(
                model_from_pretrained, tmp_dirname, is_trainable=False, config=config
            )

            assert model_from_pretrained.peft_config["default"].inference_mode
            assert model_from_pretrained.peft_config["default"] is config

    def _test_merge_layers_fp16(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig, IA3Config, AdaLoraConfig, LoHaConfig, LoKrConfig):
            # Merge layers only supported for LoRA and IA³
            return pytest.skip(f"Test not applicable for {config_cls}")

        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        model = self.transformers_class.from_pretrained(model_id, torch_dtype=torch.float16)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(device="cpu", dtype=torch.float16)

        model.eval()

        # This should simply work
        _ = model.merge_and_unload()

    def _test_merge_layers_nan(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig, IA3Config, AdaLoraConfig, LoHaConfig, LoKrConfig):
            # Merge layers only supported for LoRA and IA³
            return
        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        dummy_input = self.prepare_inputs_for_testing()

        model.eval()

        # This should work
        logits_unmerged = model(**dummy_input)[0]

        model = model.merge_and_unload()
        logits_merged = model(**dummy_input)[0]

        assert torch.allclose(logits_unmerged, logits_merged, atol=1e-3, rtol=1e-3)

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        for name, module in model.named_parameters():
            if "lora_A" in name or "ia3" in name or "lora_E" in name or "lora_B" in name:
                module.data[0] = torch.nan

        with pytest.raises(
            ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
        ):
            model = model.merge_and_unload(safe_merge=True)

        for name, module in model.named_parameters():
            if "lora_A" in name or "ia3" in name or "lora_E" in name or "lora_B" in name:
                module.data[0] = torch.inf

        with pytest.raises(
            ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
        ):
            model = model.merge_and_unload(safe_merge=True)

    def _test_merge_layers(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        dummy_input = self.prepare_inputs_for_testing()
        model.eval()
        logits = model(**dummy_input)[0]

        model.merge_adapter()
        logits_merged = model(**dummy_input)[0]
        model.unmerge_adapter()
        logits_unmerged = model(**dummy_input)[0]

        model = model.merge_and_unload()
        logits_merged_unloaded = model(**dummy_input)[0]

        atol, rtol = 1e-4, 1e-4
        if (config.peft_type == "IA3") and (model_id == "Conv2d"):
            # for some reason, the IA³ Conv2d introduces a larger error
            atol, rtol = 0.3, 0.01
        assert torch.allclose(logits, logits_merged, atol=atol, rtol=rtol)
        assert torch.allclose(logits, logits_unmerged, atol=atol, rtol=rtol)
        assert torch.allclose(logits, logits_merged_unloaded, atol=atol, rtol=rtol)

        # For this test to work, weights should not be initialized to identity transform (e.g.
        # init_lora_weights should be False).
        transformers_model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        logits_transformers = transformers_model(**dummy_input)[0]
        assert not torch.allclose(logits_merged, logits_transformers, atol=1e-10, rtol=1e-10)

        # test that the logits are identical after a save-load-roundtrip
        if hasattr(model, "save_pretrained"):
            # model is a transformers model
            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)
                model_from_pretrained = self.transformers_class.from_pretrained(tmp_dirname).to(self.torch_device)
        else:
            # model is not a transformers model
            model_from_pretrained = pickle.loads(pickle.dumps(model))

        logits_merged_from_pretrained = model_from_pretrained(**dummy_input)[0]
        assert torch.allclose(logits_merged, logits_merged_from_pretrained, atol=atol, rtol=rtol)

    def _test_merge_layers_multi(self, model_id, config_cls, config_kwargs):
        supported_peft_types = [PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.IA3, PeftType.OFT]

        if ("gpt2" in model_id.lower()) and (config_cls == IA3Config):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        if config.peft_type not in supported_peft_types:
            return

        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)

        model = model.to(self.torch_device)

        dummy_input = self.prepare_inputs_for_testing()
        model.eval()

        with torch.inference_mode():
            logits_adapter_1 = model(**dummy_input)[0]

        model.add_adapter("adapter-2", config)
        model.set_adapter("adapter-2")
        model.eval()

        with torch.inference_mode():
            logits_adapter_2 = model(**dummy_input)[0]

        assert not torch.allclose(logits_adapter_1, logits_adapter_2, atol=1e-3, rtol=1e-3)

        model.set_adapter("default")

        with torch.inference_mode():
            logits_adapter_1_after_set = model(**dummy_input)[0]

        assert torch.allclose(logits_adapter_1_after_set, logits_adapter_1, atol=1e-3, rtol=1e-3)

        model_copy = copy.deepcopy(model)
        model_copy_2 = copy.deepcopy(model)
        model_merged_all = model.merge_and_unload(adapter_names=["adapter-2", "default"])

        with torch.inference_mode():
            logits_merged_all = model_merged_all(**dummy_input)[0]

        assert not torch.allclose(logits_merged_all, logits_adapter_2, atol=1e-3, rtol=1e-3)
        assert not torch.allclose(logits_merged_all, logits_adapter_1, atol=1e-3, rtol=1e-3)

        model_merged_adapter_2 = model_copy.merge_and_unload(adapter_names=["adapter-2"])

        with torch.inference_mode():
            logits_merged_adapter_2 = model_merged_adapter_2(**dummy_input)[0]

        assert torch.allclose(logits_merged_adapter_2, logits_adapter_2, atol=1e-3, rtol=1e-3)

        model_merged_adapter_default = model_copy_2.merge_and_unload(adapter_names=["default"])

        with torch.inference_mode():
            logits_merged_adapter_default = model_merged_adapter_default(**dummy_input)[0]

        assert torch.allclose(logits_merged_adapter_default, logits_adapter_1, atol=1e-3, rtol=1e-3)

    def _test_merge_layers_is_idempotent(self, model_id, config_cls, config_kwargs):
        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        model.eval()
        torch.manual_seed(0)
        model.merge_adapter()
        logits_0 = model(**self.prepare_inputs_for_testing())[0]

        # merging again should not change anything
        # also check warning:
        with pytest.warns(UserWarning, match="All adapters are already merged, nothing to do"):
            model.merge_adapter()
        logits_1 = model(**self.prepare_inputs_for_testing())[0]

        assert torch.allclose(logits_0, logits_1, atol=1e-6, rtol=1e-6)

    def _test_safe_merge(self, model_id, config_cls, config_kwargs):
        torch.manual_seed(0)
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = model.to(self.torch_device).eval()

        inputs = self.prepare_inputs_for_testing()
        logits_base = model(**inputs)[0]

        model = get_peft_model(model, config).eval()
        logits_peft = model(**inputs)[0]

        # sanity check that the logits are different
        assert not torch.allclose(logits_base, logits_peft, atol=1e-6, rtol=1e-6)

        model_unloaded = model.merge_and_unload(safe_merge=True)
        logits_unloaded = model_unloaded(**inputs)[0]

        # check that the logits are the same after unloading
        assert torch.allclose(logits_peft, logits_unloaded, atol=1e-6, rtol=1e-6)

    def _test_generate(self, model_id, config_cls, config_kwargs):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        inputs = self.prepare_inputs_for_testing()

        # check if `generate` works
        _ = model.generate(**inputs)

    def _test_generate_pos_args(self, model_id, config_cls, config_kwargs, raises_err: bool):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        inputs = self.prepare_inputs_for_testing()
        if raises_err:
            with pytest.raises(TypeError):
                # check if `generate` raises an error if positional arguments are passed
                _ = model.generate(inputs["input_ids"])
        else:
            # check if `generate` works if positional arguments are passed
            _ = model.generate(inputs["input_ids"])

    def _test_generate_half_prec(self, model_id, config_cls, config_kwargs):
        if config_cls not in (IA3Config, LoraConfig, PrefixTuningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        if self.torch_device == "mps":  # BFloat16 is not supported on MPS
            return pytest.skip("BFloat16 is not supported on MPS")

        model = self.transformers_class.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(self.torch_device)
        attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        # check if `generate` works
        _ = model.generate(input_ids=input_ids, attention_mask=attention_mask)

    def _test_prefix_tuning_half_prec_conversion(self, model_id, config_cls, config_kwargs):
        if config_cls not in (PrefixTuningConfig,):
            return pytest.skip(f"Test not applicable for {config_cls}")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.half()

        assert model.base_model_torch_dtype == torch.float16

    def _test_training(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")
        if (config_cls == AdaLoraConfig) and ("roberta" in model_id.lower()):
            # TODO: no gradients on the "dense" layer, other layers work, not sure why
            self.skipTest("AdaLora with RoBERTa does not work correctly")

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        inputs = self.prepare_inputs_for_testing()

        # check if `training` works
        output = model(**inputs)[0]
        loss = output.sum()
        loss.backward()
        parameter_prefix = model.prefix
        for n, param in model.named_parameters():
            if (parameter_prefix in n) or ("modules_to_save" in n):
                assert param.grad is not None
            else:
                assert param.grad is None

    def _test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        if (config_cls == PrefixTuningConfig) and ("deberta" in model_id.lower()):
            # TODO: raises an error:
            # TypeError: DebertaModel.forward() got an unexpected keyword argument 'past_key_values'
            self.skipTest("DeBERTa with PrefixTuning does not work correctly")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        inputs = self.prepare_inputs_for_testing()

        # check if `training` works
        output = model(**inputs)[0]
        logits = output[0]

        loss = output.sum()
        loss.backward()

        # set to eval mode, since things like dropout can affect the output otherwise
        model.eval()
        logits = model(**inputs)[0][0]

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname, safe_serialization=True)
            assert "adapter_model.safetensors" in os.listdir(tmp_dirname)
            assert "adapter_model.bin" not in os.listdir(tmp_dirname)

            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(self.torch_device)

            logits_from_pretrained = model_from_pretrained(**inputs)[0][0]
            assert torch.allclose(logits, logits_from_pretrained, atol=1e-4, rtol=1e-4)

    def _test_training_layer_indexing(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig,):
            return pytest.skip(f"Test not applicable for {config_cls}")

        config = config_cls(
            base_model_name_or_path=model_id,
            layers_to_transform=[0],
            **config_kwargs,
        )
        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        inputs = self.prepare_inputs_for_testing()

        # check if `training` works
        output = model(**inputs)[0]
        logits = output[0]

        loss = output.sum()
        loss.backward()

        nb_trainable = 0

        for n, param in model.named_parameters():
            if "lora" in n:
                assert param.grad is not None
                nb_trainable += 1
            else:
                assert param.grad is None

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(self.torch_device)

            logits_from_pretrained = model_from_pretrained(**inputs)[0][0]
            assert torch.allclose(logits, logits_from_pretrained, atol=1e-4, rtol=1e-4)

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        nb_trainable_all = 0

        for n, param in model.named_parameters():
            if "lora" in n:
                nb_trainable_all += 1

        assert nb_trainable < nb_trainable_all

    def _test_training_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        if (config_cls == AdaLoraConfig) and ("roberta" in model_id.lower()):
            # TODO: no gradients on the "dense" layer, other layers work, not sure why
            self.skipTest("AdaLora with RoBERTa does not work correctly")

        model = self.transformers_class.from_pretrained(model_id)

        if not getattr(model, "supports_gradient_checkpointing", False):
            return pytest.skip(f"Model {model_id} does not support gradient checkpointing")

        model.gradient_checkpointing_enable()

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        inputs = self.prepare_inputs_for_testing()

        # check if `training` works
        output = model(**inputs)[0]

        loss = output.sum()
        loss.backward()
        parameter_prefix = "ia3" if config_cls == IA3Config else "lora"
        for n, param in model.named_parameters():
            if parameter_prefix in n:
                assert param.grad is not None
            else:
                assert param.grad is None

    def _test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig,):
            return pytest.skip(f"Test not applicable for {config_cls}")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        model = self.transformers_class.from_pretrained(model_id)

        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model_from_pretrained = self.transformers_class.from_pretrained(model_id)
            _ = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname, device_map={"": "cpu"}).to(
                self.torch_device
            )

    def _test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        if not issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        inputs = self.prepare_inputs_for_testing()

        # check if `training` works
        output = model(**inputs)[0]
        loss = output.sum()
        loss.backward()

        # check that prompt encoder has grads
        for param in model.prompt_encoder.parameters():
            assert param.grad is not None

    def _test_delete_adapter(self, model_id, config_cls, config_kwargs):
        supported_peft_types = [PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.IA3, PeftType.OFT]
        # IA3 does not support deleting adapters yet, but it just needs to be added
        # AdaLora does not support multiple adapters
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.peft_type not in supported_peft_types:
            return pytest.skip(f"Test not applicable for {config.peft_type}")

        model = self.transformers_class.from_pretrained(model_id)
        adapter_to_delete = "delete_me"
        model = get_peft_model(model, config)
        model.add_adapter(adapter_to_delete, config)
        model.set_adapter(adapter_to_delete)
        model = model.to(self.torch_device)
        model.delete_adapter(adapter_to_delete)
        assert adapter_to_delete not in model.peft_config
        assert model.active_adapters == ["default"]

        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            _, target, _ = _get_submodules(model, key)
            attributes_to_check = getattr(target, "adapter_layer_names", []) + getattr(target, "other_param_names", [])
            for attr in attributes_to_check:
                assert adapter_to_delete not in getattr(target, attr)

        # check that we can also delete the last remaining adapter
        model.delete_adapter("default")
        assert "default" not in model.peft_config
        assert model.active_adapters == []

        input = self.prepare_inputs_for_testing()
        # note: we cannot call model(**input) because PeftModel always expects there to be at least one adapter
        model.base_model(**input)  # should not raise an error

    def _test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        # same as test_delete_adapter, but this time an inactive adapter is deleted
        supported_peft_types = [PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.IA3, PeftType.OFT]
        # IA3 does not support deleting adapters yet, but it just needs to be added
        # AdaLora does not support multiple adapters
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.peft_type not in supported_peft_types:
            return pytest.skip(f"Test not applicable for {config.peft_type}")

        model = self.transformers_class.from_pretrained(model_id)
        adapter_to_delete = "delete_me"
        model = get_peft_model(model, config)
        model.add_adapter(adapter_to_delete, config)
        # "delete_me" is added but not activated
        model = model.to(self.torch_device)
        model.delete_adapter(adapter_to_delete)
        assert adapter_to_delete not in model.peft_config
        assert model.active_adapters == ["default"]

        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            _, target, _ = _get_submodules(model, key)
            attributes_to_check = getattr(target, "adapter_layer_names", []) + getattr(target, "other_param_names", [])
            for attr in attributes_to_check:
                assert adapter_to_delete not in getattr(target, attr)

        # check that we can also delete the last remaining adapter
        model.delete_adapter("default")
        assert "default" not in model.peft_config
        assert model.active_adapters == []

        input = self.prepare_inputs_for_testing()
        # note: we cannot call model(**input) because PeftModel always expects there to be at least one adapter
        model.base_model(**input)  # should not raise an error

    def _test_unload_adapter(self, model_id, config_cls, config_kwargs):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        if config.peft_type not in ("LORA", "ADALORA", "IA3"):
            with pytest.raises(AttributeError):
                model = model.unload()
        else:
            dummy_input = self.prepare_inputs_for_testing()
            logits_with_adapter = model(**dummy_input)[0]

            transformers_model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            logits_transformers = transformers_model(**dummy_input)[0]

            model.eval()
            model = model.unload()
            logits_unload = model(**dummy_input)[0]

            assert not torch.allclose(logits_with_adapter, logits_unload, atol=1e-10, rtol=1e-10)
            assert torch.allclose(logits_transformers, logits_unload, atol=1e-4, rtol=1e-4)

    def _test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, AdaLoraConfig):
            # AdaLora does not support adding more than 1 adapter
            return pytest.skip(f"Test not applicable for {config_cls}")

        adapter_list = ["adapter1", "adapter_2", "adapter_3"]
        weight_list = [0.5, 1.5, 1.5]
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if not isinstance(config, LoraConfig):
            return pytest.skip(f"Test not applicable for {config}")

        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config, adapter_list[0])
        model.add_adapter(adapter_list[1], config)
        model.add_adapter(adapter_list[2], replace(config, r=20))
        model = model.to(self.torch_device)

        # test re-weighting single adapter
        model.add_weighted_adapter([adapter_list[0]], [weight_list[0]], "single_adapter_reweighting")

        # test svd re-weighting with multiple adapters
        model.add_weighted_adapter(adapter_list[1:], weight_list[1:], "multi_adapter_svd_reweighting")

        # test ties_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_ties_svd_reweighting",
            combination_type="ties_svd",
            density=0.5,
        )

        # test dare_linear_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_dare_linear_svd_reweighting",
            combination_type="dare_linear_svd",
            density=0.5,
        )

        # test dare_ties_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_dare_ties_svd_reweighting",
            combination_type="dare_ties_svd",
            density=0.5,
        )

        # test magnitude_prune_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_magnitude_prune_svd_reweighting",
            combination_type="magnitude_prune_svd",
            density=0.5,
        )

        # test cat re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:], weight_list[1:], "multi_adapter_cat_reweighting", combination_type="cat"
        )

        # test linear re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2], weight_list[:2], "multi_adapter_linear_reweighting", combination_type="linear"
        )

        # test ties re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2], weight_list[:2], "multi_adapter_ties_reweighting", combination_type="ties", density=0.5
        )

        # test dare_linear re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2],
            weight_list[:2],
            "multi_adapter_dare_linear_reweighting",
            combination_type="dare_linear",
            density=0.5,
        )

        # test dare_ties re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2],
            weight_list[:2],
            "multi_adapter_dare_ties_reweighting",
            combination_type="dare_ties",
            density=0.5,
        )

        # test magnitude_prune re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2],
            weight_list[:2],
            "multi_adapter_magnitude_prune_reweighting",
            combination_type="magnitude_prune",
            density=0.5,
        )

        # test linear re-weighting with multiple adapters with only first adapter having non zero weight
        model.add_weighted_adapter(
            adapter_list[:2],
            [weight_list[0], 0],
            "multi_adapter_linear_reweighting_single_enabled",
            combination_type="linear",
        )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_linear_reweighting_uneven_r",
                combination_type="linear",
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_ties_reweighting_uneven_r",
                combination_type="ties",
                density=0.5,
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_dare_linear_reweighting_uneven_r",
                combination_type="dare_linear",
                density=0.5,
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_dare_ties_reweighting_uneven_r",
                combination_type="dare_ties",
                density=0.5,
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_magnitude_prune_reweighting_uneven_r",
                combination_type="magnitude_prune",
                density=0.5,
            )

        new_adapters = [
            "single_adapter_reweighting",
            "multi_adapter_svd_reweighting",
            "multi_adapter_ties_svd_reweighting",
            "multi_adapter_dare_linear_svd_reweighting",
            "multi_adapter_dare_ties_svd_reweighting",
            "multi_adapter_magnitude_prune_svd_reweighting",
            "multi_adapter_cat_reweighting",
            "multi_adapter_linear_reweighting",
            "multi_adapter_linear_reweighting_single_enabled",
            "multi_adapter_ties_reweighting",
            "multi_adapter_dare_linear_reweighting",
            "multi_adapter_dare_ties_reweighting",
            "multi_adapter_magnitude_prune_reweighting",
        ]
        for new_adapter in new_adapters:
            assert new_adapter in model.peft_config

        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            _, target, _ = _get_submodules(model, key)
            if isinstance(target, LoraLayer):
                for adapter_name in new_adapters:
                    if "single" in adapter_name:
                        new_delta_weight = target.get_delta_weight(adapter_name)
                        weighted_original_delta_weights = target.get_delta_weight(adapter_list[0]) * weight_list[0]
                        assert torch.allclose(new_delta_weight, weighted_original_delta_weights, atol=1e-4, rtol=1e-4)
                    elif "svd" in adapter_name:
                        assert target.r[adapter_name] == 20
                    elif "linear" in adapter_name:
                        assert target.r[adapter_name] == 8
                    elif "cat" in adapter_name:
                        assert target.r[adapter_name] == 28

        dummy_input = self.prepare_inputs_for_testing()
        model.eval()
        for adapter_name in new_adapters:
            # ensuring new adapters pass the forward loop
            model.set_adapter(adapter_name)
            assert model.active_adapter == adapter_name
            assert model.active_adapters == [adapter_name]
            model(**dummy_input)[0]

    def _test_disable_adapter(self, model_id, config_cls, config_kwargs):
        task_type = config_kwargs.get("task_type")
        if (task_type == "SEQ_2_SEQ_LM") and (config_cls in (PromptTuningConfig, PromptEncoderConfig)):
            self.skipTest("Seq2Seq + prompt tuning/prompt encoder does not work with disabling adapters")

        def get_output(model):
            # helper function that works with different model types
            torch.manual_seed(0)

            if hasattr(model, "generate"):
                # let's check the scores, not the output ids, since the latter can easily be identical even if the
                # weights are slightly changed
                output = model.generate(**input, return_dict_in_generate=True, output_scores=True).scores[0]
                # take element 0, as output is a tuple
            else:
                output = model(**input)

            if hasattr(output, "images"):  # for SD
                import numpy as np

                img = output.images[0]
                return torch.from_numpy(np.array(img))

            return output

        # initialize model
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)

        # output from BASE MODEL
        input = self.prepare_inputs_for_testing()
        output_before = get_output(model)

        # output from PEFT MODEL
        if hasattr(self, "instantiate_sd_peft"):
            # SD models are instantiated differently
            peft_model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)
        else:
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            peft_model = get_peft_model(model, config)

        output_peft = get_output(peft_model)

        # first check trivial case is not true that peft does not affect the output; for this to work, init_lora_weight
        # must be False
        if isinstance(peft_model, StableDiffusionPipeline):
            # for SD, check that most pixels have different values
            assert (output_before != output_peft).float().mean() > 0.8
        else:
            assert not torch.allclose(output_before, output_peft)

        # output with DISABLED ADAPTER
        if isinstance(peft_model, StableDiffusionPipeline):
            with peft_model.unet.disable_adapter():
                with peft_model.text_encoder.disable_adapter():
                    output_peft_disabled = get_output(peft_model)
            # for SD, very rarely, a pixel can differ
            assert (output_before != output_peft_disabled).float().mean() < 1e-4
        else:
            with peft_model.disable_adapter():
                output_peft_disabled = get_output(peft_model)
            assert torch.allclose(output_before, output_peft_disabled, atol=1e-6, rtol=1e-6)

            # after leaving the disable_adapter context, the output should be the same as with enabled adapter again
            # see #1501
            output_peft_after_disabled = get_output(peft_model)
            assert torch.allclose(output_peft, output_peft_after_disabled, atol=1e-6, rtol=1e-6)

        # TODO: add tests to check if disabling adapters works after calling merge_adapter

    def _test_adding_multiple_adapters_with_bias_raises(self, model_id, config_cls, config_kwargs):
        # When trying to add multiple adapters with bias in Lora or AdaLora, an error should be
        # raised. Also, the peft model should not be left in a half-initialized state.
        if not issubclass(config_cls, (LoraConfig, AdaLoraConfig)):
            return pytest.skip(f"Test not applicable for {config_cls}")

        config_kwargs = config_kwargs.copy()
        config_kwargs["bias"] = "all"
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config, "adapter0")
        with pytest.raises(ValueError):
            model.add_adapter("adapter1", replace(config, r=20))

        # (superficial) test that the model is not left in a half-initialized state when adding an adapter fails
        assert "adapter1" not in model.peft_config
        assert "adapter1" not in model.base_model.peft_config

    def _test_passing_input_embeds_works(self, test_name, model_id, config_cls, config_kwargs):
        # https://github.com/huggingface/peft/issues/727
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config, adapter_name="test-adapter").to(self.torch_device)
        dummy_input = self.prepare_inputs_for_testing()
        inputs_embeds = model.get_input_embeddings()(dummy_input["input_ids"])
        # just check that no error is raised
        model.forward(inputs_embeds=inputs_embeds)
