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
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import replace
from unittest import mock

import pytest
import torch
import yaml
from diffusers import StableDiffusionPipeline
from packaging import version
from safetensors.torch import save_file

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    BoneConfig,
    CPTConfig,
    FourierFTConfig,
    HRAConfig,
    IA3Config,
    LNTuningConfig,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PeftModel,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptLearningConfig,
    PromptTuningConfig,
    RandLoraConfig,
    VBLoRAConfig,
    VeraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import _get_submodules, infer_device
from peft.utils.other import AuxiliaryTrainingWrapper, ModulesToSaveWrapper, TrainableTokensWrapper

from .testing_utils import get_state_dict


HUB_MODEL_ACCESSES = {}

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
        "total_step": 1,
    },
    # BOFT
    {
        "target_modules": None,
    },
    # VeRA
    {
        "r": 8,
        "target_modules": None,
        "vera_dropout": 0.05,
        "projection_prng_key": 0xFF,
        "d_initial": 0.1,
        "save_projection": True,
        "bias": "none",
    },
    # FourierFT
    {
        "n_frequency": 10,
        "target_modules": None,
    },
    # HRA
    {
        "target_modules": None,
    },
    # VBLoRA
    {"target_modules": None, "vblora_dropout": 0.05, "vector_length": 1, "num_vectors": 2},
    # OFT
    {
        "target_modules": None,
    },
    # Bone
    {
        "target_modules": None,
        "r": 2,
    },
    # LoRA + trainable_tokens
    {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": None,
        "lora_dropout": 0.05,
        "bias": "none",
        "trainable_token_indices": [0, 1, 3],
    },
    # RandLoRA
    {
        "r": 32,
        "randlora_alpha": 64,
        "target_modules": None,
        "randlora_dropout": 0.05,
        "projection_prng_key": 0xFF,
        "save_projection": True,
        "bias": "none",
    },
    # CPT tuninig
    {
        "cpt_token_ids": [0, 1, 2, 3, 4, 5, 6, 7],  # Example token IDs for testing
        "cpt_mask": [1, 1, 1, 1, 1, 1, 1, 1],
        "cpt_tokens_type_mask": [1, 2, 2, 2, 3, 3, 4, 4],
    },
)

CLASSES_MAPPING = {
    "ia3": (IA3Config, CONFIG_TESTING_KWARGS[0]),
    "lora": (LoraConfig, CONFIG_TESTING_KWARGS[1]),
    "prefix_tuning": (PrefixTuningConfig, CONFIG_TESTING_KWARGS[2]),
    "prompt_encoder": (PromptEncoderConfig, CONFIG_TESTING_KWARGS[3]),
    "prompt_tuning": (PromptTuningConfig, CONFIG_TESTING_KWARGS[4]),
    "adalora": (AdaLoraConfig, CONFIG_TESTING_KWARGS[5]),
    "boft": (BOFTConfig, CONFIG_TESTING_KWARGS[6]),
    "vera": (VeraConfig, CONFIG_TESTING_KWARGS[7]),
    "fourierft": (FourierFTConfig, CONFIG_TESTING_KWARGS[8]),
    "hra": (HRAConfig, CONFIG_TESTING_KWARGS[9]),
    "vblora": (VBLoRAConfig, CONFIG_TESTING_KWARGS[10]),
    "oft": (OFTConfig, CONFIG_TESTING_KWARGS[11]),
    "bone": (BoneConfig, CONFIG_TESTING_KWARGS[12]),
    "lora+trainable_tokens": (LoraConfig, CONFIG_TESTING_KWARGS[13]),
    "randlora": (RandLoraConfig, CONFIG_TESTING_KWARGS[14]),
}

DECODER_MODELS_EXTRA = {"cpt": (CPTConfig, CONFIG_TESTING_KWARGS[15])}


@contextmanager
def hub_online_once(model_id: str):
    """Set env[HF_HUB_OFFLINE]=1 (and patch transformers/hugging_face_hub to think that it was always that way)
    for model ids that were seen already so that the hub is not contacted twice for the same model id in said context.
    The cache (`HUB_MODEL_ACCESSES`) also tracks the number of cache hits per model id.

    The reason for doing a context manager and not patching specific methods (e.g., `from_pretrained`) is that there
    are a lot of places (`PeftConfig.from_pretrained`, `get_peft_state_dict`, `load_adapter`, ...) that possibly
    communicate with the hub to download files / check versions / etc.

    Note that using this context manager can cause problems when used in code sections that access different resources.
    Example:

    ```
    def test_something(model_id, config_kwargs):
        with hub_online_once(model_id):
            model = ...from_pretrained(model_id)
            self.do_something_specific_with_model(model)
    ```
    It is assumed that `do_something_specific_with_model` is an absract method that is implement by several tests.
    Imagine the first test simply does `model.generate([1,2,3])`. The second call from another test suite however uses
    a tokenizer (`AutoTokenizer.from_pretrained(model_id)`) - this will fail since the first pass was online but didn't
    use the tokenizer and we're now in offline mode and cannot fetch the tokenizer. The recommended workaround is to
    extend the cache key (`model_id` passed to `hub_online_once` in this case) by something in case the tokenizer is
    used, so that these tests don't share a cache pool with the tests that don't use a tokenizer.
    """
    global HUB_MODEL_ACCESSES
    override = {}

    try:
        if model_id in HUB_MODEL_ACCESSES:
            override = {"HF_HUB_OFFLINE": "1"}
            HUB_MODEL_ACCESSES[model_id] += 1
        else:
            if model_id not in HUB_MODEL_ACCESSES:
                HUB_MODEL_ACCESSES[model_id] = 0
        with (
            # strictly speaking it is not necessary to set the environment variable since most code that's out there
            # is evaluating it at import time and we'd have to reload the modules for it to take effect. It's
            # probably still a good idea to have it if there's some dynamic code that checks it.
            mock.patch.dict(os.environ, override),
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", override.get("HF_HUB_OFFLINE", False) == "1"),
            mock.patch("transformers.utils.hub._is_offline_mode", override.get("HF_HUB_OFFLINE", False) == "1"),
        ):
            yield
    except Exception:
        # in case of an error we have to assume that we didn't access the model properly from the hub
        # for the first time, so the next call cannot be considered cached.
        if HUB_MODEL_ACCESSES.get(model_id) == 0:
            del HUB_MODEL_ACCESSES[model_id]
        raise


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

        # The Hub expects the lora tag to be set for PEFT LoRA models since they
        # have explicit support for things like inference.
        if model.active_peft_config.peft_type.value == "LORA":
            assert "lora" in dct["tags"]

    def check_config_json(self, tmp_dirname, model):
        # check the generated config.json
        filename = os.path.join(tmp_dirname, "adapter_config.json")
        assert os.path.exists(filename)
        with open(filename, encoding="utf-8") as f:
            config = json.load(f)

        if hasattr(model, "config"):  # custom models don't have a config attribute
            assert config["base_model_name_or_path"] == model.config.to_dict()["_name_or_path"]

    def perturb_trainable_token_weights_if_used(self, model, config_kwargs, adapter_name="default", scale=1.0):
        """TrainableTokensLayer is initialized to be a no-op by default. Since there's currently no way to pass
        `init_weights=False` to the trainable tokens layer when used in conjunction with LoRA, we have to do it like
        this to make sure that it is *not* a no-op (essentially simulating "training" of the adapter).
        """
        if "trainable_token_indices" not in config_kwargs:
            return

        token_wrapper = None

        if hasattr(model, "get_input_embeddings"):
            token_wrapper = model.get_input_embeddings()
        else:
            for module in model.modules():
                if isinstance(module, TrainableTokensWrapper):
                    token_wrapper = module
                    break

        # for a model with trainable_token_indices there should always be a trainable token wrapper somewhere.
        # if not, then there's something broken.
        assert token_wrapper is not None

        token_wrapper.token_adapter.trainable_tokens_delta[adapter_name].data = (
            torch.rand_like(token_wrapper.token_adapter.trainable_tokens_delta[adapter_name].data) * scale
        )

    def _test_model_attr(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
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
        if config_kwargs.get("trainable_token_indices", None) is not None:
            # incompatible because trainable tokens is marking embeddings as trainable
            self.skipTest("Trainable tokens is incompatible with this test.")

        # some tests require specific tokenizers, make sure that they can be fetched as well
        with hub_online_once(model_id + config_kwargs.get("tokenizer_name_or_path", "")):
            model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            dummy_input = self.prepare_inputs_for_testing()
            dummy_output = model.get_input_embeddings()(dummy_input["input_ids"])

            assert not dummy_output.requires_grad

            # load with `prepare_model_for_kbit_training`
            model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            model = prepare_model_for_kbit_training(model)

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

    def _test_load_model_low_cpu_mem_usage(self, model_id, config_cls, config_kwargs):
        # Ensure that low_cpu_mem_usage=True works for from_pretrained and load_adapter and that the resulting model's
        # parameters are on the correct device.
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            # note: not using the context manager here because it fails on Windows CI for some reason
            tmp_dirname = tempfile.mkdtemp()
            try:
                model.save_pretrained(tmp_dirname)

                model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
                model = PeftModel.from_pretrained(
                    model, tmp_dirname, torch_device=self.torch_device, low_cpu_mem_usage=True
                )
                assert {p.device.type for p in model.parameters()} == {self.torch_device}

                model.load_adapter(tmp_dirname, adapter_name="other", low_cpu_mem_usage=True)
                assert {p.device.type for p in model.parameters()} == {self.torch_device}
            finally:
                try:
                    shutil.rmtree(tmp_dirname)
                except PermissionError:
                    # windows error
                    pass

            # also test injecting directly
            del model
            model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            inject_adapter_in_model(config, model, low_cpu_mem_usage=True)  # check that there is no error

            if not isinstance(config, LNTuningConfig):
                # LN tuning does not add adapter layers that could be on meta device, it only changes the requires_grad.
                # Therefore, there is no meta device for LN tuning.
                assert "meta" in {p.device.type for p in model.parameters()}

    def _test_save_pretrained(self, model_id, config_cls, config_kwargs, safe_serialization=True):
        # ensure that the weights are randomly initialized
        if issubclass(config_cls, LoraConfig):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_lora_weights"] = False
        if issubclass(config_cls, IA3Config):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_ia3_weights"] = False
        if hasattr(config_cls, "init_weights"):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_weights"] = False

        with hub_online_once(model_id):
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
                with warnings.catch_warnings(record=True) as recs:
                    model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
                    # ensure that there is no warning
                    assert not any("Found missing adapter keys" in str(rec.message) for rec in recs)

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
        elif issubclass(config_cls, IA3Config):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_ia3_weights"] = False
        elif hasattr(config_cls, "init_weights"):
            config_kwargs["init_weights"] = False

        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
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

    def _test_load_multiple_adapters(self, model_id, config_cls, config_kwargs):
        # just ensure that this works and raises no error
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)
                del model

                model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
                model = PeftModel.from_pretrained(model, tmp_dirname, torch_device=self.torch_device)
                load_result1 = model.load_adapter(tmp_dirname, adapter_name="other")
                load_result2 = model.load_adapter(tmp_dirname, adapter_name="yet-another")

                # VBLoRA uses a shared "vblora_vector_bank" across all layers, causing it to appear
                # in the missing keys list, which leads to failed test cases. So
                # skipping the missing keys check for VBLoRA.
                if config.peft_type != "VBLORA":
                    assert load_result1.missing_keys == []
                    assert load_result2.missing_keys == []

    def _test_merge_layers_fp16(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig, IA3Config, AdaLoraConfig, LoHaConfig, LoKrConfig, VBLoRAConfig):
            # Merge layers only supported for LoRA and IA³
            return pytest.skip(f"Test not applicable for {config_cls}")

        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        if (self.torch_device in ["cpu"]) and (version.parse(torch.__version__) <= version.parse("2.1")):
            self.skipTest("PyTorch 2.1 not supported for Half of addmm_impl_cpu_ ")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id, torch_dtype=torch.float16)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)
            model = model.to(device=self.torch_device, dtype=torch.float16)

            model.eval()

            # This should simply work
            _ = model.merge_and_unload()

    def _test_merge_layers_nan(self, model_id, config_cls, config_kwargs):
        if config_cls not in (
            LoraConfig,
            IA3Config,
            AdaLoraConfig,
            LoHaConfig,
            LoKrConfig,
            VeraConfig,
            FourierFTConfig,
        ):
            # Merge layers only supported for LoRA and IA³
            return
        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        if "gemma" in model_id.lower():
            # TODO: could be related to tied weights
            self.skipTest("Merging currently fails with gemma")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )

            model = get_peft_model(model, config)
            model = model.to(self.torch_device)

            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

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
                if (
                    "lora_A" in name
                    or "ia3" in name
                    or "lora_E" in name
                    or "lora_B" in name
                    or "vera_lambda" in name
                    or "fourierft_spectrum" in name
                ):
                    module.data[0] = torch.nan

            with pytest.raises(
                ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
            ):
                model = model.merge_and_unload(safe_merge=True)

            for name, module in model.named_parameters():
                if (
                    "lora_A" in name
                    or "ia3" in name
                    or "lora_E" in name
                    or "lora_B" in name
                    or "vera_lambda" in name
                    or "fourierft_spectrum" in name
                ):
                    module.data[0] = torch.inf

            with pytest.raises(
                ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
            ):
                model = model.merge_and_unload(safe_merge=True)

    def _test_merge_layers(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        if issubclass(config_cls, (OFTConfig, BOFTConfig)):
            return pytest.skip(f"Test not applicable for {config_cls}")

        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        if "gemma" in model_id.lower():
            # TODO: could be related to tied weights
            self.skipTest("Merging currently fails with gemma")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )

            model = get_peft_model(model, config)
            model = model.to(self.torch_device)

            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

            dummy_input = self.prepare_inputs_for_testing()
            model.eval()
            logits = model(**dummy_input)[0]

            model.merge_adapter()
            logits_merged = model(**dummy_input)[0]
            model.unmerge_adapter()
            logits_unmerged = model(**dummy_input)[0]

            model = model.merge_and_unload()

            # check that PEFT layers are completely removed
            assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
            logits_merged_unloaded = model(**dummy_input)[0]

            conv_ids = ["Conv2d", "Conv3d", "Conv2d2"]
            atol, rtol = 1e-4, 1e-4
            if self.torch_device in ["mlu"]:
                atol, rtol = 1e-3, 1e-3  # MLU
            if config.peft_type == "ADALORA":
                # AdaLoRA is a bit flaky on CI, but this cannot be reproduced locally
                atol, rtol = 1e-2, 1e-2
            if (config.peft_type in {"IA3", "LORA"}) and (model_id in conv_ids):
                # for some reason, the Conv introduces a larger error
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
                tmp_dirname = tempfile.mkdtemp()
                # note: not using the context manager here because it fails on Windows CI for some reason
                try:
                    model.save_pretrained(tmp_dirname)
                    model_from_pretrained = self.transformers_class.from_pretrained(tmp_dirname).to(self.torch_device)
                finally:
                    try:
                        shutil.rmtree(tmp_dirname)
                    except PermissionError:
                        # windows error
                        pass
            else:
                # model is not a transformers model
                model_from_pretrained = pickle.loads(pickle.dumps(model))

            logits_merged_from_pretrained = model_from_pretrained(**dummy_input)[0]
            assert torch.allclose(logits_merged, logits_merged_from_pretrained, atol=atol, rtol=rtol)

    def _test_merge_layers_multi(self, model_id, config_cls, config_kwargs):
        supported_peft_types = [
            PeftType.LORA,
            PeftType.LOHA,
            PeftType.LOKR,
            PeftType.IA3,
            PeftType.OFT,
            PeftType.BOFT,
            PeftType.HRA,
            PeftType.BONE,
        ]

        if ("gpt2" in model_id.lower()) and (config_cls == IA3Config):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        if config_kwargs.get("trainable_token_indices", None) is not None:
            self.skipTest(
                "Merging two adapters with trainable tokens is tested elsewhere since adapters with "
                "the same token indices cannot be merged."
            )

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        if config.peft_type not in supported_peft_types:
            return

        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
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

            atol, rtol = 1e-6, 1e-6  # default
            # Initializing with LN tuning cannot be configured to change the outputs (unlike init_lora_weights=False)
            if not issubclass(config_cls, LNTuningConfig):
                # sanity check that the logits are different
                assert not torch.allclose(logits_base, logits_peft, atol=atol, rtol=rtol)

            model_unloaded = model.merge_and_unload(safe_merge=True)
            logits_unloaded = model_unloaded(**inputs)[0]

            if self.torch_device in ["mlu"]:
                atol, rtol = 1e-3, 1e-3  # MLU

            conv_ids = ["Conv2d", "Conv3d", "Conv2d2"]
            if issubclass(config_cls, (IA3Config, LoraConfig)) and model_id in conv_ids:  # more instability with Conv
                atol, rtol = 1e-3, 1e-3

            # check that the logits are the same after unloading
            assert torch.allclose(logits_peft, logits_unloaded, atol=atol, rtol=rtol)

            # Ensure that serializing with safetensors works, there was an error when weights were not contiguous
            with tempfile.TemporaryDirectory() as tmp_dirname:
                # serializing with torch.save works
                torch.save(model_unloaded.state_dict(), os.path.join(tmp_dirname, "model.bin"))

                # serializing with safetensors works
                save_file(model_unloaded.state_dict(), os.path.join(tmp_dirname, "model.safetensors"))

    def _test_mixed_adapter_batches(self, model_id, config_cls, config_kwargs):
        # Test for mixing different adapters in a single batch by passing the adapter_names argument
        if config_cls not in (LoraConfig,):
            return pytest.skip(f"Mixed adapter batches not supported for {config_cls}")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        torch.manual_seed(0)
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config, adapter_name="adapter0").eval()
            model.add_adapter("adapter1", config)
            model = model.to(self.torch_device).eval()

        self.perturb_trainable_token_weights_if_used(model, config_kwargs, adapter_name="adapter0")
        self.perturb_trainable_token_weights_if_used(model, config_kwargs, adapter_name="adapter1")

        dummy_input = self.prepare_inputs_for_testing()
        # ensure that we have at least 3 samples for this test
        dummy_input = {k: torch.cat([v for _ in range(3)]) for k, v in dummy_input.items()}

        with torch.inference_mode():
            with model.disable_adapter():
                output_base = model(**dummy_input)[0]
                logits_base = model.generate(**dummy_input, return_dict_in_generate=True, output_scores=True).scores[0]

        model.set_adapter("adapter0")
        with torch.inference_mode():
            output_adapter0 = model(**dummy_input)[0]
            logits_adapter0 = model.generate(**dummy_input, return_dict_in_generate=True, output_scores=True).scores[0]

        model.set_adapter("adapter1")
        with torch.inference_mode():
            output_adapter1 = model(**dummy_input)[0]
            logits_adapter1 = model.generate(**dummy_input, return_dict_in_generate=True, output_scores=True).scores[0]

        atol, rtol = 1e-4, 1e-4
        # sanity check that there are enough outputs and that they are different
        assert len(output_base) == len(output_adapter0) == len(output_adapter1) >= 3
        assert len(logits_base) == len(logits_adapter0) == len(logits_adapter1) >= 3
        assert not torch.allclose(output_base, output_adapter0, atol=atol, rtol=rtol)
        assert not torch.allclose(output_base, output_adapter1, atol=atol, rtol=rtol)
        assert not torch.allclose(output_adapter0, output_adapter1, atol=atol, rtol=rtol)
        assert not torch.allclose(logits_base, logits_adapter0, atol=atol, rtol=rtol)
        assert not torch.allclose(logits_base, logits_adapter1, atol=atol, rtol=rtol)
        assert not torch.allclose(logits_adapter0, logits_adapter1, atol=atol, rtol=rtol)

        # alternate between base model, adapter0, and adapter1
        adapters = ["__base__", "adapter0", "adapter1"]
        dummy_input["adapter_names"] = [adapters[i % 3] for i in (range(len(dummy_input["input_ids"])))]

        with torch.inference_mode():
            output_mixed = model(**dummy_input)[0]
            logits_mixed = model.generate(**dummy_input, return_dict_in_generate=True, output_scores=True).scores[0]

        assert torch.allclose(output_base[::3], output_mixed[::3], atol=atol, rtol=rtol)
        assert torch.allclose(output_adapter0[1::3], output_mixed[1::3], atol=atol, rtol=rtol)
        assert torch.allclose(output_adapter1[2::3], output_mixed[2::3], atol=atol, rtol=rtol)
        assert torch.allclose(logits_base[::3], logits_mixed[::3], atol=atol, rtol=rtol)
        assert torch.allclose(logits_adapter0[1::3], logits_mixed[1::3], atol=atol, rtol=rtol)
        assert torch.allclose(logits_adapter1[2::3], logits_mixed[2::3], atol=atol, rtol=rtol)

    def _test_generate_with_mixed_adapter_batches_and_beam_search(self, model_id, config_cls, config_kwargs):
        # Test generating with beam search and with mixing different adapters in a single batch by passing the
        # adapter_names argument. See #2283.
        if config_cls not in (LoraConfig,):
            return pytest.skip(f"Mixed adapter batches not supported for {config_cls}")

        if config_kwargs.get("trainable_token_indices", None) is not None:
            # for some configurations this test will fail since the adapter values don't differ.
            # this is probably a problem with the test setup and not with the implementation.
            return pytest.skip("Trainable token indices is not supported here (yet).")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        torch.manual_seed(0)
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config, adapter_name="adapter0").eval()
            model.add_adapter("adapter1", config)

            # In contrast to forward, for generate, it can sometimes happen that we get the same results as the base model
            # even with LoRA applied because the impact of LoRA is not big enough. Therefore, use this "trick" to make LoRA
            # stronger.
            for name, param in model.named_parameters():
                if model.base_model.prefix in name:
                    param.data.mul_(10.0)

            model = model.to(self.torch_device).eval()

            dummy_input = self.prepare_inputs_for_testing()
            # ensure that we have at least 3 samples for this test
            dummy_input = {k: torch.cat([v for _ in range(3)]) for k, v in dummy_input.items()}

            gen_kwargs = {**dummy_input, "max_length": 20, "num_beams": 10, "early_stopping": True}
            with torch.inference_mode():
                with model.disable_adapter():
                    gen_base = model.generate(**gen_kwargs)

            model.set_adapter("adapter0")
            with torch.inference_mode():
                gen_adapter0 = model.generate(**gen_kwargs)

            model.set_adapter("adapter1")
            with torch.inference_mode():
                gen_adapter1 = model.generate(**gen_kwargs)

        def remove_padding(seq, pad_value):
            lst = list(seq)
            while lst and (lst[-1] == pad_value):
                lst.pop()
            return lst

        def gens_are_same(gen0, gen1):
            # Special function to compare generations. We cannot use torch.allclose it will raise an error when sequence
            # lengths differ. Morevoer, we need to remove the padding from the sequences. This is because, even though
            # normally identical sequences should have the same length, when we do mixed adapter batches, each sample
            # will be padded to the longest sequence in that mixed batch, which can be different from the longest
            # sequence without mixed adapter batches.
            pad_value = model.config.eos_token_id
            for sample0, sample1 in zip(gen0, gen1):
                sample0 = remove_padding(sample0, pad_value)
                sample1 = remove_padding(sample1, pad_value)
                if (len(sample0) != len(sample1)) or (sample0 != sample1):
                    # at least one sample differs, the generations are not identical
                    return False
            return True

        # sanity check that there are enough outputs and that they are different
        assert len(gen_base) == len(gen_adapter0) == len(gen_adapter1)
        assert len(gen_adapter1) >= 3
        assert not gens_are_same(gen_base, gen_adapter0)
        assert not gens_are_same(gen_base, gen_adapter1)
        assert not gens_are_same(gen_adapter0, gen_adapter1)

        # alternate between base model, adapter0, and adapter1
        adapters = ["__base__", "adapter0", "adapter1"]
        gen_kwargs["adapter_names"] = [adapters[i % 3] for i in (range(len(dummy_input["input_ids"])))]

        with torch.inference_mode():
            gen_mixed = model.generate(**gen_kwargs)

        assert gens_are_same(gen_base[::3], gen_mixed[::3])
        assert gens_are_same(gen_adapter0[1::3], gen_mixed[1::3])
        assert gens_are_same(gen_adapter1[2::3], gen_mixed[2::3])

    def _test_generate(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
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

        with hub_online_once(model_id):
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

        with hub_online_once(model_id):
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

        with hub_online_once(model_id):
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
                if (parameter_prefix in n) or ("modules_to_save" in n) or ("token_adapter.trainable_tokens" in n):
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
        with hub_online_once(model_id):
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
                model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(
                    self.torch_device
                )

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
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config)
            model = model.to(self.torch_device)

            inputs = self.prepare_inputs_for_testing()

            # check if `training` works
            output = model(**inputs)[0]
            logits = output[0]

            loss = output.sum()
            loss.backward()

            has_trainable_tokens = config_kwargs.get("trainable_token_indices", None) is not None
            nb_trainable = 0

            for n, param in model.named_parameters():
                if model.prefix in n or (has_trainable_tokens and "trainable_tokens" in n):
                    assert param.grad is not None
                    nb_trainable += 1
                else:
                    assert param.grad is None

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(
                    self.torch_device
                )

                logits_from_pretrained = model_from_pretrained(**inputs)[0][0]
                assert torch.allclose(logits, logits_from_pretrained, atol=1e-4, rtol=1e-4)

            # check the nb of trainable params again but without layers_to_transform
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)
            nb_trainable_all = 0

            for n, param in model.named_parameters():
                if model.prefix in n or (has_trainable_tokens and "trainable_tokens" in n):
                    nb_trainable_all += 1

            mod_list = next((m for m in model.modules() if isinstance(m, torch.nn.ModuleList)), None)
            if mod_list and len(mod_list) == 1:
                # there is only a single layer
                assert nb_trainable == nb_trainable_all
            else:
                # more than 1 layer, i.e. setting layers_to_transform=[0] should target fewer layers
                assert nb_trainable < nb_trainable_all

    def _test_training_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        if config_cls == PrefixTuningConfig:
            return pytest.skip(f"Test not applicable for {config_cls}")

        if (config_cls == AdaLoraConfig) and ("roberta" in model_id.lower()):
            # TODO: no gradients on the "dense" layer, other layers work, not sure why
            self.skipTest("AdaLora with RoBERTa does not work correctly")

        if (config_cls == OFTConfig) and ("deberta" in model_id.lower()):
            # TODO: no gradients on the "dense" layer, other layers work, not sure why
            self.skipTest("OFT with Deberta does not work correctly")

        with hub_online_once(model_id):
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

            for n, param in model.named_parameters():
                if "prompt_encoder." in n:  # prompt tuning methods
                    if not issubclass(config_cls, CPTConfig):
                        assert param.grad is not None
                    elif (
                        "delta_embedding" in n
                    ):  # delta_embedding is the embedding that should be updated with grads in CPT
                        assert param.grad is not None
                elif hasattr(model, "prefix") and (model.prefix in n):  # non-prompt tuning methods
                    assert param.grad is not None
                elif "trainable_tokens_" in n:  # trainable tokens layer
                    assert param.grad is not None
                else:
                    assert param.grad is None

    def _test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig, VBLoRAConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        with hub_online_once(model_id):
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

        with hub_online_once(model_id):
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

            if issubclass(config_cls, CPTConfig):
                parameters = []
                for name, param in model.prompt_encoder.named_parameters():
                    if name != "default.embedding.weight":
                        parameters.append(param)
            else:
                parameters = model.prompt_encoder.parameters()

            # check that prompt encoder has grads
            for param in parameters:
                assert param.grad is not None

    def _test_delete_adapter(self, model_id, config_cls, config_kwargs):
        supported_peft_types = [
            PeftType.LORA,
            PeftType.LOHA,
            PeftType.LOKR,
            PeftType.IA3,
            PeftType.OFT,
            PeftType.BOFT,
            PeftType.VERA,
            PeftType.FOURIERFT,
            PeftType.HRA,
            PeftType.VBLORA,
            PeftType.BONE,
        ]
        # IA3 does not support deleting adapters yet, but it just needs to be added
        # AdaLora does not support multiple adapters
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.peft_type not in supported_peft_types:
            return pytest.skip(f"Test not applicable for {config.peft_type}")

        with hub_online_once(model_id):
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
                attributes_to_check = getattr(target, "adapter_layer_names", []) + getattr(
                    target, "other_param_names", []
                )
                for attr in attributes_to_check:
                    assert adapter_to_delete not in getattr(target, attr)

            # check auxiliary modules
            for module in model.modules():
                if isinstance(module, AuxiliaryTrainingWrapper):
                    assert adapter_to_delete not in module._adapters
                    assert module.active_adapters == ["default"]
                if isinstance(module, ModulesToSaveWrapper):
                    assert adapter_to_delete not in module.modules_to_save
                elif isinstance(module, TrainableTokensWrapper):
                    assert adapter_to_delete not in module.token_adapter.trainable_tokens_delta
                    assert adapter_to_delete not in module.token_adapter.trainable_tokens_original

            # check that we can also delete the last remaining adapter
            model.delete_adapter("default")
            assert "default" not in model.peft_config
            assert model.active_adapters == []

            for module in model.modules():
                if isinstance(module, AuxiliaryTrainingWrapper):
                    assert "default" not in module._adapters
                    assert module.active_adapters == []
                if isinstance(module, ModulesToSaveWrapper):
                    assert "default" not in module.modules_to_save
                elif isinstance(module, TrainableTokensWrapper):
                    assert "default" not in module.token_adapter.trainable_tokens_delta
                    assert "default" not in module.token_adapter.trainable_tokens_original

            input = self.prepare_inputs_for_testing()
            # note: we cannot call model(**input) because PeftModel always expects there to be at least one adapter
            model.base_model(**input)  # should not raise an error

    def _test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        # same as test_delete_adapter, but this time an inactive adapter is deleted
        supported_peft_types = [
            PeftType.LORA,
            PeftType.LOHA,
            PeftType.LOKR,
            PeftType.IA3,
            PeftType.OFT,
            PeftType.BOFT,
            PeftType.FOURIERFT,
            PeftType.HRA,
            PeftType.VBLORA,
            PeftType.BONE,
        ]
        # IA3 does not support deleting adapters yet, but it just needs to be added
        # AdaLora does not support multiple adapters
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.peft_type not in supported_peft_types:
            return pytest.skip(f"Test not applicable for {config.peft_type}")

        with hub_online_once(model_id):
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
                attributes_to_check = getattr(target, "adapter_layer_names", []) + getattr(
                    target, "other_param_names", []
                )
                for attr in attributes_to_check:
                    assert adapter_to_delete not in getattr(target, attr)

            # check auxiliary modules
            for module in model.modules():
                if isinstance(module, AuxiliaryTrainingWrapper):
                    assert adapter_to_delete not in module._adapters
                    assert module.active_adapters == ["default"]
                if isinstance(module, ModulesToSaveWrapper):
                    assert adapter_to_delete not in module.modules_to_save
                elif isinstance(module, TrainableTokensWrapper):
                    assert adapter_to_delete not in module.token_adapter.trainable_tokens_delta
                    assert adapter_to_delete not in module.token_adapter.trainable_tokens_original

            # check that we can also delete the last remaining adapter
            model.delete_adapter("default")
            assert "default" not in model.peft_config
            assert model.active_adapters == []

            for module in model.modules():
                if isinstance(module, AuxiliaryTrainingWrapper):
                    assert "default" not in module._adapters
                    assert module.active_adapters == []
                if isinstance(module, ModulesToSaveWrapper):
                    assert "default" not in module.modules_to_save
                elif isinstance(module, TrainableTokensWrapper):
                    assert "default" not in module.token_adapter.trainable_tokens_delta
                    assert "default" not in module.token_adapter.trainable_tokens_original

            input = self.prepare_inputs_for_testing()
            # note: we cannot call model(**input) because PeftModel always expects there to be at least one adapter
            model.base_model(**input)  # should not raise an error

    def _test_delete_unknown_adapter_raises(self, model_id, config_cls, config_kwargs):
        # Check that we get a nice error message when trying to delete an adapter that does not exist.
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            adapter_to_delete = "delete_me"
            model = get_peft_model(model, config)

            msg = "Adapter unknown-adapter does not exist"
            with pytest.raises(ValueError, match=msg):
                model.delete_adapter("unknown-adapter")

    def _test_unload_adapter(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
        num_params_base = len(model.state_dict())

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        if config.peft_type not in (
            "LORA",
            "ADALORA",
            "IA3",
            "BOFT",
            "OFT",
            "VERA",
            "FOURIERFT",
            "HRA",
            "VBLORA",
            "RANDLORA",
            "BONE",
            "C3A",
        ):
            with pytest.raises(AttributeError):
                model = model.unload()
        else:
            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

            dummy_input = self.prepare_inputs_for_testing()
            logits_with_adapter = model(**dummy_input)[0]

            with hub_online_once(model_id):
                transformers_model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
                logits_transformers = transformers_model(**dummy_input)[0]

                model.eval()
                model = model.unload()
                logits_unload = model(**dummy_input)[0]
                num_params_unloaded = len(model.state_dict())

                # check that PEFT layers are completely removed
                assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
                assert not torch.allclose(logits_with_adapter, logits_unload, atol=1e-10, rtol=1e-10)
                assert torch.allclose(logits_transformers, logits_unload, atol=1e-4, rtol=1e-4)
                assert num_params_base == num_params_unloaded

    def _test_weighted_combination_of_adapters_lora(self, model, config, adapter_list, weight_list):
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

    def _test_weighted_combination_of_adapters_ia3(self, model, config, adapter_list, weight_list):
        model.add_adapter(adapter_list[1], config)
        model.add_adapter(adapter_list[2], config)
        model = model.to(self.torch_device)

        # test re-weighting single adapter
        model.add_weighted_adapter([adapter_list[0]], [weight_list[0]], "single_adapter_reweighting")

        # test re-weighting with multiple adapters
        model.add_weighted_adapter(adapter_list[1:], weight_list[1:], "multi_adapter_reweighting")

        new_adapters = [
            "single_adapter_reweighting",
            "multi_adapter_reweighting",
        ]
        for new_adapter in new_adapters:
            assert new_adapter in model.peft_config

        dummy_input = self.prepare_inputs_for_testing()
        model.eval()
        for adapter_name in new_adapters:
            # ensuring new adapters pass the forward loop
            model.set_adapter(adapter_name)
            assert model.active_adapter == adapter_name
            assert model.active_adapters == [adapter_name]
            model(**dummy_input)[0]

    def _test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, AdaLoraConfig):
            # AdaLora does not support adding more than 1 adapter
            return pytest.skip(f"Test not applicable for {config_cls}")
        if model_id.endswith("qwen2"):
            # Qwen2 fails with weighted adapter combinations using SVD
            return pytest.skip(f"Test does not work with model {model_id}")
        if "gemma" in model_id.lower():
            return pytest.skip("Combining Gemma adapters with SVD is currently failing")

        adapter_list = ["adapter1", "adapter_2", "adapter_3"]
        weight_list = [0.5, 1.5, 1.5]
        # Initialize the config
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        if not isinstance(config, (LoraConfig, IA3Config)):
            # This test is only applicable for Lora and IA3 configs
            return pytest.skip(f"Test not applicable for {config}")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config, adapter_list[0])

            if isinstance(config, LoraConfig):
                self._test_weighted_combination_of_adapters_lora(model, config, adapter_list, weight_list)
            elif isinstance(config, IA3Config):
                self._test_weighted_combination_of_adapters_ia3(model, config, adapter_list, weight_list)
            else:
                pytest.skip(f"Test not applicable for {config}")

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
        with hub_online_once(model_id):
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

            # trainable_token_indices doesn't have support for `init_weights` so we have to do this manually
            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

            output_peft = get_output(peft_model)

            # first check trivial case is not true that peft does not affect the output; for this to work, init_weight
            # must be False (if the config supports it)
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
        # When trying to add multiple adapters with bias in Lora, AdaLora or BOFTConfig, an error should be
        # raised. Also, the peft model should not be left in a half-initialized state.
        if not issubclass(config_cls, (LoraConfig, AdaLoraConfig, BOFTConfig)):
            return pytest.skip(f"Test not applicable for {config_cls}")

        with hub_online_once(model_id):
            config_kwargs = config_kwargs.copy()
            config_kwargs["bias"] = "all"
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )

            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config, "adapter0")

            if config_cls == LoraConfig or config_cls == AdaLoraConfig:
                with pytest.raises(ValueError):
                    model.add_adapter("adapter1", replace(config, r=20))

            if config_cls == BOFTConfig:
                with pytest.raises(ValueError):
                    model.add_adapter("adapter1", replace(config, boft_block_num=1, boft_block_size=0))

            # (superficial) test that the model is not left in a half-initialized state when adding an adapter fails
            assert "adapter1" not in model.peft_config
            assert "adapter1" not in model.base_model.peft_config

    def _test_passing_input_embeds_works(self, test_name, model_id, config_cls, config_kwargs):
        # https://github.com/huggingface/peft/issues/727
        with hub_online_once(model_id):
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
