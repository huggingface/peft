# Copyright 2024-present the HuggingFace Inc. team.
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
import platform
import shutil
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import pytest
import torch
from parameterized import parameterized
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
)

from .testing_common import PeftCommonTester, PeftTestConfigManager


# only the PEFT methods that are explicitly supported will be tested for merging
PEFT_METHODS_SUPPORTING_MERGING = [LoraConfig]


def filter_supported_methods_supporting_merging(test_list):
    return [test for test in test_list if any(test[2] is cls for cls in PEFT_METHODS_SUPPORTING_MERGING)]


# only test a single model, it's already slow as is
PEFT_DECODER_MODELS_TO_TEST = [
    "hf-internal-testing/tiny-random-OPTForCausalLM",
]

FULL_GRID = {
    "model_ids": PEFT_DECODER_MODELS_TO_TEST,
    "task_type": "CAUSAL_LM",
}


def make_automodel_proxy(weights: str):
    """Instantiate a quanto-quantized transformers model."""
    from transformers import QuantoConfig

    class QuantoModelProxy:
        @classmethod
        def from_pretrained(self, *args, **kwargs):
            quantization_config = QuantoConfig(weights=weights)
            model = AutoModelForCausalLM.from_pretrained(*args, quantization_config=quantization_config, **kwargs)
            return model

    return QuantoModelProxy


@unittest.skipIf(platform.system() == "Darwin", "Tests are skipped on macOS")
class BasePeftQuantoModelTester:
    r"""Base class implementing tests for quanto-quantized models.

    This class is based on PeftDecoderModelTester with some quanto-specific edits, especially for the merging tests,
    which are less precise due to the quantization.

    Subclasses should implement the attributes below.
    """

    # The weights argument for quanto, should be "int2", "int4", or "int8"
    weights = "MISSING"
    # transformers class should be make_automodel_proxy(weights=weights)
    transformers_class = "MISSING"
    # expected minimum correlation between logits before and after merging
    # subclasses should override this with a float between 0 and 1
    min_correlation = "MISSING"
    # the allowed tolerance for comparing the output tensors
    tol = "MISSING"

    def _get_correlation_matrix(self, *tensors):
        return torch.corrcoef(torch.stack([t.flatten() for t in tensors]))

    def check_tensors_approximately_equal(self, *tensors):
        # Strict equality checks will fail due to the quantization, so we check:
        # 1. The correlation between the tensors is high
        # 2. Tensor equality after removing 1% of highest and lowest outliers
        cc_matrix = self._get_correlation_matrix(*tensors)
        assert cc_matrix.min() > self.min_correlation

        for tensor0, tensor1 in zip(tensors, tensors[1:]):
            tensor0, tensor1 = tensor0.flatten(), tensor1.flatten()
            diff = tensor0 - tensor1
            indices = torch.argsort(diff)
            # remove 1% outliers on both ends
            indices = indices[len(indices) // 100 : -len(indices) // 100]
            tensor0, tensor1 = tensor0[indices], tensor1[indices]
            assert torch.allclose(tensor0, tensor1, atol=self.tol, rtol=self.tol)

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)

        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return input_dict

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_attributes_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_adapter_name(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_prepare_for_training_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_prompt_tuning_text_prepare_for_training(self, test_name, model_id, config_cls, config_kwargs):
        # Test that prompt tuning works with text init
        if config_cls != PromptTuningConfig:
            return pytest.skip(f"This test does not apply to {config_cls}")

        config_kwargs = config_kwargs.copy()
        config_kwargs["prompt_tuning_init"] = PromptTuningInit.TEXT
        config_kwargs["prompt_tuning_init_text"] = "This is a test prompt."
        config_kwargs["tokenizer_name_or_path"] = model_id
        self._test_prepare_for_training(model_id, config_cls, config_kwargs)

    def test_prompt_tuning_text_tokenizer_kwargs(self):
        # Allow users to pass additional arguments to Tokenizer.from_pretrained
        # Fix for #1032
        mock = Mock()
        orig_from_pretrained = AutoTokenizer.from_pretrained

        def mock_autotokenizer_from_pretrained(*args, **kwargs):
            mock(*args, **kwargs)
            return orig_from_pretrained(config.tokenizer_name_or_path)

        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        config = PromptTuningConfig(
            base_model_name_or_path=model_id,
            tokenizer_name_or_path=model_id,
            num_virtual_tokens=10,
            prompt_tuning_init=PromptTuningInit.TEXT,
            task_type="CAUSAL_LM",
            prompt_tuning_init_text="This is a test prompt.",
            tokenizer_kwargs={"trust_remote_code": True, "foo": "bar"},
        )
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        with patch("transformers.AutoTokenizer.from_pretrained", mock_autotokenizer_from_pretrained):
            model = get_peft_model(model, config)

        expected_call = call(model_id, trust_remote_code=True, foo="bar")
        assert mock.call_args == expected_call

    def test_prompt_tuning_config_invalid_args(self):
        # Raise an error when tokenizer_kwargs is used with prompt_tuning_init!='TEXT', because this argument has no
        # function in that case
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        with pytest.raises(ValueError, match="tokenizer_kwargs only valid when using prompt_tuning_init='TEXT'."):
            PromptTuningConfig(
                base_model_name_or_path=model_id,
                tokenizer_name_or_path=model_id,
                num_virtual_tokens=10,
                task_type="CAUSAL_LM",
                prompt_tuning_init_text="This is a test prompt.",
                prompt_tuning_init=PromptTuningInit.RANDOM,  # <= should not be used together with tokenizer_kwargs
                tokenizer_kwargs={"trust_remote_code": True, "foo": "bar"},
            )

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_pickle(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs, safe_serialization=False)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_selected_adapters(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_save_pretrained_selected_adapters_pickle(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained_selected_adapters(model_id, config_cls, config_kwargs, safe_serialization=False)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_from_pretrained_config_construction(self, test_name, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "vera_kwargs": {"init_weights": [False]},
                "fourierft_kwargs": {"init_weights": [False]},
                "hra_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    def test_merge_layers(self, test_name, model_id, config_cls, config_kwargs):
        # Not using PeftCommonTester for merging tests as merging is too imprecise. So instead of checking we use a
        # custom check that relies on correlation and outlier removal
        torch.manual_seed(0)

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.is_prompt_learning:
            pytest.skip("Prompt learning models do not support merging.")

        model = self.transformers_class.from_pretrained(model_id)
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

        self.check_tensors_approximately_equal(logits, logits_merged, logits_unmerged, logits_merged_unloaded)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "vera_kwargs": {"init_weights": [False]},
                "fourierft_kwargs": {"init_weights": [False]},
                "hra_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    # TODO: enable if/when deepcopy-ing is supported
    @pytest.mark.skip("Quanto does not work (yet) with deepcopy-ing")
    def test_merge_layers_multi(self, test_name, model_id, config_cls, config_kwargs):
        # Not using PeftCommonTester for merging tests as merging is too imprecise. So instead of checking we use a
        # custom check that relies on correlation and outlier removal
        # NOTE: don't use with `torch.inference_mode()`, see: https://github.com/huggingface/optimum-quanto/issues/304
        torch.manual_seed(0)

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.is_prompt_learning:
            pytest.skip("Prompt learning models do not support merging.")

        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)

        model = model.to(self.torch_device)

        dummy_input = self.prepare_inputs_for_testing()
        model.eval()

        logits_adapter_1 = model(**dummy_input)[0]

        model.add_adapter("adapter-2", config)
        model.set_adapter("adapter-2")
        model.eval()

        logits_adapter_2 = model(**dummy_input)[0]

        assert not torch.allclose(logits_adapter_1, logits_adapter_2, atol=1e-3, rtol=1e-3)

        model.set_adapter("default")

        logits_adapter_1_after_set = model(**dummy_input)[0]

        self.check_tensors_approximately_equal(logits_adapter_1, logits_adapter_1_after_set)

        model_copy = copy.deepcopy(model)
        model_copy_2 = copy.deepcopy(model)
        model_merged_all = model.merge_and_unload(adapter_names=["adapter-2", "default"])

        logits_merged_all = model_merged_all(**dummy_input)[0]

        assert not torch.allclose(logits_merged_all, logits_adapter_2, atol=1e-3, rtol=1e-3)
        assert not torch.allclose(logits_merged_all, logits_adapter_1, atol=1e-3, rtol=1e-3)

        model_merged_adapter_2 = model_copy.merge_and_unload(adapter_names=["adapter-2"])

        logits_merged_adapter_2 = model_merged_adapter_2(**dummy_input)[0]

        self.check_tensors_approximately_equal(logits_adapter_2, logits_merged_adapter_2)

        model_merged_adapter_default = model_copy_2.merge_and_unload(adapter_names=["default"])
        logits_merged_adapter_default = model_merged_adapter_default(**dummy_input)[0]

        self.check_tensors_approximately_equal(logits_adapter_1, logits_merged_adapter_default)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    def test_merge_layers_nan(self, test_name, model_id, config_cls, config_kwargs):
        # Not using PeftCommonTester for merging tests as merging is too imprecise. So instead of checking we use a
        # custom check that relies on correlation and outlier removal
        torch.manual_seed(0)

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.is_prompt_learning:
            pytest.skip("Prompt learning models do not support merging.")

        model = self.transformers_class.from_pretrained(model_id)
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        dummy_input = self.prepare_inputs_for_testing()

        model.eval()

        # This should work
        logits_unmerged = model(**dummy_input)[0]

        model = model.merge_and_unload()
        logits_merged = model(**dummy_input)[0]

        self.check_tensors_approximately_equal(logits_unmerged, logits_merged)

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        prefixes = ["lora_A", "boft_R", "fourierft_spectrum", "hra_u", "hada_w1", "lokr_w1", "ia3_l", "oft_r"]
        prefixes += ["vera_lambda_b"]

        for name, module in model.named_parameters():
            if any(prefix in name for prefix in prefixes):
                module.data[0] = torch.nan

        with pytest.raises(
            ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
        ):
            model = model.merge_and_unload(safe_merge=True)

        for name, module in model.named_parameters():
            if any(prefix in name for prefix in prefixes):
                module.data[0] = torch.inf

        with pytest.raises(
            ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
        ):
            model = model.merge_and_unload(safe_merge=True)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "vera_kwargs": {"init_weights": [False]},
                "fourierft_kwargs": {"init_weights": [False]},
                "hra_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    @pytest.mark.xfail(strict=True)
    def test_load_merge_and_unloaded_model(self, test_name, model_id, config_cls, config_kwargs):
        # Saving and loading a quanto model that has been merged and unloaded does not work correctly. Here is the
        # reason: Quanto requires its own save_pretrained method, which, among others, saves the quantization map.
        # Without it, the model cannot be correctly loaded. To make use of this, we should thus use a quanto
        # QuantizedModel instance instead of a PretrainedModel instance. However, the QuantizedModel instance cannot be
        # used for anything else, e.g. it has no __call__ method. Therefore, we cannot use that in PEFT. Therefore,
        # users need to pass the PretrainedModel instance to get_peft_model, thus we don't have the modified
        # save_pretrained, thus loading the merged and unloaded model does not work.
        from optimum.quanto import QuantizedModelForCausalLM

        torch.manual_seed(0)

        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)
        model = model.merge_and_unload()
        model.eval()

        dummy_input = self.prepare_inputs_for_testing()
        logits = model(**dummy_input)[0]

        # model is a transformers model
        tmp_dirname = tempfile.mkdtemp()
        # note: not using the context manager here because it fails on Windows CI for some reason
        try:
            model.save_pretrained(tmp_dirname)
            # Carefuly: must use QuantizedModelForCausalLM.from_pretrained not AutoModelForCausalLM.from_pretrained
            model_from_pretrained = QuantizedModelForCausalLM.from_pretrained(tmp_dirname).to(self.torch_device)
        finally:
            try:
                shutil.rmtree(tmp_dirname)
            except PermissionError:
                # windows error
                pass

        logits_merged_from_pretrained = model_from_pretrained(**dummy_input)[0]
        self.check_tensors_approximately_equal(logits, logits_merged_from_pretrained)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    def test_mixed_adapter_batches(self, test_name, model_id, config_cls, config_kwargs):
        self._test_mixed_adapter_batches(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate(self, test_name, model_id, config_cls, config_kwargs):
        self._test_generate(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_generate_pos_args(self, test_name, model_id, config_cls, config_kwargs):
        # positional args are supported for PeftModelForCausalLM
        self._test_generate_pos_args(model_id, config_cls, config_kwargs, raises_err=False)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(FULL_GRID),
        filter_params_func=filter_supported_methods_supporting_merging,
    )
    def test_merge_layers_fp16(self, test_name, model_id, config_cls, config_kwargs):
        self._test_merge_layers_fp16(model_id, config_cls, config_kwargs)

    # this fails for a couple of methods (IA³, LoRA, prefix tuning) with segfault on GH CI
    # @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    # def test_generate_half_prec(self, test_name, model_id, config_cls, config_kwargs):
    #     self._test_generate_half_prec(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    @pytest.mark.skip("Quanto raises an error when trying to convert the dtype, skipping test.")
    def test_prefix_tuning_half_prec_conversion(self, test_name, model_id, config_cls, config_kwargs):
        self._test_prefix_tuning_half_prec_conversion(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_training_decoders(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_training_decoders_layer_indexing(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training_layer_indexing(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_training_decoders_gradient_checkpointing(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training_gradient_checkpointing(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_inference_safetensors(self, test_name, model_id, config_cls, config_kwargs):
        self._test_inference_safetensors(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_peft_model_device_map(self, test_name, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_delete_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_delete_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_delete_inactive_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_adding_multiple_adapters_with_bias_raises(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "vera_kwargs": {"init_weights": [False]},
                "fourierft_kwargs": {"init_weights": [False]},
                "hra_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    def test_unload_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_unload_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
        )
    )
    def test_weighted_combination_of_adapters(self, test_name, model_id, config_cls, config_kwargs):
        self._test_weighted_combination_of_adapters(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_training_prompt_learning_tasks(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training_prompt_learning_tasks(model_id, config_cls, config_kwargs)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "vera_kwargs": {"init_weights": [False]},
                "fourierft_kwargs": {"init_weights": [False]},
                "hra_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    def test_disable_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_disable_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(PeftTestConfigManager.get_grid_parameters(FULL_GRID))
    def test_passing_input_embeds_works(self, test_name, model_id, config_cls, config_kwargs):
        self._test_passing_input_embeds_works(test_name, model_id, config_cls, config_kwargs)

    # TODO: enable if/when deepcopy-ing is supported
    @pytest.mark.skip("Quanto does not work (yet) with deepcopy-ing")
    def test_lora_layer_replication(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        config_kwargs = {
            "target_modules": ["down_proj", "up_proj"],
            "task_type": "CAUSAL_LM",
            "lora_dropout": 0.0,
            "layer_replication": [[0, 1], [0, 2], [1, 2]],
        }
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = LoraConfig(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        assert len(model.model.layers), "Expected 2 layers in original model." == 2
        model = get_peft_model(model, config)
        layers = model.base_model.model.model.layers
        assert len(layers) == 4, "Expected 4 layers in adapted model."
        assert (
            layers[0].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            == layers[1].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            and layers[2].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            == layers[3].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
        ), "Expected layers 0-1 and 2-3 to share weights"
        assert (
            layers[0].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
            != layers[2].mlp.up_proj.base_layer.weight.data.storage().data_ptr()
        ), "Expected layers 0 and 2 to have different weights"
        assert (
            layers[0].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
            != layers[1].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
            and layers[2].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
            != layers[3].mlp.up_proj.lora_A.default.weight.data.storage().data_ptr()
        ), "Expected all LoRA adapters to have distinct weights"
        assert len([n for n, _ in model.named_parameters() if ".lora_A." in n]) == 8, (
            "Expected 8 LoRA adapters since we are adding one each for up and down."
        )
        self._test_prepare_for_training(model_id, LoraConfig, config_kwargs)
        self._test_generate(model_id, LoraConfig, config_kwargs)

    def test_prompt_learning_with_grouped_query_attention(self):
        # See 1901, fixes a bug with handling GQA
        model_id = "peft-internal-testing/tiny-dummy-qwen2"
        base_model = AutoModelForCausalLM.from_pretrained(model_id)
        peft_config = PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM")
        model = get_peft_model(base_model, peft_config)
        x = torch.tensor([[1, 2, 3]])
        # does not raise
        model(x)

    @parameterized.expand(
        PeftTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DECODER_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
                "adalora_kwargs": {"init_lora_weights": [False]},
                "ia3_kwargs": {"init_ia3_weights": [False]},
                "boft_kwargs": {"init_weights": [False]},
                "vera_kwargs": {"init_weights": [False]},
                "fourierft_kwargs": {"init_weights": [False]},
                "hra_kwargs": {"init_weights": [False]},
                "task_type": "CAUSAL_LM",
            },
            filter_params_func=filter_supported_methods_supporting_merging,
        )
    )
    def test_quanto_merge_conv2d(self, test_name, model_id, config_cls, config_kwargs):
        torch.manual_seed(0)

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.is_prompt_learning:
            pytest.skip("Prompt learning models do not support merging.")

        config.target_modules = {"seq.0", "seq.2", "seq.4"}
        config.task_type = None

        class ModelConv2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(3, 8, 3),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, 3),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, 3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(800, 64),
                )

            def forward(self, X):
                return self.seq(X)

        model = ModelConv2D()
        model = get_peft_model(model, config)
        model = model.to(self.torch_device)

        dummy_input = torch.randn(5, 3, 16, 16).to(self.torch_device)
        model.eval()
        logits = model(dummy_input)[0]

        model.merge_adapter()
        logits_merged = model(dummy_input)[0]
        model.unmerge_adapter()
        logits_unmerged = model(dummy_input)[0]

        model = model.merge_and_unload()
        logits_merged_unloaded = model(dummy_input)[0]

        self.check_tensors_approximately_equal(logits, logits_merged, logits_unmerged, logits_merged_unloaded)


class PeftQuanto2bitModelTester(unittest.TestCase, PeftCommonTester, BasePeftQuantoModelTester):
    weights = "int2"
    transformers_class = make_automodel_proxy(weights=weights)
    min_correlation = 0.9
    tol = 0.3


class PeftQuanto4bitModelTester(unittest.TestCase, PeftCommonTester, BasePeftQuantoModelTester):
    weights = "int4"
    transformers_class = make_automodel_proxy(weights=weights)
    min_correlation = 0.95
    tol = 1e-2


class PeftQuanto8bitModelTester(unittest.TestCase, PeftCommonTester, BasePeftQuantoModelTester):
    weights = "int8"
    transformers_class = make_automodel_proxy(weights=weights)
    min_correlation = 0.95
    tol = 1e-2
