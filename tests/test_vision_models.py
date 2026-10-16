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

# This is not a full on test suite of vision models, since we already run many tests on dummy models with Conv2d layers
# and on stable diffusion models. Instead, this file contains specific tests for bugs that have been found in the past.
import copy
import gc

import numpy as np
import pytest
import torch
from accelerate.utils.memory import clear_device_cache
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    HiraConfig,
    HRAConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PeftModel,
    PrefixTuningConfig,
    get_peft_model,
)

from .testing_utils import load_cat_image


CONFIGS = {
    "lora": LoraConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "dora": LoraConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"], use_dora=True),
    "loha": LoHaConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "lokr": LoKrConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "oft": OFTConfig(
        r=1, oft_block_size=0, target_modules=["convolution"], modules_to_save=["classifier", "normalization"]
    ),
    "hra": HRAConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    # Cannot target multiple layers with BOFT because some convolutional kernel dimensions vary and there is no common
    # denominator for the boft_block_size except 1, but using 1 results in an error in the fbd_cuda kernel:
    # > Error in forward_fast_block_diag_cuda_kernel: an illegal memory access was encountered
    "boft": BOFTConfig(
        target_modules=["0.layer.0.convolution"], modules_to_save=["classifier", "normalization"], boft_block_size=2
    ),
    "adalora": AdaLoraConfig(
        target_modules=["convolution"], modules_to_save=["classifier", "normalization"], total_step=1
    ),
    "hira": HiraConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "ia3": IA3Config(
        target_modules=["convolution"], feedforward_modules=[], modules_to_save=["classifier", "normalization"]
    ),
    "ia3_ff": IA3Config(
        target_modules=["convolution"],
        feedforward_modules=["convolution"],
        modules_to_save=["classifier", "normalization"],
    ),
}


# Ensure that models like Llava that pass past_key_values automatically do not fail, see #1938
class TestPastKV:
    def test_past_kv(self):
        model_id = "peft-internal-testing/tiny-LlavaForConditionalGeneration"
        prompt = "USER: <image>\nWhat are these?\nASSISTANT:"

        # prepare model and inputs
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        raw_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt")

        # get peft model
        peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
        model = get_peft_model(model, peft_config)
        # check that this does not raise
        model(**inputs, output_hidden_states=True)


class TestResnet:
    # saftensors version of the hf-internal-testing model
    model_id = "peft-internal-testing/tiny-random-ResNetForImageClassification"
    cat_image = load_cat_image()  # for caching

    @pytest.fixture(autouse=True)
    def teardown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        clear_device_cache(garbage_collection=True)
        gc.collect()

    @pytest.fixture(scope="class")
    def image_processor(self):
        image_processor = AutoImageProcessor.from_pretrained(self.model_id)
        return image_processor

    @pytest.fixture(scope="class")
    def data(self, image_processor):
        return image_processor(self.cat_image, return_tensors="pt")

    @pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
    def test_model_with_batchnorm_reproducibility(self, config, tmp_path, data):
        # see 1732
        torch.manual_seed(0)
        model = AutoModelForImageClassification.from_pretrained(self.model_id)
        model = get_peft_model(model, config)

        # record outputs before training
        model.eval()
        with torch.inference_mode():
            output_before = model(**data)
        model.train()

        # train the model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_size = 4
        max_steps = 5 * batch_size
        labels = torch.zeros(1, 3)
        labels[0, 1] = 1
        for i in range(0, max_steps, batch_size):
            optimizer.zero_grad()
            outputs = model(**data, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # record outputs after training
        model.eval()
        with torch.inference_mode():
            output_after = model(**data)
        assert torch.isfinite(output_after.logits).all()
        atol, rtol = 1e-4, 1e-4
        # sanity check: model was updated
        assert not torch.allclose(output_before.logits, output_after.logits, atol=atol, rtol=rtol)

        # check saving the model and loading it
        model.save_pretrained(tmp_path)
        del model

        torch.manual_seed(0)
        model = AutoModelForImageClassification.from_pretrained(self.model_id)
        model = PeftModel.from_pretrained(model, tmp_path).eval()
        with torch.inference_mode():
            output_loaded = model(**data)
        assert torch.allclose(output_after.logits, output_loaded.logits, atol=atol, rtol=rtol)

        # ensure that the checkpoint file contains the buffers
        model_running_mean = len([k for k in model.state_dict().keys() if "running_mean" in k])
        state_dict = load_file(tmp_path / "adapter_model.safetensors")
        checkpoint_running_mean = len([k for k in state_dict.keys() if "running_mean" in k])
        # note that the model has twice as many "running_mean", as there is one copy per ModulesToSaveWrapper, we need
        # to multiply by 2 to get the same number
        assert model_running_mean == checkpoint_running_mean * 2


class TestConvArguments:
    """
    Generic tests for PEFT methods adapting Conv2d layers.

    Omitting `groups` for now, as that easily becomes more complex.
    """

    conv_kwargs = [
        {"kernel_size": 3},
        {"kernel_size": (3, 3)},
        {"kernel_size": (1, 1)},
        {"kernel_size": (3, 2)},
        {"bias": False},  # default: True
        {"dilation": 2},  # default: 1
        {"dilation": (2, 1)},
        {"dilation": (1, 2)},
        {"stride": 2, "padding": 1},  # default stride: 1
        {"stride": (2, 1), "padding": 1},
        {"padding": 1},  # default: 0
        {"padding": (2, 0)},
        {"padding": 1, "padding_mode": "reflect"},  # default mode: "zeros"
        # combining
        {"kernel_size": 3, "bias": False, "dilation": 2, "stride": 2, "padding": 1, "padding_mode": "reflect"},
    ]

    def get_conv_model(self, in_channels=4, out_channels=16, kernel_size=3, **kwargs):

        class ModelConv2D(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
                super().__init__()
                self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

            def forward(self, x):
                return self.convolution(x)

        return ModelConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs)

    def get_inputs(
        self,
        batch_size=4,
        in_channels=4,
        kernel_size=3,
        dilation=1,
        **kwargs,
    ):
        kws_to_ignore = ("bias", "padding", "padding_mode", "stride")
        [kwargs.pop(kw, None) for kw in kws_to_ignore if kw in kwargs]
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")

        torch.manual_seed(0)

        def pair(x) -> tuple[int, int]:
            output = x if isinstance(x, (list, tuple)) else (x, x)
            assert len(output) == 2, f"Expected a pair, got {output}"
            return output

        kernel_size = pair(kernel_size)
        dilation = pair(dilation)
        # An unpadded kernel must fit in the input for OFT's rotation. Padding and stride can still change the output.
        spatial_size = tuple(d * (k - 1) + 1 for k, d in zip(kernel_size, dilation))

        return torch.randn(batch_size, in_channels, *spatial_size)

    def should_skip(self, config, conv_kwargs):
        kernel_size = conv_kwargs.get("kernel_size", 3)
        kernel_is_square = isinstance(kernel_size, int) or len(set(kernel_size)) == 1
        return config.peft_type in {"OFT", "HRA", "BOFT"} and not kernel_is_square

    @pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
    @pytest.mark.parametrize("conv_kwargs", conv_kwargs)
    def test_base_output_preserved_in_forward(self, config, conv_kwargs):
        # Compare raw convolution outputs so that shape errors and missing convolution arguments are visible.
        if self.should_skip(config, conv_kwargs):
            pytest.skip("This method assumes square convolution kernels throughout its weight handling")

        torch.manual_seed(0)
        model = self.get_conv_model(**conv_kwargs)

        inputs = self.get_inputs(**conv_kwargs)
        with torch.inference_mode():
            output_base = model(inputs)

        config = copy.deepcopy(config)
        config.target_modules = {"convolution"}

        if config.peft_type == "OFT" and conv_kwargs.get("dilation", 1) != 1:
            with pytest.raises(ValueError, match="Conv2d with dilation > 1 is not supported by OFT"):
                model = get_peft_model(model, config).eval()
            return

        model = get_peft_model(model, config).eval()
        with torch.inference_mode():
            output_peft = model(inputs)

        atol, rtol = 1e-4, 1e-4
        assert torch.allclose(output_base, output_peft, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
    @pytest.mark.parametrize("conv_kwargs", conv_kwargs)
    def test_base_output_preserved_after_merging(self, config, conv_kwargs):
        # Same test as test_base_output_preserved_in_forward, but after merging the PEFT model into the base model.
        if self.should_skip(config, conv_kwargs):
            pytest.skip("This method assumes square convolution kernels throughout its weight handling")

        torch.manual_seed(0)
        model = self.get_conv_model(**conv_kwargs)

        inputs = self.get_inputs(**conv_kwargs)
        with torch.inference_mode():
            output_base = model(inputs)

        config = copy.deepcopy(config)
        config.target_modules = {"convolution"}

        if (config.peft_type == "OFT") and (conv_kwargs.get("dilation", 1) != 1):
            with pytest.raises(ValueError, match="Conv2d with dilation > 1 is not supported by OFT"):
                model = get_peft_model(model, config).eval()
            return

        model = get_peft_model(model, config).eval()
        model.merge_adapter()

        with torch.inference_mode():
            output_merged = model(inputs)

        atol, rtol = 1e-4, 1e-4
        assert torch.allclose(output_base, output_merged, atol=atol, rtol=rtol)
