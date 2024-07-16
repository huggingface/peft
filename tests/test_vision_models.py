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
import gc

import pytest
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModelForImageClassification

from peft import HRAConfig, LoHaConfig, LoKrConfig, LoraConfig, OFTConfig, PeftModel, get_peft_model


CONFIGS = {
    "lora": LoraConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "loha": LoHaConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "lokr": LoKrConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "oft": OFTConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    "hra": HRAConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"]),
    # TODO: cannot use BOFT because some convolutional kernel dimensions are even (64) and others odd (147). There is no
    # common denominator for the boft_block_size except 1, but using 1 results in an error in the fbd_cuda kernel:
    # > Error in forward_fast_block_diag_cuda_kernel: an illegal memory access was encountered
    # "boft": BOFTConfig(target_modules=["convolution"], modules_to_save=["classifier", "normalization"], boft_block_size=2),
}


class TestResnet:
    model_id = "microsoft/resnet-18"

    @pytest.fixture(autouse=True)
    def teardown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @pytest.fixture(scope="class")
    def image_processor(self):
        image_processor = AutoImageProcessor.from_pretrained(self.model_id)
        return image_processor

    @pytest.fixture(scope="class")
    def data(self, image_processor):
        dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
        image = dataset["test"]["image"][0]
        return image_processor(image, return_tensors="pt")

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
        labels = torch.zeros(1, 1000)
        labels[0, 283] = 1
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
