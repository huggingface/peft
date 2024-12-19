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


import itertools
import platform
import re
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset, load_dataset
from huggingface_hub.utils import reset_sessions
from safetensors.torch import load_file
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import (
    AdaLoraConfig,
    EvaConfig,
    IA3Config,
    LoftQConfig,
    LoKrConfig,
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    PromptTuningConfig,
    VBLoRAConfig,
    VeraConfig,
    get_eva_state_dict,
    get_peft_model,
    initialize_lora_eva_weights,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda
from peft.tuners.lora.layer import LoraLayer
from peft.utils import infer_device
from peft.utils.constants import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils.hotswap import hotswap_adapter


class TestLoraInitialization:
    """Test class to check the initialization of LoRA adapters."""

    torch_device = infer_device()

    def get_uniform(self, amin, amax, size=(10000,)):
        unif = torch.distributions.uniform.Uniform(amin, amax)
        samples = unif.sample(size)
        return samples

    def get_normal(self, mean, std, size=(10000,)):
        normal = torch.distributions.normal.Normal(mean, std)
        samples = normal.sample(size)
        return samples

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = nn.Linear(1000, 1000)
                self.embed = nn.Embedding(1000, 1000)
                self.conv2d = nn.Conv2d(100, 100, 3)

            def forward(self, x):
                x_int = (100 * x).int()
                x_4d = x.flatten().reshape(1, 100, 10, 10)
                return self.linear(x), self.embed(x_int), self.conv2d(x_4d)

        return MyModule().eval().to(self.torch_device)

    @pytest.fixture
    def data(self):
        return torch.rand(10, 1000).to(self.torch_device)

    def test_lora_linear_init_default(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"])
        model = get_peft_model(model, config)
        weight_A = model.linear.lora_A["default"].weight
        weight_B = model.linear.lora_B["default"].weight

        # use statistical test to check if weight A is from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a normal distribution
        normal = self.get_normal(weight_A.mean().item(), weight_A.std().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_linear_init_gaussian(self):
        # use gaussian init
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.linear.lora_A["default"].weight
        weight_B = model.linear.lora_B["default"].weight

        # use statistical test to check if weight A is from a normal distribution
        normal = self.get_normal(0.0, 1 / config.r)
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())

        # import matplotlib.pyplot as plt
        # x = weight_A.detach().flatten().cpu().numpy()
        # breakpoint()

        assert p_value > 0.5

        # check that weight A is *not* from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_linear_false(self):
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_B = model.linear.lora_B["default"].weight

        # with init_lora_weights=False, weight B should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not torch.allclose(weight_B, torch.zeros_like(weight_B))

    def test_lora_embedding_default(self):
        # embedding is initialized as a normal distribution, not kaiming uniform
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["embed"])
        model = get_peft_model(model, config)
        weight_A = model.embed.lora_embedding_A["default"]
        weight_B = model.embed.lora_embedding_B["default"]

        # use statistical test to check if weight B is from a normal distribution
        normal = self.get_normal(0.0, 1.0)
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight B is *not* from a uniform distribution
        unif = self.get_uniform(weight_B.min().item(), weight_B.max().item())
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight A is zero
        assert (weight_A == 0.0).all()

    def test_lora_embedding_gaussian(self):
        # embedding does not change with init_lora_weights="gaussian" vs True
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["embed"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.embed.lora_embedding_A["default"]
        weight_B = model.embed.lora_embedding_B["default"]

        # use statistical test to check if weight B is from a normal distribution
        normal = self.get_normal(0.0, 1.0)
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight B is *not* from a uniform distribution
        unif = self.get_uniform(weight_B.min().item(), weight_B.max().item())
        _, p_value = stats.kstest(weight_B.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight A is zero
        assert (weight_A == 0.0).all()

    def test_lora_embedding_false(self):
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["embed"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_A = model.embed.lora_embedding_B["default"]

        # with init_lora_weights=False, weight A should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not torch.allclose(weight_A, torch.zeros_like(weight_A))

    def test_lora_conv2d_default(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"])
        model = get_peft_model(model, config)
        weight_A = model.conv2d.lora_A["default"].weight
        weight_B = model.conv2d.lora_B["default"].weight

        # use statistical test to check if weight A is from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a normal distribution
        normal = self.get_normal(weight_A.mean().item(), weight_A.std().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_conv2d_init_gaussian(self):
        # use gaussian init
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.conv2d.lora_A["default"].weight
        weight_B = model.conv2d.lora_B["default"].weight

        # use statistical test to check if weight A is from a normal distribution
        normal = self.get_normal(0.0, 1 / config.r)
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), normal.flatten().cpu().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.detach().flatten().cpu().numpy(), unif.flatten().cpu().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_conv2d_false(self):
        torch.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_B = model.conv2d.lora_B["default"].weight

        # with init_lora_weights=False, weight B should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not torch.allclose(weight_B, torch.zeros_like(weight_B))

    def test_lora_scaling_default(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=False
        config = LoraConfig(target_modules=["linear", "embed", "conv2d"], lora_alpha=3, r=16, use_rslora=False)
        model = get_peft_model(model, config)

        expected_scaling = config.lora_alpha / config.r

        assert model.linear.scaling["default"] == expected_scaling
        assert model.embed.scaling["default"] == expected_scaling
        assert model.conv2d.scaling["default"] == expected_scaling

    # testcase for bugfix for issue 2194
    def test_pattern_override(self):
        torch.manual_seed(0)

        layer = self.get_model()
        model = nn.Sequential(layer, layer)
        config = LoraConfig(
            target_modules=["linear"],
            lora_alpha=1,
            r=8,
            use_rslora=False,
            rank_pattern={"linear": 8},
            alpha_pattern={"0.linear": 2},
        )
        model = get_peft_model(model, config)
        scaling_with_rank_pattern = model.model[0].linear.scaling

        layer = self.get_model()
        model = nn.Sequential(layer, layer)
        config = LoraConfig(
            target_modules=["linear"], lora_alpha=1, r=8, use_rslora=False, alpha_pattern={"0.linear": 2}
        )
        model = get_peft_model(model, config)
        scaling_without_rank_pattern = model.model[0].linear.scaling

        assert scaling_with_rank_pattern == scaling_without_rank_pattern

    def test_lora_pissa_linear_init_default(self, data):
        model = self.get_model()
        output = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert torch.allclose(output, peft_model(data)[0], atol=1e-06)

        config = LoraConfig(init_lora_weights="pissa_niter_16", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert torch.allclose(output, peft_model(data)[0], atol=1e-06)

    def test_lora_olora_linear_init_default(self, data):
        model = self.get_model()
        output = model(data)[0]

        # Both OLoRA and olora should work
        config = LoraConfig(init_lora_weights="OLoRA", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert torch.allclose(output, peft_model(data)[0], atol=1e-06)

    def test_lora_pissa_conversion_same_output_after_loading(self, data, tmp_path):
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_config_keys_before = list(peft_model.peft_config.keys())
        peft_config_dict_before = peft_model.peft_config["default"].to_dict()
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        peft_config_keys_after = list(peft_model.peft_config.keys())
        peft_config_dict_after = peft_model.peft_config["default"].to_dict()
        assert peft_config_keys_before == peft_config_keys_after
        assert peft_config_dict_before == peft_config_dict_after

        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_lora_pissa_conversion_same_output_after_loading_with_rank_pattern(self, data, tmp_path):
        # same as above, but using rank_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use rank_pattern here; note that since there is only a single linear layer, r is completely overridden
        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8, rank_pattern={"linear": 32})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 32
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 64
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_lora_pissa_conversion_same_output_after_loading_with_alpha_pattern(self, data, tmp_path):
        # same as above, but using alpha_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use alpha_pattern here; note that since there is only a single linear layer, lora_alpha is completely
        # overridden
        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], alpha_pattern={"linear": 5})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 5 / 8
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        assert model_converted.base_model.model.linear.scaling["default"] == 10 / 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_lora_pissa_conversion_same_output_after_loading_with_rslora(self, data, tmp_path):
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8, use_rslora=True)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 8 / (8**0.5)
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # same scale as before with a little bit of floating point imprecision
        assert model_converted.base_model.model.linear.scaling["default"] == pytest.approx(8 / (8**0.5))
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_pissa_rank_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="pissa", target_modules=["linear"], r=8, rank_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "pissa-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    def test_pissa_alpha_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="pissa", target_modules=["linear"], r=8, alpha_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "pissa-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    def test_olora_conversion_same_output_after_loading(self, data, tmp_path):
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], r=8)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_config_keys_before = list(peft_model.peft_config.keys())
        peft_config_dict_before = peft_model.peft_config["default"].to_dict()
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        peft_config_keys_after = list(peft_model.peft_config.keys())
        peft_config_dict_after = peft_model.peft_config["default"].to_dict()
        assert peft_config_keys_before == peft_config_keys_after
        assert peft_config_dict_before == peft_config_dict_after

        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_conversion_same_output_after_loading_with_rank_pattern(self, data, tmp_path):
        # same as above, but using rank_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use rank_pattern here; note that since there is only a single linear layer, r is completely overridden
        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], r=8, rank_pattern={"linear": 32})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 32
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 64
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_conversion_same_output_after_loading_with_alpha_pattern(self, data, tmp_path):
        # same as above, but using alpha_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use alpha_pattern here; note that since there is only a single linear layer, lora_alpha is completely
        # overridden
        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], alpha_pattern={"linear": 5})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 5 / 8
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        assert model_converted.base_model.model.linear.scaling["default"] == 10 / 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_conversion_same_output_after_loading_with_rslora(self, data, tmp_path):
        # same as above, but using alpha_pattern
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], r=8, use_rslora=True)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 8 / (8**0.5)
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # same scale as before with a little bit of floating point imprecision
        assert model_converted.base_model.model.linear.scaling["default"] == pytest.approx(8 / (8**0.5))
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_rank_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="olora", target_modules=["linear"], r=8, rank_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "olora-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    def test_olora_alpha_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="olora", target_modules=["linear"], r=8, alpha_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "olora-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    @pytest.mark.parametrize(
        "config_kwargs, should_warn",
        [
            # no warning
            ({"init_lora_weights": "pissa", "target_modules": ["linear"]}, False),
            ({"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"]}, False),
            ({"init_lora_weights": "olora", "target_modules": ["linear"]}, False),
            ({"init_lora_weights": "pissa", "target_modules": ["linear"], "use_rslora": True}, False),
            ({"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"], "use_rslora": True}, False),
            ({"init_lora_weights": "olora", "target_modules": ["linear"], "use_rslora": True}, False),
            ({"init_lora_weights": "pissa", "target_modules": ["linear"], "rank_pattern": {"linear": 8}}, False),
            (
                {"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"], "rank_pattern": {"linear": 8}},
                False,
            ),
            ({"init_lora_weights": "olora", "target_modules": ["linear"], "rank_pattern": {"linear": 8}}, False),
            ({"init_lora_weights": "pissa", "target_modules": ["linear"], "alpha_pattern": {"linear": 8}}, False),
            (
                {"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"], "alpha_pattern": {"linear": 8}},
                False,
            ),
            ({"init_lora_weights": "olora", "target_modules": ["linear"], "alpha_pattern": {"linear": 8}}, False),
            # warning
            (
                {
                    "init_lora_weights": "pissa",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa_niter_3",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "olora",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa_niter_3",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "olora",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa_niter_3",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "olora",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
        ],
    )
    def test_lora_config_pissa_olora_warns(self, config_kwargs, should_warn, recwarn):
        # Using post training conversion of modified base weights to restore their initial values (PiSSA, OLoRA) cannot
        # be correctly done when using rslora + rank_pattern/alpha_pattern. We can't really know if the user intends
        # this when they'll eventually call save_pretrained (i.e. if they'll pass
        # path_initial_model_for_weight_conversionl). Therefore, we only warn but don't raise an error here.
        msg = re.escape("Using Rank-Stabilized LoRA with rank_pattern/alpha_pattern and post-training conversion")
        if should_warn:
            LoraConfig(**config_kwargs)
            assert len(recwarn.list) == 1
            with pytest.warns(UserWarning, match=msg):
                LoraConfig(**config_kwargs)
        else:
            LoraConfig(**config_kwargs)
            assert not recwarn.list

    @pytest.mark.parametrize("init_method", ["pissa", "olora"])
    @pytest.mark.parametrize("pissa_olora_loaded_first", [False, True])
    def test_load_pissa_olora_with_other_adapter_warns(self, init_method, pissa_olora_loaded_first, recwarn, tmp_path):
        # Since PiSSA/OLoRA modifies the base weights, it should not be combined with other adapters. Check for a
        # warning. See #2184.

        # create an adapter without PiSSA/OloRA
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = get_peft_model(model, LoraConfig(init_lora_weights=True))
        model.save_pretrained(tmp_path / "adapter0")
        del model

        # create a model with PiSSA/OLoRA
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = get_peft_model(model, LoraConfig(init_lora_weights=init_method))
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load the model
        if pissa_olora_loaded_first:
            path0, path1 = tmp_path / "adapter1", tmp_path / "adapter0"
        else:
            path0, path1 = tmp_path / "adapter0", tmp_path / "adapter1"

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = PeftModel.from_pretrained(model, path0)
        model = model.load_adapter(path1, adapter_name="other")

        if init_method == "pissa":
            msg = "PiSSA changes the base weights of the model and should thus not be used with other adapters"
        else:
            msg = "OLoRA changes the base weights of the model and should thus not be used with other adapters"
        assert any(str(w.message).startswith(msg) for w in recwarn.list)

    def test_lora_rslora_scaling(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear", "embed", "conv2d"], lora_alpha=3, r=16, use_rslora=True)
        model = get_peft_model(model, config)

        expected_scaling = config.lora_alpha / (config.r**0.5)

        assert model.linear.scaling["default"] == expected_scaling
        assert model.embed.scaling["default"] == expected_scaling
        assert model.conv2d.scaling["default"] == expected_scaling

    def test_lora_default_scaling_pattern(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=False with rank and alpha pattern
        config = LoraConfig(
            target_modules=["linear", "embed", "conv2d"],
            rank_pattern={"embed": 9, "conv2d": 16},
            alpha_pattern={"linear": 11, "conv2d": 13},
            lora_alpha=17,
            r=25,
            use_rslora=False,
        )
        model = get_peft_model(model, config)

        expected_scaling = {
            "linear": config.alpha_pattern["linear"] / config.r,
            "embed": config.lora_alpha / config.rank_pattern["embed"],
            "conv2d": config.alpha_pattern["conv2d"] / config.rank_pattern["conv2d"],
        }

        assert model.linear.scaling["default"] == expected_scaling["linear"]
        assert model.embed.scaling["default"] == expected_scaling["embed"]
        assert model.conv2d.scaling["default"] == expected_scaling["conv2d"]

    def test_lora_rslora_scaling_pattern(self):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=True with rank and alpha pattern
        config = LoraConfig(
            target_modules=["linear", "embed", "conv2d"],
            rank_pattern={"embed": 9, "conv2d": 16},
            alpha_pattern={"linear": 11, "conv2d": 13},
            lora_alpha=17,
            r=25,
            use_rslora=True,
        )
        model = get_peft_model(model, config)

        expected_scaling = {
            "linear": config.alpha_pattern["linear"] / (config.r**0.5),
            "embed": config.lora_alpha / (config.rank_pattern["embed"] ** 0.5),
            "conv2d": config.alpha_pattern["conv2d"] / (config.rank_pattern["conv2d"] ** 0.5),
        }

        assert model.linear.scaling["default"] == expected_scaling["linear"]
        assert model.embed.scaling["default"] == expected_scaling["embed"]
        assert model.conv2d.scaling["default"] == expected_scaling["conv2d"]

    def test_lora_use_dora_linear(self, data):
        # check that dora is a no-op when initialized
        torch.manual_seed(0)
        model = self.get_model()
        output_base, _, _ = model(data)

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear"], use_dora=True)
        model = get_peft_model(model, config)

        with model.disable_adapter():
            output_disabled, _, _ = model(data)
        output_dora, _, _ = model(data)

        assert torch.allclose(output_base, output_disabled)
        assert torch.allclose(output_base, output_dora)

    def test_lora_use_dora_linear_init_false(self, data):
        # with init_lora_weights=False, dora should not be a no-op
        torch.manual_seed(0)
        model = self.get_model()
        output_base, _, _ = model(data)

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear"], use_dora=True, init_lora_weights=False)
        model = get_peft_model(model, config)

        with model.disable_adapter():
            output_disabled, _, _ = model(data)
        output_dora, _, _ = model(data)

        assert torch.allclose(output_base, output_disabled)
        assert not torch.allclose(output_base, output_dora)

    def test_lora_use_dora_with_megatron_core_raises(self):
        megatron_config = {"does-not": "matter-here"}
        with pytest.raises(ValueError, match="DoRA does not support megatron_core"):
            LoraConfig(target_modules=["linear"], use_dora=True, megatron_config=megatron_config)

    def test_lora_with_bias_extra_params(self):
        # lora with lora_bias=True
        model = self.get_model()
        config = LoraConfig(target_modules=["linear", "conv2d"], lora_bias=False)
        model_no_bias = get_peft_model(model, config)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear", "conv2d"], lora_bias=True)
        model_bias = get_peft_model(model, config)

        # check that bias for LoRA B is set
        assert model_no_bias.base_model.model.linear.lora_B["default"].bias is None
        assert model_bias.base_model.model.linear.lora_B["default"].bias.shape == (1000,)
        assert model_no_bias.base_model.model.conv2d.lora_B["default"].bias is None
        assert model_bias.base_model.model.conv2d.lora_B["default"].bias.shape == (100,)

        # check that the same params are present except for the extra bias term
        params_no_bias = {name for name, _ in model_no_bias.named_parameters()}
        params_bias = {name for name, _ in model_bias.named_parameters()}
        extra_params = {
            "base_model.model.linear.lora_B.default.bias",
            "base_model.model.conv2d.lora_B.default.bias",
        }
        assert params_bias - params_no_bias == extra_params
        assert params_no_bias.issubset(params_bias)

    def test_lora_with_bias_embedding_raises(self):
        # lora with lora_bias=True is not supported for embedding layers
        model = self.get_model()
        config = LoraConfig(target_modules=["embed"], lora_bias=True)
        msg = "lora_bias=True is not supported for Embedding"
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)

    @pytest.mark.parametrize(
        "extra_kwargs",
        [
            {"use_dora": True},
            {"init_lora_weights": "eva"},
            {"init_lora_weights": "gaussian"},
            {"init_lora_weights": "loftq", "loftq_config": LoftQConfig()},
            {"init_lora_weights": "olora"},
            {"init_lora_weights": "pissa"},
            {"init_lora_weights": "pissa_niter_3"},
        ],
    )
    def test_lora_with_bias_incompatible_arguments(self, extra_kwargs):
        # some arguments don't work in conjunction with lora_bias and should raise
        # just check the common chunk of the error message
        msg = "The argument lora_bias=True is"
        with pytest.raises(ValueError, match=msg):
            LoraConfig(target_modules=["linear"], lora_bias=True, **extra_kwargs)


class TestLokrInitialization:
    torch_device = infer_device()

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # Choose a large weight so that averages are close to expected values.
                self.linear = nn.Linear(1000, 1000)
                self.conv2d = nn.Conv2d(100, 100, 3)

            def forward(self, x):
                x_4d = x.flatten().reshape(1, 100, 10, 10)
                return self.linear(x), self.conv2d(x_4d)

        return MyModule().eval().to(self.torch_device)

    @pytest.fixture
    def data(self):
        return torch.rand(10, 1000).to(self.torch_device)

    def test_lokr_linear_init_default(self, data):
        torch.manual_seed(0)

        model = self.get_model()
        output_before = model(data)[0]
        config = LoKrConfig(target_modules=["linear"])
        model = get_peft_model(model, config)
        output_after = model(data)[0]

        assert torch.allclose(output_before, output_after)

    def test_lokr_linear_init_false(self, data):
        torch.manual_seed(0)

        model = self.get_model()
        output_before = model(data)[0]
        config = LoKrConfig(target_modules=["linear"], init_weights=False)
        model = get_peft_model(model, config)
        output_after = model(data)[0]

        assert not torch.allclose(output_before, output_after)

    def test_lokr_linear_init_lycoris(self, data):
        torch.manual_seed(0)

        model = self.get_model()
        output_before = model(data)[0]
        config = LoKrConfig(target_modules=["linear"], init_weights="lycoris")
        model = get_peft_model(model, config)
        output_after = model(data)[0]

        assert torch.allclose(output_before, output_after)

    def test_lokr_conv2d_init_default(self, data):
        torch.manual_seed(0)

        model = self.get_model()
        output_before = model(data)[1]
        config = LoKrConfig(target_modules=["conv2d"])
        model = get_peft_model(model, config)
        output_after = model(data)[1]

        assert torch.allclose(output_before, output_after)

    def test_lokr_conv2d_init_false(self, data):
        torch.manual_seed(0)

        model = self.get_model()
        output_before = model(data)[1]
        config = LoKrConfig(target_modules=["conv2d"], init_weights=False)
        model = get_peft_model(model, config)
        output_after = model(data)[1]

        assert not torch.allclose(output_before, output_after)

    def test_lokr_conv2d_init_lycoris(self, data):
        torch.manual_seed(0)

        model = self.get_model()
        output_before = model(data)[1]
        config = LoKrConfig(target_modules=["conv2d"], init_weights="lycoris")
        model = get_peft_model(model, config)
        output_after = model(data)[1]

        assert torch.allclose(output_before, output_after)


class TestAdaLoraInitialization:
    torch_device = infer_device()

    def test_adalora_target_modules_set(self):
        config = AdaLoraConfig(target_modules=["linear", "embed", "conv2d"])
        assert config.target_modules == {"linear", "embed", "conv2d"}

    def test_adalora_use_dora_raises(self):
        with pytest.raises(ValueError, match="ADALORA does not support DoRA"):
            AdaLoraConfig(use_dora=True)

    def test_adalora_loftq_config_raises(self):
        with pytest.raises(ValueError, match="ADALORA does not support LOFTQ"):
            AdaLoraConfig(init_lora_weights="loftq", loftq_config={"loftq": "config"})

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = nn.Linear(1000, 1000)

            def forward(self, x):
                return self.linear(x)

        return MyModule().eval().to(self.torch_device)

    @pytest.fixture
    def data(self):
        return torch.rand(10, 1000).to(self.torch_device)

    def test_adalora_default_init_identity(self, data):
        # default is True
        torch.manual_seed(0)

        model = self.get_model()
        output_before = model(data)
        config = AdaLoraConfig(target_modules=["linear"])
        model = get_peft_model(model, config)
        output_after = model(data)
        assert torch.allclose(output_before, output_after)


class TestPromptTuningInitialization:
    torch_device = infer_device()

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = nn.Linear(1000, 1000)
                self.embed = nn.Embedding(1000, 1000)
                self.conv2d = nn.Conv2d(100, 100, 3)

            def forward(self, x):
                x_int = (100 * x).int()
                x_4d = x.flatten().reshape(1, 100, 10, 10)
                return self.linear(x), self.embed(x_int), self.conv2d(x_4d)

        return MyModule().eval().to(self.torch_device)

    def test_use_prompt_tuning_init_text_raises(self):
        with pytest.raises(ValueError, match="When prompt_tuning_init='TEXT', tokenizer_name_or_path can't be None"):
            PromptTuningConfig(prompt_tuning_init="TEXT", prompt_tuning_init_text="prompt tuning init text")
        with pytest.raises(ValueError, match="When prompt_tuning_init='TEXT', prompt_tuning_init_text can't be None"):
            PromptTuningConfig(prompt_tuning_init="TEXT", tokenizer_name_or_path="t5-base")


class TestVeraInitialization:
    torch_device = infer_device()

    def get_model(self):
        class MLP(nn.Module):
            def __init__(self, bias=True):
                super().__init__()
                self.lin0 = nn.Linear(10, 20, bias=bias)
                self.lin1 = nn.Linear(20, 2, bias=bias)

            def forward(self, X):
                X = self.lin0(X)
                X = self.lin1(X)
                return X

        return MLP().to(self.torch_device)

    def test_vera_mixing_save_projection_raises(self):
        # it is unclear what the right thing to do would be if some adapters save the projection weights and some don't
        # so we better raise an error

        config0 = VeraConfig(target_modules=["lin0"], init_weights=False, save_projection=True)
        model = self.get_model()
        model = get_peft_model(model, config0)
        config1 = VeraConfig(target_modules=["lin0"], init_weights=False, save_projection=False)
        msg = re.escape(
            "VeRA projection weights must be saved for all adapters or none, but got multiple different values: "
            "[False, True]"
        )
        with pytest.raises(ValueError, match=msg):
            model.add_adapter("other", config1)

    def test_vera_add_second_adapter_with_incompatible_input_shape(self):
        config0 = VeraConfig(target_modules=["lin0"], r=8)
        config1 = VeraConfig(target_modules=["lin1"])

        base_model = self.get_model()
        lin0_in_feat = base_model.lin0.in_features
        lin1_in_feat = base_model.lin1.in_features
        model = get_peft_model(base_model, config0)
        # not full message but enough to identify the error
        msg = f"vera_A has a size of {lin0_in_feat} but {lin1_in_feat} or greater is required"
        with pytest.raises(ValueError, match=msg):
            model.add_adapter("other", config1)

    def test_vera_add_second_adapter_with_higher_rank(self):
        rank0 = 123
        rank1 = 456
        config0 = VeraConfig(target_modules=["lin0"], r=rank0)
        # second adapter has higher rank
        config1 = VeraConfig(target_modules=["lin0"], r=rank1)

        model = get_peft_model(self.get_model(), config0)
        # not full message but enough to identify the error
        msg = f"vera_A has a size of {rank0} but {rank1} or greater is required"
        with pytest.raises(ValueError, match=msg):
            model.add_adapter("other", config1)


class TestVBLoraInitialization:
    torch_device = infer_device()

    def get_model(self):
        class MLP(nn.Module):
            def __init__(self, bias=True):
                super().__init__()
                self.lin0 = nn.Linear(10, 30, bias=bias)
                self.lin1 = nn.Linear(30, 2, bias=bias)

            def forward(self, X):
                X = self.lin0(X)
                X = self.lin1(X)
                return X

        return MLP().to(self.torch_device)

    def test_vblora_with_incompatible_vector_length_with_in_features(self):
        vector_length = 3
        model = self.get_model()
        config = VBLoRAConfig(target_modules=["lin0"], vector_length=vector_length)
        msg = f"`in_features` {model.lin0.in_features} must be divisible by `vector_length` {vector_length}"
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)

    def test_vblora_with_incompatible_vector_length_with_out_features(self):
        vector_length = 3
        model = self.get_model()
        config = VBLoRAConfig(target_modules=["lin1"], vector_length=vector_length)
        msg = f"`out_features` {model.lin1.out_features} must be divisible by `vector_length` {vector_length}"
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)


class TestNoInfiniteRecursionDeepspeed:
    # see #1892 for details
    classes = [
        PeftModel,
        PeftMixedModel,
        PeftModelForSequenceClassification,
        PeftModelForQuestionAnswering,
        PeftModelForTokenClassification,
        PeftModelForCausalLM,
        PeftModelForSeq2SeqLM,
        PeftModelForFeatureExtraction,
    ]

    @pytest.fixture
    def wrap_init(self):
        # emulates the wrapper from DeepSpeed
        import functools

        def decorator(f):
            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):
                hasattr(self, "abc")  # any hasattr will do
                f(self, *args, **kwargs)

            return wrapper

        return decorator

    @pytest.fixture
    def model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                # to emulate LMs:
                self.prepare_inputs_for_generation = None
                self._prepare_encoder_decoder_kwargs_for_generation = None

        return MyModule()

    @pytest.mark.parametrize("cls", classes)
    def test_no_infinite_recursion(self, cls, model, wrap_init):
        original_init = cls.__init__
        try:
            cls.__init__ = wrap_init(cls.__init__)
            # this would trigger an infinite loop before the fix in 1892
            cls(model, LoraConfig(target_modules=["linear"]))
        finally:
            # ensure there are no side effects of this test
            cls.__init__ = original_init


class TestLoadAdapterOfflineMode:
    # make sure that PEFT honors offline mode

    @contextmanager
    def hub_offline_ctx(self):
        # this is required to simulate offline mode, setting the env var dynamically inside the test does not work
        # because the value is checked only once at the start of the session
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            reset_sessions()
            yield
        reset_sessions()

    def test_load_from_hub_then_offline_model(self):
        # this uses LoRA but it's the same mechanism for other methods
        peft_model_id = "peft-internal-testing/gpt2-lora-random"
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")

        # first ensure that the adapter model has been downloaded
        PeftModel.from_pretrained(base_model, peft_model_id)

        del base_model

        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        with self.hub_offline_ctx():
            # does not raise
            PeftModel.from_pretrained(base_model, peft_model_id)


class TestCustomModelConfigWarning:
    # Check potential warnings when the user provided base_model_name_or_path is overridden by PEFT. See #2001 for
    # context. We use LoRA for this test but the same applies to other methods
    @pytest.fixture
    def custom_module(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(10, 10)

        return MyModule()

    def test_no_warning_by_default_transformers_model(self, recwarn):
        # first a sanity test that there is no warning by default when using a model from transformers
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM")
        get_peft_model(model, LoraConfig())
        for warning in recwarn.list:
            assert "renamed" not in str(warning.message)

    def test_no_warning_by_default_custom_model(self, custom_module, recwarn):
        # same as above but with a custom model
        get_peft_model(custom_module, LoraConfig(target_modules=["lin"]))
        for warning in recwarn.list:
            assert "renamed" not in str(warning.message)

    def test_warning_name_transformers_model(self, recwarn):
        # The base_model_name_or_path provided by the user is overridden.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM")
        custom_name = "custom_name"
        get_peft_model(model, LoraConfig(base_model_name_or_path=custom_name))
        msg = f"was renamed from '{custom_name}' to 'hf-internal-testing/tiny-random-OPTForCausalLM'"
        assert any(msg in str(warning.message) for warning in recwarn.list)

    def test_warning_name_custom_model(self, custom_module, recwarn):
        custom_name = "custom_name"
        get_peft_model(custom_module, LoraConfig(target_modules=["lin"], base_model_name_or_path=custom_name))
        msg = f"was renamed from '{custom_name}' to 'None'"
        assert any(msg in str(warning.message) for warning in recwarn.list)

    def test_warning_name_custom_model_with_custom_name(self, custom_module, recwarn):
        custom_name = "custom_name"
        custom_module.name_or_path = "foobar"
        get_peft_model(custom_module, LoraConfig(target_modules=["lin"], base_model_name_or_path=custom_name))
        msg = f"was renamed from '{custom_name}' to 'foobar'"
        assert any(msg in str(warning.message) for warning in recwarn.list)


class TestLowCpuMemUsage:
    """Test for the low CPU memory usage option for loading PEFT models.

    Note that we have `test_load_model_low_cpu_mem_usage` in the custom model and stable diffusion tests. Those are
    broad tests (i.e. testing all the supported PEFT methods) but not very deep (only testing if loading works and the
    device is correctly set). The test class here goes deeper but only tests LoRA, as checking all PEFT methods would
    be too much.

    """

    # test on CPU and optionally on accelerator device
    devices = ["cpu"]
    _device = infer_device()
    if _device != "cpu":
        devices.append(_device)

    model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"

    def get_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_id)

    @pytest.fixture(scope="class")
    def lora_config(self):
        return LoraConfig(init_lora_weights=False, target_modules="all-linear")

    @pytest.fixture(scope="class")
    def lora_path(self, tmp_path_factory, lora_config):
        torch.manual_seed(0)
        tmp_path = tmp_path_factory.mktemp("lora")
        model = self.get_model()
        model = get_peft_model(model, lora_config)
        model.save_pretrained(tmp_path)
        return tmp_path

    @pytest.fixture(scope="class")
    def inputs(self):
        return {"input_ids": torch.randint(0, 100, (1, 10)), "attention_mask": torch.ones(1, 10)}

    @pytest.mark.parametrize("device", devices)
    def test_from_pretrained_low_cpu_mem_usage_works(self, device, inputs, lora_path):
        model = self.get_model().to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = PeftModel.from_pretrained(model, lora_path, torch_device=device).eval()
        device_set_not_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_not_low_cpu_mem = model(**inputs).logits

        del model

        model = self.get_model().to(device)
        model = PeftModel.from_pretrained(model, lora_path, low_cpu_mem_usage=True, torch_device=device).eval()
        device_set_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_low_cpu_mem = model(**inputs).logits

        assert device_set_low_cpu_mem == device_set_not_low_cpu_mem
        assert torch.allclose(logits_low_cpu_mem, logits_not_low_cpu_mem)

    @pytest.mark.parametrize("device", devices)
    def test_load_adapter_low_cpu_mem_usage_works(self, device, inputs, lora_path, lora_config):
        model = self.get_model().to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        torch.manual_seed(0)
        model = get_peft_model(model, lora_config)
        model.load_adapter(lora_path, adapter_name="other", torch_device=device)
        model.set_adapter("other")
        model.eval()
        device_set_not_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_not_low_cpu_mem = model(**inputs).logits

        del model

        model = self.get_model().to(device)
        torch.manual_seed(0)
        model = get_peft_model(model, lora_config)
        model.load_adapter(lora_path, adapter_name="other", low_cpu_mem_usage=True, torch_device=device)
        model.set_adapter("other")
        model.eval()
        device_set_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_low_cpu_mem = model(**inputs).logits

        assert device_set_low_cpu_mem == device_set_not_low_cpu_mem
        assert torch.allclose(logits_low_cpu_mem, logits_not_low_cpu_mem)

    @pytest.mark.parametrize("device", devices)
    def test_get_peft_model_low_cpu_mem_usage_works(self, device, inputs):
        # when calling get_peft_model, the PEFT weights will not be initialized on device but remain on meta
        model = self.get_model().to(device)
        model = get_peft_model(model, LoraConfig(target_modules="all-linear"), low_cpu_mem_usage=True)

        devices_lora_weights = {p.device for n, p in model.named_parameters() if "lora_" in n}
        expected = {torch.device("meta")}
        assert devices_lora_weights == expected

    @pytest.mark.parametrize("device", devices)
    def test_get_peft_model_with_task_type_low_cpu_mem_usage_works(self, device, inputs):
        # same as the previous test, but pass the task_type argument
        model = self.get_model().to(device)
        model = get_peft_model(
            model, LoraConfig(target_modules="all-linear", task_type="CAUSAL_LM"), low_cpu_mem_usage=True
        )

        devices_lora_weights = {p.device for n, p in model.named_parameters() if "lora_" in n}
        expected = {torch.device("meta")}
        assert devices_lora_weights == expected

    @pytest.mark.parametrize("device", devices)
    def test_inject_adapter_low_cpu_mem_usage_works(self, device, inputs, lora_path, lora_config):
        # external libs like transformers and diffusers use inject_adapter_in_model, let's check that this also works
        model = self.get_model().to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        torch.manual_seed(0)
        model = get_peft_model(model, lora_config)
        model.load_adapter(lora_path, adapter_name="other", torch_device=device)
        model.set_adapter("other")
        model.eval()
        device_set_not_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_not_low_cpu_mem = model(**inputs).logits

        del model

        torch.manual_seed(0)
        model = self.get_model().to(device)
        inject_adapter_in_model(lora_config, model, low_cpu_mem_usage=True)
        device_set_before_loading = {p.device.type for p in model.parameters()}
        # at this stage, lora weights are still on meta device
        assert device_set_before_loading == {"meta", device}

        state_dict = load_file(lora_path / "adapter_model.safetensors")
        remapped_dict = {}
        prefix = "base_model.model."
        for key, val in state_dict.items():
            new_key = key[len(prefix) :]
            remapped_dict[new_key] = val.to(device)
        errors = set_peft_model_state_dict(model, remapped_dict, low_cpu_mem_usage=True)
        # sanity check: no unexpected keys
        assert not errors.unexpected_keys

        model.eval()
        device_set_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_low_cpu_mem = model(**inputs).logits

        assert device_set_low_cpu_mem == device_set_not_low_cpu_mem
        assert torch.allclose(logits_low_cpu_mem, logits_not_low_cpu_mem)

    ############################
    # tests for PeftMixedModel #
    ############################

    @pytest.mark.parametrize("device", devices)
    def test_mixed_model_from_pretrained_low_cpu_mem_usage_works(self, device, inputs, lora_path):
        model = self.get_model().to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = PeftMixedModel.from_pretrained(model, lora_path, torch_device=device).eval()
        device_set_not_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_not_low_cpu_mem = model(**inputs).logits

        del model

        model = self.get_model().to(device)
        model = PeftMixedModel.from_pretrained(model, lora_path, low_cpu_mem_usage=True, torch_device=device).eval()
        device_set_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_low_cpu_mem = model(**inputs).logits

        assert device_set_low_cpu_mem == device_set_not_low_cpu_mem
        assert torch.allclose(logits_low_cpu_mem, logits_not_low_cpu_mem)

    @pytest.mark.parametrize("device", devices)
    def test_mixed_model_load_adapter_low_cpu_mem_usage_works(self, device, inputs, lora_path, lora_config):
        model = self.get_model().to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        torch.manual_seed(0)
        model = PeftModel.from_pretrained(model, lora_path)
        model.load_adapter(lora_path, adapter_name="other", torch_device=device)
        model.set_adapter("other")
        model.eval()
        device_set_not_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_not_low_cpu_mem = model(**inputs).logits

        del model

        model = self.get_model().to(device)
        torch.manual_seed(0)
        model = PeftModel.from_pretrained(model, lora_path)
        model.load_adapter(lora_path, adapter_name="other", low_cpu_mem_usage=True, torch_device=device)
        model.set_adapter("other")
        model.eval()
        device_set_low_cpu_mem = {p.device.type for p in model.parameters()}
        logits_low_cpu_mem = model(**inputs).logits

        assert device_set_low_cpu_mem == device_set_not_low_cpu_mem
        assert torch.allclose(logits_low_cpu_mem, logits_not_low_cpu_mem)


def test_from_pretrained_missing_keys_warning(recwarn, tmp_path):
    # For more context, see issue 2115
    # When loading a PEFT adapter and we're missing a PEFT-specific weight, there should be a warning.
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM")
    config = LoraConfig()
    model = get_peft_model(model, config)
    state_dict = model.state_dict()

    # first, sanity check that there are no warnings if no key is missing
    model.save_pretrained(tmp_path)
    del model
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM")
    model = PeftModel.from_pretrained(model, tmp_path)
    msg = "Found missing adapter keys"
    assert not any(msg in str(w.message) for w in recwarn.list)

    # remove a key from the state_dict
    missing_key = "base_model.model.model.decoder.layers.0.self_attn.v_proj.lora_A.default.weight"

    def new_state_dict():
        return {k: v for k, v in state_dict.items() if k != missing_key}

    model.state_dict = new_state_dict
    model.save_pretrained(tmp_path)
    del model

    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM")
    model = PeftModel.from_pretrained(model, tmp_path)
    assert any(msg in str(w.message) for w in recwarn.list)
    assert any(missing_key in str(w.message) for w in recwarn.list)


class TestNamingConflictWarning:
    """
    Tests for warnings related to naming conflicts between adapter names and tuner prefixes. References: Issue 2252
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.peft_config = LoraConfig()
        self.prefix = PEFT_TYPE_TO_PREFIX_MAPPING[self.peft_config.peft_type]
        self.base_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM")

    def _save_and_reload_model(self, model, adapter_name, tmp_path):
        # Helper method to save and reload the PEFT model
        model.save_pretrained(tmp_path, selected_adapters=[adapter_name])
        del model
        reloaded_base_model = AutoModelForCausalLM.from_pretrained(tmp_path / adapter_name)
        return PeftModel.from_pretrained(reloaded_base_model, tmp_path / adapter_name)

    def test_no_warning_without_naming_conflict_get_peft_model(self, recwarn):
        # No warning should be raised when there is no naming conflict during get_peft_model.
        non_conflict_adapter = "adapter"
        _ = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        expected_msg = f"Adapter name {non_conflict_adapter} should not be contained in the prefix {self.prefix}."
        assert not any(expected_msg in str(w.message) for w in recwarn.list)

    def test_no_warning_without_naming_conflict_add_adapter(self, recwarn):
        # No warning should be raised when adding an adapter without naming conflict.
        non_conflict_adapter = "adapter"
        other_non_conflict_adapter = "other_adapter"
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        _ = model.add_adapter(other_non_conflict_adapter, self.peft_config)
        expected_msg = (
            f"Adapter name {other_non_conflict_adapter} should not be contained in the prefix {self.prefix}."
        )
        assert not any(expected_msg in str(w.message) for w in recwarn.list)

    def test_no_warning_without_naming_conflict_save_and_load(self, recwarn, tmp_path):
        # No warning should be raised when saving and loading the model without naming conflict.
        non_conflict_adapter = "adapter"
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        _ = self._save_and_reload_model(model, non_conflict_adapter, tmp_path)
        expected_msg = f"Adapter name {non_conflict_adapter} should not be contained in the prefix {self.prefix}."
        assert not any(expected_msg in str(w.message) for w in recwarn.list)

    def test_warning_naming_conflict_get_peft_model(self, recwarn):
        # Warning is raised when the adapter name conflicts with the prefix in get_peft_model.
        conflicting_adapter_name = self.prefix[:-1]
        _ = get_peft_model(self.base_model, self.peft_config, adapter_name=conflicting_adapter_name)
        expected_msg = f"Adapter name {conflicting_adapter_name} should not be contained in the prefix {self.prefix}."
        assert any(expected_msg in str(w.message) for w in recwarn.list)

    def test_warning_naming_conflict_add_adapter(self, recwarn):
        # Warning is raised when adding an adapter with a name that conflicts with the prefix.
        conflicting_adapter = self.prefix[1:]
        non_conflict_adapter = "adapter"
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        _ = model.add_adapter(conflicting_adapter, self.peft_config)
        expected_msg = f"Adapter name {conflicting_adapter} should not be contained in the prefix {self.prefix}."
        assert any(expected_msg in str(w.message) for w in recwarn.list)

    def test_warning_naming_conflict_save_and_load(self, recwarn, tmp_path):
        # Warning is raised when saving and loading the model with a naming conflict.
        conflicting_adapter = self.prefix[:-1]
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=conflicting_adapter)
        _ = self._save_and_reload_model(model, conflicting_adapter, tmp_path)
        expected_msg = f"Adapter name {conflicting_adapter} should not be contained in the prefix {self.prefix}."
        assert any(expected_msg in str(w.message) for w in recwarn.list)


class TestCordaInitialization:
    """Test class to check the initialization of CorDA adapters."""

    torch_device = infer_device()

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = nn.Linear(1000, 1000)

            def forward(self, x):
                return self.linear(x)

        return MyModule().eval().to(self.torch_device)

    @pytest.fixture
    def data(self):
        # larger data is required to pass KPM test
        torch.manual_seed(233)
        return torch.rand(1000, 1000).to(self.torch_device)

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_sample_count(self, data, corda_method):
        original_model = self.get_model()
        model = deepcopy(original_model)

        corda_config = CordaConfig(
            corda_method=corda_method,
        )
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=corda_config,
        )
        preprocess_corda(
            model,
            config,
            run_model=lambda: [model(data), model(data)],  # running model twice to test `sample_count`
            hooked_model=model,
        )

        # covariance of linear should be data.T @ data
        layer = model.linear
        assert hasattr(layer, "covariance_matrix")
        assert torch.allclose(layer.covariance_matrix, data.T @ data, atol=1e-06)

        # sample count of linear should be 2
        assert hasattr(layer, "sample_count")
        assert layer.sample_count == 2

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_hook_unregister(self, data, corda_method):
        original_model = self.get_model()
        model = deepcopy(original_model)

        hook_call_count = 0

        def hook(*args):
            nonlocal hook_call_count
            hook_call_count += 1

        model.linear.register_forward_hook(hook)

        corda_config = CordaConfig(
            corda_method=corda_method,
        )
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=corda_config,
        )
        preprocess_corda(
            model,
            config,
            run_model=lambda: model(data),
            hooked_model=model,
        )

        # after preprocessing, external and internal hook should be run once
        assert hook_call_count == 1
        assert model.linear.sample_count == 1

        # run preprocessed model once
        model(data)[0]

        # the external hook should be kept, but the internal hook should be gone
        assert hook_call_count == 2
        assert model.linear.sample_count == 1

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_linear_init_default(self, data, tmp_path, corda_method):
        original_model = self.get_model()
        model = deepcopy(original_model)
        output_base = model(data)[0]

        corda_config = CordaConfig(
            cache_file=tmp_path / "corda_cache.pt",
            covariance_file=tmp_path / "covariance_cache.pt",
            corda_method=corda_method,
        )
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=corda_config,
        )
        preprocess_corda(
            model,
            config,
            run_model=lambda: model(data),
            hooked_model=model,
        )
        peft_model = get_peft_model(model, config)

        # check if adapter performs an identity transformantion
        assert torch.allclose(output_base, peft_model(data)[0], atol=1e-06)

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_corda = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_corda, atol=tol, rtol=tol)

        # if load SVD result from cache, the output should be the same
        model = deepcopy(original_model)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=CordaConfig(cache_file=tmp_path / "corda_cache.pt", corda_method=corda_method),
        )
        preprocess_corda(model, config)
        peft_model = get_peft_model(model, config)
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        assert torch.allclose(output_corda, peft_model(data)[0], atol=1e-06)

        # if load covariance from cache, the output should be the same
        model = deepcopy(original_model)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=CordaConfig(covariance_file=tmp_path / "covariance_cache.pt", corda_method=corda_method),
        )
        preprocess_corda(model, config)
        peft_model = get_peft_model(model, config)
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        assert torch.allclose(output_corda, peft_model(data)[0], atol=1e-06)

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_hooked_model_linear_init_default(self, data, tmp_path, corda_method):
        original_model = self.get_model()
        model = deepcopy(original_model)
        hooked_model = deepcopy(model)
        output_base = model(data)[0]

        corda_config = CordaConfig(
            cache_file=tmp_path / "corda_cache.pt",
            covariance_file=tmp_path / "covariance_cache.pt",
            corda_method=corda_method,
        )
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=corda_config,
        )

        # difference from the above test: this test uses a copied model as hooked model
        preprocess_corda(
            model,
            config,
            run_model=lambda: hooked_model(data),
            hooked_model=hooked_model,
        )
        peft_model = get_peft_model(model, config)

        # check if adapter performs an identity transformantion
        assert torch.allclose(output_base, peft_model(data)[0], atol=1e-06)

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_corda = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_corda, atol=tol, rtol=tol)

        # if load SVD result from cache, the output should be the same
        model = deepcopy(original_model)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=CordaConfig(cache_file=tmp_path / "corda_cache.pt", corda_method=corda_method),
        )
        preprocess_corda(model, config)
        peft_model = get_peft_model(model, config)
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        assert torch.allclose(output_corda, peft_model(data)[0], atol=1e-06)

        # if load covariance from cache, the output should be the same
        model = deepcopy(original_model)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=CordaConfig(covariance_file=tmp_path / "covariance_cache.pt", corda_method=corda_method),
        )
        preprocess_corda(model, config)
        peft_model = get_peft_model(model, config)
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        assert torch.allclose(output_corda, peft_model(data)[0], atol=1e-06)

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_linear_init_default_with_rank_pattern(self, data, tmp_path, corda_method):
        original_model = self.get_model()
        model = deepcopy(original_model)
        output_base = model(data)[0]

        corda_config = CordaConfig(
            cache_file=tmp_path / "corda_cache.pt",
            covariance_file=tmp_path / "covariance_cache.pt",
            corda_method=corda_method,
        )
        config = LoraConfig(
            rank_pattern={"linear": 8, "embed": 16, "conv2d": 32},
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=corda_config,
        )
        preprocess_corda(
            model,
            config,
            run_model=lambda: model(data),
        )
        peft_model = get_peft_model(model, config)

        # check if adapter performs an identity transformantion
        assert torch.allclose(output_base, peft_model(data)[0], atol=1e-06)

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_corda = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_corda, atol=tol, rtol=tol)

        # if load SVD result from cache, the output should be the same
        model = deepcopy(original_model)
        config = LoraConfig(
            rank_pattern={"linear": 8, "embed": 16, "conv2d": 32},
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=CordaConfig(cache_file=tmp_path / "corda_cache.pt", corda_method=corda_method),
        )
        preprocess_corda(model, config)
        peft_model = get_peft_model(model, config)
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        assert torch.allclose(output_corda, peft_model(data)[0], atol=1e-06)

        # if load covariance from cache, the output should be the same
        model = deepcopy(original_model)
        config = LoraConfig(
            rank_pattern={"linear": 8, "embed": 16, "conv2d": 32},
            init_lora_weights="corda",
            target_modules=["linear"],
            corda_config=CordaConfig(covariance_file=tmp_path / "covariance_cache.pt", corda_method=corda_method),
        )
        preprocess_corda(model, config)
        peft_model = get_peft_model(model, config)
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        assert torch.allclose(output_corda, peft_model(data)[0], atol=1e-06)

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_conversion_same_output_after_loading(self, data, tmp_path, corda_method):
        model = self.get_model()
        output_base = model(data)[0]

        corda_config = CordaConfig(corda_method=corda_method)
        config = LoraConfig(init_lora_weights="corda", target_modules=["linear"], r=8, corda_config=corda_config)
        preprocess_corda(model, config, run_model=lambda: model(data), hooked_model=model)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "corda"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_corda = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_corda, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "corda-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_corda, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_config_keys_before = list(peft_model.peft_config.keys())
        peft_config_dict_before = peft_model.peft_config["default"].to_dict()
        peft_model.save_pretrained(
            tmp_path / "corda-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        peft_config_keys_after = list(peft_model.peft_config.keys())
        peft_config_dict_after = peft_model.peft_config["default"].to_dict()
        assert peft_config_keys_before == peft_config_keys_after
        assert peft_config_dict_before == peft_config_dict_after

        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_corda, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_conversion_same_output_after_loading_with_rank_pattern(self, data, tmp_path, corda_method):
        # same as above, but using rank_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use rank_pattern here; note that since there is only a single linear layer, r is completely overridden
        corda_config = CordaConfig(corda_method=corda_method)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            r=8,
            rank_pattern={"linear": 32},
            corda_config=corda_config,
        )
        preprocess_corda(model, config, run_model=lambda: model(data), hooked_model=model)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "corda"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_corda = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_corda, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "corda-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_corda, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 32
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "corda-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_corda, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 64
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_conversion_same_output_after_loading_with_alpha_pattern(self, data, tmp_path, corda_method):
        # same as above, but using alpha_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use alpha_pattern here; note that since there is only a single linear layer, lora_alpha is completely
        # overridden
        corda_config = CordaConfig(corda_method=corda_method)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            alpha_pattern={"linear": 5},
            corda_config=corda_config,
        )
        preprocess_corda(model, config, run_model=lambda: model(data), hooked_model=model)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "corda"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_corda = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_corda, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "corda-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_corda, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 5 / 8
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "corda-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_corda, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        assert model_converted.base_model.model.linear.scaling["default"] == 10 / 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_conversion_same_output_after_loading_with_rslora(self, data, tmp_path, corda_method):
        model = self.get_model()
        output_base = model(data)[0]

        corda_config = CordaConfig(corda_method=corda_method)
        config = LoraConfig(
            init_lora_weights="corda", target_modules=["linear"], r=8, use_rslora=True, corda_config=corda_config
        )
        preprocess_corda(model, config, run_model=lambda: model(data), hooked_model=model)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "corda"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_corda = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_corda, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "corda-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_corda, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 8 / (8**0.5)
        # sanity check: the base model weights were indeed changed
        assert not torch.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "corda-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "corda-model-converted")
        output_converted = model_converted(data)[0]

        assert torch.allclose(output_corda, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # same scale as before with a little bit of floating point imprecision
        assert model_converted.base_model.model.linear.scaling["default"] == pytest.approx(8 / (8**0.5))
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_rank_pattern_and_rslora_raises(self, data, tmp_path, corda_method):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        corda_config = CordaConfig(corda_method=corda_method)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            r=8,
            rank_pattern={"linear": 2},
            use_rslora=True,
            corda_config=corda_config,
        )
        preprocess_corda(model, config, run_model=lambda: model(data), hooked_model=model)
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "corda-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    @pytest.mark.parametrize("corda_method", ("ipm", "kpm"))
    def test_lora_corda_alpha_pattern_and_rslora_raises(self, data, tmp_path, corda_method):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        corda_config = CordaConfig(corda_method=corda_method)
        config = LoraConfig(
            init_lora_weights="corda",
            target_modules=["linear"],
            r=8,
            alpha_pattern={"linear": 2},
            use_rslora=True,
            corda_config=corda_config,
        )
        preprocess_corda(model, config, run_model=lambda: model(data), hooked_model=model)
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "corda-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )


class TestEvaInitialization:
    """Tests for the EVA (Explained Variance Adaptation) initialization method.

    This test suite verifies:
    1. Consistency of initialization across different seeds
    2. Proper error handling for invalid inputs
    3. Compatibility with different model architectures
    4. Reproducibility of results
    5. Proper handling of edge cases
    """

    # Constants for test configuration
    COSINE_SIMILARITY_THRESHOLD = 0.75
    NUM_SEEDS = 2
    BATCH_SIZE = 4
    MAX_LENGTH = 256
    LORA_DIM = 8
    LORA_ALPHA = 1
    DEVICE = infer_device()

    @pytest.fixture
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.fixture
    def dataset(self, tokenizer):
        dataset = load_dataset("ybelkada/english_quotes_copy", split="train")
        # concatenate examples
        examples = []
        example = ""
        for data in dataset:
            if len(example) >= self.MAX_LENGTH:
                examples.append(example)
                example = ""
            example = example + " " + data["quote"]
        dataset = Dataset.from_dict({"text": examples})
        # tokenize
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=self.MAX_LENGTH),
            batched=True,
            remove_columns=dataset.column_names,
        )
        dataset.set_format(type="torch")
        return dataset

    @pytest.fixture
    def model(self):
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        model.transformer.h = model.transformer.h[:2]  # truncate to 2 layers
        return model.to(self.DEVICE)

    @pytest.fixture
    def peft_config(self):
        return LoraConfig(
            r=self.LORA_DIM,
            lora_alpha=self.LORA_ALPHA,
            target_modules=["c_attn"],
            init_lora_weights="eva",
            eva_config=EvaConfig(rho=2),
        )

    @staticmethod
    def collate_fn(examples):
        return {k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()}

    @staticmethod
    def prepare_layer_inputs_fn(layer_input, model_input, layer_name):
        return layer_input[0].view(-1, layer_input[0].size(-1))

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.BATCH_SIZE,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    @pytest.mark.parametrize(
        "prepare_layer_inputs_keys, expected_outcome",
        [
            (None, "success"),
            (["transformer.h.0.attn.c_attn"], "success"),
            (
                ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn", "transformer.h.2.attn.c_attn"],
                "value_error",
            ),
        ],
    )
    def test_eva_state_dict_prepare_inputs_mapping(
        self, model, dataset, peft_config, prepare_layer_inputs_keys, expected_outcome
    ):
        """
        Tests for cases where prepare_layer_inputs_fn is a mapping. Checks that if not all target modules are present,
        the prepare_layer_inputs_fn for the remaining modules is set to None. Also checks that if more keys than target
        modules are present, a ValueError is raised.
        """

        def fn(x, *args):
            return x[0].view(-1, x[0].size(-1))

        if prepare_layer_inputs_keys is None:
            prepare_layer_inputs_fn = fn
        else:
            prepare_layer_inputs_fn = {k: fn for k in prepare_layer_inputs_keys}

        shuffled_dataset = dataset.shuffle(seed=0)
        dataloader = self.get_dataloader(shuffled_dataset)
        modified_peft_config = deepcopy(peft_config)
        modified_peft_config.eva_config.tau = 0  # converge immediately
        if expected_outcome == "success":
            sd = get_eva_state_dict(
                model,
                dataloader,
                modified_peft_config,
                prepare_model_inputs_fn=None,
                prepare_layer_inputs_fn=prepare_layer_inputs_fn,
            )
            assert len(sd) == 2
            assert "transformer.h.0.attn.c_attn" in sd
            assert "transformer.h.1.attn.c_attn" in sd
        else:
            with pytest.raises(
                ValueError, match="prepare_layer_inputs_fn is a mapping but the following module names were not found"
            ):
                get_eva_state_dict(
                    model,
                    dataloader,
                    modified_peft_config,
                    prepare_model_inputs_fn=None,
                    prepare_layer_inputs_fn=prepare_layer_inputs_fn,
                )

    @pytest.mark.parametrize(
        "eva_config",
        [EvaConfig(rho=2, adjust_scaling_factors=True)],
    )
    def test_eva_state_dict_adjust_scaling_factors(self, model, dataset, peft_config, eva_config):
        """
        Tests that the scaling factors are adjusted so that all LoRA gradients have the same scale regardless of their
        rank.
        """
        modified_peft_config = deepcopy(peft_config)
        modified_peft_config.eva_config = eva_config
        dataloader = self.get_dataloader(dataset)
        peft_model = get_peft_model(deepcopy(model), modified_peft_config)
        scaling_factors_before = {}
        for n, m in peft_model.named_modules():
            if isinstance(m, LoraLayer):
                scaling_factors_before[n] = m.scaling["default"]
        initialize_lora_eva_weights(peft_model, dataloader)
        for n, m in peft_model.named_modules():
            if isinstance(m, LoraLayer):
                assert m.scaling["default"] == scaling_factors_before[n]

    @pytest.mark.parametrize(
        "eva_config",
        [
            # note: lower tau to decrease number of iterations until convergence, as tests are slow on CPU
            EvaConfig(rho=2, tau=0.9),
            EvaConfig(rho=1, tau=0.9),
            EvaConfig(rho=1, whiten=True, tau=0.9),
            EvaConfig(rho=1.0001, tau=0.9),
        ],
    )
    def test_eva_initialization_consistency(self, model, dataset, peft_config, eva_config):
        """
        Tests that the state dict returned by `get_eva_state_dict` is consistent across different seeds based on the
        cosine similarity of the svd components.
        """
        modified_peft_config = deepcopy(peft_config)
        modified_peft_config.eva_config = eva_config
        state_dicts = []
        for seed in range(self.NUM_SEEDS):
            shuffled_dataset = dataset.shuffle(seed=seed)
            dataloader = self.get_dataloader(shuffled_dataset)
            sd = get_eva_state_dict(model, dataloader, modified_peft_config, show_progress_bar=False)
            state_dicts.append(sd)

        cos_sims = defaultdict(list)
        for i, j in itertools.combinations(range(self.NUM_SEEDS), 2):
            for k, v1 in state_dicts[i].items():
                v2 = state_dicts[j][k]
                min_size = min(v1.size(0), v2.size(0))
                cos_sims[k].extend(torch.cosine_similarity(v1[:min_size].abs(), v2[:min_size].abs(), dim=1).tolist())

        mean_cosine_similarities = {k: torch.tensor(v).mean() for k, v in cos_sims.items()}
        for layer_name, mean_cosine_similarity in mean_cosine_similarities.items():
            assert mean_cosine_similarity > self.COSINE_SIMILARITY_THRESHOLD, (
                f"Mean absolute cosine similarity {mean_cosine_similarity:.4f} "
                f"is not greater than {self.COSINE_SIMILARITY_THRESHOLD}"
            )

    @pytest.mark.parametrize("has_rank_zero", [True, False])
    def test_load_eva_state_dict(self, model, dataset, peft_config, tmp_path, has_rank_zero):
        """
        Tests that the `eva_state_dict` argument in `initialize_lora_eva_weights` can be used to initialize a model
        with EVA weights and that the initialized model can be saved and loaded correctly.
        """
        dataloader = self.get_dataloader(dataset)
        peft_model = get_peft_model(deepcopy(model), peft_config)
        sd = get_eva_state_dict(peft_model, dataloader)
        if has_rank_zero:
            k = "base_model.model.transformer.h.0.attn.c_attn"
            sd[k] = sd[k][:0]
        initialize_lora_eva_weights(peft_model, eva_state_dict=sd)
        if has_rank_zero:
            assert not isinstance(peft_model.model.transformer.h[0].attn.c_attn, LoraLayer)
        else:
            assert isinstance(peft_model.model.transformer.h[0].attn.c_attn, LoraLayer)
        peft_model.save_pretrained(tmp_path)
        peft_model = PeftModel.from_pretrained(model, tmp_path, torch_device=self.DEVICE, low_cpu_mem_usage=True)
        peft_model(**{k: v.to(self.DEVICE) for k, v in next(iter(dataloader)).items()})

    def test_missing_eva_inits(self, model, dataset, peft_config):
        """
        Tests that a warning is raised when some adapter modules were not initialized with EVA weights.
        """
        modified_peft_config = deepcopy(peft_config)
        modified_peft_config.target_modules = ["wte"]
        dataloader = self.get_dataloader(dataset)
        peft_model = get_peft_model(deepcopy(model), modified_peft_config)
        with pytest.warns(
            UserWarning,
            match="the following layers were initialized with init_lora_weights=True because they were not found in the eva state_dict:*",
        ):
            initialize_lora_eva_weights(peft_model, dataloader)

    def test_load_eva_model(self, model, dataset, peft_config, tmp_path):
        """
        Tests that a model initialized with EVA weights can be loaded correctly.
        """
        dataloader = self.get_dataloader(dataset)
        peft_model = get_peft_model(deepcopy(model), peft_config)
        initialize_lora_eva_weights(peft_model, dataloader)
        peft_model.save_pretrained(tmp_path)
        peft_model = PeftModel.from_pretrained(model, tmp_path, torch_device=self.DEVICE, low_cpu_mem_usage=True)
        peft_model(**{k: v.to(self.DEVICE) for k, v in next(iter(dataloader)).items()})

    def test_eva_initialization_with_invalid_dataloader(self, model, peft_config):
        """Test that appropriate error is raised when dataloader is empty."""
        empty_dataset = Dataset.from_dict({"text": []})
        dataloader = self.get_dataloader(empty_dataset)

        with pytest.raises(ValueError, match="dataloader is empty"):
            get_eva_state_dict(model, dataloader, peft_config)

    def test_eva_config_rho(self):
        """
        Tests that EvaConfig.__init__ raises a ValueError when rho is negative.
        """
        with pytest.raises(ValueError, match="`rho` must be >= 1.0"):
            EvaConfig(rho=-1)

    def test_eva_config_tau(self):
        """
        Tests that EvaConfig.__init__ raises a ValueError when tau is not between 0.0 and 1.0.
        """
        with pytest.raises(ValueError, match="`tau` must be between 0.0 and 1.0."):
            EvaConfig(tau=-0.1)
        with pytest.raises(ValueError, match="`tau` must be between 0.0 and 1.0."):
            EvaConfig(tau=1.1)

    def test_lora_config_raises_warning_with_eva_init_but_not_eva_config(self):
        """
        Tests that LoraConfig.__init__ raises a warning when init_lora_weights='eva' but eva_config is not set.
        """
        with pytest.warns(
            UserWarning,
            match="`init_lora_weights` is 'eva' but `eva_config` is not specified. Using default EVA config.",
        ):
            LoraConfig(init_lora_weights="eva")

    def test_lora_config_raises_warning_with_eva_config_but_not_eva_init(self):
        """
        Tests that LoraConfig.__init__ raises a warning when init_lora_weights is not 'eva' but eva_config is set.
        """
        with pytest.warns(
            UserWarning, match="`eva_config` specified but will be ignored when `init_lora_weights` is not 'eva'."
        ):
            LoraConfig(init_lora_weights=True, eva_config=EvaConfig())


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Out of the box, torch.compile does not work on Windows or MacOS"
)
class TestHotSwapping:
    """Tests for the hotswapping function"""

    torch_device = infer_device()

    def compile(self, model, do_compile):
        if not do_compile:
            return model
        return torch.compile(model)

    def get_model(self):
        class MLP(nn.Module):
            def __init__(self, bias=True):
                super().__init__()
                self.lin0 = nn.Linear(10, 20, bias=True)
                self.relu = nn.ReLU()
                self.lin1 = nn.Linear(20, 5, bias=False)

            def forward(self, X):
                X = X.float()
                X = self.lin0(X)
                X = self.relu(X)
                X = self.lin1(X)
                return X

        torch.manual_seed(0)
        return MLP().to(self.torch_device)

    # this works with all adapters except prompt learning, but we don't test all
    # as it is unnecessary and would be slow
    @pytest.mark.parametrize(
        "config",
        [
            LoraConfig(init_lora_weights=0, target_modules=["lin0"]),
            LoraConfig(init_lora_weights=0, target_modules=["lin0", "lin1"]),
        ],
    )
    @pytest.mark.parametrize("do_compile", [False, True])
    def test_hotswap_works(self, config, do_compile, tmp_path):
        # Load 2 different adapters and check that we can hotswap between them, with the model optionally being
        # compiled.
        atol, rtol = 1e-4, 1e-4
        inputs = torch.rand(3, 10).to(self.torch_device)

        # create adapter 0
        model = self.get_model()
        torch.manual_seed(0)
        model = get_peft_model(model, config)
        model = self.compile(model, do_compile=do_compile)
        model.eval()
        with torch.inference_mode():
            output0 = model(inputs)
        model.save_pretrained(tmp_path / "adapter0")

        del model

        # create adapter 1
        model = self.get_model()
        torch.manual_seed(1)
        model = get_peft_model(model, config)
        model = self.compile(model, do_compile=do_compile)
        model.eval()
        with torch.inference_mode():
            output1 = model(inputs)
        model.save_pretrained(tmp_path / "adapter1")

        # sanity check: they're not the same
        assert not torch.allclose(output0, output1, atol=atol, rtol=rtol)

        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")
        model = self.compile(model, do_compile=do_compile)
        with torch.inference_mode():
            output_loaded0 = model(inputs)

        # sanity check: same output after loading for adapter 0
        assert torch.allclose(output0, output_loaded0, atol=atol, rtol=rtol)

        # hotswap with adapter 1
        hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")
        with torch.inference_mode():
            output_loaded1 = model(inputs)

        # real check: model now behaves like adapter 1
        assert torch.allclose(output1, output_loaded1, atol=atol, rtol=rtol)

        # hotswap back to adapter 0
        hotswap_adapter(model, tmp_path / "adapter0", adapter_name="default")
        with torch.inference_mode():
            output_loaded_back0 = model(inputs)

        # real check: model now behaves again like adapter 0
        assert torch.allclose(output0, output_loaded_back0, atol=atol, rtol=rtol)

    def test_hotswap_incompatible_config_params_raises(self, tmp_path):
        # When the configs of the two adapters are incompatible, an error is raised
        config0 = LoraConfig(target_modules=["lin0"], lora_alpha=1.0)
        config1 = LoraConfig(target_modules=["lin0"], lora_alpha=2.0)

        model = self.get_model()
        model = get_peft_model(model, config0)
        model.save_pretrained(tmp_path / "adapter0")
        del model

        model = self.get_model()
        model = get_peft_model(model, config1)
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")

        msg = r"Configs are incompatible: for lora_alpha, 1.0 != 2.0"
        with pytest.raises(ValueError, match=msg):
            hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")

    def test_hotswap_different_peft_types_raises(self, tmp_path):
        # When the configs of the two adapters are different PEFT methods, raise
        config0 = LoraConfig(target_modules=["lin0"])
        config1 = IA3Config(target_modules=["lin0"], feedforward_modules=[])

        model = self.get_model()
        model = get_peft_model(model, config0)
        model.save_pretrained(tmp_path / "adapter0")
        del model

        model = self.get_model()
        model = get_peft_model(model, config1)
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")

        msg = r"Incompatible PEFT types found: LORA and IA3"
        with pytest.raises(ValueError, match=msg):
            hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")

    def test_hotswap_wrong_peft_types_raises(self, tmp_path):
        # Only LoRA is supported at the moment
        config0 = IA3Config(target_modules=["lin0"], feedforward_modules=[])
        config1 = IA3Config(target_modules=["lin0"], feedforward_modules=[])

        model = self.get_model()
        model = get_peft_model(model, config0)
        model.save_pretrained(tmp_path / "adapter0")
        del model

        model = self.get_model()
        model = get_peft_model(model, config1)
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")

        msg = r"Hotswapping only supports LORA but IA3 was passed"
        with pytest.raises(ValueError, match=msg):
            hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")

    def test_hotswap_missing_key_raises(self, tmp_path):
        # When a key is missing, raise
        config = LoraConfig(target_modules=["lin0", "lin1"])

        model = self.get_model()
        model = get_peft_model(model, config)
        model.save_pretrained(tmp_path / "adapter0")
        del model

        model = self.get_model()
        model = get_peft_model(model, config)

        # remove one key from the state_dict
        key = "base_model.model.lin1.lora_A.default.weight"
        state_dict = model.state_dict()
        del state_dict[key]
        model.state_dict = lambda: state_dict
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")

        msg = f"Hot swapping the adapter did not succeed. Missing keys: {key}"
        with pytest.raises(RuntimeError, match=msg):
            hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")

    def test_hotswap_extra_key_raises(self, tmp_path):
        # When there is an extra key, raise
        config = LoraConfig(target_modules=["lin0"])

        model = self.get_model()
        model = get_peft_model(model, config)
        model.save_pretrained(tmp_path / "adapter0")
        del model

        model = self.get_model()
        model = get_peft_model(model, config)

        # add an unexpected key
        state_dict = model.state_dict()
        new_key = "base_model.model.lin1.lora_A.default.weight"
        state_dict[new_key] = torch.zeros(8, 20)
        model.state_dict = lambda: state_dict
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")

        msg = f"Hot swapping the adapter did not succeed. Unexpected keys: {new_key}"
        with pytest.raises(RuntimeError, match=msg):
            hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")
