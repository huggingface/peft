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

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PeftModel, get_peft_model
from peft.import_utils import is_transformers_ge_v5
from peft.tuners import lora
from peft.utils import infer_device
from peft.utils.integrations import init_empty_weights, skip_init_on_device

from .testing_utils import hub_online_once


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 2, bias=bias)


def get_mlp():
    return MLP()


class TestInitEmptyWeights:
    def test_init_empty_weights_works(self):
        # this is a very rudimentary test, as init_empty_weights is copied almost 1:1 from accelerate and is tested
        # there
        with init_empty_weights():
            mlp = get_mlp()

        expected = torch.device("meta")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_works(self):
        # when a function is decorated with skip_init_on_device, the parameters are not moved to meta device, even when
        # inside the context
        decorated_fn = skip_init_on_device(get_mlp)
        with init_empty_weights():
            mlp = decorated_fn()

        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_works_outside_context(self):
        # same as before, but ensure that skip_init_on_device does not break when no init_empty_weights context is used
        decorated_fn = skip_init_on_device(get_mlp)
        mlp = decorated_fn()
        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_not_permanent(self):
        # ensure that after skip_init_on_device has been used, init_empty_weights reverts to its original functionality

        # with decorator => cpu
        decorated_fn = skip_init_on_device(get_mlp)
        with init_empty_weights():
            mlp = decorated_fn()

        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp.parameters())

        # without decorator => meta
        with init_empty_weights():
            mlp = get_mlp()

        expected = torch.device("meta")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_nested(self):
        # ensure that skip_init_on_device works even if the decorated function is nested inside another decorated
        # function
        @skip_init_on_device
        def outer_fn():
            @skip_init_on_device
            def inner_fn():
                return get_mlp()

            mlp0 = inner_fn()
            mlp1 = get_mlp()
            return mlp0, mlp1

        with init_empty_weights():
            mlp0, mlp1 = outer_fn()

        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp0.parameters())
        assert all(p.device == expected for p in mlp1.parameters())


@pytest.mark.skipif(not is_transformers_ge_v5, reason="Requires the right transformers version")
class TestTransformersV5:
    """Unit tests intended to test proper working of PEFT with Transformers v5"""

    torch_device = infer_device()

    @pytest.fixture
    def expected_logits(self):
        # original logits were:
        # tensor([[[ 0.2676,  0.3870,  0.2956,  ...,  0.4624,  0.1966,  0.2539],
        #          [-0.6706, -0.0969, -0.6240,  ..., -0.0201,  0.7099, -0.3099],
        #          [ 0.0663,  0.1653,  0.7189,  ...,  0.5905,  0.0649,  0.5839],
        #          ...,
        #          [-0.2712, -0.6451, -0.0219,  ..., -0.4344,  0.5471, -0.9355],
        #          [-0.3607,  0.4526,  0.2750,  ...,  0.1082,  0.7179,  0.8487],
        #          [ 0.5826, -0.1407, -0.3131,  ...,  0.1026,  0.6878, -0.3382]]],
        #        device='cuda:0')
        expected_logits_0_to_3 = torch.Tensor(
            [
                [0.2676, 0.3870, 0.2956],
                [-0.6706, -0.0969, -0.6240],
                [0.0663, 0.1653, 0.7189],
            ]
        ).to(device=self.torch_device, dtype=torch.float16)
        return expected_logits_0_to_3

    def test_mixtral_v4_lora_weight_conversion_transformers_load_adapter(self, expected_logits):
        # Load a PEFT adapter trained with transformers v4 on Mixtral, which now has converted weights (MoE), using the
        # Transformers integration (model.load_adapter).
        inputs = torch.arange(10).view(1, -1).to(device=self.torch_device)
        model_id = "hf-internal-testing/Mixtral-tiny"
        lora_id = "peft-internal-testing/mixtral-pre-v5-lora"

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)

        # test AutoModel.load_adapter
        model.load_adapter(lora_id)
        model.to(self.torch_device)
        with torch.inference_mode():
            output = model(inputs).logits

        # a little bit of deviation but that's fine
        atol, rtol = 1e-3, 1e-4
        assert torch.allclose(output[0, :3, :3], expected_logits, atol=atol, rtol=rtol)

    def test_mixtral_v4_lora_weight_conversion_peft_model_from_pretrained(self, expected_logits):
        # Load a PEFT adapter trained with transformers v4 on Mixtral, which now has converted weights (MoE), using
        # PeftModel.from_pretrained
        inputs = torch.arange(10).view(1, -1).to(device=self.torch_device)
        model_id = "hf-internal-testing/Mixtral-tiny"
        lora_id = "peft-internal-testing/mixtral-pre-v5-lora"

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)

        # test PeftModel.from_pretrained
        model = PeftModel.from_pretrained(model, lora_id)
        with torch.inference_mode():
            output = model(inputs).logits

        # a little bit of deviation but that's fine
        atol, rtol = 1e-3, 1e-4
        assert torch.allclose(output[0, :3, :3], expected_logits, atol=atol, rtol=rtol)

    def test_mixtral_v4_lora_weight_conversion_peft_model_load_adapter(self, expected_logits):
        # Same as the previous test, but using PeftModel.load_adapter
        inputs = torch.arange(10).view(1, -1).to(device=self.torch_device)
        model_id = "hf-internal-testing/Mixtral-tiny"
        lora_id = "peft-internal-testing/mixtral-pre-v5-lora"

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)

        # create a PeftModel instance
        model = get_peft_model(model, LoraConfig(target_modules=["q_proj"]))

        # test PeftModel.load_adapter
        model.load_adapter(lora_id, adapter_name="other")
        model.set_adapter("other")
        with torch.inference_mode():
            output = model(inputs).logits
        atol, rtol = 1e-3, 1e-4
        assert torch.allclose(output[0, :3, :3], expected_logits, atol=atol, rtol=rtol)

    def test_mixtral_save_load_roundtrip(self, expected_logits, tmp_path):
        # Load the v4 checkpoint with PEFT, save it (now v5 format) and load it again
        inputs = torch.arange(10).view(1, -1).to(device=self.torch_device)
        model_id = "hf-internal-testing/Mixtral-tiny"
        lora_id = "peft-internal-testing/mixtral-pre-v5-lora"

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)

        model = PeftModel.from_pretrained(model, lora_id)
        model.save_pretrained(tmp_path)
        del model

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
        model = PeftModel.from_pretrained(model, tmp_path)

        with torch.inference_mode():
            output = model(inputs).logits

        # a little bit of deviation but that's fine
        atol, rtol = 1e-3, 1e-4
        assert torch.allclose(output[0, :3, :3], expected_logits, atol=atol, rtol=rtol)

    def test_add_lora_to_mixtral_v5_works(self):
        # Ensure that using LoRA directly with a v5 model still works
        inputs = torch.arange(10).view(1, -1).to(device=self.torch_device)
        model_id = "hf-internal-testing/Mixtral-tiny"
        lora_id = "peft-internal-testing/mixtral-pre-v5-lora"

        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
        with torch.inference_mode():
            output_base = model(inputs).logits

        lora_config = LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj"],
            target_parameters=["gate.weight", "experts.gate_up_proj", "experts.down_proj"],
        )
        model = get_peft_model(model, lora_config).eval()  # no error

        with torch.inference_mode():
            output_lora = model(inputs).logits
        # sanity check
        assert torch.allclose(output_base, output_lora)

        num_lora_layers = len([m for m in model.modules() if isinstance(m, lora.LoraLayer)])
        # sanity check
        expected_num_lora_layers = 12  # 2 layers with 6 lora layers each
        assert num_lora_layers == expected_num_lora_layers
