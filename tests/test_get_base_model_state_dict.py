# Copyright 2026-present the HuggingFace Inc. team.
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

"""Tests for get_base_model_state_dict / set_base_model_state_dict.

The per-method and per-model coverage lives in ``PeftCommonTester._test_get_base_model_state_dict`` (see
``testing_common.py``), which is called from ``test_decoder_models.py``, ``test_encoder_decoder_models.py`` and
``test_custom_models.py`` and covers key matching, value matching and the get/set roundtrip for every tuner config in
those matrices. What remains here are the cases that are independent of the tuner method and are therefore not worth
running once per config: the ``strict`` bookkeeping and multiple adapters.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_base_model_state_dict, get_peft_model, set_base_model_state_dict
from peft.utils import infer_device

from .testing_utils import hub_online_once


CAUSAL_LM_MODEL_ID = "peft-internal-testing/tiny-random-OPTForCausalLM"


class TestGetBaseModelStateDict:
    torch_device = infer_device()

    def get_peft_model(self, **config_kwargs):
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)
        config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear", **config_kwargs)
        return get_peft_model(base_model, config)

    def test_strict_missing_keys(self):
        """A state dict that misses a base model key is rejected with strict=True and reported with strict=False."""
        peft_model = self.get_peft_model()
        state_dict = get_base_model_state_dict(peft_model)
        removed_key = next(iter(state_dict.keys()))
        del state_dict[removed_key]

        with pytest.raises(RuntimeError, match="Missing key"):
            set_base_model_state_dict(peft_model, state_dict, strict=True)

        result = set_base_model_state_dict(peft_model, state_dict, strict=False)
        assert removed_key in result.missing_keys

    def test_strict_unexpected_keys(self):
        """A state dict with an unknown key is rejected with strict=True and reported with strict=False."""
        peft_model = self.get_peft_model()
        state_dict = get_base_model_state_dict(peft_model)
        state_dict["unexpected.weight"] = torch.zeros(10)

        with pytest.raises(RuntimeError, match="Unexpected key"):
            set_base_model_state_dict(peft_model, state_dict, strict=True)

        result = set_base_model_state_dict(peft_model, state_dict, strict=False)
        assert "unexpected.weight" in result.unexpected_keys

    def test_multiple_adapters(self):
        """Adapters added after the first one must not leak into the base model state dict either."""
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        base_model_keys = set(base_model.state_dict().keys())

        peft_model = get_peft_model(
            base_model, LoraConfig(r=4, lora_alpha=2, target_modules=["q_proj", "v_proj"]), adapter_name="adapter1"
        )
        peft_model.add_adapter("adapter2", LoraConfig(r=8, lora_alpha=4, target_modules=["k_proj", "out_proj"]))

        assert set(get_base_model_state_dict(peft_model).keys()) == base_model_keys
