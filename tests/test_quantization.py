# Copyright 2025-present the HuggingFace Inc. team.
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

"""
Test PEFT method x quantization method matrix, focusing on basic tests.
"""

from dataclasses import dataclass

import pytest
import torch
from accelerate.utils.memory import clear_device_cache
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TorchAoConfig

from peft import MissConfig, VeraConfig, get_peft_model
from peft.import_utils import is_bnb_4bit_available, is_bnb_available, is_torchao_available
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import infer_device
from peft.utils.quantization_utils import Bnb4bitBackend, Bnb8bitBackend, TorchaoBackend

from .testing_utils import hub_online_once, set_init_weights_false


MODEL_ID = "peft-internal-testing/opt-125m"
SEED = 0
DEVICE = infer_device()


@dataclass
class Bnb8bitLoader:
    name = "bnb_8bit"
    backend_cls = Bnb8bitBackend
    supports_merge = True

    @staticmethod
    def load_model():
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        with hub_online_once(MODEL_ID):
            return AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config).to(DEVICE)


@dataclass
class Bnb4bitLoader:
    name = "bnb_4bit"
    backend_cls = Bnb4bitBackend
    supports_merge = True

    @staticmethod
    def load_model():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float32,
        )
        with hub_online_once(MODEL_ID):
            return AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config).to(DEVICE)


@dataclass
class TorchAoInt8WeightOnlyLoader:
    name = "torchao_int8_weight_only"
    backend_cls = TorchaoBackend
    supports_merge = True

    @staticmethod
    def load_model():
        from torchao.quantization import Int8WeightOnlyConfig

        quant_config = TorchAoConfig(quant_type=Int8WeightOnlyConfig())
        with hub_online_once(MODEL_ID):
            return AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config).to(DEVICE)


@dataclass
class TorchAoInt8DynamicActivationInt8WeightLoader:
    name = "torchao_int8_dynamic_activation_int8"
    backend_cls = TorchaoBackend
    supports_merge = False

    @staticmethod
    def load_model():
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig

        quant_config = TorchAoConfig(quant_type=Int8DynamicActivationInt8WeightConfig())
        with hub_online_once(MODEL_ID):
            return AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config).to(DEVICE)


QUANTIZATION_BACKENDS = []
if is_bnb_available():
    QUANTIZATION_BACKENDS.append(Bnb8bitLoader())
if is_bnb_4bit_available():
    QUANTIZATION_BACKENDS.append(Bnb4bitLoader())
if is_torchao_available():
    QUANTIZATION_BACKENDS.append(TorchAoInt8WeightOnlyLoader())
    QUANTIZATION_BACKENDS.append(TorchAoInt8DynamicActivationInt8WeightLoader())


def _quant_id(backend):
    return backend.name


TEST_CASES = [
    (
        MissConfig,
        {"r": 2},
    ),
    (
        MissConfig,
        {"r": 2, "init_weights": "bat"},
    ),
    (
        VeraConfig,
        {"r": 8, "target_modules": ["q_proj", "v_proj"]},
    ),
]


def _peft_id(val):
    """Generate test id config_cls / config_kwargs."""
    if isinstance(val, dict):
        id_ = str(val).replace(" ", "")
    else:  # the PEFT config class
        id_ = val.__name__.removesuffix("Config").lower()
    return id_


def check_outputs_similar(x, y, min_corr=0.9, max_mse=1.0):
    # As quantization introduces a lot of error, use generous tolerances
    assert x.shape == y.shape
    corr = torch.corrcoef(torch.stack((x.flatten(), y.flatten())))
    mse = ((x - y) ** 2).mean()

    corr_checks = corr[0, 1] >= min_corr
    mse_checks = mse <= max_mse
    if not corr_checks and not mse_checks:
        assert False, f"both correlation ({corr[0, 1]:.4f}>={min_corr}) and MSE ({mse:.4f}<={max_mse}) check failed"
    if not corr_checks:
        assert False, f"correlation ({corr[0, 1]:.4f}>={min_corr}) check failed"
    if not mse_checks:
        assert False, f"MSE ({mse:.4f}<={max_mse}) check failed"


class TestQuantization:
    """Test for PEFT method x quantization method

    Note: It is recommended to keep the number of tests low, as the number of combinations is already large as is. This
    means testing multiple things per test, even if this is generally not desired. The reason is that we want to keep
    the number of model initializations to a minimum, as those take time.

    """

    @pytest.fixture(autouse=True)
    def set_seed(self):
        torch.manual_seed(SEED)

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        clear_device_cache(garbage_collection=True)

    @pytest.fixture
    def dummy_input(self):
        return torch.arange(10).view(1, -1).to(DEVICE)

    @pytest.mark.parametrize("quant", QUANTIZATION_BACKENDS, ids=_quant_id)
    @pytest.mark.parametrize("config_cls,config_kwargs", TEST_CASES, ids=_peft_id)
    def test_quantization_backend_is_set_and_repr(self, config_cls, config_kwargs, quant):
        """PEFT layers should have quantization_backend set"""
        model = quant.load_model()
        config = config_cls(**config_kwargs)
        model = get_peft_model(model, config)

        quantized_layers = [
            m for m in model.modules() if isinstance(m, BaseTunerLayer) and m.quantization_backend is not None
        ]
        assert len(quantized_layers) == 24  # (q_proj, v_proj) x 12 layers
        for layer in quantized_layers:
            rep = repr(layer)
            assert "quantization_backend=" in rep

    @pytest.mark.parametrize("quant", QUANTIZATION_BACKENDS, ids=_quant_id)
    @pytest.mark.parametrize("config_cls,config_kwargs", TEST_CASES, ids=_peft_id)
    def test_forward_changes_output(self, config_cls, config_kwargs, quant, dummy_input):
        """Check that the forward pass works, also check if the results are affected"""
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        model = quant.load_model()

        with torch.inference_mode():
            out_base = model(dummy_input).logits

        config = config_cls(**config_kwargs)
        model = get_peft_model(model, config)

        with torch.inference_mode():
            out_peft = model(dummy_input).logits

        atol, rtol = 1e-3, 1e-3
        assert not torch.allclose(out_base, out_peft, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("quant", QUANTIZATION_BACKENDS, ids=_quant_id)
    @pytest.mark.parametrize("config_cls,config_kwargs", TEST_CASES, ids=_peft_id)
    def test_quantized_output_similar_to_non_quantized(self, config_cls, config_kwargs, quant, dummy_input):
        """Quantized PEFT output should be similar to non-quantized PEFT output.

        Both models use the same adapter config with non-identity init. The outputs won't match exactly due to
        quantization noise, but should be in the same ballpark.
        """
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)

        # Quantized model
        model = quant.load_model()
        config = config_cls(**config_kwargs)
        torch.manual_seed(SEED)
        model = get_peft_model(model, config).eval()

        with torch.inference_mode():
            out_quant = model(dummy_input).logits

        del model

        # Non-quantized model
        with hub_online_once(MODEL_ID):
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        config = config_cls(**config_kwargs.copy())
        torch.manual_seed(SEED)
        model = get_peft_model(model, config).eval()
        model_non_quant = model.to(quant.load_model().device)

        with torch.inference_mode():
            out_non_quant = model(dummy_input).logits

        check_outputs_similar(out_non_quant, out_quant)

    @pytest.mark.parametrize("quant", QUANTIZATION_BACKENDS, ids=_quant_id)
    @pytest.mark.parametrize("config_cls,config_kwargs", TEST_CASES, ids=_peft_id)
    def test_merge_unmerge_unload(self, config_cls, config_kwargs, quant, dummy_input):
        """Check merge and unmerge roundtrip"""
        if not quant.supports_merge:
            pytest.skip(f"{quant.name} does not support merging")

        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        model = quant.load_model()
        config = config_cls(**config_kwargs)
        torch.manual_seed(SEED)
        model = get_peft_model(model, config).eval()

        with torch.inference_mode():
            out_before = model(dummy_input).logits

        model.merge_adapter()
        with torch.inference_mode():
            out_merged = model(dummy_input).logits

        check_outputs_similar(out_before, out_merged)

        model.unmerge_adapter()
        with torch.inference_mode():
            out_unmerged = model(dummy_input).logits

        check_outputs_similar(out_before, out_unmerged)

        model = model.merge_and_unload()
        with torch.inference_mode():
            out_unloaded = model(dummy_input).logits

        check_outputs_similar(out_before, out_unloaded)
