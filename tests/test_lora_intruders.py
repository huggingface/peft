from functools import partial
from io import StringIO

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

from peft import LoraConfig, MissConfig, get_peft_model
from peft.tuners.lora.intruders import reduce_intruder_dimension

from .testing_utils import hub_online_once


class TestLoraIntruders:
    @pytest.fixture
    def model_lin(self):
        model_id = "peft-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

        cfg = LoraConfig(target_modules=["q_proj"])
        peft_model = get_peft_model(base_model, cfg)

        return peft_model

    @pytest.fixture
    def model_emb(self):
        model_id = "peft-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

        cfg = LoraConfig(target_modules=["embed_tokens"])
        peft_model = get_peft_model(base_model, cfg)

        return peft_model

    @pytest.fixture
    def model_conv1d(self):
        # Conv1D (as used in GPT-2) stores its weight as (in_features, out_features), i.e. fan_in_fan_out=True. Use
        # non-square dimensions so that a transposed weight shows up as a shape mismatch.
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = Conv1D(24, 16)

            def forward(self, x):
                return self.conv(x)

        cfg = LoraConfig(r=4, target_modules=["conv"], fan_in_fan_out=True)
        peft_model = get_peft_model(Net(), cfg)

        return peft_model

    @pytest.fixture
    def model_lin_bf16_no_autocast(self):
        model_id = "peft-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)

        cfg = LoraConfig(target_modules=["q_proj"])
        # autocast_adapter_dtype=False keeps the adapter weights in the base model's dtype
        # (bf16) instead of upcasting them to fp32.
        peft_model = get_peft_model(base_model, cfg, autocast_adapter_dtype=False)

        return peft_model

    @pytest.fixture
    def model_lin_non_lora(self):
        model_id = "peft-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

        cfg = MissConfig(target_modules=["q_proj"], r=8)
        peft_model = get_peft_model(base_model, cfg)

        return peft_model

    def test_lora_intruders_linear(self, model_lin):
        original_weights = {}
        for name, module in model_lin.named_modules():
            if "q_proj" in name and hasattr(module, "lora_B"):
                original_weights[name] = module.lora_B["default"].weight.detach().clone()

        buffer = StringIO()

        # use a high epsilon to make sure that we get a match to see whether layers get modified
        reduce_intruder_dimension(model_lin, threshold_epsilon=999, logging_sink=partial(print, file=buffer))

        # the old adapter should not be active anymore, just the new one. but the old one should still exist.
        assert model_lin.active_adapters == ["intruder_reduced"]
        assert set(model_lin.peft_config.keys()) == {"default", "intruder_reduced"}

        buffer.seek(0)
        lines = buffer.readlines()

        assert len(lines) > 0
        assert any("q_proj" in line for line in lines)

        for name, module in model_lin.named_modules():
            if name in original_weights:
                # Make sure that the original adapter was not modified
                assert torch.equal(module.lora_B["default"].weight.detach(), original_weights[name])

                # Since the epsilon is really low, we should modify every layer so the weights should differ
                new_weight = module.lora_B["intruder_reduced"].weight.detach()
                assert not torch.equal(new_weight, original_weights[name])

    def test_lora_intruders_linear_bf16_no_autocast(self, model_lin_bf16_no_autocast):
        # Regression test: with autocast_adapter_dtype=False, the base layer's weights (and thus
        # W_merged = W + dW) are bf16. torch.linalg.svd does not support half-precision dtypes, so
        # W_merged must be upcast to fp32 for the SVD calls just like W already is. Without that,
        # this call used to raise a RuntimeError.
        model_lin = model_lin_bf16_no_autocast

        original_dtypes = {}
        for name, module in model_lin.named_modules():
            if "q_proj" in name and hasattr(module, "lora_B"):
                original_dtypes[name] = module.lora_B["default"].weight.dtype

        # use a high epsilon to make sure that we get a match to see whether layers get modified
        reduce_intruder_dimension(model_lin, threshold_epsilon=999)

        assert model_lin.active_adapters == ["intruder_reduced"]

        for name, module in model_lin.named_modules():
            if name in original_dtypes:
                # The new adapter's dtype must match the old adapter's (and the base model's) dtype,
                # not be left as the float32 the SVD internally computed in. Both LoRA factors are
                # rebuilt from the SVD, so check A and B.
                assert module.lora_A["intruder_reduced"].weight.dtype == original_dtypes[name]
                assert module.lora_B["intruder_reduced"].weight.dtype == original_dtypes[name]
                assert original_dtypes[name] == torch.bfloat16

    def test_lora_intruders_embedding(self, model_emb):
        original_weights = {}
        original_shapes = {}
        for name, module in model_emb.named_modules():
            if "embed_tokens" in name and hasattr(module, "lora_B"):
                original_weights[name] = module.lora_embedding_B["default"].detach().clone()
                original_shapes[name] = (
                    module.lora_embedding_A["default"].shape,
                    module.lora_embedding_B["default"].shape,
                )

        buffer = StringIO()

        # use a high epsilon to make sure that we get a match to see whether layers get modified
        reduce_intruder_dimension(model_emb, threshold_epsilon=999, logging_sink=partial(print, file=buffer))

        # the old adapter should not be active anymore, just the new one. but the old one should still exist.
        assert model_emb.active_adapters == ["intruder_reduced"]
        assert set(model_emb.peft_config.keys()) == {"default", "intruder_reduced"}

        buffer.seek(0)
        lines = buffer.readlines()

        assert len(lines) > 0
        assert any("embed_tokens" in line for line in lines)

        for name, module in model_emb.named_modules():
            if name in original_weights:
                # Make sure that the original adapter was not modified
                assert torch.equal(module.lora_embedding_B["default"].detach(), original_weights[name])

                # Since the epsilon is really low, we should modify every layer so the weights should differ
                new_weight = module.lora_embedding_B["intruder_reduced"].detach()
                assert not torch.equal(new_weight, original_weights[name])

                # The embedding weight is stored transposed, the new adapter must keep the shapes of the old one
                assert module.lora_embedding_A["intruder_reduced"].shape == original_shapes[name][0]
                assert module.lora_embedding_B["intruder_reduced"].shape == original_shapes[name][1]

    def test_lora_intruders_conv1d(self, model_conv1d):
        layer = model_conv1d.base_model.model.conv
        original_A = layer.lora_A["default"].weight.detach().clone()
        original_B = layer.lora_B["default"].weight.detach().clone()

        buffer = StringIO()

        # use a high epsilon to make sure that we get a match to see whether layers get modified
        reduce_intruder_dimension(model_conv1d, threshold_epsilon=999, logging_sink=partial(print, file=buffer))

        # the old adapter should not be active anymore, just the new one. but the old one should still exist.
        assert model_conv1d.active_adapters == ["intruder_reduced"]
        assert set(model_conv1d.peft_config.keys()) == {"default", "intruder_reduced"}

        buffer.seek(0)
        lines = buffer.readlines()

        assert len(lines) > 0
        assert any("conv" in line for line in lines)

        # Make sure that the original adapter was not modified
        assert torch.equal(layer.lora_A["default"].weight.detach(), original_A)
        assert torch.equal(layer.lora_B["default"].weight.detach(), original_B)

        # The Conv1D weight is stored transposed, the new adapter must keep the shapes of the old one
        assert layer.lora_A["intruder_reduced"].weight.shape == original_A.shape
        assert layer.lora_B["intruder_reduced"].weight.shape == original_B.shape

        # Since the epsilon is really low, we should modify every layer so the weights should differ
        assert not torch.equal(layer.lora_B["intruder_reduced"].weight.detach(), original_B)

    @pytest.mark.parametrize(
        "layer_cls, layer_args, fan_in_fan_out",
        [
            (nn.Linear, (16, 16), False),
            (nn.Linear, (16, 24), False),
            (Conv1D, (16, 16), True),
            (Conv1D, (24, 16), True),
        ],
        ids=["linear", "linear_rect", "conv1d", "conv1d_rect"],
    )
    def test_lora_intruders_without_mitigation_keeps_output(self, layer_cls, layer_args, fan_in_fan_out):
        # With mitigation_lambda=1.0 nothing is subtracted, so the new adapter must reproduce the old adapter. A high
        # epsilon makes every top-k dimension count as an intruder, so the SVD path is taken. The tolerance covers
        # float32 error of a rank-4 SVD round trip on a 16-dim layer.
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = layer_cls(*layer_args)

            def forward(self, x):
                return self.layer(x)

        x = torch.randn(4, 16)
        config = LoraConfig(r=4, target_modules=["layer"], fan_in_fan_out=fan_in_fan_out, init_lora_weights=False)
        model = get_peft_model(Net(), config).eval()
        with torch.no_grad():
            expected = model(x)

        reduce_intruder_dimension(model, threshold_epsilon=999, mitigation_lambda=1.0, top_k=4, logging_sink=None)

        with torch.no_grad():
            output = model(x)
        torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-4)

    def test_lora_intruders_embedding_without_mitigation_keeps_output(self):
        # Same as above for an embedding layer, which needs integer inputs.
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Embedding(20, 16)

            def forward(self, x):
                return self.layer(x)

        x = torch.randint(0, 20, (4,))
        config = LoraConfig(r=4, target_modules=["layer"], init_lora_weights=False)
        model = get_peft_model(Net(), config).eval()
        with torch.no_grad():
            expected = model(x)

        reduce_intruder_dimension(model, threshold_epsilon=999, mitigation_lambda=1.0, top_k=4, logging_sink=None)

        with torch.no_grad():
            output = model(x)
        torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-4)

    def test_non_lora_intruders_linear_raises(self, model_lin_non_lora):
        with pytest.raises(ValueError) as e:
            reduce_intruder_dimension(model_lin_non_lora, threshold_epsilon=999)
            assert "The provided model is not using LoRA" in str(e)
