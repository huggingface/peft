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

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from peft import PeftModel, SupertuningConfig, get_peft_model
from peft.tuners.supertuning.layer import Linear as SupertuningLinear
from peft.utils import infer_device


class TestSupertuning:
    device = infer_device()

    def test_supertuning_config(self):
        """Test that SupertuningConfig is properly configured."""
        config = SupertuningConfig(
            peft_type="SUPERTUNING",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        assert config.peft_type.value == "SUPERTUNING"
        assert config.sparsity == 0.5
        assert config.r is None  # pure Super by default

    def test_supertuning_config_validation(self):
        """Test that SupertuningConfig validates its parameters."""
        # Invalid sparsity
        with pytest.raises(ValueError, match="sparsity must be"):
            SupertuningConfig(sparsity=1.5)

        # Invalid selection_direction
        with pytest.raises(ValueError, match="selection_direction must be"):
            SupertuningConfig(selection_direction="sideways")

        # Invalid Supra rank
        with pytest.raises(ValueError, match="r must be a positive integer"):
            SupertuningConfig(r=0)

        # lora_alpha set without r
        with pytest.raises(ValueError, match="lora_alpha is set but r is None"):
            SupertuningConfig(lora_alpha=16.0)

    def test_supertuning_identity_init(self):
        """With zero-initialized values, the sparse update is the identity and does not change the output."""
        torch.manual_seed(0)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model.eval()
        output_base = model(inputs).logits

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
            init_weights=True,
        )
        model = get_peft_model(model, config)
        model.eval()
        output_peft = model(inputs).logits

        # values start at zero, so the effective weight equals the base weight exactly
        assert torch.allclose(output_base, output_peft, atol=1e-6, rtol=1e-6)

    def test_supertuning_state_dict(self, tmp_path):
        """Test that Supertuning saves only the compact (indices, values) support and round-trips."""
        torch.manual_seed(0)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
            # non-identity update so the round-trip actually exercises the trained values
            init_weights=False,
        )
        model = get_peft_model(model, config)
        model.eval()
        output_peft = model(inputs).logits

        model.save_pretrained(tmp_path)
        del model

        # the adapter checkpoint stores the compact sparse support: 1-D values and indices sized to the trainable
        # count, and nothing resembling a full dense mask.
        state_dict = load_file(tmp_path / "adapter_model.safetensors")
        assert any("supertuning_values" in key for key in state_dict)
        assert any("supertuning_indices" in key for key in state_dict)
        assert not any("sparse_mask" in key for key in state_dict)
        values_keys = [key for key in state_dict if "supertuning_values" in key]
        assert values_keys
        for key in values_keys:
            values = state_dict[key]
            indices = state_dict[key.replace("supertuning_values", "supertuning_indices")]
            # support is a 1-D pair sized to the trainable count (sparse storage, not the full weight numel)
            assert values.ndim == 1
            assert indices.shape == values.shape

        atol, rtol = 1e-5, 1e-8
        # the trained sparse values survive the round-trip
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model = PeftModel.from_pretrained(model, tmp_path)
        output_loaded = model(inputs).logits
        assert torch.allclose(output_peft, output_loaded, atol=atol, rtol=rtol)

    def test_supertuning_get_peft_model(self):
        """Test that get_peft_model works with Supertuning."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        model = get_peft_model(model, config)

        # Check that the model has the adapter
        assert hasattr(model, "peft_config")
        assert "default" in model.peft_config
        assert model.peft_config["default"].peft_type.value == "SUPERTUNING"

    def test_supertuning_trainable_parameters_count(self):
        """Test that trainable parameter count is computed correctly."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        model = get_peft_model(model, config)

        # Get trainable parameter count
        if hasattr(model, "get_trainable_parameters_count"):
            counts = model.get_trainable_parameters_count()
            assert "total_parameters" in counts
            assert "trainable_parameters" in counts
            assert "sparsity" in counts
            # Check that sparsity is close to the configured value
            assert 0.4 <= counts["sparsity"] <= 0.6  # Allow some tolerance

    def test_supertuning_magnitude_scoring(self):
        """Test that magnitude-only scoring works."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        model = get_peft_model(model, config)

        # Magnitude is the (only) automatic scoring path — indices are populated at construction time.
        for _, module in model.named_modules():
            if hasattr(module, "supertuning_indices") and "default" in module.supertuning_indices:
                assert module.supertuning_indices["default"].numel() > 0

    def test_supertuning_multiple_adapters(self):
        """Test that multiple adapters can be added."""
        torch.manual_seed(0)

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        config1 = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.5,
        )
        model = get_peft_model(model, config1, adapter_name="adapter1")

        config2 = SupertuningConfig(
            target_modules=["q_proj", "v_proj"],
            sparsity=0.3,
        )
        model.add_adapter("adapter2", config2)

        assert "adapter1" in model.peft_config
        assert "adapter2" in model.peft_config
        assert model.peft_config["adapter1"].sparsity == 0.5
        assert model.peft_config["adapter2"].sparsity == 0.3

    def _prepare_trainable_model(self, **config_kwargs):
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        kwargs = {"target_modules": ["q_proj", "v_proj"], "sparsity": 0.5}
        kwargs.update(config_kwargs)
        config = SupertuningConfig(**kwargs)
        model = get_peft_model(model, config)
        return model

    def _supertuning_layers(self, model):
        return [module for module in model.modules() if isinstance(module, SupertuningLinear)]

    def test_supertuning_base_weight_frozen(self):
        """The base weight is frozen; only the sparse ``values`` are trainable."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model()

        saw_layer = False
        for layer in self._supertuning_layers(model):
            saw_layer = True
            weight = layer.get_base_layer().weight
            assert weight.requires_grad is False
            assert layer.supertuning_values["default"].requires_grad is True
        assert saw_layer

    def test_supertuning_gradient_only_reaches_values(self):
        """Backward must not accumulate any gradient on the frozen base weight."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model()
        inputs = torch.arange(10).view(-1, 1).to(self.device)

        model(inputs).logits.float().sum().backward()

        saw_support_signal = False
        for layer in self._supertuning_layers(model):
            weight = layer.get_base_layer().weight
            values = layer.supertuning_values["default"]
            # the frozen weight receives no gradient at all
            assert weight.grad is None
            assert values.grad is not None
            if torch.any(values.grad != 0):
                saw_support_signal = True
        # the support still learns
        assert saw_support_signal

    def test_supertuning_optimizer_step_only_updates_support(self):
        """After an optimizer step, the frozen weight is untouched and only the sparse support changes.

        Uses AdamW, whose stateful update would leak into the frozen entries under the old gradient-masking mechanism
        but cannot here since the base weight receives no gradient.
        """
        torch.manual_seed(0)
        model = self._prepare_trainable_model()
        inputs = torch.arange(10).view(-1, 1).to(self.device)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0)
        layer = self._supertuning_layers(model)[0]
        weight = layer.get_base_layer().weight
        values = layer.supertuning_values["default"]
        weight_before = weight.detach().clone()
        values_before = values.detach().clone()

        model(inputs).logits.float().sum().backward()
        optimizer.step()

        # the frozen base weight is never modified by the optimizer
        assert torch.equal(weight_before, weight.detach())
        # the sparse support (values) is updated
        assert not torch.equal(values_before, values.detach())

    def test_supertuning_forward_applies_sparse_update(self):
        """The forward pass adds the sparse ``values`` at ``indices`` on top of the base weight."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(init_weights=False)
        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model.eval()

        layer = self._supertuning_layers(model)[0]
        weight = layer.get_base_layer().weight
        indices = layer.supertuning_indices["default"].to(torch.int64)
        values = layer.supertuning_values["default"].to(weight.dtype)

        # reconstruct the effective weight and compare against the module's own linear output
        effective = weight.detach().reshape(-1).scatter_add(0, indices, values.detach()).reshape_as(weight)
        x = torch.randn(3, layer.in_features).to(self.device).to(weight.dtype)
        expected = torch.nn.functional.linear(x, effective, layer.get_base_layer().bias)
        assert torch.allclose(layer(x), expected, atol=1e-5)
        # the update is non-trivial
        assert not torch.equal(effective, weight.detach())

    def test_supertuning_bottomk_selects_least_salient_support(self):
        """selection_direction='bottom' keeps the least-salient entries as the trainable support.

        Mirrors the paper's `magnitude-bottomk` (bottom-k) variant. Verifies (a) the selected support is
        disjoint from the TopK support at the same sparsity, and (b) every selected value has magnitude at most the
        magnitude of every non-selected value.
        """
        torch.manual_seed(0)
        top_model = self._prepare_trainable_model(sparsity=0.9, selection_direction="top")
        torch.manual_seed(0)
        bot_model = self._prepare_trainable_model(sparsity=0.9, selection_direction="bottom")

        top_layer = self._supertuning_layers(top_model)[0]
        bot_layer = self._supertuning_layers(bot_model)[0]
        top_idx = set(top_layer.supertuning_indices["default"].tolist())
        bot_idx = set(bot_layer.supertuning_indices["default"].tolist())

        # at sparsity 0.9 the TopK and BottomK 10% supports do not overlap
        assert top_idx.isdisjoint(bot_idx)

        # every selected magnitude is at most every non-selected magnitude — i.e. the BottomK support is the
        # least-salient set under magnitude scoring
        weight_flat = bot_layer.get_base_layer().weight.detach().flatten().abs()
        selected_mags = weight_flat[bot_layer.supertuning_indices["default"].long()]
        mask = torch.ones_like(weight_flat, dtype=torch.bool)
        mask[bot_layer.supertuning_indices["default"].long()] = False
        non_selected_mags = weight_flat[mask]
        assert selected_mags.max() <= non_selected_mags.min()

        # and the module still trains — a backward pass reaches the BottomK values
        inputs = torch.arange(10).view(-1, 1).to(self.device)
        bot_model(inputs).logits.float().sum().backward()
        assert bot_layer.supertuning_values["default"].grad is not None
        assert bot_layer.get_base_layer().weight.grad is None


    def test_supertuning_set_precomputed_indices(self):
        """User-supplied indices override the magnitude-computed ones and reset ``values`` to zero."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(sparsity=0.9, init_weights=False)  # values non-zero at first
        # Names must be relative to the inner base model (what SupertuningModel iterates), NOT the outer PeftModel.
        # This matches how users typically compute indices — on the raw base model, before get_peft_model wraps it.
        inner = model.base_model.model
        layers_by_name = {
            name: mod for name, mod in inner.named_modules() if hasattr(mod, "supertuning_indices")
        }
        # Non-magnitude choice — first N flat positions, deterministic and disjoint from magnitude top-k on this init.
        overrides = {}
        for name, mod in layers_by_name.items():
            n = mod.supertuning_values["default"].numel()
            device = mod.get_base_layer().weight.device
            overrides[name] = torch.arange(n, dtype=torch.int32, device=device)

        model.base_model.set_precomputed_indices(overrides)

        for name, mod in layers_by_name.items():
            assert torch.equal(mod.supertuning_indices["default"], overrides[name])
            # ``set_precomputed_indices`` re-zeros values so the new support starts as an identity update
            assert torch.all(mod.supertuning_values["default"] == 0)

    def test_supertuning_set_precomputed_indices_wrong_length_raises(self):
        """Indices whose count doesn't match the sparse budget raise a clear ValueError."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(sparsity=0.5)
        # Names must be relative to the inner base model (what SupertuningModel iterates), NOT the outer PeftModel.
        # This matches how users typically compute indices — on the raw base model, before get_peft_model wraps it.
        inner = model.base_model.model
        layers_by_name = {
            name: mod for name, mod in inner.named_modules() if hasattr(mod, "supertuning_indices")
        }
        name = next(iter(layers_by_name))
        mod = layers_by_name[name]
        wrong = torch.zeros(mod.supertuning_values["default"].numel() + 1, dtype=torch.int32, device=self.device)
        with pytest.raises(ValueError, match="to match the sparse budget"):
            model.base_model.set_precomputed_indices({name: wrong})

    def test_supra_hybrid_allocates_lora_parameters(self):
        """When ``r`` is set, the layer allocates LoRA A and B parameters."""
        torch.manual_seed(0)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        config = SupertuningConfig(target_modules=["q_proj", "v_proj"], sparsity=0.5, r=4)
        model = get_peft_model(model, config)

        counts = model.base_model.get_trainable_parameters_count()
        assert counts["sparse_parameters"] > 0
        assert counts["lora_parameters"] > 0
        assert counts["trainable_parameters"] == counts["sparse_parameters"] + counts["lora_parameters"]

        # lora_alpha defaults to 2*r
        assert model.peft_config["default"].lora_alpha == 8.0

    def test_supra_hybrid_forward_composes_sparse_and_lora(self):
        """The Supra forward output equals the composed effective weight (sparse + LoRA) applied to the input."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(r=4, init_weights=False)
        model.eval()

        layer = self._supertuning_layers(model)[0]
        weight = layer.get_base_layer().weight
        indices = layer.supertuning_indices["default"].to(torch.int64)
        values = layer.supertuning_values["default"].to(weight.dtype)

        # Seed lora_B to a non-zero value so the LoRA contribution is observable
        with torch.no_grad():
            layer.lora_B["default"].normal_(std=0.02)

        sparse_delta = torch.zeros_like(weight).reshape(-1).scatter_add(0, indices, values).reshape_as(weight)
        r = layer.supertuning_rank["default"]
        alpha = layer.supertuning_lora_alpha["default"]
        lora_delta = ((alpha / r) * (layer.lora_B["default"] @ layer.lora_A["default"])).to(weight.dtype)
        effective = weight.detach() + sparse_delta.detach() + lora_delta.detach()

        x = torch.randn(3, layer.in_features).to(self.device).to(weight.dtype)
        expected = torch.nn.functional.linear(x, effective, layer.get_base_layer().bias)
        assert torch.allclose(layer(x), expected, atol=1e-5)

    def test_supra_hybrid_merge_unmerge_round_trip(self):
        """Merging then unmerging a Supra adapter returns the base weight to its original value."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(r=4, init_weights=False)
        layer = self._supertuning_layers(model)[0]

        # Seed lora_B so the LoRA contribution is non-zero
        with torch.no_grad():
            layer.lora_B["default"].normal_(std=0.02)

        weight_before = layer.get_base_layer().weight.detach().clone()
        layer.merge()
        # After merge, the base weight has been modified in place
        assert not torch.equal(layer.get_base_layer().weight.detach(), weight_before)
        layer.unmerge()
        assert torch.allclose(layer.get_base_layer().weight.detach(), weight_before, atol=1e-6)
