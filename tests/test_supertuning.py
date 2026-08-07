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

"""Super-Tuning-specific tests.

Generic PEFT behaviours (config instantiation, save / load round-trip, identity init, multiple adapters, base-weight
freezing, get_nb_trainable_parameters, etc.) are covered uniformly by tests/test_custom_models.py and
tests/test_decoder_models.py — Super-Tuning is registered in those suites via its config entry. Config validation lives
in tests/test_initialization.py::TestSupertuningInitialization.

This file keeps only the Super-Tuning-specific checks that the generic suites cannot express: magnitude-scoring's
data-free index selection, bottom-k support disjointness, Supra's sparse + LoRA compose semantics, and two regression
tests (bf16-base + fp32-LoRA dtype promotion, and LoRA A/B trainability under the tuner's prefix freeze pass).
"""

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from peft import SupertuningConfig, get_peft_model
from peft.tuners.supertuning.layer import Linear as SupertuningLinear
from peft.utils import infer_device


class TestSupertuning:
    device = infer_device()

    def _prepare_trainable_model(self, **config_kwargs):
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        kwargs = {"target_modules": ["q_proj", "v_proj"], "sparsity": 0.5}
        kwargs.update(config_kwargs)
        config = SupertuningConfig(**kwargs)
        return get_peft_model(model, config)

    def _supertuning_layers(self, model):
        return [module for module in model.modules() if isinstance(module, SupertuningLinear)]

    def test_supertuning_state_dict_stores_compact_support(self, tmp_path):
        """The adapter checkpoint stores the compact (indices, values) support — not a dense mask.

        This is Super-Tuning-specific: the storage shape (1-D pair sized to trainable count) is a design choice unique
        to this tuner, so the generic save-round-trip tests can't check it.
        """
        torch.manual_seed(0)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        config = SupertuningConfig(target_modules=["q_proj", "v_proj"], sparsity=0.5, init_weights=False)
        model = get_peft_model(model, config)
        model.save_pretrained(tmp_path)

        state_dict = load_file(tmp_path / "adapter_model.safetensors")
        assert any("supertuning_values" in key for key in state_dict)
        assert any("supertuning_indices" in key for key in state_dict)
        assert not any("sparse_mask" in key for key in state_dict)
        values_keys = [key for key in state_dict if "supertuning_values" in key]
        assert values_keys
        for key in values_keys:
            values = state_dict[key]
            indices = state_dict[key.replace("supertuning_values", "supertuning_indices")]
            assert values.ndim == 1
            assert indices.shape == values.shape
            # Indices MUST stay integer-typed. Regression guard: if PEFT's `other_param_names`
            # machinery ever casts the BufferDict to a float dtype, `scatter_add` would read
            # garbage from `.to(int64)` on those floats and produce out-of-bounds asserts on GPU.
            assert not indices.is_floating_point(), (
                f"supertuning_indices must not be cast to a floating-point dtype (got {indices.dtype})"
            )

    def test_supertuning_magnitude_scoring_populates_indices(self):
        """Magnitude scoring runs at construction time (data-free) and populates a non-empty support."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model()
        for _, module in model.named_modules():
            if hasattr(module, "supertuning_indices") and "default" in module.supertuning_indices:
                assert module.supertuning_indices["default"].numel() > 0

    def test_supertuning_bottomk_selects_disjoint_support(self):
        """`select_top=False` keeps the least-salient support — verify it's disjoint from the top-k support.

        Only the disjointness check remains; asserting that the smallest-magnitude entries have magnitude ≤ everything
        else just re-implements torch.topk's own contract, so it can't catch a bug in the selection code path it's
        supposed to test.
        """
        torch.manual_seed(0)
        top_model = self._prepare_trainable_model(sparsity=0.9, select_top=True)
        torch.manual_seed(0)
        bot_model = self._prepare_trainable_model(sparsity=0.9, select_top=False)

        top_layer = self._supertuning_layers(top_model)[0]
        bot_layer = self._supertuning_layers(bot_model)[0]
        top_idx = set(top_layer.supertuning_indices["default"].tolist())
        bot_idx = set(bot_layer.supertuning_indices["default"].tolist())
        assert top_idx.isdisjoint(bot_idx)

    def test_supra_hybrid_forward_composes_sparse_and_lora(self):
        """Supra forward equals (base + sparse + LoRA) applied as a single linear — composition semantics."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(r=4, init_weights=False)
        model.eval()

        layer = self._supertuning_layers(model)[0]
        weight = layer.get_base_layer().weight
        indices = layer.supertuning_indices["default"].to(torch.int64)
        values = layer.supertuning_values["default"].to(weight.dtype)

        # Seed lora_B to non-zero so the LoRA contribution is observable
        with torch.no_grad():
            layer.supertuning_lora_B["default"].weight.normal_(std=0.02)

        sparse_delta = torch.zeros_like(weight).reshape(-1).scatter_add(0, indices, values).reshape_as(weight)
        r = layer.supertuning_rank["default"]
        alpha = layer.supertuning_lora_alpha["default"]
        lora_delta = (
            (alpha / r) * (layer.supertuning_lora_B["default"].weight @ layer.supertuning_lora_A["default"].weight)
        ).to(weight.dtype)
        effective = weight.detach() + sparse_delta.detach() + lora_delta.detach()

        x = torch.randn(3, layer.in_features).to(self.device).to(weight.dtype)
        expected = torch.nn.functional.linear(x, effective, layer.get_base_layer().bias)
        assert torch.allclose(layer(x), expected, atol=1e-5)

    def test_supra_hybrid_merge_unmerge_round_trip(self):
        """Merging then unmerging a Supra adapter returns the base weight to its original value."""
        torch.manual_seed(0)
        model = self._prepare_trainable_model(r=4, init_weights=False)
        layer = self._supertuning_layers(model)[0]

        with torch.no_grad():
            layer.supertuning_lora_B["default"].weight.normal_(std=0.02)

        weight_before = layer.get_base_layer().weight.detach().clone()
        layer.merge()
        assert not torch.equal(layer.get_base_layer().weight.detach(), weight_before)
        layer.unmerge()
        assert torch.allclose(layer.get_base_layer().weight.detach(), weight_before, atol=1e-6)

    def test_supra_hybrid_forward_bf16_base_fp32_lora(self):
        """Regression: Supra forward under bf16 base + fp32 LoRA promotes activations correctly.

        LoRA is intentionally held in fp32 for training stability (matches PEFT LoRA convention), so the wrapper must
        promote incoming bf16 activations to the LoRA dtype and downcast the result. A prior version fed bf16
        activations directly into an fp32 matmul and crashed with `expected mat1 and mat2 to have the same dtype`.
        """
        if not torch.cuda.is_available():
            pytest.skip("bf16 requires CUDA")

        torch.manual_seed(0)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(self.device)
        config = SupertuningConfig(target_modules=["q_proj", "v_proj"], sparsity=0.5, r=4)
        model = get_peft_model(model, config)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        out = model(inputs)
        assert out.logits.dtype == torch.bfloat16

        out.logits.float().sum().backward()
        for _, module in model.named_modules():
            if hasattr(module, "supertuning_lora_A") and "default" in module.supertuning_lora_A:
                assert module.supertuning_lora_A["default"].weight.grad is not None
                assert module.supertuning_lora_B["default"].weight.grad is not None
                break

    def test_supra_lora_parameters_are_trainable(self):
        """Regression: LoRA A / B must be trainable in Supra mode.

        PEFT's `BaseTuner._mark_only_adapters_as_trainable` keys off `self.prefix` (`supertuning_`). An earlier version
        named the parameters `lora_A` / `lora_B`, which did NOT contain that prefix — the outer freeze pass then set
        their `requires_grad = False` while `save_pretrained` still serialised them (as `adapter_layer_names` was left
        broad), silently collapsing Supra to pure Super at the configured sparsity. The rename to `supertuning_lora_A`
        / `supertuning_lora_B` puts them under the tuner prefix; this test asserts both are trainable by the same
        accounting the harness uses (`sum p.numel() for p.requires_grad`).
        """
        torch.manual_seed(0)
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        config = SupertuningConfig(target_modules=["q_proj", "v_proj"], sparsity=0.5, r=4)
        model = get_peft_model(model, config)

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert any("supertuning_lora_A" in n for n in trainable_names)
        assert any("supertuning_lora_B" in n for n in trainable_names)

        for _, mod in model.named_modules():
            if hasattr(mod, "supertuning_lora_A") and "default" in mod.supertuning_lora_A:
                assert mod.supertuning_lora_A["default"].weight.requires_grad
                assert mod.supertuning_lora_B["default"].weight.requires_grad
                break

    def test_supertuning_reports_trainable_via_get_nb_trainable_parameters(self):
        """`model.get_nb_trainable_parameters()` reports the sparse + (optional) LoRA counts correctly.

        This is the canonical PEFT API for the parameter tally (what `print_trainable_parameters` prints). For Super at
        sparsity `s` on target modules `M`, trainable ≈ `sum((1-s)*numel(w))` for `w in M`.
        """
        torch.manual_seed(0)
        model = self._prepare_trainable_model(sparsity=0.5)
        trainable, total = model.get_nb_trainable_parameters()

        # every trainable parameter should be from the Super support (values) — no LoRA in this config
        support_params = sum(layer.supertuning_values["default"].numel() for layer in self._supertuning_layers(model))
        assert trainable == support_params
        assert total > trainable  # frozen base weights dominate the total
