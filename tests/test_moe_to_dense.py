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

import copy
import json
import os
import warnings

import pytest
import torch
from safetensors.torch import load_file as safe_load_file
from transformers import (
    AutoModelForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    GptOssConfig,
    GptOssForCausalLM,
    NemotronHConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

from peft import MoeToDenseConfig, PeftModel, get_peft_model
from peft.tuners.moe_to_dense.arch import ExpertsLayout, build_dense_expert_tensors, is_experts_module
from peft.tuners.moe_to_dense.layer import MoeToDenseLayer, forward_kl_divergence
from peft.tuners.moe_to_dense.scoring import conditional_prob_scores


VOCAB_SIZE = 64
HIDDEN_SIZE = 16
EXPERT_INTERMEDIATE_SIZE = 8
NUM_EXPERTS = 6
TOP_K = 2
NUM_LAYERS = 2
MODEL_TYPES = ["qwen3_moe", "gpt_oss", "gemma4"]


def build_model(model_type: str, experts_implementation: str = "eager", seed: int = 0):
    torch.manual_seed(seed)
    common = {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "experts_implementation": experts_implementation,
    }
    if model_type == "qwen3_moe":
        config = Qwen3MoeConfig(
            intermediate_size=24,
            moe_intermediate_size=EXPERT_INTERMEDIATE_SIZE,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            norm_topk_prob=True,
            **common,
        )
        model = Qwen3MoeForCausalLM(config)
    elif model_type == "gpt_oss":
        config = GptOssConfig(
            intermediate_size=EXPERT_INTERMEDIATE_SIZE,
            num_local_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            sliding_window=8,
            **common,
        )
        model = GptOssForCausalLM(config)
    elif model_type == "gemma4":
        config = Gemma4TextConfig(
            intermediate_size=24,
            enable_moe_block=True,
            num_experts=NUM_EXPERTS,
            top_k_experts=TOP_K,
            moe_intermediate_size=EXPERT_INTERMEDIATE_SIZE,
            vocab_size_per_layer_input=VOCAB_SIZE,
            hidden_size_per_layer_input=4,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            **common,
        )
        model = Gemma4ForCausalLM(config)
    else:
        raise ValueError(model_type)

    # make sure that expert and router weights are random so that routing is non-trivial
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(key in name for key in ("experts", "router", "mlp.gate.", "per_expert_scale")):
                param.normal_(0.0, 0.2)
    return model.eval()


def get_inputs(seed: int = 1):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, -2:] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_layers(peft_model) -> list[MoeToDenseLayer]:
    return [module for module in peft_model.modules() if isinstance(module, MoeToDenseLayer)]


def calibrate(peft_model, num_batches: int = 3):
    # the first batch runs under inference mode (as e.g. `generate` does), the others under no_grad
    with torch.inference_mode():
        peft_model(**get_inputs(seed=10))
    with torch.no_grad():
        for i in range(1, num_batches):
            peft_model(**get_inputs(seed=10 + i))


@pytest.fixture(params=MODEL_TYPES)
def model_type(request):
    return request.param


@pytest.fixture
def base_model(model_type):
    return build_model(model_type)


@pytest.fixture
def peft_model(base_model):
    return get_peft_model(copy.deepcopy(base_model), MoeToDenseConfig(task_type="CAUSAL_LM"))


@pytest.fixture
def allocated_model(peft_model):
    calibrate(peft_model)
    peft_model.update_and_allocate()
    return peft_model


class TestMoeToDense:
    def test_wrapping(self, peft_model, model_type):
        layers = get_layers(peft_model)
        assert len(layers) == NUM_LAYERS
        trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        assert trainable
        assert all("moe_to_dense_experts" in n for n in trainable)

        dense_intermediate = TOP_K * EXPERT_INTERMEDIATE_SIZE
        for layer in layers:
            assert not layer.is_allocated("default")
            dense = layer.moe_to_dense_experts["default"]
            assert type(dense) is type(layer.base_layer)
            if model_type == "gpt_oss":  # transposed layout with biases
                assert dense.gate_up_proj.shape == (1, HIDDEN_SIZE, 2 * dense_intermediate)
                assert dense.down_proj.shape == (1, dense_intermediate, HIDDEN_SIZE)
                assert dense.gate_up_proj_bias.shape == (1, 2 * dense_intermediate)
                assert dense.down_proj_bias.shape == (1, HIDDEN_SIZE)
            else:
                assert dense.gate_up_proj.shape == (1, 2 * dense_intermediate, HIDDEN_SIZE)
                assert dense.down_proj.shape == (1, HIDDEN_SIZE, dense_intermediate)
            # the router is not part of the adapter
            assert layer.router_name not in dict(layer.named_children())
            assert layer._router_hook_handle is not None
        peft_model.print_trainable_parameters()

    def test_explicit_target_modules(self, base_model, model_type):
        target = "experts"
        peft_model = get_peft_model(copy.deepcopy(base_model), MoeToDenseConfig(target_modules=[target]))
        assert len(get_layers(peft_model)) == NUM_LAYERS

    def test_forward_before_allocation_is_passthrough(self, base_model, peft_model):
        inputs = get_inputs()
        with torch.no_grad():
            expected = base_model(**inputs).logits
        with pytest.warns(UserWarning, match="has not been allocated yet"), torch.no_grad():
            output = peft_model(**inputs).logits
        assert torch.allclose(output, expected)
        # the warning is only given once
        with torch.no_grad():
            peft_model(**inputs)

    def test_update_and_allocate(self, allocated_model, model_type):
        layers = get_layers(allocated_model)
        for layer in layers:
            assert layer.is_allocated("default")
            assert layer._router_hook_handle is None
            selected = layer.selected_experts["default"]
            assert len(selected) == TOP_K == len(set(selected))
            scores = conditional_prob_scores(layer.stats)
            assert sorted(selected) == sorted(torch.topk(scores, TOP_K).indices.tolist())

        # calling it again is a no-op
        with pytest.warns(UserWarning, match="already been allocated"):
            allocated_model.update_and_allocate()

        with torch.no_grad():
            output = allocated_model(**get_inputs()).logits
        assert torch.isfinite(output).all()

    def test_allocate_without_statistics_raises(self, peft_model):
        with pytest.raises(RuntimeError, match="router was never called"):
            peft_model.update_and_allocate()

    def test_uninterpretable_router_output(self, peft_model):
        # simulate a router whose output structure is not understood by calling the hook directly
        layers = get_layers(peft_model)
        output = {"nope": torch.zeros(14, NUM_EXPERTS)}
        with pytest.warns(UserWarning, match="could not be interpreted.*but got dict"):
            layers[0]._router_hook(layers[0].router, (), output)
        # warned only once per layer
        layers[0]._router_hook(layers[0].router, (), output)
        with pytest.raises(RuntimeError, match=r"router .* was called 2 times, because its output could not be"):
            peft_model.update_and_allocate()

    def test_dense_ffn_is_exact_concatenation_of_selected_experts(self, allocated_model, model_type):
        # The dense FFN must compute the same as the MoE experts when the router selects exactly the kept experts with
        # uniform weights (Appendix A of the paper).
        torch.manual_seed(0)
        num_tokens = 5
        for layer in get_layers(allocated_model):
            hidden_states = torch.randn(num_tokens, HIDDEN_SIZE)
            selected = torch.tensor(layer.selected_experts["default"])
            indices = selected.unsqueeze(0).expand(num_tokens, -1)
            weights = torch.full((num_tokens, TOP_K), 1.0 / TOP_K)
            if model_type == "gemma4":
                weights = weights * layer.router.per_expert_scale.detach()[selected]
            with torch.no_grad():
                expected = layer.base_layer(hidden_states, indices, weights)
                actual = layer(hidden_states, indices, weights)
            assert torch.allclose(actual, expected, atol=1e-5)

            # sanity check: a different selection gives a different result
            other = torch.tensor([e for e in range(NUM_EXPERTS) if e not in selected.tolist()][:TOP_K])
            with torch.no_grad():
                different = layer.base_layer(hidden_states, other.unsqueeze(0).expand(num_tokens, -1), weights)
            assert not torch.allclose(actual, different, atol=1e-5)

    def test_expert_concatenation_without_gate(self):
        # None of the architectures above use a gateless FFN (`has_gate=False`), but NemotronH does. Its full model is
        # a Mamba hybrid, too heavy for a tiny test, so check the expert concatenation on its experts class directly.
        torch.manual_seed(0)
        config = NemotronHConfig(
            hidden_size=HIDDEN_SIZE, n_routed_experts=NUM_EXPERTS, moe_intermediate_size=EXPERT_INTERMEDIATE_SIZE
        )
        experts = NemotronHExperts(config)
        with torch.no_grad():
            experts.up_proj.normal_(0.0, 0.2)
            experts.down_proj.normal_(0.0, 0.2)

        assert is_experts_module(experts)
        layout = ExpertsLayout.from_module(experts)
        assert (layout.has_gate, layout.has_bias, layout.is_transposed) == (False, False, False)
        layout.validate_module(experts, NUM_EXPERTS, EXPERT_INTERMEDIATE_SIZE)

        selected = torch.tensor([4, 1])
        scales = torch.tensor([0.25, 0.75])
        tensors = build_dense_expert_tensors(experts, layout, selected, scales)
        assert set(tensors) == {"up_proj", "down_proj"}
        assert tensors["up_proj"].shape == (1, TOP_K * EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE)
        assert tensors["down_proj"].shape == (1, HIDDEN_SIZE, TOP_K * EXPERT_INTERMEDIATE_SIZE)

        # the single dense expert computes the same as the selected experts weighted by the scales (Appendix A)
        dense_config = copy.deepcopy(config)
        dense_config.n_routed_experts = 1
        dense_config.moe_intermediate_size = TOP_K * EXPERT_INTERMEDIATE_SIZE
        dense = NemotronHExperts(dense_config)
        with torch.no_grad():
            for name, tensor in tensors.items():
                getattr(dense, name).copy_(tensor)

        num_tokens = 5
        hidden_states = torch.randn(num_tokens, HIDDEN_SIZE)
        indices = selected.unsqueeze(0).expand(num_tokens, -1)
        weights = scales.unsqueeze(0).expand(num_tokens, -1)
        with torch.no_grad():
            expected = experts(hidden_states, indices, weights)
            actual = dense(
                hidden_states,
                torch.zeros((num_tokens, 1), dtype=torch.long),
                torch.ones((num_tokens, 1)),
            )
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_num_experts_to_keep(self, base_model):
        config = MoeToDenseConfig(num_experts_to_keep=3)
        peft_model = get_peft_model(copy.deepcopy(base_model), config)
        calibrate(peft_model)
        peft_model.update_and_allocate()
        for layer in get_layers(peft_model):
            assert len(layer.selected_experts["default"]) == 3
            dense = layer.moe_to_dense_experts["default"]
            assert dense.down_proj.numel() == HIDDEN_SIZE * 3 * EXPERT_INTERMEDIATE_SIZE

        with pytest.raises(ValueError, match="only has"):
            get_peft_model(copy.deepcopy(base_model), MoeToDenseConfig(num_experts_to_keep=NUM_EXPERTS + 1))

    def test_disable_adapter_gives_teacher_output(self, base_model, allocated_model):
        inputs = get_inputs()
        with torch.no_grad():
            expected = base_model(**inputs).logits
            student = allocated_model(**inputs).logits
            with allocated_model.disable_adapter():
                teacher = allocated_model(**inputs).logits
        assert torch.allclose(teacher, expected)
        assert not torch.allclose(student, expected)

    def test_generate(self, allocated_model):
        inputs = get_inputs()
        output = allocated_model.generate(**inputs, max_new_tokens=5, do_sample=False)
        assert output.shape == (2, 7 + 5)
        assert ((output >= 0) & (output < VOCAB_SIZE)).all()
        # generating with the teacher works as well
        with allocated_model.disable_adapter():
            output = allocated_model.generate(**inputs, max_new_tokens=5, do_sample=False)
        assert output.shape == (2, 7 + 5)

    def test_distillation_loss(self, allocated_model):
        inputs = get_inputs()
        loss = allocated_model.get_distillation_loss(**inputs)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss > 0

        loss.backward()
        for name, param in allocated_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, name
                assert torch.isfinite(param.grad).all(), name
            else:
                assert param.grad is None, name
        allocated_model.zero_grad()

        # passing the teacher logits explicitly gives the same result
        with torch.no_grad(), allocated_model.disable_adapter():
            teacher_logits = allocated_model(**inputs).logits
        loss_explicit = allocated_model.get_distillation_loss(**inputs, teacher_logits=teacher_logits)
        assert torch.allclose(loss, loss_explicit)

        # labels: only the tokens with a label != -100 count
        labels = inputs["input_ids"].clone()
        labels[:, :3] = -100
        loss_labels = allocated_model.get_distillation_loss(**inputs, labels=labels)
        assert torch.isfinite(loss_labels)
        assert not torch.allclose(loss, loss_labels)

        # temperature
        loss_temp = allocated_model.get_distillation_loss(**inputs, temperature=2.0)
        assert torch.isfinite(loss_temp)
        assert not torch.allclose(loss, loss_temp)

    def test_distillation_loss_is_zero_before_allocation(self, peft_model):
        with pytest.warns(UserWarning, match="has not been allocated yet"):
            loss = peft_model.get_distillation_loss(**get_inputs())
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

    def test_training_reduces_distillation_loss(self, allocated_model):
        # mini distillation run on a single batch: the loss decreases and only the adapter parameters are updated
        torch.manual_seed(0)
        frozen_before = {n: p.clone() for n, p in allocated_model.named_parameters() if not p.requires_grad}
        trainable_before = {n: p.clone() for n, p in allocated_model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.Adam((p for p in allocated_model.parameters() if p.requires_grad), lr=1e-2)
        inputs = get_inputs()

        losses = []
        for _ in range(10):
            loss = allocated_model.get_distillation_loss(**inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

        for name, param in allocated_model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(param, trainable_before[name]), name
            else:
                assert torch.equal(param, frozen_before[name]), name

    def test_forward_kl_divergence_chunking(self):
        torch.manual_seed(0)
        teacher = torch.randn(2, 9, 11)
        student = torch.randn(2, 9, 11, requires_grad=True)
        mask = torch.ones(2, 9, dtype=torch.bool)
        mask[0, :4] = False

        kl_per_token = torch.nn.functional.kl_div(
            torch.log_softmax(student, -1), torch.log_softmax(teacher, -1), log_target=True, reduction="none"
        ).sum(-1)
        expected = (kl_per_token * mask).sum() / mask.sum()
        for chunk_size in (1, 4, 100):
            loss = forward_kl_divergence(teacher, student, mask=mask, chunk_size=chunk_size)
            assert torch.allclose(loss, expected, atol=1e-6)
            (grad,) = torch.autograd.grad(loss, student)
            assert torch.isfinite(grad).all()

        # without a mask, all tokens count
        expected_no_mask = kl_per_token.mean()
        assert not torch.allclose(expected_no_mask, expected)  # sanity check: the two cases are distinguishable
        assert torch.allclose(forward_kl_divergence(teacher, student, chunk_size=4), expected_no_mask, atol=1e-6)

    def test_save_and_load(self, base_model, allocated_model, tmp_path):
        # perturb the dense weights so that the loaded adapter is distinguishable from a freshly allocated one
        with torch.no_grad():
            for layer in get_layers(allocated_model):
                for param in layer.moe_to_dense_experts["default"].parameters():
                    param.add_(0.1 * torch.randn_like(param))
        inputs = get_inputs()
        with torch.no_grad():
            expected = allocated_model(**inputs).logits

        allocated_model.save_pretrained(tmp_path)
        assert os.path.exists(tmp_path / "adapter_config.json")
        state_dict = safe_load_file(tmp_path / "adapter_model.safetensors")
        assert state_dict
        assert all("moe_to_dense_experts" in key for key in state_dict)
        assert any(key.endswith(".allocated") for key in state_dict)
        with open(tmp_path / "adapter_config.json") as f:
            config = json.load(f)
        assert config["peft_type"] == "MOE_TO_DENSE"

        loaded = PeftModel.from_pretrained(copy.deepcopy(base_model), tmp_path)
        for layer in get_layers(loaded):
            assert layer.is_allocated("default")
        with torch.no_grad():
            output = loaded(**inputs).logits  # no calibration needed, no warning
        assert torch.allclose(output, expected, atol=1e-6)
        assert not any(p.requires_grad for p in loaded.parameters())

        # loading as trainable
        loaded = PeftModel.from_pretrained(copy.deepcopy(base_model), tmp_path, is_trainable=True)
        assert any(p.requires_grad for p in loaded.parameters())

        # loading with low_cpu_mem_usage (adapter weights are created on the meta device first)
        loaded = PeftModel.from_pretrained(copy.deepcopy(base_model), tmp_path, low_cpu_mem_usage=True)
        with torch.no_grad():
            output = loaded(**inputs).logits
        assert torch.allclose(output, expected, atol=1e-6)

    def test_non_default_adapter_name(self, tmp_path):
        # the whole workflow works with an adapter name other than "default"
        base_model = build_model("qwen3_moe")
        peft_model = get_peft_model(
            copy.deepcopy(base_model), MoeToDenseConfig(task_type="CAUSAL_LM"), adapter_name="other"
        )
        calibrate(peft_model)
        peft_model.update_and_allocate()
        for layer in get_layers(peft_model):
            assert layer.is_allocated("other")
            assert set(layer.moe_to_dense_experts) == {"other"}

        inputs = get_inputs()
        loss = peft_model.get_distillation_loss(**inputs)
        assert torch.isfinite(loss)
        with torch.no_grad():
            expected = peft_model(**inputs).logits

        # non-default adapters are saved in a subdirectory named after the adapter
        peft_model.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(copy.deepcopy(base_model), tmp_path / "other", adapter_name="other")
        with torch.no_grad():
            output = loaded(**inputs).logits
        assert torch.allclose(output, expected, atol=1e-6)

        dense_model = peft_model.compress_and_unload()
        with torch.no_grad():
            output = dense_model(**inputs).logits
        assert torch.allclose(output, expected, atol=1e-6)

    def test_compress_and_unload(self, allocated_model, model_type, tmp_path):
        inputs = get_inputs()
        with torch.no_grad():
            expected = allocated_model(**inputs).logits
        num_params_before = sum(p.numel() for p in allocated_model.parameters())

        dense_model = allocated_model.compress_and_unload()
        assert not get_layers(dense_model)
        assert not hasattr(dense_model, "peft_config")
        assert sum(p.numel() for p in dense_model.parameters()) < num_params_before
        assert not any(p.requires_grad for p in dense_model.parameters())
        with torch.no_grad():
            output = dense_model(**inputs).logits
        assert torch.allclose(output, expected, atol=1e-6)

        config = dense_model.config
        dense_intermediate = TOP_K * EXPERT_INTERMEDIATE_SIZE
        if model_type == "qwen3_moe":
            # true dense export
            assert config.mlp_only_layers == list(range(NUM_LAYERS))
            assert config.intermediate_size == dense_intermediate
            assert all(type(layer.mlp).__name__ == "Qwen3MoeMLP" for layer in dense_model.model.layers)
        elif model_type == "gpt_oss":
            assert config.num_local_experts == 1
            assert config.num_experts_per_tok == 1
            assert config.intermediate_size == dense_intermediate
        else:
            assert config.num_experts == 1
            assert config.top_k_experts == 1
            assert config.moe_intermediate_size == dense_intermediate

        # the dense model can be saved and loaded without PEFT
        dense_model.save_pretrained(tmp_path)
        state_dict = safe_load_file(tmp_path / "model.safetensors")
        assert not any("moe_to_dense" in key or "allocated" in key for key in state_dict)
        loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
        with torch.no_grad():
            output_loaded = loaded(**inputs).logits
        assert torch.allclose(output_loaded, expected, atol=1e-5)

    def test_compress_before_allocation_raises(self, peft_model):
        with pytest.raises(RuntimeError, match="have not been allocated yet"):
            peft_model.compress_and_unload()

    def test_unload_restores_base_model(self, base_model, allocated_model):
        inputs = get_inputs()
        with torch.no_grad():
            expected = base_model(**inputs).logits
        unloaded = allocated_model.unload()
        assert not get_layers(unloaded)
        with torch.no_grad():
            output = unloaded(**inputs).logits
        assert torch.allclose(output, expected)

    def test_merging_is_forbidden(self, allocated_model):
        with pytest.raises(NotImplementedError, match="compress_and_unload"):
            allocated_model.merge_and_unload()
        with pytest.raises(NotImplementedError, match="compress_and_unload"):
            allocated_model.merge_adapter()
        with pytest.raises(NotImplementedError):
            allocated_model.unmerge_adapter()

    def test_multiple_active_adapters_are_forbidden(self, allocated_model):
        allocated_model.add_adapter("other", MoeToDenseConfig(num_experts_to_keep=1))
        with pytest.raises(ValueError, match="single active adapter"):
            allocated_model.base_model.set_adapter(["default", "other"])
        # switching to the other adapter works; it is not allocated yet but statistics are still being collected
        allocated_model.set_adapter("other")
        assert all(layer._router_hook_handle is not None for layer in get_layers(allocated_model))
        calibrate(allocated_model)
        allocated_model.update_and_allocate()
        for layer in get_layers(allocated_model):
            assert len(layer.selected_experts["other"]) == 1
            assert layer._router_hook_handle is None

    def test_no_export_warning_for_supported_architectures(self, base_model):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            get_peft_model(copy.deepcopy(base_model), MoeToDenseConfig())
        assert not [w for w in record if "compress_and_unload" in str(w.message)]

    def test_export_warning_for_unsupported_router(self):
        # DeepSeek-V3 uses a sigmoid router with group-limited routing, which cannot route to a single expert
        torch.manual_seed(0)
        config = DeepseekV3Config(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=24,
            moe_intermediate_size=EXPERT_INTERMEDIATE_SIZE,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=2,
            num_key_value_heads=1,
            n_routed_experts=8,
            num_experts_per_tok=TOP_K,
            n_shared_experts=1,
            n_group=2,
            topk_group=1,
            first_k_dense_replace=0,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=8,
            kv_lora_rank=8,
            q_lora_rank=None,
            experts_implementation="eager",
        )
        model = DeepseekV3ForCausalLM(config).eval()
        with pytest.warns(UserWarning, match="compress_and_unload.*does not assign a routing weight of 1"):
            peft_model = get_peft_model(model, MoeToDenseConfig())

        # distillation still works (the tiny DeepSeek-V3 model does not accept an attention mask, so only pass input_ids)
        with torch.no_grad():
            for i in range(3):
                peft_model(input_ids=get_inputs(seed=10 + i)["input_ids"])
        peft_model.update_and_allocate()

        loss = peft_model.get_distillation_loss(input_ids=get_inputs()["input_ids"])
        assert torch.isfinite(loss)
        # the export warns again
        with pytest.warns(UserWarning, match="will probably not be equivalent"):
            peft_model.compress_and_unload()

    def test_bfloat16(self, model_type):
        base_model = build_model(model_type).to(torch.bfloat16)
        peft_model = get_peft_model(base_model, MoeToDenseConfig())
        calibrate(peft_model)
        peft_model.update_and_allocate()
        for layer in get_layers(peft_model):
            dense = layer.moe_to_dense_experts["default"]
            # the dense FFNs keep the dtype of the base model, they are not autocast to float32
            assert all(param.dtype == torch.bfloat16 for param in dense.parameters())

        inputs = get_inputs()
        loss = peft_model.get_distillation_loss(**inputs)

        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in peft_model.parameters() if p.requires_grad)

        with torch.no_grad():
            expected = peft_model(**inputs).logits
        dense_model = peft_model.compress_and_unload()
        with torch.no_grad():
            output = dense_model(**inputs).logits

        assert output.dtype == torch.bfloat16
        assert torch.allclose(output, expected)

    def test_delete_adapter(self, allocated_model):
        allocated_model.add_adapter("other", MoeToDenseConfig(num_experts_to_keep=1))
        allocated_model.set_adapter("other")
        calibrate(allocated_model)
        allocated_model.update_and_allocate()
        allocated_model.delete_adapter("other")
        for layer in get_layers(allocated_model):
            assert set(layer.moe_to_dense_experts) == {"default"}
            for attr in layer.other_param_names:
                assert "other" not in getattr(layer, attr)
            assert layer.active_adapters == ["default"]

        with torch.no_grad():
            output = allocated_model(**get_inputs()).logits
        assert torch.isfinite(output).all()

        allocated_model.delete_adapter("default")
        for layer in get_layers(allocated_model):
            assert set(layer.moe_to_dense_experts) == set()
            for attr in layer.other_param_names:
                assert "default" not in getattr(layer, attr)
            assert layer.active_adapters == []

    def test_modules_to_save(self, base_model, tmp_path):
        config = MoeToDenseConfig(modules_to_save=["q_proj", "input_layernorm"], task_type="CAUSAL_LM")
        peft_model = get_peft_model(copy.deepcopy(base_model), config)
        trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
        assert any("q_proj.modules_to_save.default" in n for n in trainable)
        assert any("input_layernorm.modules_to_save.default" in n for n in trainable)
        assert not any(".original_module." in n for n in trainable)
        calibrate(peft_model)
        peft_model.update_and_allocate()

        inputs = get_inputs()
        # the trained copies receive gradients, the originals (used by the teacher) do not
        loss = peft_model.get_distillation_loss(**inputs)
        loss.backward()
        assert all(p.grad is not None for n, p in peft_model.named_parameters() if "modules_to_save" in n)
        assert all(p.grad is None for n, p in peft_model.named_parameters() if "original_module" in n)
        peft_model.zero_grad()

        # perturb the copies: the teacher is unaffected, the student changes
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "modules_to_save" in name:
                    param.add_(0.1 * torch.randn_like(param))
        with torch.no_grad():
            student_logits = peft_model(**inputs).logits
            with peft_model.disable_adapter():
                teacher_logits = peft_model(**inputs).logits
            assert torch.allclose(teacher_logits, base_model(**inputs).logits)
            assert not torch.allclose(student_logits, teacher_logits)

        # the copies are saved with the adapter
        peft_model.save_pretrained(tmp_path)
        state_dict = safe_load_file(tmp_path / "adapter_model.safetensors")
        assert any("q_proj" in key for key in state_dict)
        loaded = PeftModel.from_pretrained(copy.deepcopy(base_model), tmp_path)
        with torch.no_grad():
            assert torch.allclose(loaded(**inputs).logits, student_logits, atol=1e-6)

        # the exported model uses the trained copies, the unloaded model the originals
        dense_model = peft_model.compress_and_unload()
        assert not any(".modules_to_save." in n or ".original_module." in n for n, _ in dense_model.named_modules())
        with torch.no_grad():
            assert torch.allclose(dense_model(**inputs).logits, student_logits, atol=1e-6)

        unloaded = loaded.unload()
        assert not any(".modules_to_save." in n or ".original_module." in n for n, _ in unloaded.named_modules())
        with torch.no_grad():
            assert torch.allclose(unloaded(**inputs).logits, teacher_logits)

    @pytest.mark.parametrize("experts_implementation", ["eager", "grouped_mm", "batched_mm"])
    def test_experts_implementations_agree(self, model_type, experts_implementation):
        # The conversion must not depend on the experts implementation the model was loaded with: the collected
        # routing statistics, the selected experts, and the outputs of the PEFT model and of the exported dense model
        # must match the eager reference.
        if model_type == "gpt_oss" and experts_implementation == "grouped_mm":
            pytest.skip("grouped_mm is not supported by GPT-OSS")

        reference = get_peft_model(build_model(model_type, experts_implementation="eager"), MoeToDenseConfig())
        model = get_peft_model(
            build_model(model_type, experts_implementation=experts_implementation), MoeToDenseConfig()
        )
        for peft_model in (reference, model):
            calibrate(peft_model)
            peft_model.update_and_allocate()

        for ref_layer, layer in zip(get_layers(reference), get_layers(model)):
            assert ref_layer.selected_experts == layer.selected_experts

        inputs = get_inputs()
        with torch.no_grad():
            expected = reference(**inputs).logits
            output = model(**inputs).logits
        assert torch.allclose(output, expected, atol=1e-5)

        dense_model = model.compress_and_unload()
        with torch.no_grad():
            output = dense_model(**inputs).logits
        assert torch.allclose(output, expected, atol=1e-5)
