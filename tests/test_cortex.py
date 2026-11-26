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

from importlib import import_module

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from peft import CortexConfig, PeftModel, get_peft_model
from peft.tuners.cortex import CortexExperts, CortexMLP

from .testing_common import PeftCommonTester
from .testing_utils import set_init_weights_false


# Llama has dense FFNs. Qwen3-MoE and GPT-OSS have packed experts with different layouts: Qwen3-MoE stores gate and up
# weights concatenated, while GPT-OSS stores them interleaved and transposed, with biases and a clamped activation.
# Neither has shared experts, so `moe_target="shared"` adds channels common to all experts. Dense FFNs are also covered
# by the common tests (see TestCortexDecoderModels and test_custom_models.py), so most tests below only use the MoE
# models.
MOE_CASES = [
    (model_type, moe_target) for model_type in ("qwen3_moe", "gpt_oss") for moe_target in ("shared", "routed", "both")
]
ALL_CASES = [("llama", "shared")] + MOE_CASES


def create_tiny_model(model_type, **config_kwargs):
    """Create a small random causal LM. The same arguments always create the same weights."""
    defaults = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "moe_intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "num_experts": 3,
        "num_local_experts": 3,
        "num_experts_per_tok": 2,
        # The routing weights of Qwen3-MoE don't sum to one, to check that public branches are merged exactly anyway.
        "norm_topk_prob": False,
        # Gemma4 with a small MoE block in every layer
        "enable_moe_block": True,
        "top_k_experts": 2,
        "num_kv_shared_layers": 0,
        "global_head_dim": 8,
        "hidden_size_per_layer_input": 0,
        "max_position_embeddings": 131072 if model_type == "gpt_oss" else 64,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }
    config = AutoConfig.for_model(model_type, **(defaults | config_kwargs))
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).eval()


def get_logits(model):
    input_ids = torch.tensor([[1, 4, 7, 3, 9, 2], [1, 5, 8, 4, 6, 2]])
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


def split_gate_up(weight, experts):
    """Split a (gate_)up weight with the layout of the given experts into gate and up weights."""
    if experts.is_concatenated:
        return weight.chunk(2, dim=-2)
    return weight[..., ::2, :], weight[..., 1::2, :]


def assert_channels_are_copies(gate, up, base_gate, base_up):
    """Assert that every added channel has the gate and up weights of a single channel of the base FFN."""
    for channel_gate, channel_up in zip(gate, up):
        matches = (base_gate == channel_gate).all(dim=-1) & (base_up == channel_up).all(dim=-1)
        assert matches.any()


class SparseActivationMLP(nn.Module):
    """An FFN that sparsifies its activations based on statistics over all channels, like some layers of Gemma3n."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
        self.act_fn = nn.GELU()
        self.activation_sparsity = 0.95

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = torch.relu(gate - gate.mean(dim=-1, keepdim=True) - gate.std(dim=-1, keepdim=True))
        return self.down_proj(self.act_fn(gate) * self.up_proj(x))


class ScaledOutputMLP(nn.Module):
    """An FFN that multiplies its output by a learned scale, like the dense FFNs of Inkling."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
        self.act_fn = nn.GELU()
        self.output_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) * self.output_scale


class NormalizedMLP(nn.Module):
    """An FFN that normalizes the activations before the down projection, like the dense FFNs of BitNet."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
        self.act_fn = nn.GELU()
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        return self.down_proj(self.norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class ModelWithMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(x)


# The native MoE blocks of models with shared experts, clamped or offset activations, fused gate and up projections, or
# experts without gate, by Transformers model type.
MOE_BLOCKS = {
    "minimax": ("MiniMaxConfig", "MiniMaxSparseMoeBlock"),
    "minimax_m2": ("MiniMaxM2Config", "MiniMaxM2SparseMoeBlock"),
    "minimax_m3_vl": ("MiniMaxM3VLTextConfig", "MiniMaxM3VLSparseMoeBlock"),
    "deepseek_v2": ("DeepseekV2Config", "DeepseekV2Moe"),
    "deepseek_v3": ("DeepseekV3Config", "DeepseekV3MoE"),
    "deepseek_v32": ("DeepseekV32Config", "DeepseekV32MoE"),
    "deepseek_v4": ("DeepseekV4Config", "DeepseekV4SparseMoeBlock"),
    "kimi_linear": ("KimiLinearConfig", "KimiLinearMoE"),
    "glm4_moe": ("Glm4MoeConfig", "Glm4MoeMoE"),
    "glm4_moe_lite": ("Glm4MoeLiteConfig", "Glm4MoeLiteMoE"),
    "glm_moe_dsa": ("GlmMoeDsaConfig", "GlmMoeDsaMoE"),
    "glm5_next": ("Glm5NextTextConfig", "Glm5NextTextMoE"),
    "qwen3_next": ("Qwen3NextConfig", "Qwen3NextSparseMoeBlock"),
    "qwen3_5_moe": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeSparseMoeBlock"),
    "nemotron_h": ("NemotronHConfig", "NemotronHMoE"),
}


def create_moe_block_config(model_type):
    """Create a small config for the native MoE block of the given model.

    The configs name their fields differently, e.g. the number of experts is `num_experts`, `num_local_experts`, or
    `n_routed_experts`. So the fields of all of them are passed, and each config uses the ones that it defines.
    """
    module = pytest.importorskip(f"transformers.models.{model_type}.configuration_{model_type}")
    config = getattr(module, MOE_BLOCKS[model_type][0])(
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        dense_intermediate_size=20,
        shared_intermediate_size=18,
        shared_expert_intermediate_size=18,
        moe_shared_expert_intermediate_size=18,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        num_experts=4,
        num_local_experts=4,
        n_routed_experts=4,
        n_shared_experts=2,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        mlp_layer_types=["moe" if model_type == "deepseek_v4" else "sparse"] * 2,
        moe_latent_size=8,
        max_position_embeddings=64,
        layer_types=["sliding_attention" if model_type == "deepseek_v4" else "full_attention"] * 2,
        tie_word_embeddings=False,
    )
    return config


class MoeBlockModel(PreTrainedModel):
    """A model consisting of a single native MoE block, to test MoE models without building a whole model."""

    def __init__(self, config):
        super().__init__(config)
        # Use the reference implementation of the experts, independently of the available kernels.
        self.config._experts_implementation = "eager"

        model_type = next(key for key, (config_name, _) in MOE_BLOCKS.items() if type(config).__name__ == config_name)
        block_cls = getattr(
            import_module(f"transformers.models.{model_type}.modeling_{model_type}"), MOE_BLOCKS[model_type][1]
        )
        # The MoE block of DeepSeek-V4 needs its layer index to choose its router.
        self.mlp = block_cls(config, layer_idx=0) if model_type == "deepseek_v4" else block_cls(config)

        with torch.no_grad():
            for parameter in self.parameters():
                # Large weights, so that the FFNs that clamp their activations actually clamp them.
                parameter.normal_(std=0.15)

    def forward(self, x):
        result = self.mlp(x)
        return result[0] if isinstance(result, tuple) else result


class TestCortex:
    @pytest.mark.parametrize("init_weights", ["random", "sample"])
    @pytest.mark.parametrize("model_type, moe_target", ALL_CASES)
    def test_initialization_does_not_change_output(self, model_type, moe_target, init_weights):
        base_model = create_tiny_model(model_type)
        expected = get_logits(base_model)

        config = CortexConfig(k=4, moe_target=moe_target, init_weights=init_weights)
        model = get_peft_model(base_model, config)
        torch.testing.assert_close(get_logits(model), expected)

    @pytest.mark.parametrize(
        "model_type, moe_target, cortex_bias",
        [(model_type, moe_target, "auto") for model_type, moe_target in MOE_CASES]
        # without adapter biases, merging adds zero biases to the gate and up biases of GPT-OSS
        + [("gpt_oss", "both", "none")],
    )
    def test_merge_and_unmerge(self, model_type, moe_target, cortex_bias):
        # The merged model runs the native Transformers forward with wider experts. This also checks that the adapter
        # forward computes the channels exactly like the native experts.
        config = CortexConfig(k=4, moe_target=moe_target, cortex_bias=cortex_bias, init_weights=False)
        model = get_peft_model(create_tiny_model(model_type), config)
        expected = get_logits(model)
        assert not torch.allclose(expected, get_logits(create_tiny_model(model_type)))

        model.merge_adapter()
        torch.testing.assert_close(get_logits(model), expected, rtol=2e-4, atol=3e-5)

        # unmerging restores the original base weights, up to floating point errors of the down biases of GPT-OSS
        model.unmerge_adapter()
        torch.testing.assert_close(get_logits(model), expected)

        merged_model = model.merge_and_unload()
        assert not any("cortex_" in name for name, _ in merged_model.named_parameters())
        torch.testing.assert_close(get_logits(merged_model), expected, rtol=2e-4, atol=3e-5)

    @pytest.mark.parametrize("model_type", ["llama", "qwen3_moe", "gpt_oss"])
    def test_partial_merge_keeps_other_active_adapters(self, model_type):
        model = get_peft_model(
            create_tiny_model(model_type),
            CortexConfig(k=4, moe_target="both", init_weights=False),
        )
        model.add_adapter("other", CortexConfig(k=3, init_weights=False))
        model.base_model.set_adapter(["default", "other"])
        expected = get_logits(model)

        model.merge_adapter(["default"])
        torch.testing.assert_close(get_logits(model), expected, rtol=2e-4, atol=3e-5)
        model.merge_adapter(["other"])
        torch.testing.assert_close(get_logits(model), expected, rtol=2e-4, atol=3e-5)
        model.unmerge_adapter()
        torch.testing.assert_close(get_logits(model), expected)

    def test_safe_merge_detects_overflow_in_base_dtype(self):
        # The adapter weights are upcast to float32. A value that is finite in float32 can still overflow when it is
        # merged into the float16 base weights.
        model = get_peft_model(create_tiny_model("llama").half(), CortexConfig(k=4))
        adapter_weight = next(param for name, param in model.named_parameters() if "cortex_" in name)
        with torch.no_grad():
            adapter_weight.view(-1)[0] = 70000.0

        with pytest.raises(ValueError, match="NaNs detected in the merged weights"):
            model.merge_adapter(safe_merge=True)

    @pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
    @pytest.mark.parametrize("model_type, moe_target", MOE_CASES)
    def test_save_and_load(self, model_type, moe_target, low_cpu_mem_usage, tmp_path):
        config = CortexConfig(k=4, moe_target=moe_target, init_weights=False)
        model = get_peft_model(create_tiny_model(model_type), config)
        expected = get_logits(model)
        model.save_pretrained(tmp_path)

        loaded = PeftModel.from_pretrained(
            create_tiny_model(model_type), tmp_path, low_cpu_mem_usage=low_cpu_mem_usage
        )
        torch.testing.assert_close(get_logits(loaded), expected)

    @pytest.mark.parametrize("model_type, moe_target", MOE_CASES)
    def test_training_changes_only_the_adapter(self, model_type, moe_target):
        base_output = get_logits(create_tiny_model(model_type))
        config = CortexConfig(k=4, moe_target=moe_target, task_type="CAUSAL_LM")
        model = get_peft_model(create_tiny_model(model_type), config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        input_ids = torch.tensor([[1, 4, 7, 3, 9, 2]])

        # The down weights are initially zero, so only they receive gradients in the first step. Train with mixed
        # precision like the Trainer with `bf16=True`: the adapter weights are float32, but under autocast, the
        # projections compute in bfloat16.
        model.train()
        for _ in range(3):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()

        assert not torch.allclose(get_logits(model), base_output)
        with model.disable_adapter():
            torch.testing.assert_close(get_logits(model), base_output)

    @pytest.mark.parametrize("model_type, moe_target", ALL_CASES)
    def test_sample_initialization_copies_base_channels(self, model_type, moe_target):
        # The gate and up weights are stored in different layouts depending on the model, so check that the sampled
        # weights correspond to complete channels of the base FFN.
        config = CortexConfig(k=4, moe_target=moe_target, init_weights="sample", init_noise_std=0.0)
        model = get_peft_model(create_tiny_model(model_type), config)

        for layer in model.modules():
            if isinstance(layer, CortexMLP):
                base = layer.get_base_layer()
                gate, up = layer.cortex_gate["default"].weight, layer.cortex_up["default"].weight
                assert_channels_are_copies(gate, up, base.gate_proj.weight, base.up_proj.weight)

            elif isinstance(layer, CortexExperts):
                experts = layer.get_base_layer()
                base_weight = experts.gate_up_proj.transpose(-1, -2) if experts.is_transposed else experts.gate_up_proj
                base_gate, base_up = split_gate_up(base_weight, experts)
                if "default" in layer.cortex_routed:
                    # routed channels are sampled from their own expert
                    gate, up = split_gate_up(layer.cortex_routed["default"].up, experts)
                    for expert in range(experts.num_experts):
                        assert_channels_are_copies(gate[expert], up[expert], base_gate[expert], base_up[expert])
                if "default" in layer.cortex_public:
                    # public channels are sampled from all experts
                    gate, up = split_gate_up(layer.cortex_public["default"].up, experts)
                    assert_channels_are_copies(gate, up, base_gate.flatten(0, 1), base_up.flatten(0, 1))

    @pytest.mark.parametrize(
        "model_type, config_kwargs, expected",
        [
            # 2 layers with 3 * hidden_size * k parameters each
            ("llama", {}, 2 * 3 * 16 * 4),
            ("qwen3_moe", {}, 2 * 3 * 16 * 4),
            # routed channels are added to each of the 3 experts
            ("qwen3_moe", {"moe_target": "routed", "k_routed": 2}, 2 * 3 * 3 * 16 * 2),
            ("qwen3_moe", {"moe_target": "both", "k_routed": 2}, 2 * (3 * 16 * 4 + 3 * 3 * 16 * 2)),
            # GPT-OSS has biases, so the adapter has 2 * k gate and up biases and hidden_size down biases as well
            ("gpt_oss", {}, 2 * (3 * 16 * 4 + 2 * 4 + 16)),
            ("gpt_oss", {"cortex_bias": "none"}, 2 * 3 * 16 * 4),
        ],
    )
    def test_number_of_trainable_parameters(self, model_type, config_kwargs, expected):
        config = CortexConfig(k=4, **config_kwargs)
        model = get_peft_model(create_tiny_model(model_type), config)
        trainable_params, _ = model.get_nb_trainable_parameters()
        assert trainable_params == expected

    @pytest.mark.parametrize(
        "model_type, target_modules",
        [
            # projections of an FFN instead of the whole FFN
            ("llama", ["up_proj"]),
            # a MoE block instead of its experts
            ("qwen3_moe", ["mlp"]),
        ],
    )
    def test_unsupported_target_raises(self, model_type, target_modules):
        with pytest.raises(TypeError, match="Target module .* is not supported"):
            get_peft_model(create_tiny_model(model_type), CortexConfig(target_modules=target_modules))

    @pytest.mark.parametrize("mlp_cls", [SparseActivationMLP, ScaledOutputMLP, NormalizedMLP])
    def test_ffn_with_additional_operations_raises(self, mlp_cls):
        # Merged channels would change the statistics of the activations, or would be scaled or normalized, unlike the
        # channels of the adapter, so the merged model would differ from the trained one.
        with pytest.raises(TypeError, match="Target module .* is not supported"):
            get_peft_model(ModelWithMLP(mlp_cls()), CortexConfig(target_modules=["mlp"]))

    @pytest.mark.parametrize(
        "moe_target, expected_targets",
        [("shared", {"shared_experts"}), ("routed", {"experts"}), ("both", {"shared_experts", "experts"})],
    )
    def test_default_targets_of_moe_block_with_shared_experts(self, moe_target, expected_targets):
        model = get_peft_model(
            MoeBlockModel(create_moe_block_config("deepseek_v3")), CortexConfig(k=3, moe_target=moe_target)
        )
        assert {name.split(".")[-1] for name in model.targeted_module_names} == expected_targets
        # The shared experts are extended instead of adding channels common to all experts.
        assert not any("cortex_public" in name for name, _ in model.named_parameters())

    @pytest.mark.parametrize(
        "moe_target, target_modules",
        [
            # the shared experts are extended instead of the packed experts
            ("shared", ["experts"]),
            # the shared experts are not routed
            ("routed", ["shared_experts"]),
        ],
    )
    def test_target_without_added_channels_raises(self, moe_target, target_modules):
        config = CortexConfig(moe_target=moe_target, target_modules=target_modules)
        with pytest.raises(ValueError, match="Cortex adds no channels"):
            get_peft_model(MoeBlockModel(create_moe_block_config("deepseek_v3")), config)

    def test_shared_experts_next_to_moe_block_are_recognized(self):
        # The shared FFN of GraniteMoeShared is next to its MoE block instead of next to the packed experts. With
        # `moe_target="shared"`, it is extended instead of adding channels common to all experts.
        model = create_tiny_model("granitemoeshared", shared_intermediate_size=12)
        with pytest.raises(ValueError, match="Cortex adds no channels"):
            get_peft_model(model, CortexConfig(target_modules=["experts"]))

    @pytest.mark.parametrize(
        "target_modules, message",
        [
            (["experts"], "only routed channels are added"),
            (["shared_experts"], "only the shared experts are extended"),
        ],
    )
    def test_both_with_partial_targets_warns(self, target_modules, message):
        config = CortexConfig(moe_target="both", target_modules=target_modules)
        with pytest.warns(UserWarning, match=message):
            get_peft_model(MoeBlockModel(create_moe_block_config("deepseek_v3")), config)

    @pytest.mark.parametrize("model_type", MOE_BLOCKS)
    def test_native_moe_block(self, model_type):
        config = create_moe_block_config(model_type)
        torch.manual_seed(0)
        base_model = MoeBlockModel(config).eval()
        # Large inputs, so that the FFNs that clamp their activations actually clamp them.
        x = 20 * torch.randn(2, 5, 16)
        with torch.no_grad():
            base_output = base_model(x)

        # Add routed channels to the experts, and extend the native shared experts, or add channels common to all
        # experts if there are none. Many of these routers scale the routing weights, so that they don't sum to one.
        cortex_config = CortexConfig(
            k=3,
            k_routed=2,
            moe_target="both",
            init_weights=False,
            target_modules=["experts", "shared_experts", "shared_expert"],
        )
        model = get_peft_model(base_model, cortex_config).eval()
        with torch.no_grad():
            expected = model(x)
        assert not torch.allclose(expected, base_output)

        merged_model = model.merge_and_unload()
        with torch.no_grad():
            torch.testing.assert_close(merged_model(x), expected, rtol=2e-4, atol=3e-5)

        # The config of the merged model describes the wider FFNs.
        config = type(merged_model.config).from_dict(merged_model.config.to_dict())
        loaded_model = MoeBlockModel(config).eval()
        loaded_model.load_state_dict(merged_model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(loaded_model(x), merged_model(x))

    @pytest.mark.parametrize(
        "model_type, config_kwargs",
        [
            ("llama", {}),
            # The FFN of the second layer is padded to the width of the adapted first layer.
            ("llama", {"target_modules": ["model.layers.0.mlp"]}),
            ("qwen3_moe", {"moe_target": "both"}),
            ("gpt_oss", {"moe_target": "routed"}),
            # Gemma4 has a dense FFN, which counts as shared experts, and experts in each layer, whose widths are
            # determined by different fields.
            ("gemma4_text", {"moe_target": "both"}),
        ],
    )
    def test_merged_model_can_be_loaded_with_from_pretrained(self, model_type, config_kwargs, tmp_path):
        config = CortexConfig(k=4, k_routed=2, init_weights=False, **config_kwargs)
        model = get_peft_model(create_tiny_model(model_type), config)
        expected = get_logits(model)

        model.merge_and_unload().save_pretrained(tmp_path)
        loading_kwargs = {} if model_type == "llama" else {"experts_implementation": "eager"}
        loaded_model, info = AutoModelForCausalLM.from_pretrained(
            tmp_path, attn_implementation="eager", output_loading_info=True, **loading_kwargs
        )
        assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys"))
        torch.testing.assert_close(get_logits(loaded_model.eval()), expected, rtol=2e-4, atol=3e-5)


CORTEX_DECODER_MODELS = [
    "peft-internal-testing/tiny-random-LlamaForCausalLM",
    "peft-internal-testing/tiny-dummy-qwen2",
]
CORTEX_CONFIG_KWARGS = {"task_type": "CAUSAL_LM", "k": 4}


class TestCortexDecoderModels(PeftCommonTester):
    """Run the common PEFT tests on decoder models with dense FFNs, using the default target modules."""

    transformers_class = AutoModelForCausalLM

    def prepare_inputs_for_testing(self):
        input_ids = torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device)
        attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_model_attr(self, model_id):
        self._test_model_attr(model_id, CortexConfig, CORTEX_CONFIG_KWARGS.copy())

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_save_pretrained(self, model_id):
        config_kwargs = set_init_weights_false(CortexConfig, CORTEX_CONFIG_KWARGS)
        self._test_save_pretrained(model_id, CortexConfig, config_kwargs)

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_save_pretrained_selected_adapters(self, model_id):
        config_kwargs = set_init_weights_false(CortexConfig, CORTEX_CONFIG_KWARGS)
        self._test_save_pretrained_selected_adapters(model_id, CortexConfig, config_kwargs)

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_load_model_low_cpu_mem_usage(self, model_id):
        self._test_load_model_low_cpu_mem_usage(model_id, CortexConfig, CORTEX_CONFIG_KWARGS.copy())

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_merge_layers(self, model_id):
        config_kwargs = set_init_weights_false(CortexConfig, CORTEX_CONFIG_KWARGS)
        self._test_merge_layers(model_id, CortexConfig, config_kwargs)

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_merge_layers_multi(self, model_id):
        config_kwargs = set_init_weights_false(CortexConfig, CORTEX_CONFIG_KWARGS)
        self._test_merge_layers_multi(model_id, CortexConfig, config_kwargs)

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_generate(self, model_id):
        self._test_generate(model_id, CortexConfig, CORTEX_CONFIG_KWARGS.copy())

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_training(self, model_id):
        self._test_training(model_id, CortexConfig, CORTEX_CONFIG_KWARGS.copy())

    @pytest.mark.parametrize("use_reentrant", [True, False])
    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_training_gradient_checkpointing(self, model_id, use_reentrant):
        self._test_training_gradient_checkpointing(
            model_id, CortexConfig, CORTEX_CONFIG_KWARGS.copy(), use_reentrant=use_reentrant
        )

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_delete_adapter(self, model_id):
        self._test_delete_adapter(model_id, CortexConfig, CORTEX_CONFIG_KWARGS.copy())

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_unload_adapter(self, model_id):
        config_kwargs = set_init_weights_false(CortexConfig, CORTEX_CONFIG_KWARGS)
        self._test_unload_adapter(model_id, CortexConfig, config_kwargs)

    @pytest.mark.parametrize("model_id", CORTEX_DECODER_MODELS)
    def test_disable_adapter(self, model_id):
        config_kwargs = set_init_weights_false(CortexConfig, CORTEX_CONFIG_KWARGS)
        self._test_disable_adapter(model_id, CortexConfig, config_kwargs)
