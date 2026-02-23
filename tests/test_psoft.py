#  Copyright 2026-present the HuggingFace Inc. team.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License governing permissions and limitations under the License.

# This test file is for tests specific to PSOFT.

import pytest
import torch
import torch.nn as nn
from accelerate.utils.imports import is_bf16_available
from torch.testing import assert_close

from peft import PeftModel, get_peft_model
from peft.tuners.psoft import PSOFTConfig

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Single Transformer Block
# -------------------------


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.attn_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.ffn_fc1 = nn.Linear(hidden_dim, mlp_hidden_dim, bias=True)
        self.ffn_fc2 = nn.Linear(mlp_hidden_dim, hidden_dim, bias=True)
        self.ffn_dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, input_ids=None, **kwargs):
        x = input_ids
        batch_size, seq_length, hidden_dim = x.shape
        head_dim = hidden_dim // self.num_heads

        residual = x
        x = self.ln1(x)

        q = self.attn_q(x).reshape(batch_size, seq_length, self.num_heads, head_dim)
        k = self.attn_k(x).reshape(batch_size, seq_length, self.num_heads, head_dim)
        v = self.attn_v(x).reshape(batch_size, seq_length, self.num_heads, head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, hidden_dim)

        x = self.attn_out(attn_output)
        x = self.attn_dropout(x)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.ffn_fc1(x)
        x = self.activation(x)
        x = self.ffn_dropout(x)
        x = self.ffn_fc2(x)
        x = self.ffn_dropout(x)
        x = residual + x
        return x


# -------------------------
# Helpers
# -------------------------

TARGETS = ["attn_q", "attn_k", "attn_v", "attn_out", "ffn_fc1", "ffn_fc2"]


def _get_submodule(model: nn.Module, name: str) -> nn.Module:
    mods = dict(model.named_modules())
    assert name in mods, f"Module {name} not found."
    return mods[name]


def get_backbone(model: nn.Module) -> nn.Module:
    if hasattr(model, "base_model"):
        model = model.base_model
    if hasattr(model, "model"):
        model = model.model
    return model


def _orthlayer_requires_grad(R: nn.Module) -> bool:
    return any(p.requires_grad for p in R.parameters())


def _column_cosine_matrix(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    X = W
    norms = X.norm(dim=0, keepdim=True).clamp_min(eps)
    Xn = X / norms
    return Xn.T @ Xn


def _assert_is_psoft_linear(layer: nn.Module):
    for attr in ("psoft_R", "get_delta_weight", "get_base_layer"):
        assert hasattr(layer, attr), f"Expected PSOFT layer to have {attr}."


def _get_AB(layer: nn.Module, adapter: str = "default"):
    if hasattr(layer, "get_cache_AB"):
        A, B = layer.get_cache_AB(adapter)
        return A, B

    safe = layer._sanitize_adapter_name(adapter)
    A = getattr(layer, f"_psoft_A_cache_{safe}")
    B = getattr(layer, f"_psoft_B_cache_{safe}")
    return A, B


def _get_ARB(layer: nn.Module, adapter: str = "default"):
    A, B = _get_AB(layer, adapter)
    R = layer.psoft_R[adapter].get_matrix()  # (r, r)
    return A, R, B


def _combined_weight(layer: nn.Module, adapter: str = "default") -> torch.Tensor:
    W_pre = layer.get_base_layer().weight
    delta_W = layer.get_delta_weight(adapter).to(W_pre.dtype)
    return W_pre + delta_W


# -------------------------
# Build PSOFT model on CPU
# -------------------------


@pytest.fixture
def psoft_block():
    torch.manual_seed(0)
    base = TransformerBlock(hidden_dim=32, num_heads=8, mlp_ratio=4, dropout=0.0)

    W_pre = {name: _get_submodule(base, name).weight.detach().clone() for name in TARGETS}

    cfg = PSOFTConfig(
        r=4,
        psoft_alpha=4,
        psoft_dropout=0.0,
        target_modules=TARGETS,
        init_psoft_weights="psoft_init",
        psoft_svd="full",
        psoft_orth=True,
        psoft_mag_a=False,
        psoft_mag_b=False,
    )

    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()

    # for name, module in model.named_modules():
    #     if hasattr(module, "psoft_R"):
    #         A, B = _get_AB(module, adapter="default")
    #         R = module.psoft_R["default"].get_matrix()

    #         print(f"\n[PSOFT layer] {name}")
    #         print(f"  A shape: {tuple(A.shape)}")   # (r, in)
    #         print(f"  B shape: {tuple(B.shape)}")   # (out, r)
    #         print(f"  R shape: {tuple(R.shape)}")   # (r, r)

    # for name, module in model.named_modules():
    #     if hasattr(module, "psoft_R"):
    #         print(f"\n{name}")
    #         for pname, p in module.named_parameters():
    #             print(f"  {pname:40s} requires_grad={p.requires_grad}")

    return model, W_pre, cfg


# -------------------------
# 1) Injection / structure correctness
# -------------------------
def test_psoft_injection_structure(psoft_block):
    model, _, _ = psoft_block
    backbone = get_backbone(model)

    for name in TARGETS:
        layer = _get_submodule(backbone, name)
        _assert_is_psoft_linear(layer)

        # adapter dict should exist and include "default"
        assert "default" in layer.psoft_R

        A, R, B = _get_ARB(layer, "default")
        assert A.shape[0] == 4  # (r, in)
        assert B.shape[1] == 4  # (out, r)
        assert R.shape == (4, 4)

        param_names = [n for n, _ in layer.named_parameters()]
        assert not any("psoft_A" in n or "psoft_B" in n for n in param_names)

        assert _orthlayer_requires_grad(layer.psoft_R["default"]) is True


# -------------------------
# 2) merge and unload correctness
# -------------------------
def test_psoft_merge_and_unload_outputs_match(psoft_block):
    model, _, _ = psoft_block
    model.eval()

    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        y_before = model(input_ids=x)

    merged_model = model.merge_and_unload()
    merged_model.eval()
    with torch.no_grad():
        y_unloaded = merged_model(input_ids=x)

    assert_close(y_before, y_unloaded, atol=1e-6, rtol=1e-6)

    backbone = get_backbone(merged_model)
    for name in TARGETS:
        assert isinstance(_get_submodule(backbone, name), nn.Linear)


# -------------------------
# 3) dtype validation
# -------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_psoft_dtype_forward(dtype):
    if dtype == torch.bfloat16 and not is_bf16_available():
        pytest.skip("bfloat16 not supported on this system")

    torch.manual_seed(0)
    base = TransformerBlock(hidden_dim=32, num_heads=8, mlp_ratio=4, dropout=0.0).to(dtype)

    cfg = PSOFTConfig(
        r=4,
        psoft_alpha=4,
        psoft_dropout=0.0,
        target_modules=TARGETS,
        init_psoft_weights="psoft_init",
        psoft_svd="full",
        psoft_orth=True,
        psoft_mag_a=False,
        psoft_mag_b=False,
    )
    model = get_peft_model(base, cfg).to(dtype)
    model.eval()

    x = torch.randn(2, 8, 32, dtype=dtype)
    with torch.no_grad():
        y = model(input_ids=x)
    assert y.dtype == dtype


# -------------------------
# 4) Init Invariance: with R initialized as I, delta should be 0
# -------------------------
def test_psoft_init_invariance_delta_zero(psoft_block):
    model, _, _ = psoft_block
    backbone = get_backbone(model)
    layer = _get_submodule(backbone, "attn_q")

    A, R, B = _get_ARB(layer, "default")

    delta = layer.get_delta_weight("default")
    assert_close(delta, torch.zeros_like(delta), atol=1e-6, rtol=1e-6)


# -------------------------
# 5) After training: column-angle invariance of W_pre vs (W_pre + delta)
# -------------------------
def test_psoft_column_angle_invariance_after_training(psoft_block):
    model, W_pre, _ = psoft_block
    model.train()

    backbone = get_backbone(model)
    layer_name = "attn_q"
    layer = _get_submodule(backbone, layer_name)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-1)

    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(2, 8, 32)
        y = model(input_ids=x)
        loss = (y**2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        W0 = W_pre[layer_name]
        W1 = _combined_weight(layer, "default")

        C0 = _column_cosine_matrix(W0.T)
        C1 = _column_cosine_matrix(W1.T)

    assert_close(C0, C1, atol=1e-6, rtol=1e-6)


# -------------------------
# 6) Save and load PSOFT
# -------------------------
def _extract_pure_base_state_dict(injected_backbone: torch.nn.Module) -> dict:
    """
    Convert an injected backbone state_dict (with *.base_layer.* and adapter keys)
    into a pure TransformerBlock-compatible state_dict.
    """
    sd = injected_backbone.state_dict()
    out = {}
    for k, v in sd.items():
        if ".psoft_" in k or "psoft_R" in k or "_psoft_" in k:
            continue

        if ".base_layer.weight" in k:
            k = k.replace(".base_layer.weight", ".weight")
        elif ".base_layer.bias" in k:
            k = k.replace(".base_layer.bias", ".bias")

        out[k] = v
    return out


def test_psoft_save_and_load_adapter_roundtrip(psoft_block, tmp_path):
    model, _, _ = psoft_block
    model.train()

    backbone = get_backbone(model)
    layer_name = "attn_q"
    layer = _get_submodule(backbone, layer_name)

    base_sd = _extract_pure_base_state_dict(backbone)

    with torch.no_grad():
        layer.psoft_R["default"].weight.add_(0.01 * torch.randn_like(layer.psoft_R["default"].weight))

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-1)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(2, 8, 32)
        y = model(input_ids=x)
        (y**2).mean().backward()
        opt.step()

    model.eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        y_ref = model(input_ids=x)

    save_dir = tmp_path / "psoft_adapter"
    model.save_pretrained(save_dir)

    new_base = TransformerBlock(hidden_dim=32, num_heads=8, mlp_ratio=4, dropout=0.0)
    new_base.load_state_dict(base_sd, strict=True)

    loaded = PeftModel.from_pretrained(new_base, save_dir)
    loaded.eval()

    with torch.no_grad():
        y_loaded = loaded(input_ids=x)

    torch.testing.assert_close(y_ref, y_loaded, atol=1e-6, rtol=1e-6)

@pytest.mark.slow
def test_psoft_basic_usage_opt_125m_smoke():

    torch.manual_seed(0)

    model_id = "facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(model_id)

    config = PSOFTConfig(
        r=4,
        psoft_alpha=4,
        target_modules=["q_proj", "v_proj"],
        init_psoft_weights="psoft_init",
        psoft_svd="full",
        psoft_orth=True,
        psoft_mag_a=True,
        psoft_mag_b=True,
        use_cayley_neumann=False,
        num_cayley_neumann_terms=5,
        cayley_neumann_eps=None,
    )
    model = get_peft_model(model, config)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    loss.backward()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    for p in trainable:
        assert torch.isfinite(p).all()   
