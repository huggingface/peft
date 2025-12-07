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
from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel

from .config import ArrowConfig


TASK_ADAPTER_PREFIX = "task_"
GKS_ADAPTER_PREFIX = "gks_"


class ArrowLoraLinearLayer(nn.Module):
    """
    This class represent the main logic of the arrow routing algorithm for linear layers.
    """

    def __init__(self, in_features, arrow_config):
        super().__init__()
        # extra parameters needed for arrow
        self.in_features = in_features
        self._protos_ready = False
        self.top_k = arrow_config.top_k
        self.temperature = arrow_config.router_temperature
        self.rng_seed = arrow_config.rng_seed
        self.task_adapter_names = (
            arrow_config.task_adapter_names.copy()
        )  # Set in create_arrow_model() with this format: task_0, task_1, ...
        self.gks_adapter_names = (
            arrow_config.gks_adapter_names
        )  # Set in create_arrow_model() with this format: gks_0, gks_1, ...
        self.use_gks = arrow_config.use_gks
        self.gks_done = False
        self.gks_added_adapter_names = []
        self.in_features = in_features
        self.cast_input_dtype_enabled = True

    @torch.no_grad()
    def on_adapter_change(self, lora_A, lora_B):
        """
        Called when adapters are added/removed/renamed so Arrow can refresh its internal state before the next forward
        pass.
        """
        all_ts_adapter_names = [
            k
            for k in lora_A.keys()
            if k in lora_B and k != "arrow_router" and not (k.startswith("gks_") and k[len("gks_") :].isdigit())
        ]

        if sorted(self.task_adapter_names) == sorted(all_ts_adapter_names):  # No changes in the ts_adapters
            return

        # Getting the name(s) of added adapter(s)
        if len(self.task_adapter_names) < len(all_ts_adapter_names):  # Adapter(s) are added.
            self.gks_added_adapter_names = [x for x in all_ts_adapter_names if x not in self.task_adapter_names]

        # Updating the task_adapter_names
        self.task_adapter_names = all_ts_adapter_names.copy()
        # Invalidate caches so they’ll be rebuilt lazily on next forward()
        self._protos_ready = False
        # GKS will be handled by self.gks_added_adapter_names

    def top_right_singular_vec_from_BA(self, A, B, iters=15, eps=1e-8):
        """
        Computes the top *right* singular vector of ΔW = B @ A without forming ΔW.

        Theory:
            For any matrix M, the right singular vectors are the eigenvectors of Mᵀ M. If ΔW = B @ A (with A ∈
            ℝ^{r×in}, B ∈ ℝ^{out×r}), then
                ΔWᵀ ΔW = (B @ A)ᵀ (B @ A) = Aᵀ (Bᵀ B) A ∈ ℝ^{in×in}.
            Therefore, the dominant right singular vector of ΔW is the dominant eigenvector of M := Aᵀ (Bᵀ B) A. We
            find it by *power iteration* on the linear operator
                v ↦ Aᵀ (Bᵀ B) (A v),
            which avoids materializing ΔW (out×in) or M (in×in). The result lives in the input/token space (size =
            in_features), which is exactly what Arrow needs. (Right singular vectors ≡ eigenvectors of MᵀM; power
            iteration converges to the dominant eigenvector under mild conditions.)
        =============================== Practical notes:
            - We perform all iteration in float32 for numerical stability, then cast back
            to the LoRA dtype/device before storing/using the prototype.
            - Convergence is checked with a simple fixed-iter cap (`iters`) and/or
            `allclose` tolerance (`tol`).
            - The returned vector is unique up to sign (±), as with any singular vector.
            Downstream code should be sign-invariant.
        """

        # A: (r, in), B: (out, r)
        A32 = A.to(torch.float32)
        B32 = B.to(torch.float32)
        C = B32.T @ B32  # (r, r)

        # Private RNG on A's device
        gen = None
        if self.rng_seed is not None:
            gen = torch.Generator(device=A32.device.type)
            gen.manual_seed(int(self.rng_seed))

        # init vector in input space
        v = torch.randn(A32.size(1), dtype=A32.dtype, device=A32.device, generator=gen)
        v = v / (v.norm() + eps)

        for _ in range(iters):
            # w = (ΔWᵀΔW) v = Aᵀ (BᵀB) (A v)
            w = A32.T @ (C @ (A32 @ v))
            v = w / (w.norm() + eps)

        return v  # fp32

    @torch.no_grad()
    def build_prototypes(self, lora_A, lora_B):
        """
        Computes a prototype vector for each LoRA module in every layer by applying Singular Value Decomposition (SVD)
        to the `lora_A` matrix and extracting the top right singular vector.

        These prototypes are later used to calculate the cosine similarity between each input token and each expert.
        The resulting similarity scores serve as coefficients to compute a weighted average of the corresponding LoRA
        modules, effectively routing each token through its most relevant experts.

        ** This prototype computation is done is done once for all experts and is re-done on newly added adapters.**

        Args:
            lora_A : Matrices A in LoRA layer.
            lora_B (optional): Matrices B in LoRA layer. Defaults to None.
        """

        if self._protos_ready:
            return
        protos = []
        for name in self.task_adapter_names:
            A = lora_A[name].weight  # (r, in_features)
            B = lora_B[name].weight  # (out_features, r)

            # Efficiently computing right singular vector of A @ B
            proto32 = self.top_right_singular_vec_from_BA(A, B)

            proto = proto32.to(dtype=A.dtype, device=A.device)
            protos.append(proto)

        proto_stack = torch.stack(protos, dim=0)  # (E, in_features)

        # Register the prototypes buffer with correct dtype/device consistent with A and B weights
        self.register_buffer("prototypes", proto_stack, persistent=False)
        self._protos_ready = True

    @torch.no_grad()
    def gen_know_sub(self, lora_A, lora_B):
        """
        This function performs General Knowledge Subtraction. It takes an average of provided general_adapters, and
        subtract it from each task_adapter. This subtraction tries to purify the task adapters, based on
        "forgetting-via-negation" principle. Forgetting-via-negation is a task-arithmetic operation, explained in:
        https://huggingface.co/papers/2212.04089 The task adapters will be more focused and isolated, enhancing the
        performance on new tasks.

        Args:
            lora_A : Matrices A in LoRA layer.
            lora_B : Matrices A in LoRA layer.
        """
        if not self.use_gks:
            return
        elif self.gks_done and not self.gks_added_adapter_names:
            return
        else:
            # 1) compute average A/B over gks_adapter_names
            avg_A = torch.stack([lora_A[n].weight for n in self.gks_adapter_names], dim=0).mean(
                0
            )  # shape (r, in_features)
            avg_B = torch.stack([lora_B[n].weight for n in self.gks_adapter_names], dim=0).mean(
                0
            )  # shape (out_features, r)

            # 2) Subtract the average from task-specific experts
            if self.gks_done is False:  # GKS is done for all the experts, since it hasn't been done yet.
                for name in self.task_adapter_names:
                    lora_A[name].weight.data.sub_(avg_A)
                    lora_B[name].weight.data.sub_(avg_B)
            else:  # GKS is only done on new added experts, since GKS has been done previously.
                for name in self.gks_added_adapter_names:
                    lora_A[name].weight.data.sub_(avg_A)
                    lora_B[name].weight.data.sub_(avg_B)

            # 3) Set gks_done flag as true, so we won't do it again in ArrowLinearVariant.forward().
            self.gks_done = True
            # Clearing the self.gks_added_adapter_names
            self.gks_added_adapter_names = []

    def _cast_input_dtype(self, x, dtype: torch.dtype):
        """
        Whether to cast the dtype of the input of the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if x is None:  # useful e.g. if x is the bias, which can be None
            return None

        cast_input_dtype_enabled = getattr(self, "cast_input_dtype_enabled", True)
        if (not cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)

    def forward(self, x, lora_A, lora_B, dropout, scaling):
        """
        Applies Arrow routing inside a LoRA layer.

        Steps:
        1. Compute cosine similarity between each token representation and all adapter prototypes.
        2. Select the top-k experts per token and normalize their scores with a softmax.
        3. Project tokens into each selected expert’s low-rank space (A weights).
        4. Map back to the output space (B weights).
        5. Aggregate expert outputs via the weighted sum of their contributions.
        6. Apply dropout, scaling, and return the reshaped delta.

        - Conceptually, this is a Mixture-of-Experts (MoE) over LoRA adapters,
        where coefficients are derived from prototype similarity.

        Returns:
            delta: LoRA output adjustment computed by Arrow routing.
        """
        x = self._cast_input_dtype(x, lora_A[self.task_adapter_names[0]].weight.dtype)
        B, *rest, F_in = x.shape
        tok = x.view(-1, F_in)  # (t, F_in)
        t, E = tok.size(0), self.prototypes.size(0)

        # We now turn scaling, which is a dict, to tensors in order to use them later
        scales_tens = torch.tensor(
            [scaling[n] for n in self.task_adapter_names],
            device=tok.device,
            dtype=tok.dtype,
        )  # shape (E,)

        # 1) similarity   — sign-agnostic
        sim = torch.abs(tok @ self.prototypes.T)  # (t, E)

        # 2) top-k + softmax over full E (non-top-k = -inf)
        top_v, idx = torch.topk(sim, self.top_k, dim=1)
        full_score = tok.new_full((t, E), float("-inf"))
        full_score.scatter_(1, idx, top_v)
        coeff = torch.softmax(full_score / self.temperature, dim=1)  # (t, E)

        # 3) stack all A and B weights once
        #   A_stack: (E, r, in_features), B_stack: (E, out_features, r)
        A_stack = torch.stack([lora_A[n].weight for n in self.task_adapter_names], dim=0)
        B_stack = torch.stack([lora_B[n].weight for n in self.task_adapter_names], dim=0)

        # 4) project tokens into each expert’s low‑rank space:
        #    z[e] = tok @ A_e.T   → shape (t, E, r)
        z = torch.einsum("tf, erf -> ter", tok, A_stack)

        # 5) lift back each expert’s output:
        #    y[e] = z[e] @ B_e.T  → shape (t, E, out_features)
        y = torch.einsum("ter, eor -> teo", z, B_stack)

        # 6) apply per-expert scaling before the weighted sum
        # y_scaled[t, e, o] = scales[e] * y[t, e, o]
        y = y * scales_tens.view(1, -1, 1)

        # 6) weighted sum over experts:
        #    delta_flat[t,o] = Σ_e coeff[t,e] * y[t,e,o]
        delta_flat = torch.einsum("te, teo -> to", coeff, y)  # (t, out_features)

        # 7) dropout, scale, and reshape
        delta = dropout(delta_flat)
        out_dim = delta_flat.size(-1)
        return delta.view(B, *rest, out_dim)


def check_loaded_lora_compatibility_arrow(model, adapter_names: list[str]):
    """
    After loading all adapters into `model`, check they share:
      - the same LoRA rank (r)
      - identical weight shapes
      - identical sets of target_modules
    Returns (sorted list of target module names, agreed rank r).
    """
    reference = None  # {'r':…, 'shapes':(Ashape,Bshape), 'modules':set([...])}

    for name in adapter_names:
        curr_modules = set()
        curr_r = None
        curr_shapes = None

        for full_name, module in model.named_modules():
            if hasattr(module, "lora_A") and name in module.lora_A:
                A = module.lora_A[name].weight
                B = module.lora_B[name].weight
                mod_name = full_name.split(".")[-1]
                curr_modules.add(mod_name)
                # A has shape (r, in_features); B has shape (out_features, r)
                curr_r = A.shape[0]
                curr_shapes = (A.shape, B.shape)

        if reference is None:
            reference = {"r": curr_r, "shapes": curr_shapes, "modules": curr_modules}
        else:
            if curr_r != reference["r"]:
                raise ValueError(f"[{name}] rank mismatch: {curr_r} != {reference['r']}")
            if curr_shapes != reference["shapes"]:
                raise ValueError(f"[{name}] shape mismatch: {curr_shapes} != {reference['shapes']}")
            if curr_modules != reference["modules"]:
                raise ValueError(
                    f"[{name}] target_modules mismatch:\n"
                    f"  this adapter -> {sorted(curr_modules)}\n"
                    f"  reference   -> {sorted(reference['modules'])}"
                )

    agreed_modules = sorted(reference["modules"])
    return agreed_modules, int(reference["r"])


def ensure_adapters_target_linear_layers_only(model, adapter_names: list[str]):
    """
    Validate that every module holding LoRA weights for any of `adapter_names` is Linear-like: nn.Linear,
    bitsandbytes.nn.Linear4bit, nn.Conv1d, or transformers.models.gpt2.modeling_gpt2.Conv1D. If not, raise.
    """
    import torch.nn as nn

    Linear4bit = None
    try:
        import bitsandbytes as bnb  # type: ignore

        Linear4bit = bnb.nn.Linear4bit
    except ImportError:
        pass

    HFConv1D = None
    try:
        from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D  # type: ignore
    except ImportError:
        pass

    allowed_types = (nn.Linear, nn.Conv1d)
    if Linear4bit is not None:
        allowed_types = allowed_types + (Linear4bit,)
    if HFConv1D is not None:
        allowed_types = allowed_types + (HFConv1D,)

    offenders = []

    for full_name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            for name in adapter_names:
                if name in getattr(module, "lora_A", {}):
                    base = getattr(module, "base_layer", None) or getattr(module, "original_module", None)
                    layer_to_check = base if base is not None else module

                    if not isinstance(layer_to_check, allowed_types):
                        offenders.append((name, full_name, type(layer_to_check).__name__))

    if offenders:
        lines = [
            "LoRA adapters must only target Linear-like layers "
            "(nn.Linear, nn.Conv1d, HF Conv1D, or bitsandbytes.nn.Linear4bit). Found:"
        ]
        for name, full_name, tname in offenders:
            lines.append(f"  - adapter '{name}' on module '{full_name}' of type {tname}")
        raise TypeError("\n".join(lines))


def _resolve_adapter_source(path: str) -> tuple[str, str | None]:
    """
    Resolve a user-provided adapter `path` into (model_id, subfolder).

    Supports:
      - Local path to a folder that contains `adapter_config.json`
      - Hub path with subfolder, e.g. "user/repo/ts_expert_0[/more/...]", which becomes:
            model_id="user/repo", subfolder="ts_expert_0[/more/...]"
      - Plain Hub repo id "user/repo" (no subfolder)
    """
    if os.path.isdir(path):
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            raise ValueError(f"Local adapter path '{path}' does not contain 'adapter_config.json'.")
        return path, None

    parts = path.strip("/").split("/")
    if len(parts) >= 2:
        model_id = "/".join(parts[:2])
        if len(parts) > 2:
            subfolder = "/".join(parts[2:])
            return model_id, subfolder
        return model_id, None

    return path, None


def create_arrow_model(
    base_model: PreTrainedModel,
    task_specific_adapter_paths: list[str],
    arrow_config: ArrowConfig,
    general_adapter_paths: list[str] | None = None,
    **adapter_kwargs: Any,
):
    if task_specific_adapter_paths is None or len(task_specific_adapter_paths) == 0:
        raise ValueError("`task_specific_adapter_paths` should contain at least one adapter path")

    from peft import LoraConfig, PeftModel

    model_id0, sub0 = _resolve_adapter_source(task_specific_adapter_paths[0])
    initial_ts_expert_name = f"{TASK_ADAPTER_PREFIX}0"

    first_kwargs = dict(adapter_kwargs)
    if sub0 is not None and "subfolder" not in first_kwargs:
        first_kwargs["subfolder"] = sub0

    model = PeftModel.from_pretrained(
        base_model,
        model_id=model_id0,
        adapter_name=initial_ts_expert_name,
        **first_kwargs,
    )

    for i in range(1, len(task_specific_adapter_paths)):
        ts_expert_name = f"{TASK_ADAPTER_PREFIX}{i}"
        mid, sub = _resolve_adapter_source(task_specific_adapter_paths[i])
        more_kwargs = dict(adapter_kwargs)
        if sub is not None and "subfolder" not in more_kwargs:
            more_kwargs["subfolder"] = sub
        model.load_adapter(
            model_id=mid,
            adapter_name=ts_expert_name,
            **more_kwargs,
        )
    arrow_config.task_adapter_names = [f"{TASK_ADAPTER_PREFIX}{i}" for i in range(len(task_specific_adapter_paths))]

    if arrow_config.use_gks:
        if general_adapter_paths is None or len(general_adapter_paths) == 0:
            raise ValueError("You should provide general LoRA paths if you want to use GenKnowSub.")
        for i in range(len(general_adapter_paths)):
            gen_expert_name = f"{GKS_ADAPTER_PREFIX}{i}"
            mid, sub = _resolve_adapter_source(general_adapter_paths[i])
            gks_kwargs = dict(adapter_kwargs)
            if sub is not None and "subfolder" not in gks_kwargs:
                gks_kwargs["subfolder"] = sub
            model.load_adapter(
                model_id=mid,
                adapter_name=gen_expert_name,
                **gks_kwargs,
            )
        arrow_config.gks_adapter_names = [f"{GKS_ADAPTER_PREFIX}{i}" for i in range(len(general_adapter_paths))]
    else:
        arrow_config.gks_adapter_names = []

    target_modules, r = check_loaded_lora_compatibility_arrow(
        model, adapter_names=arrow_config.task_adapter_names + arrow_config.gks_adapter_names
    )

    ensure_adapters_target_linear_layers_only(
        model, adapter_names=arrow_config.task_adapter_names + arrow_config.gks_adapter_names
    )

    router_cfg = LoraConfig(
        arrow_config=arrow_config,
        target_modules=target_modules,
        r=r,
    )
    model.add_adapter(adapter_name="arrow_router", peft_config=router_cfg)
    model.set_adapter("arrow_router")

    return model
