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

import warnings
from collections.abc import Callable
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from peft.tuners.tuners_utils import BaseTunerLayer

from .arch import ExpertsLayout, MoeArchSpec, build_dense_expert_tensors, get_config_value, make_dense_config
from .config import MoeToDenseConfig
from .scoring import ExpertRoutingStats


def _describe_output(output: Any) -> str:
    """Describe the structure of a router output for error messages, e.g. `(int64[14, 2], float32[14, 2])`."""
    if isinstance(output, (tuple, list)):
        return "(" + ", ".join(_describe_output(o) for o in output) + ")"
    if torch.is_tensor(output):
        return f"{str(output.dtype).removeprefix('torch.')}{list(output.shape)}"
    return type(output).__name__


class MoeToDenseLayer(nn.Module, BaseTunerLayer):
    """
    Tuner layer that wraps the *experts* module of an MoE layer and replaces it by a dense FFN.

    The dense FFN corresponds to the concatenation of the `num_experts_to_keep` most important experts (Section 3.1,
    Eq. 2 of the paper). It is implemented as an instance of the same experts class as the wrapped module, but
    configured with a single expert whose intermediate size is `num_experts_to_keep` times the intermediate size of the
    original experts. This way, the activation function, the tensor layout, and the kernels of the original
    implementation are reused, and exporting the dense model is exact. The routing weights passed to the dense FFN are
    constant (the single dense "expert" is always selected with weight 1); the token-dependent routing weights of the
    original MoE layer are approximated by the static down projection scaling instead (Section 3.4 of the paper).

    Before the dense FFN is allocated via `update_and_allocate()` on the model, the layer collects routing statistics
    from the router module (which is a sibling of the experts module and is left in place) and passes calls through to
    the original experts.
    """

    adapter_layer_names = ("moe_to_dense_experts",)
    # per-adapter bookkeeping (dicts keyed by adapter name), cleaned up by `delete_adapter`
    other_param_names = ("num_experts_to_keep", "selected_experts", "_allocated_cache")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        peft_config: MoeToDenseConfig,
        *,
        router: nn.Module,
        router_name: str,
        spec: MoeArchSpec,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.moe_to_dense_experts = nn.ModuleDict({})
        self._active_adapter = adapter_name
        self.merged_adapters = []
        self.spec = spec
        self.layout = ExpertsLayout.from_module(base_layer)
        self.layout.validate_module(base_layer, self.layout.num_experts, self.layout.intermediate_size)
        self.top_k: int = int(get_config_value(base_layer.config, spec.top_k_fields))
        num_experts_from_config = int(get_config_value(base_layer.config, spec.num_experts_fields))
        if num_experts_from_config != self.layout.num_experts:
            raise ValueError(
                f"The config reports {num_experts_from_config} experts but the experts module holds "
                f"{self.layout.num_experts} experts. This architecture is probably not supported. Please open an "
                "issue on PEFT: https://github.com/huggingface/peft/issues."
            )

        # the router is not registered as a submodule, as it belongs to the base model, not to the adapter
        self.router_name = router_name
        self._router_ref = [router]
        # if the experts were deliberately left on the meta device (see `allocate`), keep the statistics on the CPU
        stats_device = None if self.base_layer.down_proj.is_meta else self.base_layer.down_proj.device
        self.stats = ExpertRoutingStats(self.layout.num_experts, device=stats_device)
        self._router_hook_handle = None
        self._allocated_cache: dict[str, bool] = {}
        self._warned_unallocated = False
        self._router_calls = 0
        self._router_output_description: str | None = None
        self.num_experts_to_keep: dict[str, int] = {}
        self.selected_experts: dict[str, list[int]] = {}

        self.register_router_hook()
        self.update_layer(adapter_name, peft_config)

    @property
    def router(self) -> nn.Module:
        return self._router_ref[0]

    def register_router_hook(self) -> None:
        if self._router_hook_handle is None:
            self._router_hook_handle = self.router.register_forward_hook(self._router_hook)

    def remove_router_hook(self) -> None:
        if self._router_hook_handle is not None:
            self._router_hook_handle.remove()
            self._router_hook_handle = None

    def _router_hook(self, module: nn.Module, args: tuple, output: Any) -> None:
        # Routers return the selected expert indices, the routing weights, and the logits or probabilities over all
        # experts, but not in a standardized order, so find them by dtype and shape.
        self._router_calls += 1
        tensors = [o for o in output if torch.is_tensor(o)] if isinstance(output, (tuple, list)) else []
        indices = next((o for o in tensors if not o.is_floating_point()), None)
        scores = next((o for o in tensors if o.is_floating_point() and o.shape[-1] == self.layout.num_experts), None)
        if (indices is None) or (scores is None):
            if self._router_output_description is None:
                self._router_output_description = _describe_output(output)
                warnings.warn(
                    f"The output of the router of type {type(module).__name__} could not be interpreted, no routing "
                    f"statistics can be collected. Expected a tuple containing an integer tensor with the indices of "
                    f"the selected experts and a float tensor of shape [..., {self.layout.num_experts}] with the "
                    f"routing logits or probabilities, but got {self._router_output_description}. This architecture "
                    f"is probably not supported. Please open an issue on PEFT with a reproducer: "
                    f"https://github.com/huggingface/peft/issues."
                )
            return

        # the scores are computed from the softmax probabilities over all experts (Section 3.2 of the paper)
        probs = scores.detach().float()
        returns_probs = self.spec.router_returns_probs
        if returns_probs is None:
            # heuristic: probabilities are non-negative and sum to 1
            sums = probs.sum(-1)
            returns_probs = bool((probs >= 0).all()) and bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-3))
        if not returns_probs:
            probs = torch.softmax(probs, dim=-1)
        self.stats.update(probs, indices.detach())

    def update_layer(self, adapter_name: str, peft_config: MoeToDenseConfig) -> None:
        """Create the (not yet allocated) dense FFN for the given adapter."""
        num_experts_to_keep = peft_config.num_experts_to_keep or self.top_k
        if num_experts_to_keep > self.layout.num_experts:
            raise ValueError(
                f"`num_experts_to_keep` is {num_experts_to_keep} but the MoE layer only has {self.layout.num_experts} "
                "experts."
            )

        # 'dense' module: expert module (e.g. GraniteMoeExperts) with a single expert
        intermediate_size = num_experts_to_keep * self.layout.intermediate_size
        dense_config = make_dense_config(self.base_layer.config, self.spec, intermediate_size)
        dense = type(self.base_layer)(dense_config)
        self.layout.validate_module(dense, 1, intermediate_size)

        with torch.no_grad():
            for param in dense.parameters():
                if param.device.type != "meta":
                    param.zero_()
        # persistent buffer so that loading an adapter checkpoint marks the layer as allocated
        dense.register_buffer("allocated", torch.zeros((), dtype=torch.bool))

        self.moe_to_dense_experts[adapter_name] = dense
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.num_experts_to_keep[adapter_name] = num_experts_to_keep
        self._allocated_cache.pop(adapter_name, None)
        # the new adapter needs routing statistics (unless its weights are loaded from a checkpoint afterwards)
        self.register_router_hook()
        self.set_adapter(self.active_adapters)

    def _get_base_layer_device_and_dtype(self, base_layer):
        # the experts module has no `weight` attribute, which the default implementation relies on
        param = base_layer.down_proj
        if param.is_meta:
            # The experts may deliberately be left on the meta device (their values are only needed by `allocate`,
            # e.g. when a trained adapter is loaded for `compress_and_unload`). Fall back to the router, which shares
            # device and dtype with the rest of the model; never report the meta device, as the adapter would be
            # moved to (and thereby destroyed on) it.
            param = next((p for p in self.router.parameters() if not p.is_meta), None)
            if param is None:
                return None, None
        return param.device, param.dtype

    def is_allocated(self, adapter_name: str) -> bool:
        """Whether the dense FFN of the given adapter has been allocated (or loaded from a checkpoint)."""
        if not self._allocated_cache.get(adapter_name, False):
            dense = self.moe_to_dense_experts[adapter_name]
            self._allocated_cache[adapter_name] = bool(dense.allocated.item())
            if self._allocated_cache[adapter_name] and self._all_allocated():
                # e.g. after loading the adapter weights from a checkpoint, no more statistics are needed
                self.remove_router_hook()
        return self._allocated_cache[adapter_name]

    def _all_allocated(self) -> bool:
        return all(bool(dense.allocated.item()) for dense in self.moe_to_dense_experts.values())

    def allocate(
        self,
        adapter_name: str,
        scoring_fn: Callable[[ExpertRoutingStats], torch.Tensor],
    ) -> bool:
        """
        Score the experts using the collected routing statistics, select the top scoring experts, and build the dense
        FFN from them.

        This implements the per-layer MoE-to-dense conversion of the paper (Section 3.1, Algorithm 1 in Appendix B) for
        the pure pruning case, i.e. every kept expert forms its own group and no experts are merged (`K = k` in the
        paper's notation, which Section 4.2 found to be the best configuration).

        Returns `True` if the dense FFN was allocated, `False` if it was already allocated before.
        """
        if self.is_allocated(adapter_name):
            return False

        if self.stats.num_tokens == 0:
            if self._router_calls == 0:
                raise RuntimeError(
                    "No routing statistics have been collected yet because the router was never called. Run forward "
                    "passes on calibration data with the MoE model before calling `update_and_allocate()`."
                )
            raise RuntimeError(
                f"No routing statistics have been collected although the router of type {type(self.router).__name__} "
                f"was called {self._router_calls} times, because its output could not be interpreted: expected a "
                f"tuple containing an integer tensor with the indices of the selected experts and a float tensor of "
                f"shape [..., {self.layout.num_experts}] with the routing logits or probabilities, but got "
                f"{self._router_output_description}. This architecture is probably not supported. Please open an "
                f"issue on PEFT with a reproducer: https://github.com/huggingface/peft/issues."
            )

        if self.base_layer.down_proj.is_meta:
            raise RuntimeError(
                "The MoE experts are on the meta device, but allocating the dense FFN requires their weights. Load "
                "the model with real expert weights, or load an already-allocated adapter checkpoint instead of "
                "calling `update_and_allocate()`."
            )

        dense = self.moe_to_dense_experts[adapter_name]
        num_experts_to_keep = self.num_experts_to_keep[adapter_name]
        scores = scoring_fn(self.stats).float().cpu()
        # top-K selection by importance score (Section 3.1 of the paper); the selected experts are sorted by index so
        # that the order of concatenation is deterministic (the order does not affect the function of the dense FFN)
        selected = torch.topk(scores, num_experts_to_keep).indices.sort().values

        # Uniform down projection scaling (Section 3.4 and Appendix H of the paper): each kept expert contributes
        # 1/K, which matches the average routing weight when the router normalizes its top-k weights to sum to 1.
        scales = torch.full((num_experts_to_keep,), 1.0 / num_experts_to_keep)
        if self.spec.expert_scale_fn is not None:
            # some routers additionally apply a static per-expert output scale (e.g. Gemma 4), fold it in
            expert_scale = self.spec.expert_scale_fn(self.router)
            if expert_scale is not None:
                scales = scales * expert_scale.float().cpu()[selected]

        tensors = build_dense_expert_tensors(self.base_layer, self.layout, selected, scales)
        with torch.no_grad():
            for name, tensor in tensors.items():
                param = getattr(dense, name)
                if param.shape != tensor.shape:
                    raise RuntimeError(
                        f"Shape mismatch when allocating the dense FFN: parameter '{name}' has shape "
                        f"{tuple(param.shape)} but the concatenated experts have shape {tuple(tensor.shape)}."
                    )
                param.data.copy_(tensor)
            dense.allocated.fill_(True)

        self.selected_experts[adapter_name] = selected.tolist()
        self._allocated_cache[adapter_name] = True
        if self._all_allocated():
            # no more statistics needed
            self.remove_router_hook()
        return True

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters or not self.active_adapters:
            return self.base_layer(hidden_states, *args, **kwargs)

        if len(self.active_adapters) != 1:
            raise ValueError(
                f"MoE-to-dense only supports a single active adapter, but {len(self.active_adapters)} adapters are "
                "active."
            )

        adapter_name = self.active_adapters[0]
        if not self.is_allocated(adapter_name):
            if not self._warned_unallocated:
                warnings.warn(
                    "The dense FFN has not been allocated yet, the original MoE experts are used instead. Run forward "
                    "passes on calibration data and then call `update_and_allocate()` on the PEFT model to allocate "
                    "the dense FFN."
                )
                self._warned_unallocated = True
            return self.base_layer(hidden_states, *args, **kwargs)

        # sanity check on the arguments: we assume that the MoE module is called only with 3 arguments:
        # - hidden_states
        # - top_k_index
        # - top_k_weights
        # `hidden_states, *args, **kwargs` should thus total 3 arguments (there are no defaults)
        num_fw_args = 1 + len(args) + len(kwargs)
        if num_fw_args != 3:
            raise RuntimeError(
                f"The signature of the MoE layer is different from what PEFT expects ({num_fw_args} arguments "
                "instead of 3 arguments). We cannot be sure that we're calling it correctly."
            )

        dense = self.moe_to_dense_experts[adapter_name]
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        num_tokens = hidden_states.shape[0]
        # the single dense "expert" is always selected with routing weight 1
        top_k_index = torch.zeros((num_tokens, 1), dtype=torch.long, device=hidden_states.device)
        top_k_weights = torch.ones((num_tokens, 1), dtype=hidden_states.dtype, device=hidden_states.device)
        return dense(hidden_states, top_k_index, top_k_weights).reshape(shape)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moe_to_dense." + rep


def forward_kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute the forward KL divergence `KL(p_teacher || p_student)` between the teacher and student logits, averaged
    over the (unmasked) tokens.

    This is the knowledge distillation objective of the paper (Section 3.5, Eq. 5); the paper found the forward KL
    divergence on the logits to clearly outperform both the reverse KL divergence and an additional intermediate
    hidden-state loss (Section 4.3, Table 5).

    The computation is chunked over tokens and uses activation checkpointing for the chunks, so that the full float32
    log-probabilities are never materialized at once.

    Args:
        teacher_logits (`torch.Tensor`):
            Logits of the teacher, shape `[..., vocab_size]`.
        student_logits (`torch.Tensor`):
            Logits of the student, same shape as `teacher_logits`.
        mask (`torch.Tensor`, *optional*):
            Mask of shape `[...]` (matching the leading dimensions of the logits), tokens with value 0/`False` are
            ignored. If `None`, all tokens are used.
        temperature (`float`):
            Softmax temperature. The loss is scaled by `temperature**2`. The paper uses `temperature=1` (Appendix J).
        chunk_size (`int`):
            Number of tokens per chunk.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"Teacher and student logits must have the same shape, got {tuple(teacher_logits.shape)} and "
            f"{tuple(student_logits.shape)}. If you passed the teacher logits explicitly, ensure that they were "
            f"determined correctly (same dataset, same order of samples, same batch size, etc.)."
        )

    vocab_size = student_logits.shape[-1]
    teacher = teacher_logits.reshape(-1, vocab_size)
    student = student_logits.reshape(-1, vocab_size)
    num_tokens = student.shape[0]
    if mask is None:
        mask = torch.ones(num_tokens, dtype=torch.bool, device=student.device)
    mask = mask.reshape(-1).bool()
    if mask.shape[0] != num_tokens:
        raise ValueError(f"The mask has {mask.shape[0]} entries but there are {num_tokens} tokens.")

    # memory optimization, especially for large vocabs: don't materialize and (for backward) store, big tensors
    def kl_chunk(t: torch.Tensor, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(t.float() / temperature, dim=-1)
        log_q = torch.log_softmax(s.float() / temperature, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        return (kl * m).sum()

    total = student.new_zeros((), dtype=torch.float32)
    use_checkpoint = torch.is_grad_enabled() and student.requires_grad
    for start in range(0, num_tokens, chunk_size):
        end = start + chunk_size
        t, s, m = teacher[start:end], student[start:end], mask[start:end]
        if use_checkpoint:
            total = total + checkpoint(kl_chunk, t, s, m, use_reentrant=False)
        else:
            total = total + kl_chunk(t, s, m)
    return total / mask.sum().clamp(min=1) * (temperature**2)
