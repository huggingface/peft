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
from typing import Any, Optional

import torch
from torch import nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner
from peft.utils.other import AuxiliaryTrainingWrapper, _get_submodules

from .arch import find_experts_modules, find_router, get_arch_spec
from .config import MoeToDenseConfig
from .export import EXPORT_STRATEGIES, export_problem
from .layer import MoeToDenseLayer, forward_kl_divergence
from .scoring import SCORING_FUNCTIONS


class MoeToDenseModel(BaseTuner):
    """
    Converts a Mixture-of-Experts (MoE) model into a dense model by pruning experts and distilling from the MoE model.

    The method follows "Pruning and Distilling Mixture-of-Experts into Dense Language Models" (Kim et al., 2026,
    https://arxiv.org/abs/2605.28207). The typical workflow is:

    1. Wrap the MoE model with `get_peft_model`. The experts modules of the MoE layers are wrapped and start collecting
       routing statistics during forward passes.
    2. Run forward passes on calibration data (a few hundred batches of training data are sufficient) and call
       `update_and_allocate()`. The experts are scored by importance, the top scoring experts are concatenated into a
       dense FFN per layer, and the model switches from the MoE experts to the dense FFN.
    3. Distill the MoE model (the teacher) into the dense model (the student) by minimizing
       `get_distillation_loss(**batch)`. Since the teacher and the student share all parameters except for the FFNs,
       the teacher forward pass is simply the forward pass with disabled adapters.
    4. Call `compress_and_unload()` to obtain a standalone dense model that can be saved with `save_pretrained`.

    Note that, unlike most PEFT methods, the trainable "adapter" (the dense FFNs) is large: its size corresponds to the
    number of *active* parameters of the MoE model. The parameter efficiency comes from dropping the inactive experts
    at inference time.

    Args:
        model ([`transformers.PreTrainedModel`]):
            The MoE model to be converted.
        config ([`MoeToDenseConfig`]):
            The configuration of the conversion.
        adapter_name (`str`):
            The name of the adapter, defaults to `"default"`.

    Example:

    ```py
    >>> from transformers import AutoModelForCausalLM
    >>> from peft import MoeToDenseConfig, get_peft_model

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", dtype="bfloat16")
    >>> model = get_peft_model(model, MoeToDenseConfig())
    >>> with torch.no_grad():
    ...     for batch in calibration_batches:
    ...         model(**batch)
    >>> model.update_and_allocate()
    >>> for batch in train_batches:
    ...     loss = model.get_distillation_loss(**batch)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    >>> dense_model = model.compress_and_unload()
    >>> dense_model.save_pretrained("qwen3-30b-a3b-dense")
    ```
    """

    prefix: str = "moe_to_dense_"
    tuner_layer_cls = MoeToDenseLayer
    target_module_mapping: dict[str, Any] = {}

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            target_modules = find_experts_modules(self.model)
            if not target_modules:
                raise ValueError(
                    "Could not find any MoE experts modules in the model. Please specify the names of the experts "
                    "modules via `target_modules` (the modules that hold the weights of all experts as 3D tensors) or "
                    "check that the model is an MoE model with a transformers-style MoE implementation."
                )
            peft_config.target_modules = set(target_modules)
        return peft_config

    def _create_and_replace(
        self,
        peft_config: MoeToDenseConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        if isinstance(target, MoeToDenseLayer):
            target.update_layer(adapter_name, peft_config)
            return

        spec = get_arch_spec(getattr(target, "config", None))
        router_name, router = find_router(parent, target_name, spec)
        new_module = MoeToDenseLayer(
            target, adapter_name, peft_config, router=router, router_name=router_name, spec=spec
        )
        if adapter_name != self.active_adapter:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        # The default implementation additionally moves the adapter submodules to the device of the wrapped module,
        # which must be avoided when the experts were deliberately left on the meta device (their values are only needed
        # for `allocate`). Device placement of the dense FFN is already handled in `update_layer` via
        # `_move_adapter_to_device_of_base_layer`.
        setattr(parent, child_name, new_module)

    def _post_injection_hook(self, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        # Warn early if the architecture cannot be exported to a standalone dense model, so that the user does not find
        # out only after distillation.
        layers = self._get_layers()
        if not layers:
            return

        problem = export_problem(model, layers, adapter_name, layers[0][1].spec)
        if problem is not None:
            warnings.warn(
                f"`compress_and_unload()` will most likely not produce a working standalone dense model for this "
                f"architecture because {problem}. Training and saving the adapter with `save_pretrained` still "
                "works."
            )

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        # The dense FFNs are copies of base model weights, they keep the dtype of the base model.
        pass

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        if not isinstance(adapter_name, str) and (len(adapter_name) > 1):
            raise ValueError(
                f"MoE-to-dense only supports a single active adapter, but got {len(adapter_name)} adapters: "
                f"{list(adapter_name)}."
            )
        super().set_adapter(adapter_name, inference_mode=inference_mode)

    def _get_layers(self) -> list[tuple[str, MoeToDenseLayer]]:
        return [(key, module) for key, module in self.model.named_modules() if isinstance(module, MoeToDenseLayer)]

    def _get_single_active_adapter(self) -> str:
        active_adapters = self.active_adapters
        if len(active_adapters) != 1:
            raise ValueError(
                f"MoE-to-dense requires exactly one active adapter, but {len(active_adapters)} adapters are active."
            )
        return active_adapters[0]

    def update_and_allocate(self) -> None:
        """
        Allocate the dense FFNs from the routing statistics collected so far.

        For each MoE layer, the experts are scored by their conditional probability (Section 3.2, Eq. 3 of the paper),
        the top scoring experts are selected and concatenated into the dense FFN with uniform down projection scaling
        (Sections 3.1 and 3.4, Algorithm 1 in Appendix B), and the layer switches from the MoE experts to the dense
        FFN. This is a one-time operation, calling the method again is a no-op. Routing statistics are collected during
        all forward passes of the model until the dense FFNs are allocated, so run forward passes on calibration data
        first (the paper uses 512 sequences of 2048 tokens, Section 4.1).
        """
        adapter_name = self._get_single_active_adapter()
        scoring_fn = SCORING_FUNCTIONS[self.peft_config[adapter_name].scoring]
        num_allocated = 0
        for _, layer in self._get_layers():
            num_allocated += layer.allocate(adapter_name, scoring_fn)
        if num_allocated == 0:
            warnings.warn("The dense FFNs have already been allocated, `update_and_allocate()` has no effect.")

    def get_distillation_loss(
        self,
        *args: Any,
        teacher_logits: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        chunk_size: int = 4096,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the distillation loss between the MoE teacher and the dense student for a batch of inputs: the forward
        KL divergence of the next-token distributions, averaged over the tokens (Section 3.5, Eq. 5 of the paper).

        The arguments are passed to the forward call of the model. Unless `teacher_logits` are passed, the teacher
        logits are computed by a forward pass with disabled adapters (i.e. with the original MoE experts) without
        gradients. If `labels` are passed, tokens whose label is -100 are ignored; otherwise, if an `attention_mask` is
        passed, padding tokens are ignored.

        Args:
            teacher_logits (`torch.Tensor`, *optional*):
                Precomputed logits of the teacher. Pass these to control the teacher forward pass yourself (e.g. to
                distill from expanded teacher routing as in Section 3.5, Eq. 6 of the paper, or to cache the teacher
                logits). Ensure that these logits were created with the exact same model and data (same dataset, same
                order of samples, same batch size).
            temperature (`float`, *optional*):
                Softmax temperature for the KL divergence, defaults to 1.0 (the paper's setting, Appendix J).
            chunk_size (`int`, *optional*):
                Number of tokens per chunk for the KL computation, which limits the peak memory usage.
        """
        labels = kwargs.pop("labels", None)
        attention_mask = kwargs.get("attention_mask", None)

        if teacher_logits is None:
            with torch.no_grad():
                self.disable_adapter_layers()
                try:
                    teacher_logits = self.model(*args, **kwargs).logits
                finally:
                    self.enable_adapter_layers()
        student_logits = self.model(*args, **kwargs).logits

        if labels is not None:
            # the logits at position t predict the token at position t+1, so the mask must be shifted accordingly
            mask = labels[..., 1:] != -100
            teacher_logits = teacher_logits[..., :-1, :]
            student_logits = student_logits[..., :-1, :]
        elif attention_mask is not None:
            mask = attention_mask
        else:
            mask = None
        return forward_kl_divergence(
            teacher_logits, student_logits, mask=mask, temperature=temperature, chunk_size=chunk_size
        )

    def compress_and_unload(self, progressbar: bool = False) -> nn.Module:
        """
        Replace the MoE layers by the dense FFNs and return the resulting dense model without any PEFT modules.

        The model config is adjusted so that the returned model can be saved with `save_pretrained` and loaded again
        with `from_pretrained` (best effort). Depending on the architecture, either the whole MoE block is replaced by
        the architecture's dense MLP class (e.g. Qwen3-MoE), or the MoE block is kept with a single expert and a router
        that always selects it (generic fallback). In the latter case, some routing overhead remains at inference time,
        which can be removed with e.g. `torch.compile`.

        It is important to assign the returned model to a variable and use it, this is not an in-place operation!

        Args:
            progressbar (`bool`):
                Whether to show a progressbar (default: False).
        """
        adapter_name = self._get_single_active_adapter()
        layers = self._get_layers()
        if not layers:
            raise RuntimeError("No MoE-to-dense layers found in the model.")

        not_allocated = [key for key, layer in layers if not layer.is_allocated(adapter_name)]
        if not_allocated:
            raise RuntimeError(
                f"The dense FFNs of {len(not_allocated)} layers have not been allocated yet. Call "
                "`update_and_allocate()` after collecting routing statistics before compressing the model."
            )

        for _, layer in layers:
            layer.remove_router_hook()

        exports = {layer.spec.export for _, layer in layers}
        if len(exports) != 1:
            raise ValueError(f"All MoE layers must use the same export strategy, got {sorted(exports)}.")

        export = EXPORT_STRATEGIES[exports.pop()]
        export(self.model, layers, adapter_name, layers[0][1].spec, progressbar)
        # replace the `modules_to_save` wrappers by the trained copies
        self._unwrap_auxiliary_modules(keep_trained=True, adapter_names=[adapter_name])

        if hasattr(self.model, "peft_config"):
            del self.model.peft_config
        return self.model

    def _unwrap_auxiliary_modules(self, keep_trained: bool, adapter_names: Optional[list[str]] = None) -> None:
        """Replace `modules_to_save` wrappers by the trained copy (`keep_trained=True`) or by the original module."""
        wrapped = [
            (key, module) for key, module in self.model.named_modules() if isinstance(module, AuxiliaryTrainingWrapper)
        ]
        for key, module in wrapped:
            parent, _, target_name = _get_submodules(self.model, key)
            if keep_trained:
                new_module = module.unload_and_optionally_merge_module(
                    merge=True, safe_merge=False, adapter_names=adapter_names
                )
            else:
                new_module = module.original_module
            setattr(parent, target_name, new_module)

    def unload(self) -> nn.Module:
        """
        Return the original MoE model by removing all the PEFT modules.

        It is important to assign the returned model to a variable and use it, this is not an in-place operation!
        """
        for key, layer in self._get_layers():
            layer.remove_router_hook()
            parent, _, target_name = _get_submodules(self.model, key)
            setattr(parent, target_name, layer.get_base_layer())
        self._unwrap_auxiliary_modules(keep_trained=False)
        if hasattr(self.model, "peft_config"):
            del self.model.peft_config
        return self.model

    def _unload_and_optionally_merge(self, *args: Any, **kwargs: Any) -> nn.Module:
        # this should never be called with unload and merge_and_unload overridden, raising here just to be safe
        raise NotImplementedError("Use `compress_and_unload()` or `unload()` instead.")

    def merge_and_unload(self, *args: Any, **kwargs: Any) -> nn.Module:
        raise NotImplementedError(
            "MoE-to-dense does not support merging, as the dense FFNs replace the MoE layers instead of being added "
            "to them. Use `compress_and_unload()` to obtain the dense model."
        )

    def merge_adapter(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "MoE-to-dense does not support merging, as the dense FFNs replace the MoE layers instead of being added "
            "to them. Use `compress_and_unload()` to obtain the dense model."
        )

    def unmerge_adapter(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("MoE-to-dense does not support merging, hence there is nothing to unmerge.")
