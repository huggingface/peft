# Copyright 2023-present the HuggingFace Inc. team.
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

import inspect
import re
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from functools import update_wrapper
from types import MethodType
from typing import Any, Optional, Union

import torch
from torch import nn


try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from tqdm.auto import tqdm

from .peft_model import PeftConfig, PeftModel
from .tuners.lora import LoraLayer, LoraModel, dora
from .tuners.lora.conversion import _find_cutoff_index
from .tuners.tuners_utils import BaseTunerLayer


def update_forward_signature(model: PeftModel) -> None:
    """
    Updates the forward signature of the PeftModel to include parents class signature
        model (`PeftModel`): Peft model to update the forward signature

    Example:

    ```python
    >>> from transformers import WhisperForConditionalGeneration
    >>> from peft import get_peft_model, LoraConfig
    >>> from peft.helpers import update_forward_signature

    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])

    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_forward_signature(peft_model)
    ```
    """

    # Only update signature when the current forward signature only has *args and **kwargs
    current_signature = inspect.signature(model.forward)
    if (
        len(current_signature.parameters) == 2
        and "args" in current_signature.parameters
        and "kwargs" in current_signature.parameters
    ):
        forward = deepcopy(model.forward.__func__)
        update_wrapper(
            forward, type(model.get_base_model()).forward, assigned=("__doc__", "__name__", "__annotations__")
        )
        model.forward = MethodType(forward, model)


def update_generate_signature(model: PeftModel) -> None:
    """
    Updates the generate signature of a PeftModel with overriding generate to include parents class signature
        model (`PeftModel`): Peft model to update the generate signature

    Example:

    ```python
    >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    >>> from peft import get_peft_model, LoraConfig, TaskType
    >>> from peft.helpers import update_generate_signature

    >>> model_name_or_path = "bigscience/mt0-large"
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    >>> model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    >>> peft_config = LoraConfig(
    ...     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    ... )
    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_generate_signature(peft_model)
    >>> help(peft_model.generate)
    ```
    """
    if not hasattr(model, "generate"):
        return
    current_signature = inspect.signature(model.generate)
    if (
        len(current_signature.parameters) == 2
        and "args" in current_signature.parameters
        and "kwargs" in current_signature.parameters
    ) or (len(current_signature.parameters) == 1 and "kwargs" in current_signature.parameters):
        generate = deepcopy(model.generate.__func__)
        update_wrapper(
            generate,
            type(model.get_base_model()).generate,
            assigned=("__doc__", "__name__", "__annotations__"),
        )
        model.generate = MethodType(generate, model)


def update_signature(model: PeftModel, method: str = "all") -> None:
    """
    Updates the signature of a PeftModel include parents class signature for forward or generate method
        model (`PeftModel`): Peft model to update generate or forward signature method (`str`): method to update
        signature choose one of "forward", "generate", "all"

    Example:
    ```python
    >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    >>> from peft import get_peft_model, LoraConfig, TaskType
    >>> from peft.helpers import update_signature

    >>> model_name_or_path = "bigscience/mt0-large"
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    >>> model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    >>> peft_config = LoraConfig(
    ...     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    ... )
    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_signature(peft_model)
    >>> help(peft_model.generate)
    ```
    """
    if method == "forward":
        update_forward_signature(model)
    elif method == "generate":
        update_generate_signature(model)
    elif method == "all":
        update_forward_signature(model)
        update_generate_signature(model)
    else:
        raise ValueError(f"method {method} is not supported please choose one of ['forward', 'generate', 'all']")


def check_if_peft_model(model_name_or_path: str) -> bool:
    """
    Check if the model is a PEFT model.

    Args:
        model_name_or_path (`str`):
            Model id to check, can be local or on the Hugging Face Hub.

    Returns:
        `bool`: True if the model is a PEFT model, False otherwise.
    """
    is_peft_model = True
    try:
        PeftConfig.from_pretrained(model_name_or_path)
    except Exception:
        # allow broad exceptions so that this works even if new exceptions are added on HF Hub side
        is_peft_model = False

    return is_peft_model


@contextmanager
def rescale_adapter_scale(model: nn.Module, multiplier: Union[float, int]) -> Iterator[None]:
    """
    Context manager to temporarily rescale the scaling of the LoRA adapter in a model.

    The original scaling values are restored when the context manager exits. This context manager works with the
    transformers and diffusers models that have directly loaded LoRA adapters.

    For LoRA, applying this context manager with multiplier in [0, 1] is strictly equivalent to applying
    [wise-ft](https://huggingface.co/papers/2109.01903) (see [#1940](https://github.com/huggingface/peft/issues/1940)
    for details). It can improve the performances of the model if there is a distribution shiftbetween the training
    data used for fine-tuning, and the test data used during inference.

    Warning: It has been reported that when using Apple's MPS backend for PyTorch, it is necessary to add a short sleep
        time after exiting the context before the scales are fully restored.

    Args:
        model: The model containing `LoraLayer` modules whose scaling is to be adjusted.
        multiplier (float or int):
            The multiplier that rescales the `scaling` attribute. Must be of type float or int.

    Raises:
        ValueError: If the model does not contain any `LoraLayer`
            instances, indicating that the model does not support scaling.

    Example:

    ```python
    >>> model = ModelWithLoraLayer()
    >>> multiplier = 0.5
    >>> with rescale_adapter_scale(model, multiplier):
    ...     outputs = model(**inputs)  # Perform operations with the scaled model
    >>> outputs = model(**inputs)  # The original scaling values are restored here
    ```
    """
    # check if multiplier has a valid data type
    if not isinstance(multiplier, (float, int)):
        raise TypeError(f"Argument multiplier should be of type float, got {type(multiplier)}")

    # iterate on the model's modules and grab the original scaling attribute
    # from the lora layers if present
    original_scaling = {}
    for module in model.modules():
        if isinstance(module, LoraLayer):
            original_scaling[module] = module.scaling.copy()
            module.scaling = {k: v * multiplier for k, v in module.scaling.items()}

    # check whether scaling is prohibited on model
    # the original scaling dictionary should be empty
    # if there were no lora layers
    if not original_scaling:
        raise ValueError("scaling is only supported for models with `LoraLayer`s")
    try:
        yield

    finally:
        # restore original scaling values after exiting the context
        for module, scaling in original_scaling.items():
            module.scaling = scaling


@contextmanager
def disable_input_dtype_casting(model: nn.Module, active: bool = True):
    """
    Context manager disables input dtype casting to the dtype of the weight.

    Parameters:
        model (nn.Module):
            The model containing PEFT modules whose input dtype casting is to be adjusted.
        active (bool):
            Whether the context manager is active (default) or inactive.

    """
    # Additional info: Normally, the dtype of the weight and input need to match, which is why the dtype is cast.
    # However, in certain circumustances, this is handled by forward hooks, e.g. when using layerwise casting in
    # diffusers. In that case, PEFT casting the dtype interferes with the layerwise casting, which is why the option to
    # disable it is given.
    if not active:
        yield
        return

    original_values = {}
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        original_values[name] = module.cast_input_dtype_enabled
        module.cast_input_dtype_enabled = False

    try:
        yield
    finally:
        for name, module in model.named_modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            if name in original_values:
                module.cast_input_dtype_enabled = original_values[name]


class MontecloraTrainerMixin:
    """
    Mixin class for adding Monteclora variational loss to the Trainer's compute_loss method.

    This mixin can be used with any Trainer class (e.g., Trainer, SFTTrainer) to add support for Monteclora's
    variational regularization during training.

    Example:
        ```python
        from transformers import Trainer
        from peft import get_peft_model, LoraConfig
        from peft.helpers import MontecloraTrainerMixin


        # custom trainer that supports Monteclora
        class MontecloraTrainer(MontecloraTrainerMixin, Trainer):
            pass


        # Configure LoRA with Monteclora
        monteclora_config = MontecloraConfig(
            num_samples=8,
            sample_scaler=1e-4,
            kl_loss_weight=1e-5,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            monteclora_config=monteclora_config,
        )

        # Get PEFT model and train
        model = get_peft_model(base_model, lora_config)
        trainer = MontecloraTrainer(model=model, args=training_args)
        trainer.train()
        ```
    """

    def compute_loss(
        self, model: nn.Module, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        """
        Compute loss with Monteclora variational regularization.

        This method extends the standard compute_loss by adding the variational loss (KL divergence + entropy) from
        Monteclora samplers to the task loss.

        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return model outputs along with loss
            **kwargs: Additional arguments

        Returns:
            loss or (loss, outputs) depending on return_outputs
        """
        if return_outputs:
            task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            task_loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
            outputs = None

        # `_get_monteclora_loss` already normalizes by the number of samplers within a single LoraModel.
        # In the typical case there is exactly one LoraModel (PeftModel.base_model) so this loop runs once
        # and matches the original behavior. For unusual setups with multiple tuners we additionally average
        # across the LoraModels so the regularization magnitude does not scale with the number of tuners.
        # `_get_monteclora_loss` returns 0.0 when no MonteCLoRA samplers are present, so this is a no-op
        # for plain LoRA training.
        monteclora_loss = 0.0
        num_lora_models = 0
        for module in model.modules():
            if isinstance(module, LoraModel):
                monteclora_loss = monteclora_loss + module._get_monteclora_loss()
                num_lora_models += 1
        if num_lora_models > 1:
            monteclora_loss = monteclora_loss / num_lora_models

        total_loss = task_loss + monteclora_loss

        return (total_loss, outputs) if return_outputs else total_loss


class DoraCaching:
    """Context manager to enable DoRA caching, which improves speed of DoRA inference at the expense of memory.

    With active caching, the materialized LoRA weight (B @ A) and the weight norm (base weight + LoRA weight) are
    cached.

    Even within the caching context, if the model is in training mode, caching is disabled. When the model switches to
    training mode, the cache will be cleared.

    Example:

        ```py
        >>> from peft.helpers import DoraCaching

        >>> model.eval()  # put in eval model for caching to work

        >>> with DoraCaching():  # use as a context manager
        ...     output = model(inputs)

        >>> dora_caching = DoraCaching()
        >>> dora_caching(enabled=True)  # permanently enable caching
        >>> output = model(inputs)
        >>> dora_caching(enabled=False)  # permanently disable caching
        >>> output = model(inputs)
        ```

    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.prev_value = None

    def __enter__(self):
        self.prev_value = dora.ENABLE_DORA_CACHING
        dora.ENABLE_DORA_CACHING = self.enabled

    def __exit__(self, type, value, traceback):
        dora.ENABLE_DORA_CACHING = self.prev_value
        self.prev_value = None

    def __call__(self, enabled: bool = True):
        dora.ENABLE_DORA_CACHING = enabled


class KappaTuneSelector:
    """
    Lightweight utility to compute per-module / per-parameter condition numbers and return the best LoRA targets.

    Supports:
    - Classic nn.Linear modules (target_modules in LoraConfig)
    - Modern fused MoE weights stored as 3D nn.Parameter (gate_up_proj / down_proj, gate_proj / up_proj, etc.) used in
      Llama-4, Qwen2_MoE, Qwen3_MoE, Mixtral, OLMoE and similar models. These are returned via target_parameters.

    Notes:
    - Condition-number computation requires running SVD and can take several minutes on very large models. A progress
    bar can be shown/disabled via `show_progress`.

    """

    def __init__(
        self,
        model: nn.Module,
        max_dim_size_to_analyze: int = 16384,
        moe_param_suffixes: Optional[tuple[str, ...]] = None,
        show_progress: bool = True,
    ):
        self.model = model
        self.max_dim_size_to_analyze = max_dim_size_to_analyze
        self.show_progress = show_progress
        self.moe_param_suffixes = moe_param_suffixes or (
            ".gate_up_proj",
            ".down_proj",
            ".gate_proj",
            ".up_proj",
        )
        self._condition_numbers: Optional[dict[str, float]] = None
        self._parameter_condition_numbers: Optional[dict[str, float]] = None

    def _compute_kappas(self) -> None:
        if self._condition_numbers is not None:
            return

        # === 1. nn.Linear modules ===
        condition_numbers: dict[str, float] = {}
        linear_modules = [
            (module_name, module)
            for module_name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        linear_iter = (
            tqdm(linear_modules, desc="Computing SVD (linear layers)", unit="layer")
            if self.show_progress
            else linear_modules
        )
        for module_name, module in linear_iter:
            weight = module.weight
            if bnb is not None:
                if hasattr(weight, "quant_state"):  # 4-bit
                    w = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).float()
                elif hasattr(weight, "state") and hasattr(weight.state, "CB"):  # int8
                    w = bnb.functional.int8_vectorwise_dequant(weight.state.CB, weight.state.SCB).float()
                else:
                    w = weight.data.detach().float()
            else:
                w = weight.data.detach().float()

            if any(dim > self.max_dim_size_to_analyze for dim in w.shape):
                continue

            S = torch.linalg.svdvals(w.view(w.size(0), -1))
            kappa = (S[0] / (S[-1] + 1e-8)).item()
            condition_numbers[module_name] = kappa

        self._condition_numbers = condition_numbers

        # === 2. fused MoE parameters (3D nn.Parameter) ===
        parameter_condition_numbers: dict[str, float] = {}
        moe_params = [
            (param_name, param)
            for param_name, param in self.model.named_parameters()
            if any(param_name.endswith(s) for s in self.moe_param_suffixes) and param.dim() == 3
        ]
        moe_iter = (
            tqdm(moe_params, desc="Computing SVD (MoE parameters)", unit="param") if self.show_progress else moe_params
        )
        for param_name, param in moe_iter:
            w = param.data.detach().float()
            num_experts, *expert_shape = w.shape

            if any(dim > self.max_dim_size_to_analyze for dim in expert_shape):
                continue

            kappas = []
            for expert_idx in range(num_experts):
                expert_w = w[expert_idx]
                S = torch.linalg.svdvals(expert_w)
                kappa = (S[0] / (S[-1] + 1e-8)).item()
                kappas.append(kappa)
            kappa = sum(kappas) / len(kappas)
            parameter_condition_numbers[param_name] = kappa

        self._parameter_condition_numbers = parameter_condition_numbers

    def get_best_targets(
        self, top_p: Optional[float] = None, num_modules: Optional[int] = None, threshold: Optional[float] = None
    ) -> list[str]:
        self._compute_kappas()
        if not self._condition_numbers:
            return []

        sorted_modules = sorted(self._condition_numbers.items(), key=lambda x: x[1])

        if num_modules is not None:
            k = min(num_modules, len(sorted_modules))
            return [name for name, _ in sorted_modules[:k]]

        if top_p is not None:
            k = max(1, int(len(sorted_modules) * top_p))
            return [name for name, _ in sorted_modules[:k]]

        if threshold is not None:
            return [name for name, kappa in sorted_modules if kappa <= threshold]

        return [name for name, _ in sorted_modules]

    def get_best_target_parameters(
        self, top_p: Optional[float] = None, num_modules: Optional[int] = None, threshold: Optional[float] = None
    ) -> list[str]:
        self._compute_kappas()
        if not self._parameter_condition_numbers:
            return []

        sorted_params = sorted(self._parameter_condition_numbers.items(), key=lambda x: x[1])

        if num_modules is not None:
            k = min(num_modules, len(sorted_params))
            return [name for name, _ in sorted_params[:k]]

        if top_p is not None:
            k = max(1, int(len(sorted_params) * top_p))
            return [name for name, _ in sorted_params[:k]]

        if threshold is not None:
            return [name for name, kappa in sorted_params if kappa <= threshold]

        return [name for name, _ in sorted_params]


def find_kappa_target_modules(
    model: nn.Module,
    top_p: float = 0.2,
    max_dim_size_to_analyze: int = 16384,
    moe_param_suffixes: Optional[tuple[str, ...]] = None,
    show_progress: bool = True,
) -> dict[str, Optional[list[str]]]:
    """
    One-liner convenience function for KappaTune target selection. Returns both target_modules and target_parameters.

    Args:
        model (nn.Module):
            Base model whose weights will be analyzed for condition numbers.
        top_p (float, optional):
            Select the top fraction of candidate modules/parameters with the lowest condition numbers.
        max_dim_size_to_analyze (int, optional):
            Upper bound on the maximum matrix dimension analyzed via SVD. Defaults to 16384.
        moe_param_suffixes (Optional[tuple[str, ...]], optional):
            Parameter-name suffixes used to identify fused MoE tensors that should be returned via `target_parameters`.
            If None, sensible defaults are used.
        show_progress (bool, optional):
            Whether to display a progress bar while computing condition numbers (SVD-based) across candidate
            tensors/modules. Disable in CI or other non-interactive environments. Defaults to True.
    """
    selector = KappaTuneSelector(
        model,
        max_dim_size_to_analyze=max_dim_size_to_analyze,
        moe_param_suffixes=moe_param_suffixes,
        show_progress=show_progress,
    )

    target_modules = selector.get_best_targets(top_p=top_p)
    target_parameters = selector.get_best_target_parameters(top_p=top_p)

    # Return None when there are no MoE layers (PEFT convention)
    if not target_parameters:
        target_parameters = None

    return {
        "target_modules": target_modules,
        "target_parameters": target_parameters,
    }


class _SVDLinear(nn.Linear):
    """Low-rank approximation of an `nn.Linear` layer using (optionally activation-aware) SVD.

    Replaces `W` (shape `[out_features, in_features]`) with two sequential linear layers `v` and `u`
    such that `u(v(x)) ≈ W x`. When a *scaling* matrix `S` (derived from input activations) is provided,
    the SVD is performed on `W @ S`; the resulting `V` factor is then mapped back to the original input
    space by `S^{-1}`, following the *activation-aware SVD* approach of EMLoC (Lin et al., NeurIPS 2025).

    Inherits from `nn.Linear` so that PEFT methods (e.g. LoRA) can be applied to the emulator via
    `isinstance` checks. The inherited `weight` and `bias` parameters are replaced by the factored
    `v` and `u` sub-layers; `forward` uses only `v` and `u`, not the inherited parameters.

    Note:
        The `weight` property returns the materialized effective weight `u.weight @ v.weight`. This
        allows PEFT merge operations to read the correct weight, but setting `weight.data` (as merge
        does) is a no-op — the factored form cannot be updated in-place. For LoRA training and
        inference, this is not an issue. For merge, the adapter delta is not folded into the factors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device="meta", dtype=dtype)
        # Remove the inherited weight and bias parameters — replaced by v, u sub-layers
        del self._parameters["weight"]
        if bias and "bias" in self._parameters:
            del self._parameters["bias"]
        self.rank = rank
        # Use meta device during init to avoid allocating memory for weights that are immediately
        # overwritten after construction. The caller is responsible for assigning real weights.
        meta_kwargs = {"device": "meta", "dtype": dtype}
        self.v = nn.Linear(in_features, rank, bias=False, **meta_kwargs)
        self.u = nn.Linear(rank, out_features, bias=bias, **meta_kwargs)

    @property
    def weight(self) -> torch.Tensor:
        """Materialized effective weight (u.weight @ v.weight). Read-only — use v and u to modify."""
        return self.u.weight @ self.v.weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        # No-op: the weight is factored into u and v. Setting it directly is not supported.
        pass

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.u.bias

    @bias.setter
    def bias(self, value: Optional[torch.Tensor]) -> None:
        self.u.bias = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u(self.v(x))


def _get_input_stats(inputs: list[torch.Tensor]) -> torch.Tensor:
    """Compute the input covariance/scaling matrix `X^T X` from collected input activations."""
    xtx = torch.zeros(inputs[0].shape[-1], inputs[0].shape[-1], dtype=torch.float32)
    for x in inputs:
        x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
        xtx += x.t() @ x
    return xtx


def _get_scaling(xtx: torch.Tensor) -> torch.Tensor:
    """Compute the activation-aware scaling matrix `S = Q * sqrt(L)` via eigendecomposition of `X^T X`."""
    factor = torch.trace(xtx) / xtx.shape[0]
    eps = 1e-7
    eigvals = torch.zeros(xtx.shape[0])
    eigvecs = torch.zeros_like(xtx)
    for _retry in range(5):
        try:
            eigvals, eigvecs = torch.linalg.eigh(xtx / factor)
            eigvals = torch.clamp(eigvals, min=1e-7) * factor
            break
        except Exception:
            if _retry == 4:
                raise
            xtx = xtx + torch.eye(xtx.shape[0]) * eps
            eps *= 5
    scaling = eigvecs * torch.sqrt(eigvals)
    return scaling


def _apply_activation_aware_svd(
    weight: torch.Tensor,
    rank: Union[int, float],
    scaling: Optional[torch.Tensor] = None,
    fast_svd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Perform (optionally activation-aware) truncated SVD on a weight matrix.

    Args:
        weight: The weight matrix of shape `[out_features, in_features]`.
        rank: Either an `int` for a fixed rank, or a `float` in `(0, 1]` interpreted as an energy
            threshold (the smallest `k` such that the top-`k` singular values account for at least that
            fraction of total squared singular values).
        scaling: Optional scaling matrix `S` of shape `[in_features, in_features]`. When provided, SVD
            is performed on `weight @ S` and the `V` factor is mapped back with `S^{-1}`.
        fast_svd: When `True` and `rank` is an `int`, use `torch.svd_lowrank` (a randomized SVD) instead
            of the full `torch.linalg.svd`. This is much faster for large matrices with small target
            ranks, at the cost of a slightly less accurate approximation. Ignored when `rank` is a
            `float` (the full singular value spectrum is needed to determine the cutoff).

    Returns:
        A tuple `(lora_A, lora_B, effective_rank)` where `lora_A` has shape `[rank, in_features]`
        and `lora_B` has shape `[out_features, rank]`, so that `lora_B @ lora_A` approximates the
        (scaled) weight.
    """
    weight = weight.to(dtype=torch.float32)

    if scaling is not None:
        scaling = scaling.to(dtype=torch.float32, device=weight.device)
        scaled_weight = weight @ scaling
    else:
        scaled_weight = weight

    use_fast_svd = fast_svd and isinstance(rank, int)

    if use_fast_svd:
        # torch.svd_lowrank expects [n, m] and returns U [n, q], S [q], V [m, q]
        effective_rank = int(rank)
        max_rank = min(scaled_weight.shape)
        if effective_rank > max_rank:
            raise ValueError(
                f"The chosen rank {effective_rank} is larger than the weight shape ({max_rank}), "
                "please choose a lower rank."
            )
        if effective_rank == 0:
            effective_rank = 1

        u_truncated, s_truncated, v_truncated = torch.svd_lowrank(scaled_weight, q=effective_rank)
        # v_truncated is [in_features, rank], need Vh = v_truncated.t() [rank, in_features]
        vh_truncated = v_truncated.t()
    else:
        u, s, vh = torch.linalg.svd(scaled_weight, full_matrices=False)

        if isinstance(rank, float):
            if not (0 < rank <= 1):
                raise ValueError(f"Float rank must be in (0, 1], got {rank}.")
            effective_rank = _find_cutoff_index(s, threshold=rank)
        else:
            effective_rank = int(rank)

        max_rank = u.shape[1]
        if effective_rank > max_rank:
            raise ValueError(
                f"The chosen rank {effective_rank} is larger than the weight shape ({max_rank}), "
                "please choose a lower rank."
            )
        if effective_rank == 0:
            effective_rank = 1  # at least one component

        s_truncated = s[:effective_rank]
        u_truncated = u[:, :effective_rank]
        vh_truncated = vh[:effective_rank, :]

    sqrt_sigma = torch.sqrt(torch.diag(s_truncated))

    if scaling is not None:
        scaling_inv = torch.linalg.inv(scaling)
        vh_truncated = vh_truncated @ scaling_inv

    lora_b = u_truncated @ sqrt_sigma  # [out_features, rank]
    lora_a = sqrt_sigma @ vh_truncated  # [rank, in_features]
    return lora_a, lora_b, effective_rank


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a child module by name."""
    setattr(parent, name, new_module)


def _get_parent_and_child(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    """Given a dotted module name, return the parent module and the child attribute name."""
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _find_tied_linear_modules(model: nn.Module) -> set[str]:
    """Return the set of `nn.Linear` module names whose weight parameter is shared with another module.

    This detects weight-tying (e.g. `lm_head` sharing weights with `embed_tokens`) by comparing data pointers
    of all parameters in the model. When a linear layer's weight is tied to a non-linear module (such as an
    embedding), replacing the linear would silently break the tying.
    """
    seen_ptrs: dict[int, str] = {}
    tied_names: set[str] = set()
    for param_name, param in model.named_parameters(remove_duplicate=False):
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            # This parameter shares storage with another — check if it belongs to an nn.Linear
            # We look up the module that owns this parameter by matching the parameter name prefix
            # to module names.
            module_name = param_name.rpartition(".")[0]
            if module_name:
                module = model.get_submodule(module_name)
                if isinstance(module, nn.Linear):
                    tied_names.add(module_name)
        else:
            seen_ptrs[ptr] = param_name
    return tied_names


@torch.no_grad()
def get_emulator_model(
    model: nn.Module,
    rank: Union[int, float],
    data_loader: Optional[Iterable] = None,
    target_modules: Optional[list[str]] = None,
    ignore_modules: Optional[list[str]] = None,
    num_samples: int = 64,
    inplace: bool = True,
    progressbar: bool = False,
    fast_svd: bool = False,
) -> nn.Module:
    """Construct a lightweight *emulator* of `model` by replacing `nn.Linear` layers with low-rank SVD factorizations.

    This implements the emulator construction approach from the EMLoC paper (Lin et al., NeurIPS 2025). Each
    `nn.Linear` layer is replaced by two sequential linear layers `v` (in→rank) and `u` (rank→out) whose
    weights are derived from a truncated SVD of the original weight. When a calibration `data_loader` is
    provided, the SVD is made *activation-aware*: a scaling matrix is computed from input activations and the
    SVD is performed on the rescaled weight, which preserves the directions most relevant to the task data.

    The `rank` argument controls the compression:

    - `int`: use a fixed rank `k` for every layer; the top-`k` singular values are retained.
    - `float`: interpreted as an *energy threshold* in `(0, 1]`; for each layer, the smallest `k` is chosen
      such that the top-`k` singular values account for at least that fraction of the total squared singular
      values. This can result in different ranks per layer, similar to the logic in `peft.tuners.lora.conversion`.

    The emulator approximates the original model's outputs — higher ranks yield closer approximations, at the
    cost of more parameters.

    Note:
        This function is inspired by EMLoC but is a generalized, standalone implementation that works with
        arbitrary `nn.Module` instances. It does not require PEFT layers.

        Layers with tied weights (e.g. `lm_head` sharing weights with `embed_tokens` when
        `tie_word_embeddings=True`) are automatically skipped to avoid silently breaking the tying.
        A warning is emitted listing the skipped layers.

    Args:
        model (`nn.Module`):
            The model to compress. Should contain `nn.Linear` layers.
        rank (`int` or `float`):
            The desired rank for the SVD factorization. An `int` uses a fixed rank for all layers. A
            `float` in `(0, 1]` is interpreted as an energy threshold: for each layer, the smallest rank
            `k` is chosen such that the top `k` singular values account for at least that fraction of the
            total squared singular values. Higher values (closer to 1.0) result in a better approximation
            but more parameters.
        data_loader (`Iterable`, *optional*):
            An iterator over calibration batches. Each batch should be a dict (or tuple) that can be passed
            to `model(**batch)` or `model(*batch)`. When provided, the SVD is activation-aware. When
            `None`, a plain SVD on the weight matrix is used (no activation information).
        target_modules (`list[str]`, *optional*):
            List of module name patterns (regex) to compress. If `None`, all `nn.Linear` layers are
            compressed. Module names matching these patterns are included.
        ignore_modules (`list[str]`, *optional*):
            List of module name patterns (regex) to exclude from compression. Takes precedence over
            `target_modules`.
        num_samples (`int`):
            Maximum number of calibration samples to use for activation collection. Only relevant when
            `data_loader` is provided. Defaults to 64.
        inplace (`bool`):
            If `True`, modify the model in-place. If `False`, work on a deep copy. Defaults to `True`.
        progressbar (`bool`):
            Whether to show a progress bar during compression. Defaults to `False`.
        fast_svd (`bool`):
            When `True` and `rank` is an `int`, use `torch.svd_lowrank` (a randomized SVD) instead of
            the full `torch.linalg.svd`. This is significantly faster for large matrices with small
            target ranks, at the cost of a slightly less accurate approximation. Ignored when `rank`
            is a `float`, since the full singular value spectrum is needed to determine the cutoff.
            Defaults to `False`.

    Returns:
        `nn.Module`: The emulator model with `nn.Linear` layers replaced by low-rank factorizations.

    Raises:
        `ValueError`: If the rank is invalid (0, or a float outside `(0, 1]`).
        `TypeError`: If no `nn.Linear` layers are found in the model.

    Example:

    ```python
    >>> from peft import get_emulator_model
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> # Using a fixed rank
    >>> emulator = get_emulator_model(model, rank=4)
    >>> # Using an energy threshold (activation-aware, with calibration data)
    >>> emulator = get_emulator_model(model, rank=0.95, data_loader=loader)
    ```
    """
    # --- validation ---
    if rank == 0:
        raise ValueError("Passing a rank of 0 doesn't make sense, please pass a valid value.")
    if isinstance(rank, float) and not (0 < rank <= 1):
        raise ValueError(f"If rank is a float, it is interpreted as a threshold. It must be between 0 and 1 but got {rank}.")
    if not inplace:
        model = deepcopy(model)
    model.eval()

    # --- identify linear layers to compress ---
    tied_modules = _find_tied_linear_modules(model)
    if tied_modules:
        warnings.warn(
            "The following nn.Linear layers have tied weights and will be skipped to avoid "
            f"breaking weight-tying: {sorted(tied_modules)}. If you want to compress these "
            "layers, untie the weights first (e.g. by loading with `tie_word_embeddings=False`)."
        )

    linear_modules: dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in tied_modules:
            continue
        if ignore_modules and any(re.fullmatch(pattern, name) for pattern in ignore_modules):
            continue
        if target_modules and not any(re.fullmatch(pattern, name) for pattern in target_modules):
            continue
        linear_modules[name] = module

    if not linear_modules:
        raise TypeError("Could not detect any nn.Linear layer to compress.")

    # --- collect input activations (if data_loader provided) ---
    activation_stats: dict[str, torch.Tensor] = {}
    if data_loader is not None:
        activation_stats = _collect_activations(model, linear_modules, data_loader, num_samples, progressbar)

    # --- replace each linear with SVD factorization ---
    iterator = tqdm(linear_modules.items(), desc="Building emulator", disable=not progressbar)
    for name, linear in iterator:
        weight = linear.weight.data
        has_bias = linear.bias is not None
        dtype = weight.dtype
        device = weight.device

        scaling = activation_stats.get(name, None)

        lora_a, lora_b, effective_rank = _apply_activation_aware_svd(
            weight, rank=rank, scaling=scaling, fast_svd=fast_svd
        )

        # Construct on meta device, then materialize on the target device
        svd_layer = _SVDLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=effective_rank,
            bias=has_bias,
            dtype=dtype,
            device=device,
        )
        svd_layer = svd_layer.to_empty(device=device)
        svd_layer.v.weight.data = lora_a.to(dtype=dtype, device=device).contiguous()
        svd_layer.u.weight.data = lora_b.to(dtype=dtype, device=device).contiguous()
        if has_bias:
            svd_layer.u.bias.data = linear.bias.data.to(device=device)

        parent, child_name = _get_parent_and_child(model, name)
        _replace_module(parent, child_name, svd_layer)

    return model


def _collect_activations(
    model: nn.Module,
    linear_modules: dict[str, nn.Linear],
    data_loader: Iterable,
    num_samples: int,
    progressbar: bool,
) -> dict[str, torch.Tensor]:
    """Collect input activations for each target `nn.Linear` module via forward hooks."""
    inputs_collected: dict[str, list[torch.Tensor]] = {name: [] for name in linear_modules}
    sample_count = 0

    handles = []
    for name, module in linear_modules.items():
        target_module = module

        def make_hook(module_name):
            def hook(mod, inp, out):
                inputs_collected[module_name].append(inp[0].detach().cpu())

            return hook

        handle = target_module.register_forward_hook(make_hook(name))
        handles.append(handle)

    try:
        with torch.no_grad():
            loader_iter = iter(data_loader)
            for batch in tqdm(loader_iter, desc="Collecting activations", total=num_samples, disable=not progressbar):
                if sample_count >= num_samples:
                    break
                if isinstance(batch, dict):
                    model(**batch)
                elif isinstance(batch, (tuple, list)):
                    model(*batch)
                else:
                    model(batch)
                sample_count += 1
    finally:
        for handle in handles:
            handle.remove()

    # compute scaling matrices
    scaling_matrices: dict[str, torch.Tensor] = {}
    for name, inputs in inputs_collected.items():
        if not inputs:
            continue
        xtx = _get_input_stats(inputs)
        scaling_matrices[name] = _get_scaling(xtx)

    return scaling_matrices
