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

import copy
import os
import pathlib
import warnings

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import SAFETENSORS_WEIGHTS_NAME

from .config import LoraConfig


def _find_cutoff_index(S: torch.Tensor, threshold: float) -> int:
    # assumes that the singular values are sorted
    if S.dim() != 1:
        raise ValueError("Input vector must be 1d.")

    energy = S**2
    cs = torch.cumsum(energy, dim=0)
    total = cs[-1]
    k = (cs / total > threshold).nonzero()[0].item()
    # k is the index of the first item that surpasses the threshold; since we want to include it, add + 1
    return k + 1


@torch.no_grad()
def _convert_module_to_lora(
    module: BaseTunerLayer, rank: int | float, adapter_name: str = "default"
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert a single BaseTunerLayer's adapter weight to a LoRA weight, return A, B, and the effective rank."""
    if not hasattr(module, "get_delta_weight"):
        # if we arrive here, it means that the layer actually does not support LoRA conversion, which should not happen
        raise TypeError(
            f"Module of type {type(module)} does not have a get_delta_weight method, which is required for conversion. "
            "Please open an issue: https://github.com/huggingface/peft/issues"
        )

    delta_weight = module.get_delta_weight(adapter_name)
    U, S, V = torch.linalg.svd(delta_weight, full_matrices=False)
    if isinstance(rank, int):
        effective_rank = rank
    else:
        # float => interpret as threshold
        effective_rank = _find_cutoff_index(S, threshold=rank)

    if effective_rank > U.shape[1]:
        raise ValueError(
            f"The chosen rank {effective_rank} is larger then the weight shape ({U.shape[1]}), please choose a lower "
            "rank."
        )

    lora_B = U[:, :effective_rank] * S[:effective_rank]
    lora_A = V[:effective_rank]
    return lora_A.contiguous(), lora_B.contiguous(), effective_rank


def convert_to_lora(
    model: torch.nn.Module, rank: int | float, adapter_name: str = "default", progressbar: bool = False
) -> tuple[LoraConfig, dict[str, torch.Tensor]]:
    """
    Convert a non-LoRA model with PEFT layers to a LoRA checkpoint.

    This is only supported for some specific PEFT methods that allow an equivalent conversion. Essentially, this comes
    down to PEFT methods that work by updating the base weight with a delta weight. Also, right now, only linear layers
    are supported.

    The LoRA adapter will try to approximate the initial adapter as close as possible. The higher the rank, the better
    the approximation. It is expected that the approximation will never reach the full performance of the original
    adapter, and that the parameter efficiency of the LoRA adapter will be less than that of the original adapter (i.e.
    for a similar performance, it will require more parameters). The conversion can still be useful in many situations:

    - In PEFT, LoRA supports more features than most other methods, e.g. mixed adapter batches. Thus the converted
      adapter can be used with those features.
    - Some downstream packages support LoRA adapters, but not other PEFT methods, e.g. Diffusers. The conversion allows
      to use a non-LoRA adapter with those packages.

    The LoRA scaling factor is already baked into the LoRA weights, thus the scaling will always be one (i.e. rank and
    alpha are chosen to be identical).

    Args:
        model:
            The model to be converted. Should be a model that has PEFT layers that support conversion.
        rank (`int` or `float`):
            The desired rank for the returned LoRA adapter. A higher rank results in a LoRA adapter that more
            accurately mirrors the original adapter. It will, however, also require more memory, compute, and disk
            space. Therefore, choose a value that represents the best trade off for your use case and validate the
            final adapter. If a float is passed, it will be interpreted as the desired minimum threshold for underlying
            singular values to be included. So e.g. a threshold of 0.5 means that the top K singular values are
            included so that they explained variance is greater or equal to that threshold. This effectively results in
            lower ranks being assigned if a few singular can capture the adaptation of this layer. A lower float means
            the rank is lower and vice versa. Be aware that dynamic ranks can lead to very unequal ranks per layer,
            which means that some layers may require a disproportionally high amount of memory for activations.
            Choosing a fixed (int) rank is better to achieve predictable memory requirement.
        adapter_name (`str`, *optional*):
            The name of the adapter to be converted. Can only convert a single adapter at a time. Defaults to
            `"default"`.
        progressbar (`bool`):
            whether to show a progressbar indicating the progress of the conversion (it can take a few minutes for big
            models).

    Returns:
        lora_config (`LoraConfig`)
            The `LoraConfig` that corresponds to the converted LoRA adapter.
        state_dict (`dict[str, torch.Tensor]`)
            The `state_dict` containing the LoRA weights.

    Raises
        TypeError:
            If the provided model does not have any layers that can be converted to LoRA, a `TypeError` is raised.
        ValueError:
            If a dynamic threshold was chosen that's too high, so that no layer can be converted, raise a `ValueError`.
    """
    from peft import PeftType  # local to avoid circular import

    ##########
    # CHECKS #
    ##########

    if isinstance(rank, float) and not (0 < rank <= 1):
        raise ValueError(
            f"If rank is a float, it is interpreted as a threshold. It must be between 0 and 1 but got {rank}."
        )

    # check if LoRA conversion is supported at all
    modules_not_supporting_lora = []
    num_modules_with_support = 0
    num_modules_total = 0
    for module in model.modules():
        num_modules_total += 1
        if not isinstance(module, BaseTunerLayer):
            continue

        if module.supports_lora_conversion(adapter_name):
            num_modules_with_support += 1
        else:
            modules_not_supporting_lora.append(module)

    unsupported = {repr(type(module)) for module in modules_not_supporting_lora}
    if unsupported:
        raise TypeError(f"Some module types on this model do not support LoRA conversion: {', '.join(unsupported)}.")

    if num_modules_with_support == 0:
        raise TypeError("Could not detect any layer that supports LoRA conversion.")

    peft_config = getattr(model, "peft_config", {}).get(adapter_name)
    if (peft_config is not None) and (peft_config.peft_type == PeftType.LORA):
        warnings.warn(
            "Converting a PEFT adapter to LoRA that is already a LoRA adapter. There is typically no need for that."
        )

    ###############
    # PREPARATION #
    ###############

    peft_prefix = "base_model.model."

    if peft_config is not None:
        # use the model's PEFT config, if it exists, to initialize the new LoraConfig
        peft_config = model.peft_config[adapter_name]
        config_kwargs = {
            "target_modules": copy.copy(peft_config.target_modules),
            "rank_pattern": {},
            "alpha_pattern": {},
            "exclude_modules": set(),
            "base_model_name_or_path": peft_config.base_model_name_or_path,
        }
        if hasattr(peft_config, "layers_patter"):
            # those two go hand in hand
            config_kwargs["layers_pattern"] = peft_config.layers_pattern
            config_kwargs["layers_to_transform"] = peft_config.layers_to_transform
        if isinstance(rank, int):
            # hard-coded rank
            lora_config = LoraConfig(r=rank, lora_alpha=rank, **config_kwargs)
        else:
            # r and lora_alpha shouldn't matter, as the rank will be determined by rank/alpha pattern
            lora_config = LoraConfig(r=1, lora_alpha=1, **config_kwargs)
    else:
        # create a new LoraConfig from scratch, inferring the target modules from the model
        lora_config = LoraConfig(
            r=rank if isinstance(rank, int) else 1,  # 1 is a dummy value, actual values will come from rank_pattern
            target_modules=[],
            rank_pattern={},
            alpha_pattern={},
            exclude_modules=set(),
        )

    ##############
    # CONVERSION #
    ##############

    state_dict = {}
    for name, module in tqdm(
        model.named_modules(), disable=not progressbar, desc="Converting to LoRA", total=num_modules_total
    ):
        if not isinstance(module, BaseTunerLayer):
            continue

        lora_A, lora_B, effective_rank = _convert_module_to_lora(module, rank=rank, adapter_name=adapter_name)
        lora_config.target_modules.add(name)

        if effective_rank == 0:
            # This shouldn't really happen, as we ensure that the rank is greater than 0 (int) or, for tresholds
            # (float), at least one SV is included. But better be safe than sorry, as, in principle, it is fine to
            # exclude some layers.
            lora_config.exclude_modules.add(name.removeprefix(peft_prefix))
            continue

        # the rank was dynamically adjusted, store it in rank and alpha pattern
        if effective_rank != rank:
            lora_config.rank_pattern[name.removeprefix(peft_prefix)] = effective_rank
            lora_config.alpha_pattern[name.removeprefix(peft_prefix)] = effective_rank

        # don't include adapter_name in key
        state_dict[f"{name}.lora_A.weight"] = lora_A
        state_dict[f"{name}.lora_B.weight"] = lora_B

    if not state_dict:
        # no layer was converted, which should not happen
        raise ValueError(
            "Did not convert a single layer, this means that something went wrong. Please open an issue: "
            "Please open an issue: https://github.com/huggingface/peft/issues"
        )

    return lora_config, state_dict


def save_as_lora(
    path: str | os.PathLike,
    model: torch.nn.Module,
    rank: int | float,
    adapter_name: str = "default",
    peft_config=None,
    progressbar: bool = False,
) -> None:
    """
    Convert a non-LoRA model with PEFT layers to a LoRA, then save the checkpoint file and PEFT config.

    This is only supported for some specific PEFT methods that allow an equivalent conversion. Essentially, this comes
    down to PEFT methods that work by updating the base weight with a delta weight. Also, right now, only linear layers
    are supported.

    The LoRA adapter will try to approximate the initial adapter as close as possible. The higher the rank, the better
    the approximation. It is expected that the approximation will never reach the full performance of the original
    adapter, and that the parameter efficiency of the LoRA adapter will be less than that of the original adapter (i.e.
    for a similar performance, it will require more parameters). The conversion can still be useful in many situations:

    - In PEFT, LoRA supports more features than most other methods, e.g. mixed adapter batches. Thus the converted
      adapter can be used with those features.
    - Some downstream packages support LoRA adapters, but not other PEFT methods, e.g. Diffusers. The conversion allows
      to use a non-LoRA adapter with those packages.

    The LoRA scaling factor is already baked into the LoRA weights, thus the scaling will always be one (i.e. rank and
    alpha are chosen to be identical).

    You can load the converted LoRA weight like this:

    ```py
    >>> lora_path = ...
    >>> save_as_lora(lora_path, model, rank=...)
    >>> base_model = AutoModel.from_pretrained(...)
    >>> lora_model = PeftModel.from_pretrained(base_model, lora_path)
    ```

    Args:
        model:
            The model to be converted. Should be a model that has PEFT layers that support conversion.
        rank (`int` or `float`):
            The desired rank for the returned LoRA adapter. A higher rank results in a LoRA adapter that more
            accurately mirrors the original adapter. It will, however, also require more memory, compute, and disk
            space. Therefore, choose a value that represents the best trade off for your use case and validate the
            final adapter. If a float is passed, it will be interpreted as the desired minimum threshold for underlying
            singular values to be included. So e.g. a threshold of 0.5 means that the top K singular values are
            included so that they explained variance is greater or equal to that threshold. This effectively results in
            lower ranks being assigned if a few singular can capture the adaptation of this layer. A lower float means
            the rank is lower and vice versa. Be aware that dynamic ranks can lead to very unequal ranks per layer,
            which means that some layers may require a disproportionally high amount of memory for activations.
            Choosing a fixed (int) rank is better to achieve predictable memory requirement.
        adapter_name (`str`, *optional*):
            The name of the adapter to be converted. Can only convert a single adapter at a time. Defaults to
            `"default"`.
        progressbar (`bool`):
            whether to show a progressbar indicating the progress of the conversion (it can take a few minutes for big
            models).

    Returns:
        lora_config (`LoraConfig`)
            The `LoraConfig` that corresponds to the converted LoRA adapter.
        state_dict (`dict[str, torch.Tensor]`)
            The `state_dict` containing the LoRA weights.

    Raises
        TypeError:
            If the provided model does not have any layers that can be converted to LoRA, a `TypeError` is raised.
        ValueError:
            If a dynamic threshold was chosen that's too high, so that no layer can be converted, raise a `ValueError`.
    """
    path = pathlib.Path(path)
    if not path.exists():
        os.makedirs(path)

    lora_config, state_dict = convert_to_lora(model, rank=rank, adapter_name=adapter_name, progressbar=progressbar)
    save_file(state_dict, path / SAFETENSORS_WEIGHTS_NAME)
    lora_config.save_pretrained(str(path))
