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

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class ActivationSteeringConfig(PeftConfig):
    """Configuration for additive activation steering of individual neurons.

    A neuron is an input channel of a targeted linear projection, typically an MLP's down projection. Steering adds a
    signed, trainable dose to the selected channels before the projection, which edits the neurons' activations
    directly. The dose normalization follows Lyra's neuron-explorer implementation:
    https://github.com/lyramakesmusic/neuron-explorer/blob/5a8f8267f6bd3c8caa87d598870949ab0d25b914/explorer/runtime.py#L653
    A dose of 1 on neuron `n` adds `median * W[:, n] / ||W[:, n]||` to the projection's output, where `W` is the
    projection's weight and `median` is the median L2 norm of the stream the addition lands in, measured on baseline
    calibration data. Call `calibrate_activation_steering` from `peft.tuners.activation_steering` before using a new
    adapter. The calibrated scales are saved with the adapter and are not trainable; the doses can be fine-tuned.

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The projections whose input channels are steered, e.g. `["down_proj"]`. When passing a string, a regex
            match is performed. When passing a list of strings, either an exact match is performed or it is checked if
            the name of the module ends with any of the passed strings. If not specified, the keys of
            `steering_pattern` are used.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter to. When passing a string, a regex match is performed.
            When passing a list of strings, either an exact match is performed or it is checked if the name of the
            module ends with any of the passed strings.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, the adapter is applied to the layer indices in
            this list. If a single integer is passed, the adapter is applied to this layer index only.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The name of the `nn.ModuleList` that holds the layers, usually `"layers"` or `"h"`, if it is not one of the
            common names. It is used by `layers_to_transform` and to find the layer whose output sets the unit for
            `dose_unit="layer_output"`.
        steering_pattern (`dict`):
            Maps projection names (exact or suffix matches) to specifications containing `indices` and `init_val`. The
            `indices` select the projection's input channels, and `init_val` initializes their independent, signed
            doses; a number broadcasts, and a list must match the number of indices. Targeted projections without an
            entry are not steered. If empty, every input channel of every targeted projection gets a dose of 0, which
            leaves the model unchanged. `select_neurons` from `peft.tuners.activation_steering` then keeps the most
            useful neurons and writes them into this field.
        dose_unit (`str`):
            Where calibration measures the median L2 norm that sets the unit of the doses. `"layer_output"` uses the
            output of the numbered layer that contains the projection, i.e. the residual stream. `"module_output"` uses
            the projection's own output; use it for models that normalize the MLP output before adding it to the
            residual stream (e.g. Gemma 2/3 and OLMo 2), or for models without numbered layers. Calibration pools all
            non-padding tokens of the calibration batches, excluding the first valid token of each sequence.

    Inputs to the projections must have the shape `(batch, sequence, features)`, or `(batch * sequence, features)` for
    models that flatten them, such as OPT. Model inputs after the primary input, including `attention_mask` and
    `past_key_values`, must be passed by keyword. Padding positions are not steered.

    Example:

    ```py
    >>> from peft import ActivationSteeringConfig, TaskType

    >>> config = ActivationSteeringConfig(
    ...     base_model_name_or_path="Qwen/Qwen3.5-4B",
    ...     task_type=TaskType.CAUSAL_LM,
    ...     steering_pattern={"layers.15.mlp.down_proj": {"indices": [3920], "init_val": 0.5}},
    ... )
    ```
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the projections whose input channels are steered, e.g. "
                "['down_proj']. If not specified, the keys of `steering_pattern` are used."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from steering."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indices to transform. If a list of ints is passed, the adapter is applied to the layer "
                "indices in this list. If a single integer is passed, the adapter is applied to this layer index only."
            ),
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The name of the `nn.ModuleList` that holds the layers, usually 'layers' or 'h', if it is not one of "
                "the common names. It is used by `layers_to_transform` and to find the layer whose output sets the "
                "unit for `dose_unit='layer_output'`."
            ),
        },
    )
    steering_pattern: dict[str, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Maps projection names (exact or suffix matches) to specifications containing `indices` and "
                "`init_val`. The `indices` select the projection's input channels, and `init_val` initializes their "
                "independent, signed doses; a number broadcasts, and a list must match the number of indices. "
                "Targeted projections without an entry are not steered. If empty, every input channel of every "
                "targeted projection gets a dose of 0, which leaves the model unchanged. `select_neurons` from "
                "`peft.tuners.activation_steering` then keeps the most useful neurons and writes them into this field."
            ),
        },
    )
    dose_unit: Literal["layer_output", "module_output"] = field(
        default="layer_output",
        metadata={
            "help": (
                "Where calibration measures the median L2 norm that sets the unit of the doses. 'layer_output' uses "
                "the output of the numbered layer that contains the projection, i.e. the residual stream. "
                "'module_output' uses the projection's own output; use it for models that normalize the MLP output "
                "before adding it to the residual stream (e.g. Gemma 2/3 and OLMo 2), or for models without numbered "
                "layers."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ACTIVATION_STEERING
        if self.dose_unit not in ("layer_output", "module_output"):
            raise ValueError(f"`dose_unit` must be 'layer_output' or 'module_output', got {self.dose_unit!r}.")
        if not isinstance(self.steering_pattern, dict):
            raise TypeError("`steering_pattern` must be a dictionary.")

        for name, spec in self.steering_pattern.items():
            if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
                raise ValueError("`steering_pattern` keys must be non-empty module paths.")
            if not isinstance(spec, dict):
                raise TypeError(f"The steering specification for '{name}' must be a dictionary.")

            required = {"indices", "init_val"}
            if required - spec.keys():
                raise ValueError(f"Missing steering fields for '{name}': {sorted(required - spec.keys())}.")
            if spec.keys() - required:
                raise ValueError(f"Unknown steering fields for '{name}': {sorted(spec.keys() - required)}.")

            indices = spec["indices"]
            if (
                not isinstance(indices, list)
                or not indices
                or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices)
            ):
                raise ValueError(f"`indices` for '{name}' must be a non-empty list of distinct non-negative integers.")

            values = spec["init_val"]
            values = values if isinstance(values, list) else [values] * len(indices)
            if len(values) != len(indices):
                raise ValueError(f"`init_val` for '{name}' must have the same length as `indices`.")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"`init_val` for '{name}' must contain finite numbers.")

        if self.target_modules is None:
            if not self.steering_pattern:
                raise ValueError(
                    "Specify `target_modules`, the projections whose input channels are steered, or `steering_pattern`."
                )
            self.target_modules = list(self.steering_pattern)
