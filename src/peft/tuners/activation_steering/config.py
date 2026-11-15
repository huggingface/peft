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
    signed, trainable dose, multiplied by a fixed scale, to each selected channel before the projection, which edits
    the neurons' activations directly. The scale of neuron `n` is `median / ||W[:, n]||`, where `W` is the projection's
    weight and `median` is the median L2 norm of the stream given by `dose_unit`, measured on baseline calibration
    data. A dose of 1 thus adds `median * W[:, n] / ||W[:, n]||` to the projection's output. This normalization follows
    Lyra's neuron-explorer implementation:
    https://github.com/lyramakesmusic/neuron-explorer/blob/5a8f8267f6bd3c8caa87d598870949ab0d25b914/explorer/runtime.py#L653
    Call `calibrate_activation_steering` from `peft.tuners.activation_steering` to compute the scales before using a
    new adapter. The scales are saved with the adapter and are not trainable; the doses can be fine-tuned.

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The projections whose input channels are steered, e.g. `["down_proj"]`. When passing a string, a regex
            match is performed. When passing a list of strings, either an exact match is performed or it is checked if
            the name of the module ends with any of the passed strings. Must not be set together with
            `steering_pattern`, whose keys are then the targets.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter to. When passing a string, a regex match is performed.
            When passing a list of strings, either an exact match is performed or it is checked if the name of the
            module ends with any of the passed strings. Must not be set together with `steering_pattern`.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, the adapter is applied to the layer indices in
            this list. If a single integer is passed, the adapter is applied to this layer index only. Must not be set
            together with `steering_pattern`.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The name of the `nn.ModuleList` that holds the layers, usually `"layers"` or `"h"`, if it is not one of the
            common names. It is used by `layers_to_transform` and to find the layer whose output sets the unit for
            `dose_unit="residual_stream"`.
        steering_pattern (`dict`):
            Maps projection names (exact or suffix matches) to specifications containing `indices` and `init_val`.
            These projections are the targets. The `indices` select the projection's input channels, and `init_val`
            initializes their independent, signed doses; a number broadcasts, and a list must match the number of
            indices. If empty, `target_modules` selects the projections, and every input channel gets a dose of 0,
            which leaves the model unchanged. `select_neurons` from `peft.tuners.activation_steering` then keeps the
            most useful neurons and writes them into this field.
        dose_unit (`Literal["projection_output", "residual_stream"]`):
            Where calibration measures the median L2 norm that sets the unit of the doses. `"projection_output"` uses
            the targeted projection's own output. `"residual_stream"` uses the output of the numbered layer that
            contains the projection, which is the unit of neuron-explorer: on models that add the MLP output directly
            to the residual stream, e.g. Llama and Qwen, doses found there keep their meaning, as in the presets of the
            activation steering example. The ratio of the two medians differs between layers, so such doses cannot be
            converted with a single factor. Do not use `"residual_stream"` for models that normalize the MLP output
            before adding it to the residual stream, e.g. Gemma 2/3 and OLMo 2. Calibration pools all non-padding
            tokens of the calibration batches, excluding the first valid token of each sequence.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from activation steering layers to be set as trainable and saved in the final
            checkpoint.

    Inputs to the projections must have the shape `(batch, sequence, features)`, or `(batch * sequence, features)` for
    models that flatten them, such as OPT. Model inputs after the primary input, including `attention_mask` and `cache`
    (previously called `past_key_values`), must be passed by keyword. Padding positions are not steered.
    Encoder-decoder models are not supported. With gradient checkpointing, run the backward pass of a steered forward
    pass before the next steered forward pass, since the recomputation uses the attention mask of the most recent model
    call.

    Example:

    Without pre-selected neurons, every input channel of the targeted projections starts at a dose of 0, and
    `select_neurons` keeps the most useful neurons after calibration:

    ```py
    >>> from peft import ActivationSteeringConfig, TaskType

    >>> config = ActivationSteeringConfig(task_type=TaskType.CAUSAL_LM, target_modules=["down_proj"])
    ```

    With pre-selected neurons, only these neurons are steered. Doses found with neuron-explorer are in units of the
    residual stream:

    ```py
    >>> config = ActivationSteeringConfig(
    ...     base_model_name_or_path="Qwen/Qwen3.5-4B",
    ...     task_type=TaskType.CAUSAL_LM,
    ...     steering_pattern={"layers.15.mlp.down_proj": {"indices": [3920], "init_val": 0.5}},
    ...     dose_unit="residual_stream",
    ... )
    ```
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the projections whose input channels are steered, e.g. "
                "['down_proj']. Must not be set together with `steering_pattern`, whose keys are then the targets."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to exclude from steering. Must not be "
                "set together with `steering_pattern`."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indices to transform. If a list of ints is passed, the adapter is applied to the layer "
                "indices in this list. If a single integer is passed, the adapter is applied to this layer index "
                "only. Must not be set together with `steering_pattern`."
            ),
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The name of the `nn.ModuleList` that holds the layers, usually 'layers' or 'h', if it is not one of "
                "the common names. It is used by `layers_to_transform` and to find the layer whose output sets the "
                "unit for `dose_unit='residual_stream'`."
            ),
        },
    )
    steering_pattern: dict[str, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Maps projection names (exact or suffix matches) to specifications containing `indices` and "
                "`init_val`. These projections are the targets. The `indices` select the projection's input "
                "channels, and `init_val` initializes their independent, signed doses; a number broadcasts, and a "
                "list must match the number of indices. If empty, `target_modules` selects the projections, and every "
                "input channel gets a dose of 0, which leaves the model unchanged. `select_neurons` from "
                "`peft.tuners.activation_steering` then keeps the most useful neurons and writes them into this field."
            ),
        },
    )
    dose_unit: Literal["projection_output", "residual_stream"] = field(
        default="projection_output",
        metadata={
            "help": (
                "Where calibration measures the median L2 norm that sets the unit of the doses. "
                "'projection_output' uses the targeted projection's own output. 'residual_stream' uses the output of "
                "the numbered layer that contains the projection, which is the unit of neuron-explorer: on models "
                "that add the MLP output directly to the residual stream, e.g. Llama and Qwen, doses found there keep "
                "their meaning, as in the presets of the activation steering example. The ratio of the two medians "
                "differs between layers, so such doses cannot be converted with a single factor. Do not use "
                "'residual_stream' for models that normalize the MLP output before adding it to the residual stream, "
                "e.g. Gemma 2/3 and OLMo 2."
            ),
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from activation steering layers to be set as trainable and saved in the "
            "final checkpoint. For example, in Sequence Classification or Token Classification tasks, the final layer "
            "`classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ACTIVATION_STEERING
        if self.dose_unit not in ("projection_output", "residual_stream"):
            raise ValueError(f"`dose_unit` must be 'projection_output' or 'residual_stream', got {self.dose_unit!r}.")
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

        if self.steering_pattern:
            # The steering pattern is the single source of truth for the targets.
            filters = (self.target_modules, self.layers_to_transform, self.exclude_modules)
            if any(value is not None for value in filters):
                raise ValueError(
                    "`target_modules`, `layers_to_transform` and `exclude_modules` cannot be combined with "
                    "`steering_pattern`, whose keys are the targets."
                )
        elif self.target_modules is None:
            raise ValueError(
                "Specify `target_modules`, the projections whose input channels are steered, or `steering_pattern`."
            )
