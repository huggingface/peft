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

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class ActivationSteeringConfig(PeftConfig):
    """Configuration for additive activation steering at block outputs.

    The dose normalization follows Lyra's neuron-explorer implementation:
    https://github.com/lyramakesmusic/neuron-explorer/blob/5a8f8267f6bd3c8caa87d598870949ab0d25b914/explorer/runtime.py#L653

    Args:
        steering_pattern (`dict`):
            Maps block module names (exact or suffix matches) to specifications containing `source_module`, `indices`,
            `init_val`, and optionally `statistic`. `source_module` names a linear projection relative to the block.
            Its input-channel `indices` select weight columns, each normalized to unit L2 norm. `init_val` initializes
            independent, trainable signed doses; a number broadcasts, and a list must match the number of indices.
            `statistic` also broadcasts or accepts a matching list. Currently, only `"median_l2"` is supported and is
            the default: scale the addition by the median outgoing residual L2 norm measured on baseline calibration
            data, excluding each sequence's first valid token and padding. Call `calibrate_activation_steering` from
            `peft.tuners.activation_steering` before using a new adapter. The helper pools eligible tokens across all
            calibration batches and saves a fixed scale per block and adapter in the checkpoint. The scale is not
            trainable; the doses can be fine-tuned after calibration. Apply additions at all non-padding positions.

    Blocks must return a tensor of shape `(batch, sequence, hidden)` or a tuple whose first element is that tensor.
    Steering adds normalized source weight columns to the block output. This is equivalent to an additive neuron edit
    before the projection only when its output feeds directly into the block's final residual addition. Model inputs
    after the primary input, including `attention_mask` and `past_key_values`, must be passed by keyword.

    Example:

    ```py
    >>> from peft import ActivationSteeringConfig, TaskType

    >>> config = ActivationSteeringConfig(
    ...     base_model_name_or_path="Qwen/Qwen3.5-4B",
    ...     task_type=TaskType.CAUSAL_LM,
    ...     steering_pattern={
    ...         "layers.15": {
    ...             "source_module": "mlp.down_proj",
    ...             "indices": [3920],
    ...             "init_val": 0.5,
    ...             "statistic": "median_l2",
    ...         },
    ...     },
    ... )
    ```
    """

    steering_pattern: dict[str, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Maps block module names (exact or suffix matches) to specifications containing `source_module`, "
                "`indices`, `init_val`, and optionally `statistic`. `source_module` names a linear projection "
                "relative to the block. Its input-channel `indices` select weight columns, each normalized to unit L2 "
                "norm. `init_val` initializes independent, trainable signed doses; a number broadcasts, and a list "
                "must match the number of indices. `statistic` also broadcasts or accepts a matching list. Currently, "
                "only `'median_l2'` is supported and is the default: scale the addition by the median outgoing "
                "residual L2 norm measured on baseline calibration data, excluding each sequence's first valid token "
                "and padding. Call `calibrate_activation_steering` from `peft.tuners.activation_steering` before "
                "using a new adapter. The helper pools eligible tokens across all calibration batches and saves a "
                "fixed scale per block and adapter in the checkpoint. The scale is not trainable; the doses can be "
                "fine-tuned after calibration. Apply additions at all non-padding positions."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ACTIVATION_STEERING
        if not isinstance(self.steering_pattern, dict) or not self.steering_pattern:
            raise ValueError("`steering_pattern` must be a non-empty dictionary.")

        for name, spec in self.steering_pattern.items():
            if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
                raise ValueError("`steering_pattern` keys must be non-empty module paths.")
            if not isinstance(spec, dict):
                raise TypeError(f"The steering specification for '{name}' must be a dictionary.")

            required = {"source_module", "indices", "init_val"}
            if required - spec.keys():
                raise ValueError(f"Missing steering fields for '{name}': {sorted(required - spec.keys())}.")
            if spec.keys() - required - {"statistic"}:
                raise ValueError(
                    f"Unknown steering fields for '{name}': {sorted(spec.keys() - required - {'statistic'})}."
                )

            source = spec["source_module"]
            if not isinstance(source, str) or not source or any(not part for part in source.split(".")):
                raise ValueError(f"`source_module` for '{name}' must be a non-empty relative module path.")

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

            statistics = spec.get("statistic", "median_l2")
            statistics = statistics if isinstance(statistics, list) else [statistics] * len(indices)
            if len(statistics) != len(indices):
                raise ValueError(f"`statistic` for '{name}' must have the same length as `indices`.")
            if any(not isinstance(statistic, str) or statistic != "median_l2" for statistic in statistics):
                raise ValueError(f"Only 'median_l2' is supported for `statistic` (in '{name}').")

    @property
    def target_modules(self):
        # Keep the targets derived so that adapter_config.json has a single source of truth.
        return tuple(self.steering_pattern)
