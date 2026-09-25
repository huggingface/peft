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

import platform
from collections import Counter
from typing import Any, Literal

import accelerate
import torch
import transformers

from .. import __version__
from ..peft_model import get_layer_status, get_model_status


SEVERITY = Literal["warning", "error"]


def _format_finding(code: str, severity: SEVERITY, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        values = [_json_value(item) for item in value]
        return sorted(values) if isinstance(value, set) else values
    if hasattr(value, "value"):
        return _json_value(value.value)
    return str(value)


def _get_adapter_configurations(model) -> dict[str, dict[str, Any]]:
    config_attributes = [
        "task_type",
        "inference_mode",
        "target_modules",
        "target_parameters",
        "modules_to_save",
        "layers_to_transform",
        "layers_pattern",
    ]
    configurations = {}
    for adapter_name, config in model.peft_config.items():
        configurations[adapter_name] = {
            attribute: _json_value(val)
            for attribute in config_attributes
            if (val := getattr(config, attribute, None)) is not None
        }
    return configurations


def _get_model_runtime(model) -> dict[str, Any]:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = getattr(base_model, "config", None)
    parameter_dtypes = Counter()
    parameter_devices = set()
    for parameter in base_model.parameters():
        parameter_dtypes[str(parameter.dtype).replace("torch.", "")] += parameter.numel()
        parameter_devices.add(parameter.device.type)
    hf_device_map = getattr(base_model, "hf_device_map", None)

    return {
        "model_name_or_path": getattr(config, "_name_or_path", None),
        "model_type": getattr(config, "model_type", None),
        "revision": getattr(config, "_commit_hash", None),
        "training": model.training,
        "gradient_checkpointing": bool(getattr(base_model, "is_gradient_checkpointing", False)),
        "parameter_dtypes": dict(sorted(parameter_dtypes.items())),
        "parameter_devices": sorted(parameter_devices),
        "hf_device_map_devices": (
            None if hf_device_map is None else sorted({str(device) for device in hf_device_map.values()})
        ),
        "quantization": {
            "is_loaded_in_4bit": bool(getattr(base_model, "is_loaded_in_4bit", False)),
            "is_loaded_in_8bit": bool(getattr(base_model, "is_loaded_in_8bit", False)),
            "method": _json_value(getattr(base_model, "quantization_method", None)),
        },
    }


def run_healthcheck(model, min_trainable_params: int = 1, max_trainable_params_percent: int = 50) -> dict[str, Any]:
    """
    Inspect a PEFT model for states that are suspicious before training.

    This function summarizes the model and layer status APIs, and includes findings for inconsistent adapter state,
    merged adapters, and implausible numbers of trainable parameters. It does not validate the training loop,
    optimizer, or dataset. A status will be reported as `"irregular"` if inconsistencies are found in the model, e.g.
    when for the same adapter, some layers are enabled and some layers are disabled. This almost always means that
    something went wrong and that you should check that you didn't accidentally change some attributes on the model
    incorrectly.

    If the check found something suspicious, it will be reported in the `"findings"` field. An empty `"findings""` list
    means that no suspicious adapter state was detected, not that a training run is guaranteed to succeed.

    Args:
        model (`nn.Module`)
            The model to be checked. Must be adapted with PEFT.
        min_trainable_params (`int`, *optional*, default=`1`)
            Minimum expected number of parameters. If less than those are found, this is reported as an error.
        max_trainable_params_percent (`int`, *optional*, default=`50`)
            The maximum percentage of parameters that should be trainable. If more than those are found, this is
            reported as a warning.

    Returns:
        result
            Dictionary containing the different findings. The returned dictionary is JSON-serializable.
    """
    model_status = get_model_status(model)
    layer_status = get_layer_status(model)

    trainable_percent = 100 * model_status.trainable_params / model_status.total_params
    findings: list[dict[str, str]] = []

    if model_status.trainable_params < min_trainable_params:
        findings.append(
            _format_finding(
                "NO_TRAINABLE_PARAMETERS",
                "error",
                f"The model should have at least {min_trainable_params} trainable parameter(s).",
            )
        )
    elif trainable_percent > max_trainable_params_percent:
        findings.append(
            _format_finding(
                "HIGH_TRAINABLE_PARAMETER_FRACTION",
                "warning",
                (
                    f"{trainable_percent:.2f}% of model parameters are trainable; expected no more than "
                    f"{max_trainable_params_percent}% for PEFT training."
                ),
            )
        )

    if model_status.enabled == "irregular":
        findings.append(
            _format_finding(
                "IRREGULAR_ADAPTER_ENABLED_STATE",
                "error",
                "Adapter layers are not consistently enabled.",
            )
        )
    elif not model_status.enabled:
        findings.append(
            _format_finding(
                "ADAPTERS_DISABLED",
                "warning",
                "All adapter layers are disabled.",
            )
        )

    if model_status.active_adapters == "irregular":
        findings.append(
            _format_finding(
                "IRREGULAR_ACTIVE_ADAPTERS",
                "error",
                "Active adapters differ across adapter layers.",
            )
        )

    if model_status.merged_adapters == "irregular":
        findings.append(
            _format_finding(
                "IRREGULAR_MERGED_ADAPTERS",
                "error",
                "Merged adapters differ across adapter layers.",
            )
        )
    elif model_status.merged_adapters:
        merged_adapters = ", ".join(f"'{name}'" for name in model_status.merged_adapters)
        findings.append(
            _format_finding(
                "MERGED_ADAPTERS",
                "warning",
                f"Adapter(s) {merged_adapters} is/are merged; unmerge them before training.",
            )
        )

    if any(requires_grad == "irregular" for requires_grad in model_status.requires_grad.values()):
        findings.append(
            _format_finding(
                "IRREGULAR_REQUIRES_GRAD",
                "error",
                "Adapter parameters do not consistently have requires_grad set across adapter layers.",
            )
        )
    elif isinstance(model_status.active_adapters, list):
        frozen_active_adapters = [
            adapter_name
            for adapter_name in model_status.active_adapters
            if model_status.requires_grad.get(adapter_name) is False
        ]
        if frozen_active_adapters:
            findings.append(
                _format_finding(
                    "ACTIVE_ADAPTERS_NOT_TRAINABLE",
                    "warning",
                    f"Active adapter(s) {', '.join(frozen_active_adapters)} are not trainable.",
                )
            )

    layer_types_per_adapter = {}
    for adapter_name in model_status.available_adapters:
        layer_types = Counter(
            status.module_type for status in layer_status if adapter_name in status.available_adapters
        )
        layer_types_per_adapter[adapter_name] = dict(layer_types.most_common())

    # works for Transformers, needs refinement for other types like Diffusers, just shows 'other' for now
    model_id = getattr(getattr(model, "config", None), "name_or_path", "other")

    result = {
        "is_ready": not any(finding["severity"] == "error" for finding in findings),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "peft_version": __version__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "accelerate_version": accelerate.__version__,
            "cuda_version": torch.version.cuda,
        },
        "summary": {
            "model_id": model_id,
            "base_model_type": model_status.base_model_type,
            "adapter_model_type": model_status.adapter_model_type,
            "peft_types": model_status.peft_types,
            "trainable_params": model_status.trainable_params,
            "total_params": model_status.total_params,
            "trainable_percent": trainable_percent,
            "num_adapter_layers": model_status.num_adapter_layers,
            "adapter_layer_types": layer_types_per_adapter,
        },
        "adapter_configurations": _get_adapter_configurations(model),
        "model_runtime": _get_model_runtime(model),
        "adapter_state": {
            "enabled": model_status.enabled,
            "active_adapters": model_status.active_adapters,
            "merged_adapters": model_status.merged_adapters,
            "requires_grad": model_status.requires_grad,
            "available_adapters": model_status.available_adapters,
            "devices": model_status.devices,
            "quantization_backend": model_status.quantization_backend,
        },
        "findings": findings,
    }
    return result


def format_healthcheck(healthcheck: dict[str, Any], sep=", ", indent="  ") -> str:
    """
    Format `run_healthcheck` output as a compact human-readable report.

    Args:
        healthcheck: dict
            The healthcheck dictionary as returned by `run_healthcheck`.
        sep (`str`, *optional*, defaults to `", "`)
            The separator to use between listed items.
        indent (`str`, *optional*, defaults to `"  "`)
            The indentation level for nested items.
        sink (`callable`, *optiona*, default=`print`)
            Function which is called to print the output. By default, just the builtin `print` function, but can also
            be something else like `logger.info`.

    Returns:
        Result (`str`)
            The formatted string in human-readable format.
    """
    sep = ", "
    indent = "  "

    summary = healthcheck["summary"]
    adapter_state = healthcheck["adapter_state"]
    environment = healthcheck["environment"]
    model_runtime = healthcheck["model_runtime"]
    devices = ", ".join(model_runtime["parameter_devices"])
    adapter_types = ", ".join(f"{name} ({peft_type})" for name, peft_type in summary["peft_types"].items())
    dtypes = ", ".join(f"{k}={v:,}" for k, v in model_runtime["parameter_dtypes"].items())

    active_adapters = adapter_state["active_adapters"]
    if not isinstance(active_adapters, str):
        active_adapters = ", ".join(active_adapters) or "none"
    merged_adapters = adapter_state["merged_adapters"]
    if not isinstance(merged_adapters, str):
        merged_adapters = ", ".join(merged_adapters) or "none"
    layer_types = ""
    for adapter_name, dct in summary["adapter_layer_types"].items():
        layer_types += f"{indent}{adapter_name}:\n"
        layer_types += f"{2 * indent}" + sep.join(f"{k}={v}" for k, v in dct.items())

    lines = [
        "PEFT healthcheck",
        sep.join(
            (
                f"Model:\n{indent}ID: {summary['model_id']}",
                f"type: {summary['base_model_type']}",
                f"adapters: {adapter_types}",
            )
        ),
        sep.join(
            (
                f"Environment:\n{indent}peft: {environment['peft_version']}",
                f"transformers: {environment['transformers_version']}",
                f"torch: {environment['torch_version']}",
                f"Python: {environment['python_version']}",
            )
        ),
        sep.join(
            (
                f"Runtime:\n{indent}training: {model_runtime['training']}",
                f"devices: {devices}",
                f"dtypes: {dtypes}",
                f"gradient checkpointing: {model_runtime['gradient_checkpointing']}",
            )
        ),
        sep.join(
            (
                f"Parameters:\n{indent}trainable: {summary['trainable_params']:,}",
                f"total: {summary['total_params']:,}",
                f"percent trainable: {summary['trainable_percent']:.4f}%",
                f"adapter layers: {summary['num_adapter_layers']}",
            )
        ),
        sep.join(
            (
                f"State:\n{indent}adapter is enabled: {adapter_state['enabled']}",
                f"active: {active_adapters}",
                f"merged: {merged_adapters}",
            )
        ),
        f"Layer types:\n{layer_types}",
    ]

    findings = healthcheck["findings"]
    if findings:
        lines.append("Findings:")
        lines.extend(f"  {finding['severity'].upper()}: {finding['message']}" for finding in findings)

    result = "\n".join(lines)
    return result
