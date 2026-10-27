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

"""Checks for execution modes that change the audited FFN contract."""

from functools import lru_cache, partial
from types import CodeType
from typing import Any

from accelerate.hooks import add_hook_to_module
from torch import nn


@lru_cache
def _accelerate_forward_code() -> CodeType | None:
    # Accelerate creates this function on every installation. Its code object is
    # stable, unlike the function identity or metadata copied by update_wrapper.
    return next(
        (
            value
            for value in add_hook_to_module.__code__.co_consts
            if isinstance(value, CodeType) and value.co_name == "new_forward"
        ),
        None,
    )


@lru_cache
def _native_forward(cls):
    # Remember the first observed class forward to detect later replacements.
    # Class-level patches made before that observation cannot be distinguished
    # from the native implementation by this check.
    return cls.forward


def check_no_offload(model: nn.Module) -> None:
    def offloads(hook):
        return getattr(hook, "offload", False) or any(offloads(child) for child in getattr(hook, "hooks", ()))

    if "disk" in (getattr(model, "hf_device_map", None) or {}).values() or any(
        offloads(getattr(module, "_hf_hook", None)) for module in model.modules()
    ):
        raise ValueError(
            "Cortex does not support active Accelerate CPU/disk offload hooks. Reload a materialized model "
            "without offload before injecting, merging, or unmerging Cortex. Ordinary CPU placement is supported."
        )


def check_moe_forward(module: nn.Module) -> None:
    # kernels.kernelize binds a replacement forward to the instance. A decorator
    # alone, or kernelize falling back to the native class forward, is harmless.
    forward = module.forward
    if (
        isinstance(forward, partial)
        and _accelerate_forward_code() is not None
        and getattr(forward.func, "__code__", None) is _accelerate_forward_code()
        and getattr(forward.func, "__globals__", None) is add_hook_to_module.__globals__
        and len(forward.args) == 1
        and forward.args[0] is module
        and not forward.keywords
        and hasattr(module, "_hf_hook")
    ):
        # Follow the actual dispatch target, not __wrapped__, which is only
        # metadata and can remain native after a bypassing replacement.
        forward = getattr(module, "_old_forward", None)
    if getattr(forward, "__self__", None) is not module or getattr(forward, "__func__", None) is not _native_forward(
        type(module)
    ):
        raise ValueError(
            f"Cortex requires the native {type(module).__name__}.forward. A replaced MoE parent forward "
            "(for example a MegaBlocks kernel) can bypass the adapter's expert wrapper. Reload with the "
            "native MoE forward; kernels limited to attention or normalization are not rejected by this check."
        )


def check_moe_forward_hook(module: nn.Module, args: tuple[Any, ...]) -> None:
    # Also catch kernelization after injection, before a bypassing parent runs.
    check_moe_forward(module)
