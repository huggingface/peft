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

"""Target module suggester.

When a user passes a `target_modules` value that does not match any module in the model, `suggest_target_modules`
returns a tuner-aware suggestion of valid candidates, compressed via `_find_minimal_target_modules`. Behavior is purely
additive: callers should treat `None` returns as "no decoration" and leave the original error message unchanged.

Extension: per-tuner candidate enumeration is registered via the `register_candidate_collector` decorator. Each tuner
package registers its own collector at import time. To add a new tuner, create a candidate collector module under
`src/peft/tuners/<tuner>/` and import it from that package's `__init__.py`.

IMPORTANT: do NOT add module-top-level imports from `peft.tuners.tuners_utils`. The local import of
`_find_minimal_target_modules` inside `suggest_target_modules` is required for circular-import safety.
"""

from collections.abc import Callable


TUNER_TYPE_TO_CANDIDATE_COLLECTOR: dict[str, Callable] = {}


def register_candidate_collector(tuner_type: str) -> Callable:
    """Decorator to register a per-tuner candidate enumerator.

    Args:
        tuner_type: The `PeftType` value as a string, e.g. "LORA", "IA3".

    The decorated function should accept a `model` argument and yield full module names (str) that this tuner could
    validly adapt.
    """

    def decorator(fn: Callable) -> Callable:
        TUNER_TYPE_TO_CANDIDATE_COLLECTOR[tuner_type] = fn
        return fn

    return decorator


def suggest_target_modules(tuner, model, peft_config, unmatched_modules) -> str | None:
    """Return a compressed human-readable suggestion of valid target modules,
    or None if no suggestion can be made.

    Returns None when:
        - peft_config.peft_type is not registered with a candidate collector
        - the model has zero candidate modules for this tuner
        - _find_minimal_target_modules returns empty (defensive; given the upstream candidate-set non-empty check, this
          branch is technically unreachable for the current implementation of _find_minimal_target_modules)
    """
    peft_type = getattr(peft_config, "peft_type", None)
    if peft_type is None:
        return None
    collector = TUNER_TYPE_TO_CANDIDATE_COLLECTOR.get(peft_type)
    if collector is None:
        return None

    candidate_names = list(collector(model))
    if not candidate_names:
        return None

    candidate_set = set(candidate_names)
    non_candidate_names = [n for n, _ in model.named_modules() if n and n not in candidate_set]

    # Local import for circular-import safety. Do NOT promote to module top.
    from peft.tuners.tuners_utils import _find_minimal_target_modules

    minimal = _find_minimal_target_modules(candidate_names, non_candidate_names)
    if not minimal:
        return None

    return _format_suggestion(sorted(minimal), peft_type)


def _format_suggestion(minimal_targets: list[str], peft_type) -> str:
    """First-version output format. Bossan invited iteration via tests."""
    # `peft_type` is typically a `PeftType` enum member (a `str`-mixin Enum).
    # In Python 3.12, `f"{peft_type}"` returns the enum's __str__ ("PeftType.LORA")
    # rather than the value ("LORA"). Use `.value` when available to keep the
    # user-facing message clean. Fall back to str() for plain-string callers
    # (e.g. test fixtures that use a fabricated `peft_type`).
    type_name = getattr(peft_type, "value", peft_type)
    # Render the minimal targets as a deterministic set-shaped string. The caller
    # passes `sorted(minimal)`, but using `set(minimal_targets)` in the f-string
    # would re-randomize the order via Python's hash randomization. Build the
    # set-shaped repr by hand to preserve the sort and stay run-to-run stable.
    sorted_repr = "{" + ", ".join(repr(t) for t in minimal_targets) + "}"
    return (
        f"Did you mean one of these? Valid `target_modules` candidates for {type_name} on this model: {sorted_repr}."
    )
