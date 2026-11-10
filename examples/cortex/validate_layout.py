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

"""Probe Cortex layouts and native export on a small random model built from a local or Hub config."""

import argparse
import json
import re
import tempfile
import warnings
from copy import copy, deepcopy
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from peft import CortexConfig, get_peft_model
from peft.tuners.cortex.layer import CortexLayer, ExpertLinear
from peft.tuners.cortex.model import _iter_target_candidates
from peft.tuners.cortex.structure import (
    UnresolvedSharedExpertsError,
    resolve_target_branches,
    text_backbone_prefixes,
)
from peft.tuners.tuners_utils import check_target_module_exists


TOLERANCES = {"atol": 3e-5, "rtol": 2e-4}


def text_config(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def build_base(model_name, revision=None, ffn_specs=None):
    config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    text = text_config(config)
    composite = text is not config
    factory = AutoModelForCausalLM
    if composite:
        # Older Transformers releases can still probe text-only models.
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise ValueError("This Transformers version cannot construct the native image-text wrapper.") from exc
        factory = AutoModelForImageTextToText
    if type(config) not in factory._model_mapping:
        raise ValueError(f"Cannot probe {config.model_type} with its original module paths using {factory.__name__}.")

    # Keep all layers, their order, expert counts, and top-k. Truncating a hybrid
    # model to its first two layers can quietly turn a MoE probe into a dense one.
    if text.num_hidden_layers > 256:
        raise ValueError("This probe retains all layers. Supply a smaller representative local config.")
    heads = min(text.num_attention_heads, 2)
    tiny = {
        "num_attention_heads": heads,
        "num_key_value_heads": heads if getattr(text, "num_key_value_heads", None) == text.num_attention_heads else 1,
        "hidden_size": heads * 8,
        "vocab_size": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": 128,
    }
    # Split attention can use head_dim for the rotary slice alone. Keep its native width and frequencies aligned.
    if not hasattr(text, "qk_rope_head_dim"):
        tiny["head_dim"] = 8

    # Keep distinct FFN width fields distinguishable when testing native reload.
    width_fields = {name for name in text.to_dict() if "intermediate_size" in name}
    for spec in (ffn_specs or {}).values():
        width_fields.add(spec.width_field)
        if spec.shared_width_field is not None:
            width_fields.add(spec.shared_width_field)
    widths = {
        name: value
        for name in width_fields
        if isinstance(value := getattr(text, name, None), int) and not isinstance(value, bool) and value > 0
    }
    small_width = {v: min(v, 16 + 4 * i) for i, v in enumerate(sorted(set(widths.values())))}
    tiny.update({k: small_width[v] for k, v in widths.items()})
    text.update(tiny)
    if composite:
        # Keep the multimodal wrapper: user paths (including exclusions) resolve in its namespace.
        vision = getattr(config, "vision_config", None)
        if vision is not None:
            for key, value in {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "head_dim": 8,
            }.items():
                if hasattr(vision, key):
                    setattr(vision, key, value)
    config = type(config).from_dict(config.to_dict())

    backend = {"attn_implementation": "eager", "trust_remote_code": False}
    # Older versions can deserialize this attribute without supporting the factory keyword.
    if hasattr(transformers.PretrainedConfig, "_experts_implementation"):
        backend["experts_implementation"] = "eager"
    # Some models have large tables or auxiliary towers outside the fields above.
    # Count parameters and buffers before allocating real CPU weights.
    with torch.device("meta"):
        preview = factory.from_config(config, **backend)
    elements = sum(t.numel() for t in (*preview.parameters(), *preview.buffers()))
    if elements > 12_000_000:
        raise ValueError(f"Reduced config still has {elements:,} tensor elements; supply a smaller local config.")
    return factory.from_config(config, **backend).float().eval(), factory, backend


def probe_branches(model, tokens):
    # Sample initialization zeros every down projection. Give each a nonzero
    # weight and record which expert projections actually run on these inputs.
    hits = {}
    handles = []
    groups = {}

    def observe(name):
        def record(module, inputs, output):
            if torch.count_nonzero(output):
                hits[name].add(inputs[1] if isinstance(module, ExpertLinear) else None)

        return record

    for path, layer in model.get_base_model().named_modules():
        if not isinstance(layer, CortexLayer):
            continue
        for attr in ("cortex_down", "cortex_shared_down"):
            projections = getattr(layer, attr)
            if "default" not in projections:
                continue
            projection = projections["default"]
            role = layer.role
            if role == "routed_public":
                role = "public" if attr == "cortex_shared_down" else "routed"
            groups.setdefault(role, []).append(projection)
            name = f"{path} ({layer.role}, {attr}, k={projection.weight.shape[-1]})"
            hits[name] = set()
            projection.weight.normal_(std=0.2)
            handles.append(projection.register_forward_hook(observe(name)))

    try:
        logits = model(tokens, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()

    if not hits:
        raise ValueError("No Cortex branches found.")
    missing = [name for name, seen in hits.items() if not seen]
    if missing:
        raise ValueError(f"No nonzero projection output observed for {missing}.")
    for name, seen in hits.items():
        suffix = ""
        if None not in seen:
            suffix = f", nonzero projections from {len(seen)} experts before routing weights"
            if len(seen) <= 8:
                suffix += f" {sorted(seen)}"
        print(f"  {name}{suffix}")

    if len(groups) > 1:
        check_branch_effects(model, tokens, logits, groups)
    return logits


def check_branch_effects(model, tokens, logits, groups):
    # A routed down projection can run but have zero router weight. Disable each kind in turn
    # so that a live dense or shared branch cannot hide a missing effect on the model output.
    def silence(module, inputs, output):
        return torch.zeros_like(output)

    for role, projections in groups.items():
        handles = [projection.register_forward_hook(silence) for projection in projections]
        try:
            without_branch = model(tokens, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
        if not torch.isfinite(without_branch).all():
            raise ValueError(f"Non-finite logits with {role} branches disabled.")
        if torch.allclose(without_branch, logits, **TOLERANCES):
            raise ValueError(f"No measurable {role} branch effect on logits for the probe inputs.")


def matched_paths(paths, selection, option):
    """Resolve every CLI pattern in the original model namespace, including exclusions."""
    found = set()
    patterns = [selection] if isinstance(selection, str) else selection or []
    for pattern in patterns:
        if isinstance(selection, str):
            hits = {p for p in paths if re.fullmatch(pattern, p)}
        else:
            hits = {p for p in paths if p == pattern or p.endswith("." + pattern)}
        if not hits:
            raise ValueError(f"{option} pattern {pattern!r} did not match any module in the probe model.")
        found.update(hits)
    return found


def selected_moe_branches(base, targets, excluded, config):
    """Separate mode-dependent parents from leaves whose branch role is fixed."""
    if config.target_modules is None:
        candidates = _iter_target_candidates(base.named_modules(), config, text_roots=text_backbone_prefixes(base))
    else:
        candidates = ((path, module) for path, module in base.named_modules() if path and path in targets)
    parent_branches, leaf_branches = set(), set()
    for path, _ in candidates:
        if path in excluded:
            continue

        kind, routed_path, shared_paths = resolve_target_branches(base, path, config.ffn_specs, config.moe_specs)
        if kind == "moe":
            routed = routed_path not in excluded
            if routed:
                parent_branches.add("routed")
            if shared_paths is None:
                # Unknown ownership must still reach the tuner's unsupported-layout check.
                parent_branches.add("shared")
            elif shared_paths:
                if any(shared_path not in excluded for shared_path in shared_paths):
                    parent_branches.add("shared")
            elif routed:
                # A public branch is installed on the routed experts module.
                parent_branches.add("shared")
        elif kind in {"routed", "shared"}:
            leaf_branches.add(kind)
    return parent_branches, leaf_branches


def _read_config(path):
    path = path / "adapter_config.json" if path.is_dir() else path
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a Cortex configuration JSON object.")
    if data.get("peft_type", "CORTEX") != "CORTEX":
        raise ValueError(f"{path} must contain a Cortex configuration.")
    return data


def saved_config(path):
    data = _read_config(path)
    if data.get("peft_type") != "CORTEX" or not data.get("target_roles"):
        raise ValueError(f"{path} must be a saved Cortex adapter config with resolved target_roles.")
    return CortexConfig(**data)


def fresh_config(path):
    data = _read_config(path)
    if data.get("target_roles"):
        raise ValueError("A configuration with resolved target_roles must use --adapter-config for exact replay.")
    # Keep only the supplied keys: omitted k and moe_target request a probe matrix,
    # whereas serialized defaults are explicit choices in a full configuration.
    CortexConfig(**data)
    return data


@torch.no_grad()
def validate_case(template, factory, backend, config, tokens, modes):
    base = deepcopy(template)
    expected_roles = dict(config.target_roles)
    baseline = base(tokens, use_cache=False).logits

    # Suppress the inferred-layout reminder while performing its numerical validation.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Cortex inferred unregistered Transformers layouts:", category=UserWarning
        )
        model = get_peft_model(base, config).eval()
    roles = {name: module.role for name, module in base.named_modules() if isinstance(module, CortexLayer)}
    if expected_roles and roles != expected_roles:
        raise ValueError(f"Saved roles {expected_roles} do not match actual roles {roles}.")
    for mode in modes:
        compatible = {"shared", "public", "routed_public"} if mode == "shared" else {"routed", "routed_public"}
        if compatible.isdisjoint(roles.values()):
            raise ValueError(f"No selected {mode} MoE branch was exercised. Actual roles: {roles}.")

    if config.init_weights is not False:
        torch.testing.assert_close(model(tokens, use_cache=False).logits, baseline, **TOLERANCES)
    expected = probe_branches(model, tokens)
    if not torch.isfinite(expected).all() or torch.allclose(baseline, expected, **TOLERANCES):
        raise ValueError("Expected a finite adapter effect larger than the merge tolerance.")

    try:
        native = model.merge_and_unload(safe_merge=True).eval()
    except ValueError as exc:
        # Some trainable layouts have no lossless native export (e.g. an unweighted
        # public branch without proven routing mass). Keep this distinct from a pass.
        print(f"EXPORT REFUSED: {exc}")
        return False
    merged = native(tokens, use_cache=False).logits
    torch.testing.assert_close(merged, expected, **TOLERANCES)
    with tempfile.TemporaryDirectory() as directory:
        native.save_pretrained(directory)
        restored, info = factory.from_pretrained(directory, output_loading_info=True, **backend)
        errors = {
            k: v
            for k, v in info.items()
            if k in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"} and v
        }
        if errors:
            raise ValueError(f"Native reload failed: {errors}")
        reloaded = restored.eval()(tokens, use_cache=False).logits
        torch.testing.assert_close(reloaded, expected, **TOLERANCES)
    print(
        f"PASS: maximum absolute logit differences from the nonzero adapter: "
        f"merged={(merged - expected).abs().max().item():.3g}, reloaded={(reloaded - expected).abs().max().item():.3g}."
    )
    return True


def probe_cases(base, args, checkpoint, config_values=None):
    paths = [name for name, _ in base.named_modules() if name]
    if checkpoint is not None:
        # An expert path selected afresh would mean routed, even if it was saved as public.
        matched_paths(paths, checkpoint.target_modules, "saved target_modules")
        matched_paths(paths, checkpoint.exclude_modules, "saved exclude_modules")
        saved_class = (checkpoint.auto_mapping or {}).get("base_model_class")
        if saved_class is not None and saved_class != type(base).__name__:
            raise ValueError(f"Checkpoint uses {saved_class}, but the probe constructed {type(base).__name__}.")
        return [(checkpoint, [])]

    values = {"init_weights": "sample"}
    values.update(deepcopy(config_values or {}))
    for field, names in (
        ("target_modules", ("target_modules", "target_regex")),
        ("exclude_modules", ("exclude_modules", "exclude_regex")),
        ("k_shared", ("k_shared",)),
        ("k_routed", ("k_routed",)),
        ("public_weighting", ("public_weighting",)),
    ):
        for name in names:
            value = getattr(args, name)
            if value is not None:
                values[field] = value

    template = CortexConfig(**values)
    requested_mode = args.moe_target if args.moe_target is not None else template.moe_target
    selected = matched_paths(paths, template.target_modules, "target_modules")
    excluded = matched_paths(paths, template.exclude_modules, "exclude_modules")
    # Use PEFT's matching rules for modules_to_save and its descendants as well.
    availability = copy(template)
    availability.target_modules = ".*"
    excluded.update(path for path in paths if not check_target_module_exists(availability, path))
    parent_branches, leaf_branches = selected_moe_branches(base, selected, excluded, template)
    if parent_branches:
        if args.moe_target is not None:
            modes = [args.moe_target]
        elif "moe_target" in values:
            mode = template.moe_target
            modes = [[mode] if isinstance(mode, str) else mode]
        else:
            modes = [mode for mode in (["shared"], ["routed"], ["shared", "routed"]) if set(mode) <= parent_branches]
    else:
        # Leaf selections already fix their roles; moe_target only controls selected parents.
        modes = [[]]

    widths = args.k
    if widths is None:
        widths = [template.k] if "k" in values else [2, 4]
    cases = []
    for mode in modes:
        for k in widths:
            current = deepcopy(values)
            current.update(k=k, moe_target=mode or requested_mode)
            config = CortexConfig(**current)
            expected = leaf_branches | (set(mode) & parent_branches)
            cases.append((config, [branch for branch in ("shared", "routed") if branch in expected]))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", help="Local config directory or Hub model ID (config only)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--adapter-config", type=Path, help="Replay a saved adapter_config.json or its directory")
    source.add_argument(
        "--cortex-config",
        type=Path,
        help="Fresh Cortex JSON or config directory; CLI probe options override its values",
    )
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target-modules", nargs="+", help="Complete FFN or MoE module paths or suffixes")
    targets.add_argument("--target-regex", help="Regular expression matching complete FFN or MoE module paths")
    exclusions = parser.add_mutually_exclusive_group()
    exclusions.add_argument("--exclude-modules", nargs="+", help="Module paths or suffixes to exclude")
    exclusions.add_argument("--exclude-regex", help="Regular expression matching module paths to exclude")
    parser.add_argument(
        "--moe-target", nargs="+", choices=["shared", "routed"], help="Branches to adapt on selected MoE parents"
    )
    parser.add_argument("--public-weighting", choices=["unweighted", "routing"])
    parser.add_argument("--k", type=int, nargs="+", help="Widths to probe (default: JSON k, otherwise 2 4)")
    parser.add_argument("--k-shared", type=int, help="Override the shared or public branch width")
    parser.add_argument("--k-routed", type=int, help="Override the per-expert branch width")
    args = parser.parse_args()

    checkpoint = None
    config_values = None
    if args.adapter_config is not None:
        overrides = {k for k, v in vars(args).items() if k not in {"adapter_config", "model_name"} and v is not None}
        if overrides:
            parser.error(f"Saved config mode cannot override {', '.join(sorted(overrides))}.")
        checkpoint = saved_config(args.adapter_config)
    elif args.cortex_config is not None:
        config_values = fresh_config(args.cortex_config)
    template = checkpoint if checkpoint is not None else CortexConfig(**(config_values or {}))
    if template.task_type not in (None, "CAUSAL_LM"):
        parser.error(f"The probe cannot validate task type {template.task_type}.")
    model_name = args.model_name or template.base_model_name_or_path
    if not model_name:
        parser.error("Supply --model-name or a Cortex config containing base_model_name_or_path.")

    torch.manual_seed(42)
    torch.set_num_threads(1)
    base, factory, backend = build_base(model_name, template.revision, template.ffn_specs)
    text = text_config(base.config)
    tokens = torch.randint(3, text.vocab_size, (4, 16))
    print(
        f"Checking {model_name} with {type(base).__name__} on Transformers {transformers.__version__}: "
        f"{text.num_hidden_layers} text layers, hidden size {text.hidden_size}. All weights are random."
    )
    cases = probe_cases(base, args, checkpoint, config_values)

    passed = 0
    for config, mode in cases:
        label = "saved roles" if checkpoint is not None else "+".join(mode) or "dense"
        print(f"\nCase {label}, k={config.k}")
        try:
            if validate_case(base, factory, backend, config, tokens, mode):
                passed += 1
        except UnresolvedSharedExpertsError as exc:
            print(f"UNSUPPORTED: {exc}")
        except (ValueError, TypeError, RuntimeError, AssertionError) as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
    print(f"Validated {passed}/{len(cases)} cases.")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
