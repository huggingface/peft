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
"""Generate a machine-readable capability matrix of all PEFT methods.

For each registered PEFT method, a fixed set of checks ("tasks") determines which user-facing features the method
supports, e.g. which quantization backends it integrates with, which layer types it can target, or whether its adapters
can be merged into the base weights. The result is written as JSON and is intended as a generic data source, e.g. for
documentation pages or the PEFT shop app (method_comparison/peft-shop).

Each value is annotated with the *source* of the information:

- "introspection": determined by statically inspecting the classes of the installed PEFT package
- "file_check":    determined from the presence of integration modules in the PEFT source tree
- "probe":         determined empirically by exercising the feature on a tiny model on CPU
- "error":         the check itself failed; the value is "unknown" and the note contains the reason

Values are never guessed: if a check cannot determine a feature, the value is reported as "unknown" together with a
note. In particular, probing a method requires that its config can be instantiated with default arguments; methods that
need more than that must have an entry in PROBE_CONFIG_OVERRIDES.

The script requires PEFT to be installed (e.g. `pip install -e .`), runs on CPU, downloads nothing, and is idempotent:
running it twice on the same environment produces identical output.

Usage examples:

    # check all methods, write JSON to method_capabilities.json
    python scripts/generate_method_capabilities.py

    # check only LoRA and IA3, write to a custom file
    python scripts/generate_method_capabilities.py --methods lora ia3 --output capabilities.json

    # show which checks would run, without running them
    python scripts/generate_method_capabilities.py --dry-run

Tasks are collected up-front before any of them runs (see `collect_tasks`). This makes `--dry-run` trivial and leaves
the door open to run independent tasks in parallel later, should runtime ever become an issue.
"""

import argparse
import dataclasses
import enum
import inspect
import json
import logging
import re
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.pytorch_utils import Conv1D

import peft
from peft import get_peft_model
from peft.config import PeftConfig, PromptLearningConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_MIXED_MODEL_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils.hotswap import CONFIG_KEYS_TO_CHECK
from peft.utils.peft_types import PeftType


logger = logging.getLogger("generate_method_capabilities")

UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"

# Probe models use this hidden dimension throughout. It is chosen to be highly divisible, since several methods have
# divisibility constraints between their block/rank settings and the layer dimensions (e.g. C3A, VBLoRA, RoAd).
HIDDEN_DIM = 64

# Quantization integration modules inside a tuner package, mapped to the backend names they provide. The bnb module
# always contains both the 8-bit and the 4-bit integration.
QUANT_FILE_BACKENDS: dict[str, tuple[str, ...]] = {
    "aqlm": ("aqlm",),
    "awq": ("awq",),
    "bnb": ("bnb_8bit", "bnb_4bit"),
    "eetq": ("eetq",),
    "gptq": ("gptq",),
    "hqq": ("hqq",),
    "inc": ("inc",),
    "torchao": ("torchao",),
}

# Backends covered by the generic quantization integration (peft.utils.quantization_utils). Methods that call
# resolve_quantization_backend don't need per-backend integration modules; they support everything the resolver
# handles (merging may still be unavailable for the forward-only backends, which is a property of the backend, not of
# the PEFT method).
GENERIC_QUANT_BACKENDS: tuple[str, ...] = (
    "aqlm",
    "awq",
    "bnb_4bit",
    "bnb_8bit",
    "eetq",
    "gptq",
    "hqq",
    "inc",
    "torchao",
)

# Extra config arguments required to instantiate a method's config for probing, on top of the generic arguments
# (target_modules for adapter methods, task_type/num_virtual_tokens for prompt learning). If probing a method reports
# "unknown" because its config could not be instantiated, add an entry here.
PROBE_CONFIG_OVERRIDES: dict[PeftType, dict[str, Any]] = {
    # total_step is a required argument
    PeftType.ADALORA: {"total_step": 10},
    # IA3 needs feedforward_modules
    PeftType.IA3: {"feedforward_modules": []},
    # the default block_size of 256 does not divide HIDDEN_DIM
    PeftType.C3A: {"block_size": 16},
    # token_indices defaults to an empty list, which trains nothing
    PeftType.TRAINABLE_TOKENS: {"token_indices": [0, 1]},
    # the default vector_length of 256 does not divide HIDDEN_DIM
    PeftType.VBLORA: {"num_vectors": 32, "vector_length": 16},
    PeftType.ADAPTION_PROMPT: {"adapter_layers": 1, "adapter_len": 4, "task_type": "CAUSAL_LM"},
}

# Methods that cannot be probed on a self-contained tiny model. Probe-based checks report "unknown" for these.
PROBE_SKIP: dict[PeftType, str] = {
    PeftType.XLORA: "requires pre-trained LoRA adapter checkpoints to instantiate",
}

# Method-specific config switches that are worth surfacing as "extras". This is a curated list: reporting every config
# field would drown the relevant information in noise. Note that target_parameters is not listed here, as it is
# already covered by the target_layer_types check.
NOTABLE_CONFIG_FIELDS: tuple[str, ...] = (
    "alpha_pattern",
    "layer_replication",
    "rank_pattern",
    "use_dora",
    "use_rslora",
)

# Docs page slugs that differ from the lower-cased PEFT method name.
DOCS_SLUG_OVERRIDES: dict[str, str] = {
    "ADAPTION_PROMPT": "llama_adapter",
    "CARTRIDGE": "cartridges",
    "LN_TUNING": "layernorm_tuning",
}

# paper links as they appear in the docs intro paragraphs and in the config/model class docstrings
PAPER_URL_RE = re.compile(
    r"https://(?:huggingface\.co/papers/|arxiv\.org/(?:abs|pdf)/|openreview\.net/forum\?id=)[^\s)\"'>]+"
)


class Source(enum.StrEnum):
    INTROSPECTION = "introspection"
    FILE_CHECK = "file_check"
    PROBE = "probe"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    value: Any
    source: Source
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"value": self.value, "source": str(self.source)}
        if self.note:
            result["note"] = self.note
        return result


@dataclass(frozen=True)
class MethodInfo:
    peft_type: PeftType
    config_cls: type[PeftConfig]
    model_cls: type | None
    category: str  # "adapter", "prompt_learning", or "other"

    @property
    def name(self) -> str:
        return self.peft_type.value

    @classmethod
    def from_peft_type(cls, peft_type: PeftType) -> "MethodInfo":
        config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
        model_cls = PEFT_TYPE_TO_TUNER_MAPPING.get(peft_type)
        if issubclass(config_cls, PromptLearningConfig):
            category = "prompt_learning"
        elif (model_cls is not None) and issubclass(model_cls, BaseTuner):
            category = "adapter"
        else:
            # e.g. adaption prompt, whose model class manages adapters without subclassing BaseTuner
            category = "other"
        return cls(peft_type=peft_type, config_cls=config_cls, model_cls=model_cls, category=category)


def _layer_classes(method: MethodInfo) -> list[type[BaseTunerLayer]]:
    """Return the tuner layer classes defined in the method's main layer module.

    Quantization-specific layer variants (bnb.py etc.) are deliberately not considered: importing them depends on the
    installed quantization libraries, and the main layer module is what determines baseline support.
    """
    tuner_layer_cls = getattr(method.model_cls, "tuner_layer_cls", None)
    if tuner_layer_cls is None:
        return []
    module = sys.modules[tuner_layer_cls.__module__]
    return [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, BaseTunerLayer) and obj.__module__ == module.__name__
    ]


def _format_exception(exc: BaseException, limit: int = 250) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= limit else msg[: limit - 3] + "..."


class ProbeError(Exception):
    """Raised when a probe cannot be set up; results in an 'unknown' finding, never in a false positive/negative."""


class SingleLayerModel(nn.Module):
    """Minimal host model providing a single named module ("layer") for PEFT to target."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


@dataclass(frozen=True)
class LayerSpec:
    label: str
    build: Callable[[], nn.Module]


# The layer types whose support is probed per method. Probing checks injection only (i.e. whether the layer gets
# wrapped), no forward pass, since a successful wrap is the support signal and a broken forward would be a bug.
LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec("Linear", lambda: nn.Linear(HIDDEN_DIM, HIDDEN_DIM)),
    LayerSpec("Embedding", lambda: nn.Embedding(16, HIDDEN_DIM)),
    LayerSpec("Conv1d", lambda: nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, 3)),
    LayerSpec("Conv2d", lambda: nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 3)),
    LayerSpec("Conv3d", lambda: nn.Conv3d(HIDDEN_DIM, HIDDEN_DIM, 3)),
    LayerSpec("LayerNorm", lambda: nn.LayerNorm(HIDDEN_DIM)),
    LayerSpec("MultiheadAttention", lambda: nn.MultiheadAttention(HIDDEN_DIM, num_heads=4)),
    LayerSpec("Conv1D (transformers)", lambda: Conv1D(HIDDEN_DIM, HIDDEN_DIM)),
)


class ProbeContext:
    """Builds tiny throwaway models to exercise features on CPU.

    The tiny transformer used for prompt learning methods is constructed from a config (no download) and cached; each
    probe receives a deepcopy so that probes cannot contaminate each other.
    """

    def __init__(self) -> None:
        self._tiny_lm: nn.Module | None = None

    def make_config(self, method: MethodInfo, **kwargs: Any) -> PeftConfig:
        if method.peft_type in PROBE_SKIP:
            raise ProbeError(f"not probed: {PROBE_SKIP[method.peft_type]}")
        kwargs = PROBE_CONFIG_OVERRIDES.get(method.peft_type, {}) | kwargs
        try:
            return method.config_cls(**kwargs)
        except Exception as exc:
            raise ProbeError(
                f"could not instantiate {method.config_cls.__name__} for probing "
                f"(consider adding an entry to PROBE_CONFIG_OVERRIDES): {_format_exception(exc)}"
            ) from exc

    def _probe_layer_and_input(self, method: MethodInfo) -> tuple[nn.Module, torch.Tensor]:
        if method.peft_type == PeftType.TRAINABLE_TOKENS:
            # trainable tokens only target embedding layers
            return nn.Embedding(16, HIDDEN_DIM), torch.randint(0, 16, (2, 5))
        return nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.randn(2, HIDDEN_DIM)

    def adapter_model(self, method: MethodInfo) -> tuple[nn.Module, torch.Tensor]:
        """Return a PEFT model wrapping a single-layer host, plus a suitable example input."""
        torch.manual_seed(0)
        layer, example_input = self._probe_layer_and_input(method)
        host = SingleLayerModel(layer)
        config = self.make_config(method, target_modules=["layer"])
        try:
            return get_peft_model(host, config), example_input
        except Exception as exc:
            raise ProbeError(f"could not build probe model: {_format_exception(exc)}") from exc

    def transformer_model(self, method: MethodInfo) -> nn.Module:
        """Return a PEFT model on a tiny transformer, for methods that require one (prompt learning etc.)."""
        if self._tiny_lm is None:
            torch.manual_seed(0)
            tiny_config = LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=64,
            )
            self._tiny_lm = LlamaForCausalLM(tiny_config)
        kwargs: dict[str, Any] = {"task_type": "CAUSAL_LM"}
        if method.category == "prompt_learning":
            kwargs["num_virtual_tokens"] = 4
        config = self.make_config(method, **kwargs)
        try:
            return get_peft_model(deepcopy(self._tiny_lm), config)
        except Exception as exc:
            raise ProbeError(f"could not build probe model: {_format_exception(exc)}") from exc

    def second_config(self, method: MethodInfo) -> PeftConfig:
        """A config suitable for adding a second adapter to a model built by this context."""
        if method.category == "adapter":
            return self.make_config(method, target_modules=["layer"])
        kwargs: dict[str, Any] = {"task_type": "CAUSAL_LM"}
        if method.category == "prompt_learning":
            kwargs["num_virtual_tokens"] = 4
        return self.make_config(method, **kwargs)


class Task(ABC):
    """A single feature check for a single method. Never raises; failures become 'unknown' findings."""

    feature: ClassVar[str]
    description: ClassVar[str]

    def __init__(self, method: MethodInfo, probe: ProbeContext) -> None:
        self.method = method
        self.probe = probe

    @abstractmethod
    def check(self) -> Finding: ...

    def run(self) -> Finding:
        try:
            # Probing emits plenty of warnings that are expected and irrelevant here (e.g. about adapter
            # initialization or fan_in_fan_out). Suppression is re-asserted per task instead of once globally, since
            # libraries imported lazily during probing may manipulate the global warning filters.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.check()
        except ProbeError as exc:
            return Finding(value=UNKNOWN, source=Source.PROBE, note=str(exc))
        except Exception as exc:
            return Finding(value=UNKNOWN, source=Source.ERROR, note=_format_exception(exc))


class CategoryTask(Task):
    feature = "category"
    description = "whether the method is a layer-wrapping adapter, a prompt learning method, or something else"

    def check(self) -> Finding:
        return Finding(value=self.method.category, source=Source.INTROSPECTION)


class TargetLayerTypesTask(Task):
    feature = "target_layer_types"
    description = (
        "which layer types (incl. nn.Parameter) can be targeted, probed by injecting into single-layer models"
    )

    def check(self) -> Finding:
        if self.method.category != "adapter":
            return Finding(
                value=NOT_APPLICABLE, source=Source.INTROSPECTION, note="method does not wrap target layers"
            )

        results: dict[str, bool] = {}
        first_error: str | None = None
        for spec in LAYER_SPECS:
            torch.manual_seed(0)
            host = SingleLayerModel(spec.build())
            config = self.probe.make_config(self.method, target_modules=["layer"])
            try:
                model = get_peft_model(host, config)
            except Exception as exc:
                results[spec.label] = False
                if first_error is None:
                    first_error = f"{spec.label}: {_format_exception(exc)}"
            else:
                results[spec.label] = any(isinstance(module, BaseTunerLayer) for module in model.modules())

        if not any(results.values()):
            # if not even one layer type can be wrapped, the probe setup is likely at fault, not the method
            raise ProbeError(f"no layer type could be wrapped, probe presumably mis-configured; {first_error}")

        # Directly targeting nn.Parameter (crucial e.g. for MoE layers) is governed by the target_parameters config
        # option; its presence is the support signal, no injection probe is needed.
        field_names = {f.name for f in dataclasses.fields(self.method.config_cls)}
        results["nn.Parameter"] = "target_parameters" in field_names
        return Finding(
            value=results,
            source=Source.PROBE,
            note="nn.Parameter support is based on the presence of the target_parameters config option",
        )


class QuantizationTask(Task):
    feature = "quantization_backends"
    description = "supported quantization backends, from integration modules and use of the generic backend resolver"

    def check(self) -> Finding:
        if self.method.category == "prompt_learning":
            return Finding(
                value=NOT_APPLICABLE,
                source=Source.INTROSPECTION,
                note="prompt learning does not wrap target layers and generally works regardless of quantization",
            )
        if self.method.category != "adapter":
            return Finding(value=UNKNOWN, source=Source.INTROSPECTION, note="no known quantization signal")

        backends: set[str] = set()
        package_dir = Path(inspect.getfile(self.method.model_cls)).parent
        for stem, names in QUANT_FILE_BACKENDS.items():
            if (package_dir / f"{stem}.py").exists():
                backends.update(names)

        # methods using the generic quantization integration support all backends the resolver handles
        module_names = {self.method.model_cls.__module__}
        if (tuner_layer_cls := getattr(self.method.model_cls, "tuner_layer_cls", None)) is not None:
            module_names.add(tuner_layer_cls.__module__)
        for module_name in module_names:
            if "resolve_quantization_backend" in inspect.getsource(sys.modules[module_name]):
                backends.update(GENERIC_QUANT_BACKENDS)
                break

        return Finding(value=sorted(backends), source=Source.FILE_CHECK)


class MultipleAdaptersTask(Task):
    feature = "multiple_adapters"
    description = "whether several adapters can be loaded on the same model, probed via add_adapter"

    def check(self) -> Finding:
        if self.method.category == "adapter":
            model, _ = self.probe.adapter_model(self.method)
        else:
            model = self.probe.transformer_model(self.method)
        try:
            model.add_adapter("second", self.probe.second_config(self.method))
            model.set_adapter("second")
        except Exception as exc:
            return Finding(value=False, source=Source.PROBE, note=_format_exception(exc))
        return Finding(value=True, source=Source.PROBE)


class MixedAdapterBatchesTask(Task):
    feature = "mixed_adapter_batches"
    description = "whether one batch can mix several adapters via the adapter_names argument"

    def check(self) -> Finding:
        if self.method.category != "adapter":
            return Finding(
                value=False, source=Source.INTROSPECTION, note="only supported by layer-wrapping adapter methods"
            )
        # Support requires both halves of the mechanism: the model must install the forward hooks that distribute
        # adapter_names, and the tuner layers must implement _mixed_batch_forward (possibly inherited, e.g. from
        # LoRA). Checking only one of them would over-report.
        model_supports_hooks = hasattr(self.method.model_cls, "_enable_peft_forward_hooks")
        layer_classes = _layer_classes(self.method)
        layer_supports = any(hasattr(cls, "_mixed_batch_forward") for cls in layer_classes)
        return Finding(value=model_supports_hooks and layer_supports, source=Source.INTROSPECTION)


class MergeTask(Task):
    feature = "merging"
    description = "whether adapters can be merged into the base weights, verified via merge_and_unload"

    def check(self) -> Finding:
        if self.method.category == "prompt_learning":
            return Finding(
                value=False, source=Source.INTROSPECTION, note="virtual tokens cannot be merged into base weights"
            )
        if self.method.category != "adapter":
            return Finding(value=False, source=Source.INTROSPECTION, note="method does not implement merging")

        layer_classes = _layer_classes(self.method)
        # BaseTunerLayer.merge raises NotImplementedError, so an unchanged merge attribute means no support
        implemented = any(cls.merge is not BaseTunerLayer.merge for cls in layer_classes)
        if not implemented:
            return Finding(value=False, source=Source.INTROSPECTION, note="no tuner layer class implements merge()")

        try:
            model, example_input = self.probe.adapter_model(self.method)
            model.eval()
            with torch.no_grad():
                output_before = model(example_input)
                merged = model.merge_and_unload()
                output_after = merged(example_input)
        except ProbeError as exc:
            return Finding(
                value=True, source=Source.INTROSPECTION, note=f"merge() is implemented, but probing failed: {exc}"
            )
        except NotImplementedError as exc:
            return Finding(value=False, source=Source.PROBE, note=_format_exception(exc))
        except Exception as exc:
            return Finding(
                value=True,
                source=Source.INTROSPECTION,
                note=f"merge() is implemented, but probing failed: {_format_exception(exc)}",
            )

        note = None
        if not torch.allclose(output_before, output_after, atol=1e-4):
            note = "merged model outputs deviate from unmerged outputs beyond tolerance"
        return Finding(value=True, source=Source.PROBE, note=note)


class MixedMethodModelTask(Task):
    feature = "peft_mixed_model"
    description = "whether the method can be combined with other method types in a PeftMixedModel"

    def check(self) -> Finding:
        # filled by register_peft_method(..., is_mixed_compatible=True)
        value = self.method.peft_type in PEFT_TYPE_TO_MIXED_MODEL_MAPPING
        return Finding(value=value, source=Source.INTROSPECTION)


class LoraConversionTask(Task):
    feature = "lora_conversion"
    description = "whether adapters of this method can be converted to a LoRA adapter"

    def check(self) -> Finding:
        if self.method.peft_type == PeftType.LORA:
            return Finding(value=True, source=Source.INTROSPECTION, note="already a LoRA adapter")
        if self.method.category != "adapter":
            return Finding(value=False, source=Source.INTROSPECTION, note="method does not wrap target layers")

        try:
            model, _ = self.probe.adapter_model(self.method)
        except ProbeError as exc:
            # fall back to a static signal; an override does not guarantee support, hence the note
            layer_classes = _layer_classes(self.method)
            overridden = any(
                cls.supports_lora_conversion is not BaseTunerLayer.supports_lora_conversion for cls in layer_classes
            )
            return Finding(
                value=overridden,
                source=Source.INTROSPECTION,
                note=f"based on presence of a supports_lora_conversion override; probing failed: {exc}",
            )

        layer = next(module for module in model.modules() if isinstance(module, BaseTunerLayer))
        return Finding(value=bool(layer.supports_lora_conversion("default")), source=Source.PROBE)


class WeightedAdapterTask(Task):
    feature = "add_weighted_adapter"
    description = "whether several adapters can be combined into a new one, probed via add_weighted_adapter"

    def check(self) -> Finding:
        # Methods without the API don't support the feature; methods that inherit it but override it with a stub
        # that raises (e.g. AdaLoRA) are caught by the probe below.
        if getattr(self.method.model_cls, "add_weighted_adapter", None) is None:
            return Finding(value=False, source=Source.INTROSPECTION)

        model, _ = self.probe.adapter_model(self.method)
        # combining several adapters requires loading several adapters in the first place; methods that already fail
        # here cannot support add_weighted_adapter either
        try:
            model.add_adapter("second", self.probe.second_config(self.method))
        except Exception as exc:
            return Finding(
                value=False,
                source=Source.PROBE,
                note=f"a second adapter could not be added: {_format_exception(exc)}",
            )
        try:
            model.add_weighted_adapter(adapters=["default", "second"], weights=[0.5, 0.5], adapter_name="combined")
        except Exception as exc:
            return Finding(value=False, source=Source.PROBE, note=_format_exception(exc))
        return Finding(value=True, source=Source.PROBE)


class HotswapTask(Task):
    feature = "hotswapping"
    description = "whether adapters can be hot-swapped in place (peft.utils.hotswap)"

    def check(self) -> Finding:
        return Finding(value=self.method.peft_type in CONFIG_KEYS_TO_CHECK, source=Source.INTROSPECTION)


class AuxiliaryModulesTask(Task):
    feature = "auxiliary_modules"
    description = "whether the config supports modules_to_save and trainable_token_indices"

    def check(self) -> Finding:
        field_names = {f.name for f in dataclasses.fields(self.method.config_cls)}
        value = {
            "modules_to_save": "modules_to_save" in field_names,
            "trainable_token_indices": "trainable_token_indices" in field_names,
        }
        return Finding(value=value, source=Source.INTROSPECTION)


class ExtrasTask(Task):
    feature = "extras"
    description = "notable method-specific config options (curated list)"

    def check(self) -> Finding:
        field_names = {f.name for f in dataclasses.fields(self.method.config_cls)}
        value = sorted(field_names.intersection(NOTABLE_CONFIG_FIELDS))
        return Finding(value=value, source=Source.INTROSPECTION)


class PaperLinkTask(Task):
    feature = "paper_url"
    description = "link to the method's paper, from the docs intro or class docstrings (omitted when ambiguous)"

    # default assumes the script lives in scripts/ of a repository checkout; can be overridden via --docs-dir
    docs_dir: ClassVar[Path] = Path(__file__).parent.parent / "docs" / "source" / "package_reference"

    def _docs_intro(self) -> str | None:
        """The first paragraph after the first heading of the method's docs page (the pages start with a license
        comment, a `# Title` heading, and a prose paragraph)."""
        slug = DOCS_SLUG_OVERRIDES.get(self.method.name, self.method.name.lower())
        try:
            text = (self.docs_dir / f"{slug}.md").read_text()
        except OSError:
            return None
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        match = re.search(r"^# .+?$\s+(.+?)(?:\n\s*\n|$)", text, flags=re.MULTILINE | re.DOTALL)
        return match.group(1) if match else None

    def check(self) -> Finding:
        # Sources in order of preference; the first one containing exactly one distinct paper URL wins. A source
        # with no link, or with several different ones (e.g. docstrings that also cite related methods), is skipped
        # as ambiguous -- a missing paper link is better than a wrong one.
        sources: list[tuple[str, str | None]] = [
            ("docs intro", self._docs_intro()),
            ("config class docstring", self.method.config_cls.__doc__),
            ("model class docstring", self.method.model_cls.__doc__ if self.method.model_cls is not None else None),
        ]
        for source_name, text in sources:
            if not text:
                continue
            urls = set(PAPER_URL_RE.findall(text))
            if len(urls) == 1:
                return Finding(value=urls.pop(), source=Source.FILE_CHECK, note=f"from the {source_name}")
        return Finding(
            value=None,
            source=Source.FILE_CHECK,
            note="no unambiguous paper link in the docs intro or the config/model class docstrings",
        )


# the order here determines the order of the features in the output
TASK_CLASSES: tuple[type[Task], ...] = (
    CategoryTask,
    TargetLayerTypesTask,
    QuantizationTask,
    MultipleAdaptersTask,
    MixedAdapterBatchesTask,
    MergeTask,
    MixedMethodModelTask,
    LoraConversionTask,
    WeightedAdapterTask,
    HotswapTask,
    AuxiliaryModulesTask,
    ExtrasTask,
    PaperLinkTask,
)


def collect_methods(selected: list[str] | None) -> list[MethodInfo]:
    registered = sorted(PEFT_TYPE_TO_CONFIG_MAPPING, key=lambda peft_type: peft_type.value)
    if selected is not None:
        valid = {peft_type.value for peft_type in registered}
        requested = [name.upper() for name in selected]
        if unknown := [name for name in requested if name not in valid]:
            raise SystemExit(
                f"Unknown PEFT method(s): {', '.join(unknown)}. Valid choices: {', '.join(sorted(valid))}"
            )
        registered = [peft_type for peft_type in registered if peft_type.value in requested]
    return [MethodInfo.from_peft_type(peft_type) for peft_type in registered]


def collect_tasks(methods: list[MethodInfo], probe: ProbeContext) -> list[Task]:
    return [task_cls(method, probe) for method in methods for task_cls in TASK_CLASSES]


def run_tasks(tasks: list[Task]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task in tqdm(tasks, desc="Checking capabilities", unit="check"):
        method = task.method
        entry = results.setdefault(
            method.name,
            {
                "config_class": method.config_cls.__name__,
                "model_class": method.model_cls.__name__ if method.model_cls is not None else None,
                "features": {},
            },
        )
        entry["features"][task.feature] = task.run().to_json()
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a JSON capability matrix of all PEFT methods.",
        epilog="Example: python scripts/generate_method_capabilities.py --methods lora ia3 --output capabilities.json",
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="restrict the analysis to these PEFT methods (case-insensitive, e.g. 'lora ia3'); default: all",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("method_capabilities.json"),
        help="output JSON file (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="only list the checks that would run")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=PaperLinkTask.docs_dir,
        help="directory containing the package_reference docs pages, used for the paper link check",
    )
    args = parser.parse_args(argv)
    PaperLinkTask.docs_dir = args.docs_dir
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # logs to stderr

    methods = collect_methods(args.methods)
    probe = ProbeContext()
    tasks = collect_tasks(methods, probe)

    if args.dry_run:
        for method in methods:
            print(f"{method.name}: {', '.join(task_cls.feature for task_cls in TASK_CLASSES)}")
        print(f"\n{len(methods)} methods x {len(TASK_CLASSES)} checks = {len(tasks)} tasks")
        return

    output = {
        "schema_version": 1,
        "peft_version": peft.__version__,
        "methods": run_tasks(tasks),
    }
    # The results deliberately go to a file, not to stdout: probing can trigger subprocesses whose output is written
    # directly to stdout (e.g. BOFT compiling its CUDA extension via ninja), which would interleave with the JSON.
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    logger.info(f"Wrote capabilities of {len(methods)} methods to {args.output}")


if __name__ == "__main__":
    main()
