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
"""The PEFT shop: a Gradio app to browse PEFT methods like an online store.

Users can filter methods by their capabilities (merging, multi-adapter support, quantization backends, targetable
layer types, ...), check benchmark results (switchable between the benchmarks of the method comparison suite, e.g.
MetaMathQA or image generation), and jump to the PEFT docs. The app has two tabs: one to browse the shop, one for the
cart, which shows usage code snippets and a feature comparison table for the collected methods. In keeping with the
shop theme, every method has a (crossed-out) price tag, benchmark results double as customer star ratings, and
checkout is free.

The app reads a single `data.json` file. If that file does not exist (or --rebuild is passed), it is built by merging
three sources from the PEFT repository checkout:

1. The capability matrix produced by `scripts/generate_method_capabilities.py` (run that first; it needs an
   environment with PEFT installed, which the app itself does not).
2. The results of every benchmark registered in BENCHMARKS, aggregated to the best run (by benchmark score) per PEFT
   method. Supporting a new benchmark only requires adding a BenchmarkSpec entry, provided its result files follow
   the common JSON layout of the method comparison suite.
3. Short method descriptions, extracted from the first paragraph of each method's documentation page. Sourcing the
   descriptions from the docs instead of maintaining them here keeps them consistent with the official docs and
   avoids drift; DESCRIPTION_OVERRIDES exists for cases where the extracted text reads poorly.

A deployed Space therefore only needs `app.py`, `data.json`, and gradio, while a repository checkout can rebuild the
data on the fly. The method cards live in a fixed pool of slots created once at startup, each consisting of a card
body (HTML, since a catalog-style card look is not achievable with Gradio's native components) and a real gr.Button
to add the method to the cart; filtering and sorting update the slots' content and visibility.

Usage:

    python app.py                # load data.json, build it first if missing
    python app.py --rebuild      # force rebuilding data.json, then launch
    python app.py --build-only   # only (re)build data.json, don't launch
"""

import argparse
import hashlib
import html
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr


logger = logging.getLogger("method_explorer")

HERE = Path(__file__).parent

GIB = 1024**3

# ---------------------------------------------------------------------------------------------------------------
# data building
# ---------------------------------------------------------------------------------------------------------------

# Version of the data.json layout produced by build_data; to be bumped when the layout changes, so that loading a
# stale data.json fails loudly instead of crashing the app in odd places.
DATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MetricSpec:
    """One benchmark-specific metric, reported in the final "metrics" entry of a benchmark run."""

    field: str  # key in the final "metrics" entry of a run
    label: str  # display name
    higher_is_better: bool
    is_percent: bool = False  # whether the value is displayed as a percentage


@dataclass(frozen=True)
class BenchmarkSpec:
    """Everything the app needs to know about one benchmark of the method comparison suite.

    To add a new benchmark, append an entry to BENCHMARKS and rebuild data.json -- the benchmark dropdown, the star
    ratings, and the cart's comparison table all derive from the spec. The only requirement is that the benchmark's
    result files follow the common JSON layout of the method comparison suite (one file per run with run_info and
    train_info, the final "metrics" entry holding the benchmark-specific metrics, full fine-tuning identified by a
    missing peft_config).
    """

    key: str  # identifier used in data.json
    label: str  # display name
    model_name: str  # base model the benchmark runs on, for display
    results_subdir: str  # location of the result files, relative to the method_comparison directory
    # the benchmark-specific metrics; the first one is the headline score, used to pick each method's best run
    metrics: tuple[MetricSpec, ...]


BENCHMARKS = (
    BenchmarkSpec(
        key="metamathqa",
        label="MetaMathQA",
        model_name="Llama-3.2-3B",
        results_subdir="MetaMathQA/results",
        metrics=(
            MetricSpec(field="test accuracy", label="test accuracy", higher_is_better=True, is_percent=True),
            MetricSpec(field="forgetting", label="forgetting", higher_is_better=False),
        ),
    ),
    BenchmarkSpec(
        key="image-gen",
        label="Image generation",
        model_name="FLUX.2-klein-4B",
        results_subdir="image-gen/results",
        metrics=(
            MetricSpec(field="test dino_similarity", label="DINO similarity", higher_is_better=True),
            MetricSpec(field="drift", label="drift", higher_is_better=False),
        ),
    ),
)
BENCHMARKS_BY_KEY = {spec.key: spec for spec in BENCHMARKS}

# Docs page slugs that differ from the lower-cased PEFT method name.
DOCS_SLUG_OVERRIDES = {
    "ADAPTION_PROMPT": "llama_adapter",
    "CARTRIDGE": "cartridges",
    "LN_TUNING": "layernorm_tuning",
}
DOCS_BASE_URL = "https://huggingface.co/docs/peft/main/en/package_reference"

# Manual replacements for method descriptions where the text extracted from the docs is unsuitable. Maps the PEFT
# method name (upper case) to the description.
DESCRIPTION_OVERRIDES: dict[str, str] = {}

BENCHMARK_SPACE_URL = "https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison"


def extract_description(docs_path: Path) -> str | None:
    """Extract a short description from a method's documentation page.

    The package_reference pages consistently start with a license comment, a `# Title` heading, and then a prose
    paragraph describing the method. That first paragraph (clipped to at most two sentences) makes a good card
    description. Markdown links are flattened to their text.
    """
    try:
        text = docs_path.read_text()
    except OSError:
        return None

    # drop the HTML license comment, then find the first paragraph after the first heading
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    match = re.search(r"^# .+?$\s+(.+?)(?:\n\s*\n|$)", text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        return None
    paragraph = " ".join(match.group(1).split())
    paragraph = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", paragraph)  # [text](url) -> text

    # clip to at most two sentences to keep the cards compact
    sentences = re.split(r"(?<=[.!?]) ", paragraph)
    return " ".join(sentences[:2]).strip() or None


def load_benchmark_results(
    results_dir: Path, spec: BenchmarkSpec
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Aggregate one benchmark's runs to one entry per PEFT method.

    Per method, the run with the best headline score is kept: a benchmark can contain multiple hyper-parameter
    settings per method and users browsing methods are interested in what a method can achieve, not in its worst
    setting. The full fine-tuning run is returned separately as a baseline for comparison.
    """
    score_field = spec.metrics[0].field
    best_per_method: dict[str, dict[str, Any]] = {}
    baseline: dict[str, Any] | None = None

    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text())
        run_info = result["run_info"]
        train_info = result["train_info"]
        if train_info.get("status") != "success":
            continue
        metrics = train_info.get("metrics") or []
        last = metrics[-1] if metrics else {}
        if last.get(score_field) is None:
            logger.warning(f"Skipping {path.name}: no '{score_field}' found")
            continue

        entry = {"experiment_name": run_info["experiment_name"]}
        for metric in spec.metrics:
            value = last.get(metric.field)
            # e.g. the drift metric of the image generation benchmark is NaN for the full fine-tuning run; store
            # null instead so that data.json remains valid JSON
            if isinstance(value, float) and math.isnan(value):
                value = None
            entry[metric.field] = value
        entry |= {
            "peak_memory_bytes": train_info["accelerator_memory_max"],
            "train_time_sec": train_info["train_time"],
            "num_trainable_params": train_info["num_trainable_params"],
            "adapter_file_size_bytes": train_info["file_size"],
            "num_runs": 1,
        }

        peft_config = run_info.get("peft_config")
        if not peft_config:  # full fine-tuning has no PEFT config
            baseline = entry
            continue

        method = peft_config["peft_type"]
        previous = best_per_method.get(method)
        if previous is None:
            best_per_method[method] = entry
        else:
            entry["num_runs"] = previous["num_runs"] + 1
            if entry[score_field] >= previous[score_field]:
                best_per_method[method] = entry
            else:
                previous["num_runs"] = entry["num_runs"]

    return best_per_method, baseline


def build_data(capabilities_path: Path, benchmarks_dir: Path, docs_dir: Path) -> dict[str, Any]:
    capabilities = json.loads(capabilities_path.read_text())

    baselines: dict[str, dict[str, Any] | None] = {}
    method_benchmarks: dict[str, dict[str, Any]] = {}
    for spec in BENCHMARKS:
        best_per_method, baseline = load_benchmark_results(benchmarks_dir / spec.results_subdir, spec)
        baselines[spec.key] = baseline
        for method, entry in best_per_method.items():
            method_benchmarks.setdefault(method, {})[spec.key] = entry

    methods = {}
    for name, info in sorted(capabilities["methods"].items()):
        slug = DOCS_SLUG_OVERRIDES.get(name, name.lower())
        docs_path = docs_dir / f"{slug}.md"
        description = DESCRIPTION_OVERRIDES.get(name) or extract_description(docs_path)
        if description is None:
            logger.warning(f"No description found for {name} (looked at {docs_path})")
            description = "See the PEFT documentation for details on this method."

        # The paper link is determined by scripts/generate_method_capabilities.py (the "paper_url" feature). Older
        # capability files may not contain it, and its value can be None or "unknown" -- only accept real URLs.
        paper_value = info["features"].get("paper_url", {}).get("value")
        paper_url = paper_value if isinstance(paper_value, str) and paper_value.startswith("http") else None

        methods[name] = {
            "config_class": info["config_class"],
            "model_class": info["model_class"],
            "features": info["features"],
            "description": description,
            # link to the index page when no dedicated docs page exists (e.g. brand-new methods)
            "docs_url": f"{DOCS_BASE_URL}/{slug}" if docs_path.exists() else f"{DOCS_BASE_URL}/tuners",
            "paper_url": paper_url,
            "benchmarks": method_benchmarks.pop(name, {}),
        }

    for leftover in method_benchmarks:
        logger.warning(f"Benchmark results for {leftover} have no matching capability entry, ignored")

    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "peft_version": capabilities["peft_version"],
        "baselines": baselines,
        "methods": methods,
    }


# ---------------------------------------------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------------------------------------------

# Populated once at startup by _set_data; module-level state keeps the many small helpers below free of an explicit
# `data` parameter.
DATA: dict[str, Any] = {}
METHODS: dict[str, Any] = {}


def _set_data(data: dict[str, Any]) -> None:
    global DATA, METHODS
    DATA = data
    METHODS = data["methods"]


CATEGORY_LABELS = {
    "adapter": "Adapter",
    "prompt_learning": "Prompt learning",
    "other": "Other",
}

# boolean capability features offered as filters; (label, feature key)
CAPABILITIES = [
    ("Mergeable into base weights", "merging"),
    ("Multiple adapters", "multiple_adapters"),
    ("Mixed adapter batches", "mixed_adapter_batches"),
    ("Convertible to LoRA", "lora_conversion"),
    ("Weighted adapter combination", "add_weighted_adapter"),
    ("Hot-swappable", "hotswapping"),
    ("Mixable with other methods", "peft_mixed_model"),
    ("modules_to_save", "aux:modules_to_save"),
    ("trainable_token_indices", "aux:trainable_token_indices"),
]

# the subset of capabilities rendered as badges on the cards, to keep them scannable
SHORT_BADGES = [
    ("merge", "merging"),
    ("multi-adapter", "multiple_adapters"),
    ("mixed batches", "mixed_adapter_batches"),
    ("→ LoRA", "lora_conversion"),
    ("weighted combine", "add_weighted_adapter"),
    ("hotswap", "hotswapping"),
]

# Plain-language explanations of the capabilities, shown when hovering over the badges. Per capability: the text for
# a supporting method, the text for a non-supporting method, and a neutral "whether ..." phrase for unknown support.
CAPABILITY_EXPLAINERS: dict[str, tuple[str, str, str]] = {
    "merging": (
        "This method's adapter weights can be merged into the base model, eliminating the adapter's inference "
        "overhead.",
        "This method's adapter weights cannot be merged into the base model, so the adapter always incurs some "
        "inference overhead.",
        "whether the adapter weights can be merged into the base model",
    ),
    "multiple_adapters": (
        "Several adapters of this method can be loaded on the same model at the same time and switched between.",
        "Only a single adapter of this method can be loaded on a model at a time.",
        "whether several adapters can be loaded on the same model",
    ),
    "mixed_adapter_batches": (
        "A single inference batch can mix samples that use different adapters, via the adapter_names argument.",
        "All samples in a batch must use the same adapter; the adapter_names argument is not supported.",
        "whether one batch can mix samples that use different adapters",
    ),
    "lora_conversion": (
        "A trained adapter of this method can be converted into an (approximately) equivalent LoRA adapter.",
        "A trained adapter of this method cannot be converted into a LoRA adapter.",
        "whether a trained adapter can be converted into a LoRA adapter",
    ),
    "add_weighted_adapter": (
        "Several trained adapters can be combined into a new adapter via a weighted combination.",
        "Combining several adapters into a new one (add_weighted_adapter) is not supported.",
        "whether several adapters can be combined into a new one",
    ),
    "hotswapping": (
        "Adapter weights can be swapped in-place without re-creating the model, e.g. to avoid torch.compile "
        "recompilation.",
        "Adapter weights cannot be hot-swapped; loading different weights requires re-creating the adapter.",
        "whether adapter weights can be hot-swapped in-place",
    ),
    "peft_mixed_model": (
        "This method can be combined with adapters of other PEFT method types on the same model (PeftMixedModel).",
        "This method cannot be combined with other PEFT method types on the same model (PeftMixedModel).",
        "whether the method can be combined with other PEFT method types",
    ),
    "aux:modules_to_save": (
        "Additional base model layers (e.g. a classification head) can be made trainable and stored together with "
        "the adapter (modules_to_save).",
        "The modules_to_save option for training additional base model layers is not supported.",
        "whether additional base model layers can be trained via modules_to_save",
    ),
    "aux:trainable_token_indices": (
        "Selected token embeddings can be trained alongside the adapter without training the full embedding matrix "
        "(trainable_token_indices).",
        "The trainable_token_indices option for training selected token embeddings is not supported.",
        "whether selected token embeddings can be trained via trainable_token_indices",
    ),
}


def explain_capability(key: str, value: bool | None, note: str | None = None) -> str:
    """The hover text for a capability badge, matching the badge's value."""
    supported, unsupported, neutral = CAPABILITY_EXPLAINERS[key]
    if value is True:
        return supported
    if value is False:
        return unsupported
    text = f"It could not be determined {neutral}."
    if note:
        text += f" Reason: {note}"
    return text


SORT_CHOICES = [
    ("name", "name"),
    ("benchmark score (best first)", "score"),
    ("peak memory (lowest first)", "memory"),
    ("trainable parameters (fewest first)", "params"),
    ("checkpoint size (smallest first)", "size"),
    ("train time (fastest first)", "time"),
]


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def feature(method: str, key: str) -> dict:
    return METHODS[method]["features"][key]


def capability_value(method: str, key: str) -> bool | None:
    """Return True/False for a boolean capability, None if unknown or not applicable."""
    if key.startswith("aux:"):
        aux = feature(method, "auxiliary_modules")["value"]
        return bool(aux[key.removeprefix("aux:")]) if isinstance(aux, dict) else None
    value = feature(method, key)["value"]
    return value if isinstance(value, bool) else None


def layer_types(method: str) -> dict[str, bool] | None:
    value = feature(method, "target_layer_types")["value"]
    return value if isinstance(value, dict) else None


def quant_backends(method: str) -> list[str] | None:
    """The supported quantization backends; None means "any".

    Prompt learning methods report quantization as "not_applicable" because they don't wrap layers at all and
    therefore work with (almost) any quantized base model -- treat that as supporting every backend.
    """
    value = feature(method, "quantization_backends")["value"]
    if value == "not_applicable":
        return None
    return value if isinstance(value, list) else []


def supports_quant(method: str, backend: str) -> bool:
    backends = quant_backends(method)
    return True if backends is None else backend in backends


# ---------------------------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------------------------


def fmt_bytes(num: float) -> str:
    return f"{num / GIB:.1f} GB"


def fmt_megabytes(num: float) -> str:
    return f"{num / 1024**2:.1f} MB"


def fmt_params(num: float) -> str:
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    if num >= 1e6:
        return f"{num / 1e6:.1f}M"
    return f"{num / 1e3:.0f}k"


def fmt_minutes(seconds: float) -> str:
    return f"{round(seconds / 60)} min"


def fmt_metric(metric: MetricSpec, value: float) -> str:
    return f"{100 * value:.1f}%" if metric.is_percent else f"{value:.3f}"


# Deterministic "random" typography for the method names, to give the tiles an individual look: separate bytes of a
# hash of the method name pick the color, font, and extra style from these fixed lists. md5 is used instead of the
# builtin hash(), which is salted per process and would not be stable between sessions. All colors are mid-tone so
# they stay readable on light and dark backgrounds; all fonts are common system fonts, no webfonts needed.
TITLE_COLORS = ("#e74c3c", "#e67e22", "#d4a017", "#10b981", "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899")
TITLE_FONTS = (
    "Georgia, serif",
    "'Palatino Linotype', Palatino, serif",
    "'Courier New', Courier, monospace",
    "Verdana, Geneva, sans-serif",
    "'Trebuchet MS', sans-serif",
    "Impact, 'Arial Black', sans-serif",
)
TITLE_STYLES = (
    "",
    "font-style: italic;",
    "text-shadow: 2px 2px 3px rgba(0, 0, 0, 0.3);",
    "letter-spacing: 0.12em;",
    "border-bottom: 3px dotted currentcolor;",
)


def title_style(method: str) -> str:
    digest = hashlib.md5(method.encode()).digest()
    color = TITLE_COLORS[digest[0] % len(TITLE_COLORS)]
    font = TITLE_FONTS[digest[1] % len(TITLE_FONTS)]
    extra = TITLE_STYLES[digest[2] % len(TITLE_STYLES)]
    return f"color: {color}; font-family: {font}; {extra}"


def fantasy_price(method: str) -> float:
    """The shop-themed, hash-derived fantasy price of a method; always displayed crossed-out in favor of "FREE"."""
    digest = hashlib.md5(method.encode()).digest()
    return 9.99 + 10 * (digest[3] % 10)


def price_html(method: str) -> str:
    """The shop-themed price tag: the crossed-out fantasy price next to "FREE"."""
    return f'<span class="price"><s>${fantasy_price(method):.2f}</s> <span class="free">FREE</span></span>'


def _stars_span(n_stars: int, title: str) -> str:
    return f'<span class="stars" title="{esc(title)}">{"★" * n_stars}{"☆" * (5 - n_stars)}</span>'


def benchmark_stars(spec: BenchmarkSpec, field: str, value: float, title: str, lower_is_better: bool = False) -> str:
    """A benchmark metric as a customer star rating, based on quantiles among the benchmarked PEFT methods.

    The best 20% of the methods get five stars, the next 20% four, and so on; even the worst method keeps one star.
    Quantiles are used instead of e.g. min-max scaling so that a single outlier cannot compress everyone else's
    rating. The full fine-tuning baseline is deliberately not part of the ranking -- it would hand every PEFT method
    five memory/time stars while capping the achievable score rating.
    """
    values = [info["benchmarks"][spec.key][field] for info in METHODS.values() if spec.key in info["benchmarks"]]
    rank = sum(other < value if lower_is_better else other > value for other in values)
    n_stars = 5 - int(5 * rank / len(values))
    return _stars_span(n_stars, title)


def badge(value: bool | None, label: str, title: str | None = None) -> str:
    cls = "yes" if value is True else "no" if value is False else "unknown"
    mark = "✓" if value is True else "✗" if value is False else "?"
    return f'<span class="badge {cls}" title="{esc(title or "")}">{mark} {esc(label)}</span>'


def render_card(method: str, spec: BenchmarkSpec) -> str:
    info = METHODS[method]
    category = feature(method, "category")["value"]
    bench = info["benchmarks"].get(spec.key)

    # hovering over a badge explains what its value means; for "?" badges, the finding's note states why support
    # could not be determined
    badge_list = []
    for label, key in SHORT_BADGES:
        value = capability_value(method, key)
        badge_list.append(badge(value, label, explain_capability(key, value, feature(method, key).get("note"))))
    badges = "".join(badge_list)

    backends = quant_backends(method)
    if backends is None:
        quant_html = '<span class="chip">any quantization</span>'
    elif backends:
        quant_html = "".join(f'<span class="chip">{esc(b)}</span>' for b in backends)
    else:
        quant_html = '<span class="chip muted">no quantized training</span>'

    if bench:
        baseline = DATA["baselines"][spec.key]
        score = spec.metrics[0]
        reference = (
            f" (for reference, full fine-tuning reaches {fmt_metric(score, baseline[score.field])})"
            if baseline
            else ""
        )

        # per row: field in the benchmark entry, display label, formatted value, description for the hover text,
        # whether lower values are better, and an optional reference note
        bench_rows = [
            (
                metric.field,
                metric.label,
                fmt_metric(metric, bench[metric.field]),
                f"{metric.label} on the PEFT {spec.label} benchmark",
                not metric.higher_is_better,
                reference if metric is score else "",
            )
            for metric in spec.metrics
        ]
        bench_rows += [
            (
                "peak_memory_bytes",
                "max memory allocated",
                fmt_bytes(bench["peak_memory_bytes"]),
                f"Peak accelerator memory while training on the PEFT {spec.label} benchmark",
                True,
                "",
            ),
            (
                "adapter_file_size_bytes",
                "checkpoint size",
                fmt_megabytes(bench["adapter_file_size_bytes"]),
                f"Size of the saved checkpoint on the PEFT {spec.label} benchmark",
                True,
                "",
            ),
            (
                "train_time_sec",
                "train time",
                fmt_minutes(bench["train_time_sec"]),
                f"Training time on the PEFT {spec.label} benchmark",
                True,
                "",
            ),
        ]
        # the hover text sits on the whole row (and, redundantly, on the stars span inside it), so hovering the
        # metric name or the value explains the metric as well
        row_html = []
        for field, label, value, description, lower_is_better, ref in bench_rows:
            direction = "lower is better" if lower_is_better else "higher is better"
            title = f"{description}; {direction}{ref}. Stars rank the method among the other PEFT methods."
            stars = benchmark_stars(spec, field, bench[field], title, lower_is_better=lower_is_better)
            row_html.append(
                f'<div class="bench-row" title="{esc(title)}"><span>{stars} {esc(label)}:</span>'
                f"<strong>{esc(value)}</strong></div>"
            )
        bench_html = f"""
        <div class="bench" title="Best of {bench["num_runs"]} run(s): {esc(bench["experiment_name"])}">
          {"".join(row_html)}
        </div>"""
    else:
        bench_html = f'<div class="bench muted">No reviews yet (no {esc(spec.label)} benchmark results).</div>'

    # The description is clamped to three lines by CSS. Texts long enough to likely be clamped get a button that
    # shows the full text in an overlay, using the native HTML popover API -- a real centered overlay with backdrop
    # and click-outside dismissal, without any JS or Gradio event machinery.
    description = info["description"]
    more_html = ""
    if len(description) > 160:
        more_html = (
            f'<button class="more-link" popovertarget="desc-{esc(method)}">Show full description</button>'
            f'<div popover id="desc-{esc(method)}" class="desc-popover">'
            f'<h4 style="{title_style(method)}">{esc(method)}</h4><p>{esc(description)}</p></div>'
        )

    paper_url = info.get("paper_url")  # .get: tolerate a data.json built before paper links were added
    paper_html = (
        f'<a class="cta secondary" href="{esc(paper_url)}" target="_blank" rel="noopener">Paper ↗</a>'
        if paper_url
        else ""
    )

    return f"""
    <article class="card">
      <h3 style="{title_style(method)}">{esc(method)}</h3>
      <p class="description">{esc(description)}</p>
      {more_html}
      <div class="badges"><span class="chip category">{esc(CATEGORY_LABELS.get(category, category))}</span>{badges}</div>
      <div class="quant-row">{quant_html}</div>
      {bench_html}
      <div class="card-footer">
        <a class="cta" href="{esc(info["docs_url"])}" target="_blank" rel="noopener">Docs ↗</a>
        {paper_html}
        {price_html(method)}
      </div>
    </article>"""


def matches_filters(
    method: str,
    search: str,
    categories: list[str],
    capabilities: list[str],
    layers: list[str],
    quant: list[str],
    benchmarked_only: bool,
    bench_key: str,
) -> bool:
    """Filter semantics ("e-commerce" style):

    - within "category" and "quantization": OR (any selected value matches)
    - within "capabilities" and "layer types": AND (the method must support everything selected, like mandatory
      product features)
    - across filter groups: AND
    - values reported as "unknown" by the capability script never match a positive filter: users filtering for a
      feature should only see methods where support is established.
    """
    info = METHODS[method]
    if search:
        haystack = f"{method} {info['config_class']} {info['description']}".lower()
        if search.lower() not in haystack:
            return False

    if categories and feature(method, "category")["value"] not in categories:
        return False

    if any(capability_value(method, key) is not True for key in capabilities):
        return False

    if layers:
        supported = layer_types(method)
        if supported is None or any(not supported.get(layer) for layer in layers):
            return False

    if quant and not any(supports_quant(method, backend) for backend in quant):
        return False

    return not (benchmarked_only and bench_key not in info["benchmarks"])


def sort_key(sort_by: str, bench_key: str):
    score_field = BENCHMARKS_BY_KEY[bench_key].metrics[0].field
    metric = {
        "score": lambda b: -b[score_field],
        "memory": lambda b: b["peak_memory_bytes"],
        "params": lambda b: b["num_trainable_params"],
        "size": lambda b: b["adapter_file_size_bytes"],
        "time": lambda b: b["train_time_sec"],
    }.get(sort_by)
    if metric is None:  # sort by name
        return lambda method: (0, 0, method)

    # methods without results on the selected benchmark sort last for metric-based sorts
    def key(method: str):
        bench = METHODS[method]["benchmarks"].get(bench_key)
        return (bench is None, metric(bench) if bench else 0, method)

    return key


ADD_TO_CART_LABEL = "🛒 Add to cart"
IN_CART_LABEL = "✅ In cart"


def update_cards(search, categories, capabilities, layers, quant, benchmarked_only, bench_key, sort_by, cart):
    """Assign the filtered, sorted methods to the fixed pool of card slots.

    Returns the count markdown followed by (visibility, card HTML, method name, add button) for every slot; the
    button label shows whether the slot's method is already in the cart. Slots beyond the number of matching methods
    are hidden and get an empty method name, which add_to_cart treats as a no-op.
    """
    spec = BENCHMARKS_BY_KEY[bench_key]
    cart = cart or []
    selected = [
        method
        for method in METHODS
        if matches_filters(method, search, categories, capabilities, layers, quant, benchmarked_only, bench_key)
    ]
    selected.sort(key=sort_key(sort_by, bench_key))
    count = f"**{len(selected)} of {len(METHODS)} items** — all free, all in stock"
    if not selected:
        count = f"**0 of {len(METHODS)} items** — no method matches the current filters."
    # gr.update() is used here on purpose: updating layout blocks like Column through component constructors proved
    # to be silently ignored
    updates: list = [count]
    for i in range(len(METHODS)):
        if i < len(selected):
            method = selected[i]
            card = f'<div class="explorer">{render_card(method, spec)}</div>'
            button_label = IN_CART_LABEL if method in cart else ADD_TO_CART_LABEL
            updates.extend([gr.update(visible=True), card, method, gr.update(value=button_label)])
        else:
            updates.extend([gr.update(visible=False), "", "", gr.update(value=ADD_TO_CART_LABEL)])
    return updates


def reset_filters():
    """Default values for all filter components, in the order of filter_inputs.

    Programmatically resetting the components triggers their change listeners, which re-render the cards.
    """
    return "", [], [], [], [], False, BENCHMARKS[0].key, "name"


# ---------------------------------------------------------------------------------------------------------------
# cart
# ---------------------------------------------------------------------------------------------------------------

# Config arguments used in the cart's usage snippets. Methods not listed get generic arguments based on their
# category; these are demo values and the snippet says so.
SNIPPET_CONFIG_ARGS = {
    "ADALORA": 'target_modules=["q_proj", "v_proj"], total_step=1000',
    "ADAPTION_PROMPT": 'adapter_len=16, adapter_layers=8, task_type="CAUSAL_LM"',
    "TRAINABLE_TOKENS": 'target_modules=["embed_tokens"], token_indices=[0, 1]',
}

# Complete snippet replacements for methods whose usage does not follow the common pattern at all.
SNIPPET_OVERRIDES = {
    "XLORA": """\
from transformers import AutoModelForCausalLM
from peft import XLoraConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype="bfloat16")
# X-LoRA learns to mix already trained LoRA adapters
config = XLoraConfig(
    task_type="CAUSAL_LM",
    adapters={"adapter_0": "path/to/lora_0", "adapter_1": "path/to/lora_1"},
)
model = get_peft_model(base_model, config)""",
}

EMPTY_CART_SNIPPET = '# Your cart is empty - add PEFT methods in the "Browse methods" tab to see how to use them.'


def usage_snippet(method: str) -> str:
    if method in SNIPPET_OVERRIDES:
        return SNIPPET_OVERRIDES[method]
    info = METHODS[method]
    if method in SNIPPET_CONFIG_ARGS:
        args = SNIPPET_CONFIG_ARGS[method]
    elif feature(method, "category")["value"] == "prompt_learning":
        args = 'task_type="CAUSAL_LM", num_virtual_tokens=20'
    else:
        args = 'target_modules=["q_proj", "v_proj"]'
    return f"""\
from transformers import AutoModelForCausalLM
from peft import {info["config_class"]}, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype="bfloat16")
config = {info["config_class"]}({args})  # adjust the arguments to your needs
model = get_peft_model(base_model, config)
model.print_trainable_parameters()

# train the model with your favorite training loop or trainer, then:
model.save_pretrained("my-{method.lower()}-adapter")"""


def build_comparison(methods: list[str], spec: BenchmarkSpec) -> str:
    """The cart's feature comparison table: feature sets as rows, the collected methods as columns.

    The table is wrapped in a horizontally scrollable container (see the .compare-scroll CSS), since any number of
    methods can be in the cart.
    """
    if not methods:
        return '<div class="explorer"><p class="muted">Add methods to the cart to compare their features.</p></div>'

    def row(label: str, cells: list[str]) -> str:
        return f"<tr><th scope='row'>{esc(label)}</th>{''.join(f'<td>{c}</td>' for c in cells)}</tr>"

    rows = [row("Category", [esc(CATEGORY_LABELS[feature(m, "category")["value"]]) for m in methods])]
    for label, key in CAPABILITIES:
        cells = []
        for m in methods:
            value = capability_value(m, key)
            cells.append(badge(value, "", explain_capability(key, value)))
        rows.append(row(label, cells))

    def layers_cell(method: str) -> str:
        supported = layer_types(method)
        if supported is None:
            return '<span class="muted">n/a</span>'
        return ", ".join(esc(name) for name, ok in supported.items() if ok) or "—"

    def quant_cell(method: str) -> str:
        backends = quant_backends(method)
        return "any" if backends is None else ", ".join(map(esc, backends)) or "—"

    rows.append(row("Layer types", [layers_cell(m) for m in methods]))
    rows.append(row("Quantization", [quant_cell(m) for m in methods]))

    bench_rows = [
        # the default argument binds the current metric, as default-less closures in a loop would all end up
        # referring to the last one
        (metric.label[0].upper() + metric.label[1:], lambda b, metric=metric: fmt_metric(metric, b[metric.field]))
        for metric in spec.metrics
    ]
    bench_rows += [
        ("Peak memory", lambda b: fmt_bytes(b["peak_memory_bytes"])),
        ("Trainable params", lambda b: fmt_params(b["num_trainable_params"])),
        ("Checkpoint size", lambda b: fmt_megabytes(b["adapter_file_size_bytes"])),
        ("Train time", lambda b: fmt_minutes(b["train_time_sec"])),
    ]
    rows.append(f"<tr><th scope='row' class='section'>{esc(spec.label)} benchmark</th></tr>")
    for label, fmt in bench_rows:
        cells = [
            esc(fmt(b)) if (b := METHODS[m]["benchmarks"].get(spec.key)) else '<span class="muted">—</span>'
            for m in methods
        ]
        rows.append(row(label, cells))
    rows.append(
        row("Docs", [f'<a href="{esc(METHODS[m]["docs_url"])}" target="_blank" rel="noopener">↗</a>' for m in methods])
    )

    header = "".join(f"<th>{esc(m)}</th>" for m in methods)
    return f"""
    <div class="explorer compare-scroll">
      <table class="compare-table">
        <thead><tr><th></th>{header}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


def build_receipt(methods: list[str]) -> str:
    """The order confirmation opened by the pay button: a floating window (the same native popover mechanism as the
    full-description overlays on the cards) showing a shop-style receipt with the crossed-out fantasy prices."""
    if methods:
        rows = "".join(
            f"<tr><td>{esc(method)}</td><td><s>${fantasy_price(method):.2f}</s></td>"
            f'<td><span class="free">FREE</span></td></tr>'
            for method in methods
        )
        total = sum(fantasy_price(method) for method in methods)
        body = f"""
        <table class="receipt-table">{rows}</table>
        <div class="receipt-total">
          <span>Total ({len(methods)} item{"s" if len(methods) != 1 else ""})</span>
          <span><s>${total:.2f}</s> $0.00</span>
        </div>
        <p>📦 Estimated delivery: today. For express delivery: <code>pip install peft</code></p>
        <p class="muted">Thank you for shopping parameter-efficiently! 🤗</p>"""
    else:
        body = (
            "<p>Your cart is empty — but don't worry, even a full one would be free. "
            "Add methods in the “Browse the shop” tab.</p>"
        )
    return (
        '<div class="explorer"><div popover id="pay-receipt" class="receipt-popover">'
        f"<h4>🧾 Order receipt — PEFT Shop</h4>{body}</div></div>"
    )


def update_cart(methods: list[str], bench_key: str) -> tuple[str, str, str, dict]:
    """Return the cart's code snippet, the feature comparison table, the (hidden) pay receipt, and the cart tab
    label, which shows the number of collected methods."""
    spec = BENCHMARKS_BY_KEY[bench_key]
    methods = methods or []
    tab = gr.update(label=f"🛒 Cart ({len(methods)})" if methods else "🛒 Cart")
    if not methods:
        return EMPTY_CART_SNIPPET, build_comparison([], spec), build_receipt([]), tab
    snippet = "\n\n\n".join(f"# ========== {method} ==========\n{usage_snippet(method)}" for method in methods)
    return snippet, build_comparison(methods, spec), build_receipt(methods), tab


def add_to_cart(cart: list[str] | None, method: str) -> list[str]:
    cart = list(cart or [])
    if method and method not in cart:
        cart.append(method)
        gr.Info(f"{method} added to the cart 🛒")
    return cart


# Styling for the HTML-rendered parts. The palette is mapped to Gradio's own theme CSS variables where possible, so
# light and dark mode adapt automatically (a hand-rolled `.dark` override proved unreliable). The badge and chip
# backgrounds use translucent colors that work on top of both light and dark surfaces.
CSS = """
.explorer, .method-card { --x-surface: var(--background-fill-primary);
  --x-surface-2: var(--background-fill-secondary); --x-text: var(--body-text-color);
  --x-muted: var(--body-text-color-subdued); --x-border: var(--border-color-primary);
  --x-accent: var(--color-accent); --x-accent-soft: rgba(255, 157, 0, 0.16);
  --x-yes: #22c55e; --x-yes-bg: rgba(34, 197, 94, 0.16); --x-no: #ef4444; --x-no-bg: rgba(239, 68, 68, 0.14);
  --x-unknown: #9ca3af; --x-unknown-bg: rgba(156, 163, 175, 0.2); color: var(--x-text); }

.explorer .card { display: flex; flex-direction: column; gap: 0.6rem; }
.explorer .card h3 { margin: 0; font-size: 1.5rem; line-height: 1.15; letter-spacing: 0.02em; }
.explorer .description { margin: 0; color: var(--x-muted); font-size: 0.86rem; display: -webkit-box;
  -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.explorer .badges, .explorer .quant-row { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.explorer .badge { font-size: 0.74rem; padding: 0.12rem 0.45rem; border-radius: 999px; white-space: nowrap; }
.explorer .badge.yes { color: var(--x-yes); background: var(--x-yes-bg); }
.explorer .badge.no { color: var(--x-no); background: var(--x-no-bg); opacity: 0.85; }
.explorer .badge.unknown { color: var(--x-unknown); background: var(--x-unknown-bg); }
.explorer .chip { font-size: 0.72rem; padding: 0.1rem 0.45rem; border-radius: 6px; background: var(--x-surface-2);
  border: 1px solid var(--x-border); white-space: nowrap; }
.explorer .chip.category { background: var(--x-accent-soft); border-color: transparent; color: var(--x-accent);
  font-weight: 600; }
.explorer .stars { color: #f5b50a; letter-spacing: 0.05em; }
.explorer .price { margin-left: auto; align-self: center; font-size: 0.85rem; color: var(--x-muted); }
.explorer .free { color: var(--x-accent); font-weight: 800; margin-left: 0.2rem; }
.explorer .muted { color: var(--x-muted); }
.explorer .bench { background: var(--x-surface-2); border-radius: 8px; padding: 0.6rem 0.7rem; font-size: 0.84rem; }
.explorer .bench-row { display: flex; justify-content: space-between; }
.explorer .card-footer { margin-top: auto; display: flex; gap: 0.5rem; }
.explorer .cta { display: inline-block; background: var(--x-accent); color: #1f2328; text-decoration: none;
  font-weight: 600; font-size: 0.85rem; padding: 0.35rem 0.8rem; border-radius: 8px; }
.explorer .cta.secondary { background: var(--x-surface-2); color: var(--x-text);
  border: 1px solid var(--x-border); }

/* "show full description" button and its overlay (native HTML popover, rendered in the browser's top layer) */
.explorer .more-link { background: none; border: none; color: var(--x-accent); cursor: pointer; padding: 0;
  font-size: 0.8rem; text-align: left; }
.explorer .desc-popover { max-width: min(540px, 90vw); padding: 1.2rem 1.4rem; border: 1px solid var(--x-border);
  border-radius: 12px; background: var(--x-surface); color: var(--x-text); }
.explorer .desc-popover h4 { margin: 0 0 0.6rem; font-size: 1.3rem; }
.explorer .desc-popover p { margin: 0; line-height: 1.55; }
.explorer .desc-popover::backdrop { background: rgba(0, 0, 0, 0.45); }

/* The pay button's order receipt -- like the description overlay, a native popover in the browser's top layer.
   Both popovers are re-centered explicitly: the browser stylesheet would center them via `margin: auto`, but
   Gradio's styles interfere with that, leaving the popover stuck at the viewport edge. Centering via
   top/left/transform (with !important) depends on fewer properties and survives the interference. */
.explorer .desc-popover, .explorer .receipt-popover { position: fixed !important; top: 50% !important;
  left: 50% !important; bottom: auto !important; right: auto !important; margin: 0 !important;
  transform: translate(-50%, -50%); max-height: 85vh; overflow-y: auto; }
.explorer .receipt-popover { min-width: min(480px, 92vw); padding: 1.6rem 2rem; border: 1px solid var(--x-border);
  border-radius: 14px; background: var(--x-surface); color: var(--x-text); font-size: 1rem;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.4); }
.explorer .receipt-popover::backdrop { background: rgba(0, 0, 0, 0.55); }
.explorer .receipt-popover h4 { margin: 0 0 1rem; font-size: 1.5rem; }
.explorer .receipt-popover s { color: var(--x-muted); }
.explorer .receipt-popover p { margin: 0.9rem 0 0; }
.explorer .receipt-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.explorer .receipt-table td { padding: 0.25rem 0; }
.explorer .receipt-table td:nth-child(n+2) { text-align: right; padding-left: 1.4rem; white-space: nowrap; }
.explorer .receipt-total { display: flex; justify-content: space-between; gap: 1.2rem; font-weight: 700;
  border-top: 1px solid var(--x-border); margin-top: 0.6rem; padding-top: 0.6rem; }

/* The comparison table grows by one column per method in the cart, so its wrapper scrolls horizontally; the sticky
   first column keeps the feature labels visible while scrolling. */
.compare-scroll { overflow-x: auto; }
.explorer .compare-table { border-collapse: collapse; width: max-content; min-width: 100%; font-size: 0.86rem; }
.explorer .compare-table th, .explorer .compare-table td { border-top: 1px solid var(--x-border);
  padding: 0.4rem 0.8rem; text-align: left; vertical-align: top; min-width: 8rem; max-width: 18rem; }
.explorer .compare-table thead th { border-top: none; font-size: 0.95rem; }
.explorer .compare-table tbody th, .explorer .compare-table thead th:first-child { position: sticky; left: 0;
  background: var(--x-surface); z-index: 1; color: var(--x-muted); font-weight: 500; white-space: nowrap; }
/* zebra striping for readability; the sticky label cells repeat the row shade with an opaque background so the
   stripes stay consistent while scrolling horizontally */
.explorer .compare-table tbody tr:nth-child(even) td { background: var(--x-surface-2); }
.explorer .compare-table tbody tr:nth-child(even) th { background: var(--x-surface-2); }
/* the divider row announcing which benchmark the metric rows below it belong to */
.explorer .compare-table tbody th.section { padding-top: 1rem; color: var(--x-text); font-weight: 700; }
/* The slot column is the visual tile: it carries the card chrome (instead of the HTML card body inside it), so that
   the add-to-cart button -- a separate Gradio component below the card body -- appears inside the tile. */
.method-card { background: var(--x-surface); border: 1px solid var(--x-border); border-radius: 12px;
  padding: 0.9rem !important; gap: 0.5rem !important; transition: transform 0.12s ease, box-shadow 0.12s ease; }
.method-card:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18); }
/* Belt and braces: a slot whose card body is empty (i.e. not assigned a method by the current filter) is fully
   hidden, independently of whether Gradio applies the visible=False update to the column. */
.method-card:not(:has(.card)) { display: none !important; }

/* Enlarge the tab labels (Browse methods / Cart) so they are hard to miss. Tab buttons carry the ARIA role "tab";
   the .tab-nav fallback covers Gradio versions that don't set it. */
button[role="tab"], .tab-nav button { font-size: 1.3rem !important; font-weight: 600 !important;
  padding: 0.5rem 1.4rem !important; }
"""


def build_demo() -> gr.Blocks:
    # filter options for layer types and quantization are derived from the data, so new layer types/backends appear
    # automatically
    layer_names: list[str] = []
    quant_names: set[str] = set()
    for method in METHODS:
        for name in layer_types(method) or {}:
            if name not in layer_names:  # keep the original (probe) order rather than sorting alphabetically
                layer_names.append(name)
        quant_names.update(quant_backends(method) or [])

    with gr.Blocks(title="PEFT Shop", css=CSS) as demo:
        gr.Markdown(
            "# 🤗 PEFT Shop\n"
            "Your one-stop shop for parameter-efficient fine-tuning — every method free, always in stock. "
            f"[Docs](https://huggingface.co/docs/peft/main/en/index) · "
            f"[GitHub](https://github.com/huggingface/peft) · "
            f"[Benchmarks]({BENCHMARK_SPACE_URL})"
        )
        # browsing and the cart are two top-level tabs; the tab labels are enlarged via CSS so they're hard to miss
        with gr.Tabs():
            with gr.Tab("🛍️ Browse the shop"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=270):
                        reset_button = gr.Button("Reset filters", size="sm")
                        search = gr.Textbox(label="Search", placeholder="Search methods…")
                        categories = gr.CheckboxGroup(
                            choices=[(label, value) for value, label in CATEGORY_LABELS.items()], label="Category"
                        )
                        capabilities = gr.CheckboxGroup(
                            choices=[(label, key) for label, key in CAPABILITIES],
                            label="Capabilities (must support all selected)",
                        )
                        layers = gr.CheckboxGroup(
                            choices=layer_names, label="Target layer types (must support all selected)"
                        )
                        quant = gr.CheckboxGroup(
                            choices=sorted(quant_names), label="Quantization (any selected backend)"
                        )
                        benchmarked_only = gr.Checkbox(label="Only methods with results on the selected benchmark")
                    with gr.Column(scale=3):
                        with gr.Row():
                            count_md = gr.Markdown()
                            benchmark = gr.Dropdown(
                                choices=[(f"{spec.label} ({spec.model_name})", spec.key) for spec in BENCHMARKS],
                                value=BENCHMARKS[0].key,
                                label="Benchmark",
                            )
                            sort_by = gr.Dropdown(choices=SORT_CHOICES, value="name", label="Sort by")

                        # The card grid is a fixed pool of slots, one per method, created once at startup: a column
                        # with the card body (HTML), an "add to cart" button, and a State holding the method
                        # currently displayed in the slot. Filtering and sorting update the slots' content and
                        # visibility instead of re-creating components (e.g. via gr.render) -- event listeners on
                        # components that are dynamically created proved unreliable, whereas listeners attached to
                        # static components always work. Columns with min_width inside a row wrap automatically,
                        # resulting in a responsive card grid.
                        slots = []
                        with gr.Row():
                            for _ in range(len(METHODS)):
                                with gr.Column(
                                    min_width=340, visible=False, elem_classes="method-card"
                                ) as slot_column:
                                    slot_html = gr.HTML()
                                    slot_method = gr.State("")
                                    slot_button = gr.Button(ADD_TO_CART_LABEL, size="sm")
                                slots.append((slot_column, slot_html, slot_method, slot_button))
            with gr.Tab("🛒 Cart") as cart_tab:
                cart_select = gr.Dropdown(
                    choices=sorted(METHODS),
                    multiselect=True,
                    label="Cart contents (deselect to remove, or add directly)",
                )
                cart_code = gr.Code(value=EMPTY_CART_SNIPPET, language="python", label="How to use")
                with gr.Row():
                    copy_button = gr.Button("📋 Copy code")
                    pay_button = gr.Button("Pay 💳", variant="primary")
                    clear_button = gr.Button("Clear cart")
                # the receipt popover is invisible until the pay button opens it
                receipt_html = gr.HTML(build_receipt([]))
                gr.Markdown("### Feature comparison")
                compare_html = gr.HTML(build_comparison([], BENCHMARKS[0]))
        benchmark_list = ", ".join(f"{spec.label} on {spec.model_name}" for spec in BENCHMARKS)
        gr.Markdown(
            f"<small>Capability data is generated from the PEFT code base (peft v{DATA['peft_version']}) by "
            "`scripts/generate_method_capabilities.py`; values marked “?” could not be determined automatically. "
            f"Benchmark numbers come from the [PEFT method comparison]({BENCHMARK_SPACE_URL}) ({benchmark_list}); "
            "each method shows its best run on the selected benchmark.</small>"
        )

        filter_inputs = [search, categories, capabilities, layers, quant, benchmarked_only, benchmark, sort_by]
        # the cart is an extra input to the card rendering (for the "in cart" button labels) but deliberately not
        # part of filter_inputs: resetting the filters must not clear the cart
        card_inputs = filter_inputs + [cart_select]
        slot_outputs = [count_md]
        for slot_column, slot_html, slot_method, slot_button in slots:
            slot_outputs.extend([slot_column, slot_html, slot_method, slot_button])
        for component in filter_inputs:
            # The search box needs both listeners: .input fires per keystroke but (in Gradio 6) not when characters
            # are deleted, .change fires on any value change -- including deletions and programmatic ones like
            # "reset filters" -- but might lag behind while typing. Double-firing is harmless, the render is
            # idempotent. trigger_mode="always_last" makes sure the render runs again with the final value when
            # events arrive while a previous render is still in flight -- the default "once" of .input would
            # silently drop them, leaving the cards filtered by a stale prefix of the search query.
            listeners = [component.change]
            if isinstance(component, gr.Textbox):
                listeners.append(component.input)
            for listener in listeners:
                listener(
                    update_cards,
                    inputs=card_inputs,
                    outputs=slot_outputs,
                    show_progress="hidden",
                    trigger_mode="always_last",
                )
        demo.load(update_cards, inputs=card_inputs, outputs=slot_outputs, show_progress="hidden")

        for _, _, slot_method, slot_button in slots:
            slot_button.click(
                add_to_cart, inputs=[cart_select, slot_method], outputs=cart_select, show_progress="hidden"
            )

        # Updating cart_select programmatically (from the cards' add buttons or "clear cart") also triggers these
        # listeners, so the code snippet, the comparison table, the pay receipt, and the "in cart" button labels
        # always follow along; the comparison table and receipt follow the benchmark dropdown as well (the benchmark
        # dropdown already re-renders the cards through the filter listeners above).
        cart_outputs = [cart_code, compare_html, receipt_html, cart_tab]
        for trigger in (cart_select.change, benchmark.change):
            trigger(update_cart, inputs=[cart_select, benchmark], outputs=cart_outputs, show_progress="hidden")
        cart_select.change(update_cards, inputs=card_inputs, outputs=slot_outputs, show_progress="hidden")
        clear_button.click(list, outputs=cart_select, show_progress="hidden")
        # the receipt is kept up to date by update_cart, so paying only needs to open the popover, which is a purely
        # client-side affair (fn=None + js); same for writing to the clipboard, which only the browser can do
        pay_button.click(None, js="() => document.getElementById('pay-receipt')?.togglePopover(true)")
        copy_button.click(None, inputs=cart_code, js="(code) => { navigator.clipboard.writeText(code); }")
        reset_button.click(reset_filters, outputs=filter_inputs, show_progress="hidden")

    return demo


# ---------------------------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The PEFT shop; builds its data file on demand.")
    parser.add_argument(
        "--data", type=Path, default=HERE / "data.json", help="data file to load or build (default: %(default)s)"
    )
    parser.add_argument("--rebuild", action="store_true", help="rebuild the data file even if it exists")
    parser.add_argument("--build-only", action="store_true", help="only (re)build the data file, don't launch the app")
    parser.add_argument(
        "--capabilities",
        type=Path,
        default=Path("method_capabilities.json"),
        help="capability matrix produced by scripts/generate_method_capabilities.py, used when building "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=HERE.parent,
        help="directory containing the benchmarks, i.e. one result folder per BENCHMARKS entry (default: the "
        "method_comparison directory), used when building",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=HERE.parent.parent / "docs" / "source" / "package_reference",
        help="directory containing the package_reference docs pages (for the method descriptions), used when building",
    )
    return parser.parse_args(argv)


def load_or_build_data(args: argparse.Namespace) -> dict[str, Any]:
    if args.data.exists() and not (args.rebuild or args.build_only):
        data = json.loads(args.data.read_text())
        if data.get("schema_version") != DATA_SCHEMA_VERSION:
            raise SystemExit(
                f"{args.data} has schema version {data.get('schema_version')}, but this version of the app expects "
                f"{DATA_SCHEMA_VERSION}. Rebuild the file with --rebuild."
            )
        return data

    if not args.capabilities.exists():
        raise SystemExit(
            f"{args.data} does not exist and cannot be built because the capability matrix {args.capabilities} is "
            "also missing. Generate it first with scripts/generate_method_capabilities.py or pass --capabilities."
        )
    data = build_data(args.capabilities, args.benchmarks_dir, args.docs_dir)
    args.data.write_text(json.dumps(data, indent=2) + "\n")
    logger.info(f"Wrote data for {len(data['methods'])} methods to {args.data}")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    data = load_or_build_data(args)
    if not args.build_only:
        _set_data(data)
        build_demo().launch()
