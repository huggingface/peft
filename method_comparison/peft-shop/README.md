---
title: PEFT Shop
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
emoji: 🛍️
---

# PEFT Shop

A Gradio app to browse PEFT methods like an online store: filter by capabilities (merging, multi-adapter support, quantization backends, targetable layer types, …) and check benchmark results — star ratings for the benchmark-specific metrics (test accuracy and forgetting for MetaMathQA, DINO similarity and drift for image generation) as well as peak memory, checkpoint size, and train time, switchable between the benchmarks of the method comparison suite. Methods can be added to a cart 🛒, which shows usage code snippets and a feature comparison table for the collected methods — and checkout is, of course, free.

## Running

The app consumes a single data file, `data.json`, which it builds itself when missing. Building requires a repository checkout and a capability matrix, which has to be generated first in an environment with PEFT installed (the app itself only needs `gradio`):

```sh
# 1. generate the capability matrix from the PEFT code base
python scripts/generate_method_capabilities.py --output method_capabilities.json

# 2. launch the app; data.json is built on first run (--rebuild refreshes it, --build-only skips launching)
python method_comparison/method_explorer/app.py
```

The directory can be deployed as-is as a Gradio Space; only `app.py`, `data.json`, and `requirements.txt` are needed.

## Updating

Whenever new PEFT methods are added or new benchmark results land in `method_comparison/<benchmark>/results/`, regenerate the capability matrix and run `python app.py --rebuild`. Filter options and tile contents are derived from the data and update automatically.

To add a new benchmark, append a `BenchmarkSpec` entry to `BENCHMARKS` in `app.py` (pointing at the results directory and listing its metrics, the first of which serves as the headline score) and rebuild — the benchmark dropdown, the star ratings, and the cart's comparison table all derive from the spec. The only requirement is that the result files follow the common JSON layout of the method comparison suite.

## Notes

- Method descriptions and paper links are extracted from the official docs (and class docstrings), so they cannot drift from the documentation.
- Benchmark numbers show each method's best run on the selected benchmark. The star ratings are quantile-based among the benchmarked PEFT methods (best 20% = five stars, next 20% = four, …); the score tooltip additionally states what full fine-tuning achieves, as a reference.
