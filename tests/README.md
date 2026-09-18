# PEFT test suite

This document describes how the test suite is organized, where a new test belongs, and which shared infrastructure exists. Follow this document if you add or modify PEFT tests.

## Layout

### Common test matrix

Most coverage comes from a shared battery of tests that runs each PEFT method against a set of models. The battery lives in the `PeftCommonTester` parent class in`testing_common.py`, whose `_test_*` methods check the behavior every method must support: creating the adapter, training, saving and loading, merging, disabling, multiple adapters, generation, and so on.

Several files instantiate this battery for different model types. Each contains a list of test cases -- combinations of model, config class, and config kwargs -- and a test class that parametrizes over them and delegates to `PeftCommonTester`:

- `test_custom_models.py`: small, handcrafted `torch.nn` models (MLPs, convnets, RNNs). These are the broadest and fastest tests and the first place a new method is added.
- `test_decoder_models.py`, `test_encoder_decoder_models.py`, `test_feature_extraction_models.py`, `test_seq_classifier.py`: tiny Transformers models of the respective architecture class.

A fix or feature that concerns behavior common to all methods (e.g. something about merging or saving) belongs in `PeftCommonTester`, where every method is tested against it.

### Cross-cutting features

Library features that are not tied to a single method each have their own file, e.g. `test_config.py` (config round-tripping for all methods), `test_initialization.py` (PEFT model initializatoin, adapter injection, model validation), `test_tuners_utils.py` (target module matching), `test_low_level_api.py`, `test_hub_features.py`, etc.

### Method-specific tests

Files named `test_<method>.py` (e.g. `test_vera.py`) hold tests for behavior that is particular to one method and cannot be expressed in the common matrix, like algorithmic properties. Only add a new test file after ensuring that the new tests don't fit in the existing tests.

### GPU and quantization tests

Tests that require an accelerator go into `test_gpu_examples.py` (end-to-end training scenarios) or `test_common_gpu.py`, marked with `pytest.mark.single_gpu_tests` or `pytest.mark.multi_gpu_tests`. Quantization integrations using the generic PEFT quantization backend are tested in `test_quantization.py`, gated by `require_*` decorators (see below). These tests do not run in the default CI on PRs but on nightly runs with GPU runners. Never put a test that needs a GPU into the CPU test files, as it would fail the regular CI.

Notes:

- CPU tests should also pass on accelerator hardware.
- GPU tests should be written in a hardware agnostic way (not depending on CUDA specifically).
- Multi GPU tests will run with two GPUs, but write the test so that it won't fail even when more GPUs are available.

### Special suites

A few directories exist for specific purposes and are run separately from the main suite; most contributors will not need to touch them:

- `regression/`: regression tests against artifacts created by earlier PEFT versions, skipped unless `--regression` is passed.
- `training/`: multi-process training tests (DeepSpeed, FSDP) launched via `accelerate` from Makefile targets, not covered by pytest.

## Where does a new test go?

1. Does it test behavior that all (or many) methods should have? Add a `_test_*` method to `PeftCommonTester` and corresponding parametrized tests to the matrix files, or extend an existing battery test.
2. Is it specific to one method? Put it in `test_<method>.py`, creating the file if needed.
3. Does it concern a library feature (configs, saving, injection, ...)? Extend the corresponding cross-cutting file.
4. Does it need a GPU or a quantization backend? It goes to `test_gpu_examples.py` / `test_common_gpu.py` / `test_quantization.py` with the appropriate marker or decorator.
5. Does the test extend an existing test class? Put the test next to similar test methods (e.g. a new merging test should follow existing merging tests); if there are no similar test methods, put it at the end of the test class.

Do not add standalone test scripts or new files that duplicate what the matrix already covers. If in doubt, search for a similar existing test (e.g. an older, comparable bug fix) and put the new test next to it.

### Tests for a new PEFT method

When adding a new method, coverage means adding it to the existing matrices rather than writing a separate suite:

1. Add test cases to `TEST_CASES` and `MULTIPLE_ACTIVE_ADAPTERS_TEST_CASES` in `test_custom_models.py` first; this covers most of the PEFT functionality and is quick to iterate on.
2. Add entries to the model-architecture files (`test_decoder_models.py` etc.) as applicable to the method's task types.
3. Add the config class to `ALL_CONFIG_CLASSES` in `test_config.py` and to the cases in `test_initialization.py` where applicable.
4. If the method supports quantization, extend `test_quantization.py`.
5. Only add `test_<method>.py` for genuinely method-specific behavior.

Tip: A recently merged PR that adds a new PEFT method is the best template. Check which tests were added in that PR as a guide.

## Infrastructure and conventions

`testing_utils.py` provides shared helpers:

- `hub_online_once(model_id)`: a context manager that allows the first download of a given model id and forces offline mode afterwards, so repeated hub access for the same model is caught and CI does not hammer the HF hub. Wrap model loading in it. If the test also loads a tokenizer, extend the cache key (e.g. `hub_online_once(model_id + "_tokenizer")`) so it does not share the cache pool with tests that only load the model.
- `require_*` decorators (`require_torch_gpu`, `require_bitsandbytes`, `require_multi_accelerator`, ...): skip tests when the hardware or optional package is missing. Use these rather than ad hoc skips.
- Cached data loaders (`load_dataset_english_quotes`, `load_cat_image`) for tests that need real inputs; `temp_seed` for local determinism.
- `set_init_weights_false`: many battery tests need the adapter to change the model output, which requires `init_weights=False` (or the method-specific equivalent); this helper knows the right argument per config class.
- Normal CI is tested on Ubuntu and Windows and with the four oldest Python versions that are still maintained (i.e. not EOL).
- Normal CI tests will generally run with the latest release of PyTorch, Transformers, Diffusers, Accelerate, etc. Nightly tests run on the main branch of Transformers and Accelerate.

Conventions to follow:

- For testing Transformers or Diffusers models, use only "tiny random" models, typically from `hf-internal-testing` or `peft-internal-testing`. Tests must run on CPU within seconds. Where possible, re-use models that are already part of the test suite. If you need a model architecture that doesn't have a tiny random variant, notify the maintainers so that they can create an uplod it. Never use models are datasets from random uploaders on the HF Hub.
- Re-use custom `nn.Module` definition and pytest fixtures as much as possible instead of creating new ones.
- Write pytest-style tests: plain `assert`, `pytest.mark.parametrize`, fixtures; no `unittest.TestCase`.
- `conftest.py` turns Transformers deprecation warnings into errors, so a test failing with a deprecation message is intentional and needs fixing.
- Choose numerical tolerances with other devices in mind; exact equality rarely holds across hardware.
- Add comments to explain what the test does if it's not obvious from the test name or context of the test. If non-trivial choices are made in testing code, add a comment to explain them. Comments should not refer the PR discussion but should make sense on their own; otherwise link to the relevant discussion.
- Individual tests should run fast on CPU. If you add a test that requires more than a few seconds, think about optimizing its runtime, especially if the test is run dozens of times because it is parametrized. Avoid disk reads and writes unless specifically needed for the test (e.g. tests that load checkpoints).
