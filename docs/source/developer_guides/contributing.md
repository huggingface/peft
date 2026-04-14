<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Contribute to PEFT

We are happy to accept contributions to PEFT. If you plan to contribute, please read this to make the process as smooth as possible.

## Installation

For code contributions to PEFT, you should choose the ["source"](../install#source) installation method.

If you are new to creating a pull request, follow the [Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) guide by GitHub.

## Tests and code quality checks

Regardless of the contribution type (unless it’s only about the docs), you should run tests and code quality checks before creating a PR to ensure your contribution doesn’t break anything and follows the project standards.

We provide a Makefile to execute the necessary tests. Run the code below for the unit test:

```sh
make test
```

Run one of the following to either only check or check and fix code quality and style:

```sh
make quality  # just check
make style  # check and fix
```

You can also set up [`pre-commit`](https://pre-commit.com/) to run these fixes
automatically as Git commit hooks.

```bash
$ pip install pre-commit
$ pre-commit install
```

Running all the tests can take a while, so during development it can be more efficient to only [run tests specific to your change](https://docs.pytest.org/en/6.2.x/usage.html#specifying-tests-selecting-tests), e.g. via:

```sh
pytest tests/<test-file-name> -k <name-of-test>
```

This should finish much quicker and allow for faster iteration.

If your change is specific to a hardware setting (e.g., it requires CUDA), take a look at [`tests/test_gpu_examples.py`](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_gpu_examples.py) and [`tests/test_common_gpu.py`](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_common_gpu.py) to see if it makes sense to add tests there. If your change could have an effect on saving and loading models, please run the tests with the `--regression` flag to trigger regression tests.

It can happen that while you’re working on your PR, the underlying code base changes due to other changes being merged. If that happens – especially when there is a merge conflict – please update your branch with the latest changes. This can be a merge or a rebase, and we'll squash and merge the PR once it’s ready. If possible, **avoid force pushes** to make reviews easier.

## PR description

When opening a PR, please provide a nice description of the change you're proposing. If it relates to other issues or PRs, please reference them. Providing a good description not only helps the reviewers review your code better and faster, it can also be used later (as a basis) for the commit message which helps with long term maintenance of the project.

If your code makes some non-trivial changes, it may also be a good idea to add comments to the code to explain those changes. For example, if you had to iterate on your implementation multiple times because the most obvious way didn’t work, it’s a good indication that a code comment is needed.

## Bugfixes

Please give a description of the circumstances that led to the bug. If there is an existing issue, please link to it (e.g., “Resolves #12345”).

Ideally when a bugfix is provided, it should be accompanied by a test for the bug. The test should fail with the current code and pass with the bugfix. Add a comment to the test that references the issue or PR. Without a test, it is more difficult to prevent regressions in the future.

## Documentation improvements

We are happy to have fixes for broken links and missing or unclear documentation. Taking care of examples, making sure that they are up-to-date and running fine in this fast moving environment is also highly appreciated.

Please refrain from sending pull requests that *only* correct typing errors as these generally create more work than they safe. Such changes are better combined with more substantial fixes (such as fixing broken links or extending/updating documentation).

## Add a new PEFT fine-tuning method

New parameter-efficient fine-tuning methods are developed all the time. If you would like to add a new and promising method to PEFT, please follow these steps.

1. If you're _not_ an author of the original paper, check for existing implementations and double check with the authors that they don't plan to submit a PR themselves.
2. Start with the core integration work listed below.
3. Check recent commits for new PEFT methods being added to take as inspiration.
4. It can be useful to open a draft PR early once the method basically works and first tests pass, then ask for feedback.
5. After working through reviewer feedback, ping the reviewer so that they know the PR is ready to review.

### Core integration of a new PEFT method

- [ ] Open a proposal issue on `huggingface/peft` before investing too much work.
- [ ] Link the source of the method, usually the final paper or another stable primary reference. We want to avoid work that is still under review, as the implementation should be stable.
- [ ] Add a new `PeftType` entry in `src/peft/utils/peft_types.py`.
- [ ] Create a new tuner package under `src/peft/tuners/` with the files your method needs (typically:  `config.py`, `model.py`, `layer.py`, and `__init__.py`).
- [ ] Register the method in the tuner `__init__.py` with `register_peft_method(...)`.
- [ ] Export the new config/model from `src/peft/tuners/__init__.py` and `src/peft/__init__.py`.
- [ ] If the method needs default target modules for Transformers models, add the mapping in `src/peft/utils/constants.py`.
- [ ] Add the method to the test matrix in `tests/test_custom_models.py` as these are the broadest and quickest tests. Check that the tests pass with `pytest tests/test_custom_models.py -k <method-name> -v`, fix failures if any.
- [ ] Run style/quality checks with `make style` before pushing.
- [ ] In the PR description, explain the method, link the paper, summarize tradeoffs, and list what was added.

### Full PR to add a new PEFT method

- [ ] Ensure that the configuration arguments that are specific to the method are well named and explained, don't assume that the user knows the paper inside out.
- [ ] Follow the naming and coding conventions of PEFT.
- [ ] Ensure that you didn't accidentally check in unrelated changes, e.g. the code formatter changing unrelated files.
- [ ] If some implementation choices are non-trivial, document them with a code comment.
- [ ] Complete the full test suite (`test_config.py`, `test_decoder_models.py`, etc.) by adding the PEFT method to the test matrix. Ensure that the tests pass.
- [ ] Add docs in `docs/source/package_reference/` with a short explanation, paper link, usage snippet, and autodoc blocks. Explain the pros and cons compared to other methods like LoRA. Register that doc page in `docs/source/_toctree.yml`.
- [ ] Add a runnable example under `examples/` (can be a copy of an existing example), with a short `README.md`.
- [ ] Check the benchmarks in `method_comparison/` and add experiment settings for your new method. This is a good place to sanity check that the PEFT method trains as expected. Include one or two reasonable benchmark configurations (one default, one optimized for the benchmark).

## Add other features

It is best if you first open an issue on GitHub with a proposal to add the new feature. This way, you can discuss with the maintainers if it makes sense to add the feature before spending too much time on implementing it.

New features should generally be accompanied by tests and documentation or examples. Without the latter, users will have a hard time discovering your cool new feature.

Changes to the code should be implemented in a backward-compatible way. For example, existing code should continue to work the same way after the feature is merged.
