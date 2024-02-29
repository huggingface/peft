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

Running all the tests can take a couple of minutes, so during development it can be more efficient to only run tests specific to your change:

```sh
pytest tests/ -k <name-of-test>
```

This should finish much quicker and allow for faster iteration. However, you should still run the whole test suite before creating a PR because your change can inadvertently break tests that at first glance are unrelated.

If your change is specific to a hardware setting (e.g., it requires CUDA), take a look at [tests/test_gpu_examples.py](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_gpu_examples.py) and [tests/test_common_gpu.py](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_common_gpu.py) to see if it makes sense to add tests there. If your change could have an effect on saving and loading models, please run the tests with the `--regression` flag to trigger regression tests.

It can happen that while you’re working on your PR, the underlying code base changes due to other changes being merged. If that happens – especially when there is a merge conflict – please update your branch with the latest changes. This can be a merge or a rebase, and we'll squash and merge the PR once it’s ready.

## PR description

When opening a PR, please provide a nice description of the change you're proposing. If it relates to other issues or PRs, please reference them. Providing a good description not only helps the reviewers review your code better and faster, it can also be used later (as a basis) for the commit message which helps with long term maintenance of the project.

If your code makes some non-trivial changes, it may also be a good idea to add comments to the code to explain those changes. For example, if you had to iterate on your implementation multiple times because the most obvious way didn’t work, it’s a good indication that a code comment is needed.

## Bugfixes

Please give a description of the circumstances that led to the bug. If there is an existing issue, please link to it (e.g., “Resolves #12345”).

Ideally when a bugfix is provided, it should be accompanied by a test for the bug. The test should fail with the current code and pass with the bugfix. Add a comment to the test that references the issue or PR. Without a test, it is more difficult to prevent regressions in the future.

## Add a new fine-tuning method

New parameter-efficient fine-tuning methods are developed all the time. If you would like to add a new and promising method to PEFT, please follow these steps.

1. Before you start to implement the new method, please open a GitHub issue with your proposal. This way, the maintainers can give you some early feedback.
2. Please add a link to the source (usually a paper) of the method. Some evidence should be provided there is general interest in using the method. We will not add new methods that are freshly published, but there is no evidence of demand for it.
3. When implementing the method, it makes sense to look for existing implementations that already exist as a guide. Moreover, when you structure your code, please take inspiration from the other PEFT methods. For example, if your method is similar to LoRA, it makes sense to structure your code similarly or even reuse some functions or classes where it makes sense (some code duplication is okay, but don’t overdo it).
4. Ideally, in addition to the implementation of the new method, there should also be examples (notebooks, scripts), documentation, and an extensive test suite that proves the method works with a variety of tasks. However, this can be more challenging so it is acceptable to only provide the implementation and at least one working example. Documentation and tests can be added in follow up PRs.
5. Once you have something that seems to be working, don’t hesitate to create a draft PR even if it’s not in a mergeable state yet. The maintainers are happy to give you feedback and guidance along the way.

## Add other features

It is best if you first open an issue on GitHub with a proposal to add the new feature. This way, you can discuss with the maintainers if it makes sense to add the feature before spending too much time on implementing it.

New features should generally be accompanied by tests and documentation or examples. Without the latter, users will have a hard time discovering your cool new feature.

Changes to the code should be implemented in a backward-compatible way. For example, existing code should continue to work the same way after the feature is merged.
