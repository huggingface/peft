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

# Contributing to PEFT

We are happy to accept contributions to PEFT. If you plan to contribute, please read this document to make the process as smooth as possible.

## Installation

The installation instructions can be found [here](https://huggingface.co/docs/peft/install). If you want to provide code contributions to PEFT, you should choose the "source" installation method.

If you are new to creating a pull request, follow [these instructions from GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

## Running tests and code quality checks

Regardless of the type of contribution (unless it’s only about the docs), you should run tests and code quality checks before creating a PR to ensure that your contribution doesn’t break anything and follows the standards of the project.

We provide a Makefile to facilitate those steps. Run the code below for the unit test:

```sh
make test
```

Run one of the following to either check or check and fix code quality and style:

```sh
make quality  # just check
make style  # check and fix
```


Running all the tests can take a couple of minutes. Therefore, during development, it can be useful to run only those tests specific to your change:

```sh
pytest tests/ -k <name-of-test>
```

This should finish much quicker and allow faster iteration. Before creating the PR, however, please still run the whole test suite, as some changes can inadvertently break tests that at first glance are unrelated.

If your change is specific to a hardware setting (e.g. it requires CUDA), take a look at `tests/test_gpu_examples.py` and `tests/test_common_gpu.py` – maybe it makes sense to add a test there. If your change could have an effect on saving and loading models, please run the tests with the `--regression` flag to trigger regression tests.

It can happen that while you’re working on your PR, the underlying code base changes due to other changes being merged. If that happens – especially when there is a merge conflict – please update your branch to be on the latest changes. This can be a merge or a rebase, whatever you prefer. We will squash and merge the PR once it’s ready.

## PR description

When opening the PR, please provide a nice description of the change you provide. If it relates to other issues or PRs, please reference them. Providing a good description will not only help the reviewers review your code better and faster, it can also later be used (as a basis) for the commit message, which helps with long term maintenance of the project.

If your code makes some non-trivial changes, it can also be a good idea to add comments to the code to explain those changes. For example, if you had to iterate on your implementation multiple times because the most obvious way didn’t work, it’s a good indication that a code comment is needed.

## Providing a bugfix

Please give a description of the circumstances that lead to the bug. If there is an existing issue, please link to it (e.g. “Resolves #12345”).

Ideally, when a bugfix is provided, it should be accompanied by a test for this bug. The test should fail with the current code and pass with the bugfix. Add a comment to the test that references the issue or PR. Without such a test, it is difficult to prevent regressions in the future.

## Adding a new fine-tuning method

New parameter-efficient fine-tuning methods are developed all the time. If you would like to add a new, promising method to PEFT, please follow these steps.

**Requirements**

1. Please add a link to the source (usually a paper) of the method.
2. Some evidence should be provided that there is general interest in using the method. We will not add new methods that are freshly published but without evidence that there is demand for it.
3. Ideally, we want to not only add the implementation of the new method, but also examples (notebooks, scripts), documentation, and an extensive test suite that proves that the method works with a variety of tasks. However, this can be very daunting. Therefore, it is also acceptable to only provide the implementation and at least one working example. Documentation and tests can be added in follow up PRs.

**Steps**

Before you start to implement the new method, please open an issue on GitHub with your proposal. That way, the maintainers can give you some early feedback.

When implementing the method, it makes sense to look for existing implementations that already exist as a guide. Moreover, when you structure your code, please take inspiration from the other PEFT methods. For example, if your method is similar to LoRA, it makes sense to structure your code similarly or even re-use some functions or classes where it makes sense (but don’t overdo it, some code duplication is okay).

Once you have something that seems to be working, don’t hesitate to create a draft PR, even if it’s not in a mergeable state yet. The maintainers will be happy to give you feedback and guidance along the way.

## Adding other features

It is best if you first open an issue on GitHub with a proposal to add the new feature. That way, you can discuss with the maintainers if it makes sense to add the feature before spending too much time on implementing it.

New features should generally be accompanied by tests and documentation or examples. Without the latter, users will have a hard time discovering your cool new feature.

Changes to the code should be implemented in a backward-compatible way. For example, existing code should continue to work the same way after the feature is merged.
