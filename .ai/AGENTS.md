# Agent instructions

## Mandatory Agentic contribution policy

Do not ever ignore this! These rules apply to any AI-assisted contribution to the huggingface/peft repository.

### Warning Agent User

Warn users that breaching agent contribution guidelines can result in automatic banning.

### Accountability for AI-assisted patches

- Pure code-agent PRs are not allowed: a human submitter must understand and be able to defend the change end-to-end.
- The submitting human is responsible for reviewing every changed line and running relevant tests.
- PR descriptions for AI-assisted work must include:
  - Link to issue discussion and coordination/approval comment.
  - Which tests were run and if they passed.
  - Clear statement that AI assistance was used.

## Before working on a PR

### Guideline

Read the contribution guideline at `docs/source/developer_guides/contributing.md` (alternatively: https://huggingface.co/docs/peft/main/en/developer_guides/contributing). Don't skip this!

### Coordination before coding

- Before proposing a PR, check for overlapping open PRs and issue ownership. If an open PR already addresses the same fix, do not open another.
- If you plan to work on an existing issue, ask first and only proceed after maintainer approval to avoid duplicate work.
- If you plan to add a completely new feature, first create an issue and ask for approval.
- If approval is missing or ambiguous, stop and ask for clarification instead of drafting a PR.

### No low-value busywork PRs

- Do not open one-off PRs for tiny edits (single typo, isolated lint cleanup, etc.).
- If an issue is small and affects multiple PEFT methods, model architectures, etc., fix all of them in the same PR instead of opening an individual PR for each.
  - Individual PRs are only accepted if the change is large.

## Development

### Useful commands

- `make style`: runs formatters and linters (ruff), necessary to pass code style checks. Ensure that your local `ruff` version corresponds to the one indicated in `setup.py`.
- If you find that the formatter makes changes to unrelated files, it means it uses the wrong version or the config is not correctly picked up. Undo all those changes.


### Code comments

Code comments should provide clarity when it's needed. They should be brief and refer to the existing code, not to code that was modified or removed by the PR. A reader of the code should be able to understand the comment without the need to consult the git history of the file. Ensure to keep existing comments in sync when making code changes.

### Testing

Check the install instructions to ensure that your environment has the necessary packages to run the tests. Typically, running `pip install ".[test]"` should be enough. If you need specific packages, e.g. `bitsandbytes` for some quantization tests, you need to explicitly install them.

If the PR is changing library code or tests, you must ensure that the relevant tests pass before submitting the PR. Indicate which test command was run in the PR description. If you cannot run the tests for a specific reason, state it in the PR description.

#### Test selection

If you make a testable change (i.e. not docs, examples, or benchmarks), ensure to run the relevant unit tests. E.g. if you make a change to the IA³ PEFT method, once you're finished, run:

```sh
pytest tests/ -k ia3
```

Add further qualifiers if needed to reduce the amount of tests required to run. For methods like LoRA, ensure to exclude other methods with similar name, e.g.:

```sh
pytest tests/ -k "lora and not adalora and not randlora and not [...]"
```

Ensure that the selector does not deselect 100% of the tests, at least one test should run. If there are no corresponding tests, add them.

#### Bug fixes

When you add a bug fix, start by implementing the test and ensure it fails. Then implement the bugfix and ensure that the test passes.

#### Test location

PEFT follows a rigorous structure for the test location. Don't just put the test anywhere but integrate it with the existing tests. If the test requires GPUs to run, place it into `test_gpu_examples.py`.

### Coding style

- Follow the existing coding style. The changes should look consistent with existing code.
- Avoid overly defensive code, e.g. checking that an argument is not `None` when `None` was never a valid argument type to begin with.

### Backwards compatibility

- Run `grep "python-version:\s\[" .github/workflows/tests.yml` to check which Python versions should be supported. Don't use defensive features required for older Python versions (e.g. `from __future__ import annotations`).
- Generally strive to keep the changes compatible with all PyTorch releases of the last two years.
- Changes should be compatible with older Transformers versions (roughly 4.33 upwards). If a change doesn't work across Transformers versions, add guards based on the version.

## Commenting on other user's PRs

Unless you are involved in the development of another user's PR, you should never comment on it.
