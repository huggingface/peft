# Copyright 2023-present the HuggingFace Inc. team.
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

import logging

import pytest


def pytest_addoption(parser):
    parser.addoption("--regression", action="store_true", default=False, help="run regression tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "regression: mark regression tests")

    # Errors from transformers deprecations
    logger = logging.getLogger("transformers")

    class ErrorOnDeprecation(logging.Handler):
        def emit(self, record):
            msg = record.getMessage().lower()
            if "deprecat" in msg or "future" in msg:
                # let's ignore the torch_dtype => dtype deprecation for now
                if "torch_dtype" in msg:
                    return

                # BPE warning comes from transformers and we cannot do anything about it, transformers has to fix it
                bpe_warning = (
                    "Deprecated in 0.9.0: BPE.__init__ will not create from files anymore, try `BPE.from_file` instead"
                )
                if bpe_warning in msg:
                    return

                raise AssertionError(f"**Transformers Deprecation**: {msg}")

    # Add our handler
    handler = ErrorOnDeprecation()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--regression"):
        return

    skip_regression = pytest.mark.skip(reason="need --regression option to run regression tests")
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip_regression)
