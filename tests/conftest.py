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

import platform
import re

import pytest


def pytest_addoption(parser):
    parser.addoption("--regression", action="store_true", default=False, help="run regression tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "regression: mark regression tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--regression"):
        return

    skip_regression = pytest.mark.skip(reason="need --regression option to run regression tests")
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip_regression)


# TODO: remove this once support for PyTorch 2.2 (the latest one still supported by GitHub MacOS x86_64 runners) is
# dropped, or if MacOS is removed from the test matrix, see https://github.com/huggingface/peft/issues/2431.
# Note: the function name is fixed by the pytest plugin system, don't change it
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Plug into the pytest test report generation to skip a specific MacOS failure caused by transformers.

    The error was introduced by https://github.com/huggingface/transformers/pull/37785, which results in torch.load
    failing when using torch < 2.6.

    Since the MacOS x86 runners need to use an older torch version, those steps are necessary to get the CI green.
    """
    outcome = yield
    rep = outcome.get_result()
    # ref:
    # https://github.com/huggingface/transformers/blob/858ce6879a4aa7fa76a7c4e2ac20388e087ace26/src/transformers/utils/import_utils.py#L1418
    error_msg = re.compile(r"Due to a serious vulnerability issue in `torch.load`")

    # notes:
    # - pytest uses hard-coded strings, we cannot import and use constants
    #   https://docs.pytest.org/en/stable/reference/reference.html#pytest.TestReport
    # - errors can happen during call (running the test) but also setup (e.g. in fixtures)
    if rep.failed and (rep.when in ("setup", "call")) and (platform.system() == "Darwin"):
        exc_msg = str(call.excinfo.value)
        if error_msg.search(exc_msg):
            # turn this failure into an xfail:
            rep.outcome = "skipped"
            # for this attribute, see:
            # https://github.com/pytest-dev/pytest/blob/bd6877e5874b50ee57d0f63b342a67298ee9a1c3/src/_pytest/reports.py#L266C5-L266C13
            rep.wasxfail = "Error known to occur on MacOS with older torch versions, won't be fixed"
