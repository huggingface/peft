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
"""Regression tests for `peft.utils.transformers_weight_conversion`."""

import pytest

from peft.utils.transformers_weight_conversion import _convert_peft_config_moe


class _FakeConfig:
    """Minimal stand-in for a peft config that has the attributes the helper mutates."""

    def __init__(self, target_modules, target_parameters=None):
        self.target_modules = target_modules
        self.target_parameters = target_parameters
        # The helper also touches these, so they need to exist for downstream iteration.
        self.rank_pattern = None
        self.alpha_pattern = None


@pytest.mark.parametrize(
    "target_modules",
    [
        "all-linear",
        "q_proj",
        ["q_proj", "v_proj"],
        ("q_proj", "k_proj"),
    ],
)
def test_target_modules_string_not_split_into_characters(target_modules):
    """Regression test for https://github.com/huggingface/peft/issues/3229.

    Before the fix, ``set(target_modules or [])`` would silently turn a string
    like ``"all-linear"`` into a set of individual characters, breaking the MoE
    target conversion logic and producing confusing downstream errors.
    """
    config = _FakeConfig(target_modules=target_modules)

    _convert_peft_config_moe(config, "mixtral")

    if isinstance(target_modules, str):
        assert config.target_modules == {target_modules}
    else:
        assert config.target_modules == set(target_modules)


@pytest.mark.parametrize("target_parameters", [None, "q_proj", ["q_proj", "v_proj"]])
def test_target_parameters_string_not_split_into_characters(target_parameters):
    """Same regression for ``target_parameters`` (used by DoRA-like configs)."""
    config = _FakeConfig(target_modules=["q_proj"], target_parameters=target_parameters)

    _convert_peft_config_moe(config, "mixtral")

    if target_parameters is None:
        assert config.target_parameters == set()
    elif isinstance(target_parameters, str):
        assert config.target_parameters == {target_parameters}
    else:
        assert config.target_parameters == set(target_parameters)


def test_target_modules_none_is_empty_set():
    config = _FakeConfig(target_modules=None)

    _convert_peft_config_moe(config, "mixtral")

    assert config.target_modules == set()
