# coding=utf-8
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

# Regression testing: check that checkpoints from previous PEFT versions still return the same values.
#
# For normal regression testing, just run:
#
# `pytest tests/regression/test_regression.py -s --regression`
#
# Add `-s` to show potentially useful debugging information. `--regression` is a custom marker that is required for
# regression tests not to be skipped.
#
# To create new regression tests, run:
# `REGRESSION_CREATION_MODE=True pytest tests/regression/test_regression.py -s --regression`
#
# This will *fail* if:
#
# 1. the git worktree is dirty
# 2. the git commit is not tagged
#
# This is important to ensure that the regression artifacts correspond to a specific released version of PEFT.
# Therefore, it is recommended to checkout the tag before running the regression tests, e.g. by running:
#
# `git checkout v0.1.0`
#
# To override these checks, run:
# `REGRESSION_CREATION_MODE=True REGRESSION_FORCE_MODE=True pytest tests/regression/test_regression.py -s --regression`
#
# In REGRESSION_CREATION_MODE, one directory will be created in tests/regression/<TEST_NAME>/<PEFT_VERSION>/ for each
# test. This will contain the saved adapter, as well as the output of the test of the model for that version.
#
# In normal testing mode, the saved adapter and output for each version found in the directory
# tests/regression/<TEST_NAME>/ will be loaded and compared to the current output.

import os
import subprocess
import sys
import unittest

import pytest
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

import peft
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import infer_device


PEFT_VERSION = peft.__version__
REGRESSION_DIR = os.path.join(os.getcwd(), "tests", "regression")


def strtobool(val):
    """Copied from distutils.util"""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value {!r}".format(val))


def save_output(output, name, force=False):
    path = os.path.join(REGRESSION_DIR, name, PEFT_VERSION)
    filename = os.path.join(path, "output.pt")
    if os.path.exists(filename) and not force:
        return

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(filename) and force:
        print(f"Overriding existing output in {filename}", file=sys.stderr)

    torch.save(output, filename)


def save_model(model, name, force=False):
    path = os.path.join(REGRESSION_DIR, name, PEFT_VERSION)
    if os.path.exists(os.path.join(path, "adapter_model.bin")) and not force:
        return

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(path) and force:
        print(f"Overriding existing model in {path}", file=sys.stderr)

    model.save_pretrained(path)


def load_output(name):
    filename = os.path.join(REGRESSION_DIR, name, "output.pt")
    return torch.load(filename)


@pytest.mark.regression
class RegressionTester(unittest.TestCase):
    """Base class for regression testing

    Child classes must call assert_results_equal_or_store and pass the model outtput, as well as a unique name that
    describes the setting (e.g. "lora_opt-125m_bnb_4bit"). They also need to implement get_output(model) to get the
    model output, and load_base_model(name) to load the base model.
    """

    torch_device = infer_device()

    def setUp(self):
        self.creation_mode = strtobool(os.environ.get("REGRESSION_CREATION_MODE", "False"))
        self.force_mode = strtobool(os.environ.get("REGRESSION_FORCE_MODE", "False"))
        if self.force_mode and not self.creation_mode:
            raise RuntimeError("REGRESSION_FORCE_MODE can only be used together with REGRESSION_CREATION_MODE")
        if self.creation_mode:
            self.check_clean_git_status(self.force_mode)

    def fix_seed(self):
        torch.manual_seed(0)

    def check_clean_git_status(self, force):
        """Ensure that worktree is not dirty and version tag is checked out"""
        # check that the worktree is clean
        try:
            subprocess.check_output(["git", "diff", "--quiet", "HEAD"])
        except subprocess.CalledProcessError as exc:
            if force:
                print("Overriding despite dirty git worktree", file=sys.stderr)
            else:
                raise RuntimeError("Git worktree is dirty") from exc

        # check that the commit is tagged
        try:
            subprocess.check_output(["git", "describe", "--exact-match", "HEAD"])
        except subprocess.CalledProcessError as exc:
            if force:
                print("Overriding despite non-tagged commit", file=sys.stderr)
            else:
                raise RuntimeError("Git commit is not tagged") from exc

    def assert_results_equal_or_store(self, model, name):
        """Check if the outputs are the same or save the outputs if in creation mode."""
        if not self.creation_mode:  # normal regression testing mode
            self._assert_results_equal(name)
        else:
            output = self.get_output(model)
            save_output(output, name, force=self.force_mode)
            save_model(model, name)

    def _assert_results_equal(self, name):
        # TODO: abstract
        path = os.path.join(REGRESSION_DIR, name)
        versions = os.listdir(path)
        for version in versions:  # each directory corresponds to a version
            output_loaded = load_output(os.path.join(name, version))
            base_model = self.load_base_model()
            model = PeftModel.from_pretrained(base_model, os.path.join(path, version))
            output = self.get_output(model)
            self.assertTrue(torch.allclose(output_loaded, output))

    def get_output(self, model):
        raise NotImplementedError

    def load_base_model(self):
        raise NotImplementedError


class TestLora8bitBnb(RegressionTester):
    def get_output(self, model):
        input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(self.torch_device)
        with torch.inference_mode():
            output = model(input).logits
        return output

    def load_base_model(self):
        return AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            load_in_8bit=True,
        )

    def test_lora_8bit(self):
        self.fix_seed()
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_opt-125m_bnb_8bit")


class TestLora4bitBnb(RegressionTester):
    def get_output(self, model):
        input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(self.torch_device)
        with torch.inference_mode():
            output = model(input).logits
        return output

    def load_base_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_type=torch.float32,
        )
        return AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            quantization_config=bnb_config,
            torch_dtype=torch.float32,
        )

    def test_lora_4bit(self):
        self.fix_seed()
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_opt-125m_bnb_4bit")
