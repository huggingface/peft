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
# `HF_TOKEN=<token> REGRESSION_CREATION_MODE=True pytest tests/regression/test_regression.py -s --regression`
#
# This will *fail* if:
#
# 1. the git worktree is dirty
# 2. the git commit is not tagged
#
# Note: A Hugging Face Hub token is required to upload the regression artifacts to our
# https://huggingface.co/peft-internal-testing repo. This can be done by anyone with write access to the repo but
# apparently it is not possible to create a technical token with write access.
#
# This is important to ensure that the regression artifacts correspond to a specific released version of PEFT.
# Therefore, it is recommended to checkout the tag before running the regression tests, e.g. by running:
#
# `git checkout v0.1.0`
#
# To override these checks, run:
# ``HF_TOKEN=<token> REGRESSION_CREATION_MODE=True REGRESSION_FORCE_MODE=True pytest tests/regression/test_regression.py -s --regression`
#
# In REGRESSION_CREATION_MODE, one directory will be created in tests/regression/<TEST_NAME>/<PEFT_VERSION>/ for each
# test. This will contain the saved adapter, as well as the output of the test of the model for that version.
#
# In normal testing mode, the saved adapter and output for each version found in the directory
# tests/regression/<TEST_NAME>/ will be loaded and compared to the current output.
#
# When implementing new tests, check the existing ones as well as the description in the docstring of RegressionTester.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import pytest
import torch
from huggingface_hub import snapshot_download, upload_folder
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.pytorch_utils import Conv1D

import peft
from peft import AdaLoraConfig, IA3Config, LoHaConfig, LoKrConfig, LoraConfig, PeftModel, get_peft_model
from peft.utils import infer_device


PEFT_VERSION = peft.__version__
REGRESSION_DIR = tempfile.mkdtemp(prefix="peft_regression_")
HF_TOKEN = os.environ.get("HF_TOKEN")
# the repo has to be created manually once, it is not automatically created
HF_REPO = "peft-internal-testing/regression-tests"


@pytest.fixture(scope="session", autouse=True)
def setup_tearndown():
    # Use a pytest session-scoped fixture to setup and teardown exactly once per session. AFAICT, unittest does not
    # provide such a feature

    # download regression artifacts from Hugging Face Hub at the start
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=REGRESSION_DIR,
        # Don't use symlink, because this prevents us from properly cleaning up the files once finished
        local_dir_use_symlinks=False,
    )

    yield

    # delete regression artifacts at the end of the test session; optionally, upload them first if in creation mode
    creation_mode = strtobool(os.environ.get("REGRESSION_CREATION_MODE", "False"))
    if creation_mode:
        # upload the regression directory to Hugging Face Hub, will overwrite by default
        upload_folder(
            repo_id=HF_REPO,
            folder_path=REGRESSION_DIR,
            token=HF_TOKEN,
        )

    shutil.rmtree(REGRESSION_DIR)


def strtobool(val):
    """Copied from distutils.util"""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")


# same as in ..testing_utils.py but cannot be imported
def require_torch_gpu(test_case):
    """
    Decorator marking a test that requires a GPU. Will be skipped when no GPU is available.

    Copies from tsting_utils.py.

    """
    if not torch.cuda.is_available():
        return unittest.skip("test requires GPU")(test_case)
    else:
        return test_case


# same as in ..testing_utils.py but cannot be imported
def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires the bitsandbytes library. Will be skipped when the library is not installed.

    Copies from tsting_utils.py.

    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return unittest.skip("test requires bitsandbytes")(test_case)
    else:
        return test_case


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
    filename = os.path.join(path, peft.utils.SAFETENSORS_WEIGHTS_NAME)
    if os.path.exists(filename) and not force:
        return

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(filename) and force:
        print(f"Overriding existing model in {path}", file=sys.stderr)

    model.save_pretrained(path)


def load_output(name):
    filename = os.path.join(REGRESSION_DIR, name, "output.pt")
    return torch.load(filename)


@pytest.mark.regression
class RegressionTester(unittest.TestCase):
    """Base class for regression testing

    Child classes must call assert_results_equal_or_store and pass the model outtput, as well as a unique name that
    describes the setting (e.g. "lora_opt-350m_bnb_4bit"). They also need to implement get_output(model) to get the
    model output, and load_base_model(name) to load the base model. Don't forget to fix the seed in load_base_model.
    """

    torch_device = infer_device()

    def setUp(self):
        self.tol = 1e-4
        self.creation_mode = strtobool(os.environ.get("REGRESSION_CREATION_MODE", "False"))
        self.force_mode = strtobool(os.environ.get("REGRESSION_FORCE_MODE", "False"))
        if self.force_mode and not self.creation_mode:
            raise RuntimeError("REGRESSION_FORCE_MODE can only be used together with REGRESSION_CREATION_MODE")
        if self.creation_mode:
            self.check_clean_git_status(self.force_mode)
            if HF_TOKEN is None:
                raise RuntimeError("HF_TOKEN environment variable must be set in creation mode")

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
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Model output for {name} is not finite")

            output2 = self.get_output(model)
            if not torch.allclose(output, output2):
                raise RuntimeError(f"Model output for {name} is not deterministic")

            save_output(output, name, force=self.force_mode)
            save_model(model, name, force=self.force_mode)

    def _assert_results_equal(self, name):
        path = os.path.join(REGRESSION_DIR, name)
        versions = os.listdir(path)
        for version in versions:  # each directory corresponds to a version
            output_loaded = load_output(os.path.join(name, version))
            base_model = self.load_base_model()
            model = PeftModel.from_pretrained(base_model, os.path.join(path, version))
            output = self.get_output(model)
            assert torch.allclose(output_loaded, output, atol=self.tol, rtol=self.tol)

    def get_output(self, model):
        raise NotImplementedError

    def load_base_model(self):
        raise NotImplementedError


##############
# TEST CASES #
##############


class TestMlp(RegressionTester):
    def get_output(self, model):
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        with torch.inference_mode():
            output = model(input)
        return output

    def load_base_model(self):
        class MLP(nn.Module):
            def __init__(self, bias=True):
                super().__init__()
                self.lin0 = nn.Linear(10, 20, bias=bias)
                self.relu = nn.ReLU()
                self.lin1 = nn.Linear(20, 2, bias=bias)
                self.sm = nn.LogSoftmax(dim=-1)

            def forward(self, X):
                X = X.float()
                X = self.lin0(X)
                X = self.relu(X)
                X = self.lin1(X)
                X = self.sm(X)
                return X

        self.fix_seed()
        return MLP().to(self.torch_device)

    def test_lora(self):
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
            target_modules=["lin0"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_mlp")

    def test_adalora(self):
        base_model = self.load_base_model()
        config = AdaLoraConfig(
            r=8,
            init_lora_weights=False,
            target_modules=["lin0"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "adalora_mlp")

    def test_ia3(self):
        base_model = self.load_base_model()
        config = IA3Config(
            init_ia3_weights=False,
            target_modules=["lin0"],
            feedforward_modules=["lin0"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "ia3_mlp")

    def test_ia3_no_ff(self):
        base_model = self.load_base_model()
        config = IA3Config(
            init_ia3_weights=False,
            target_modules=["lin0"],
            feedforward_modules=[],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "ia3_no_ff_mlp")

    def test_loha(self):
        # TODO
        self.skipTest("Skipping LoHa for now because init is not seedable")
        base_model = self.load_base_model()
        config = LoHaConfig(
            r=8,
            init_weights=False,
            target_modules=["lin0"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "loha_mlp")

    def test_lokr(self):
        # TODO
        self.skipTest("Skipping LoKr for now because init is not seedable")
        base_model = self.load_base_model()
        config = LoKrConfig(
            r=8,
            target_modules=["lin0"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lokr_mlp")

    def test_lora_modules_to_save(self):
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
            target_modules=["lin0"],
            modules_to_save=["lin1"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_mlp_modules_to_save")


class TestLoraEmbConv1D(RegressionTester):
    def get_output(self, model):
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        with torch.inference_mode():
            output = model(input)
        return output

    def load_base_model(self):
        class ModelEmbConv1D(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(100, 5)
                self.conv1d = Conv1D(1, 5)
                self.relu = nn.ReLU()
                self.flat = nn.Flatten()
                self.lin0 = nn.Linear(10, 2)
                self.sm = nn.LogSoftmax(dim=-1)

            def forward(self, X):
                X = self.emb(X)
                X = self.conv1d(X)
                X = self.relu(X)
                X = self.flat(X)
                X = self.lin0(X)
                X = self.sm(X)
                return X

        self.fix_seed()
        return ModelEmbConv1D().to(self.torch_device)

    def test_lora(self):
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
            target_modules=["emb", "conv1d"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_emb_conv1d")


class TestLoraConv2D(RegressionTester):
    def get_output(self, model):
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)
        with torch.inference_mode():
            output = model(input)
        return output

    def load_base_model(self):
        class ModelConv2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = nn.Conv2d(5, 10, 3)
                self.relu = nn.ReLU()
                self.flat = nn.Flatten()
                self.lin0 = nn.Linear(10, 2)
                self.sm = nn.LogSoftmax(dim=-1)

            def forward(self, X):
                X = X.float().reshape(2, 5, 3, 3)
                X = self.conv2d(X)
                X = self.relu(X)
                X = self.flat(X)
                X = self.lin0(X)
                X = self.sm(X)
                return X

        self.fix_seed()
        return ModelConv2D().to(self.torch_device)

    def test_lora(self):
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
            target_modules=["conv2d"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_conv2d")

    def test_ia3(self):
        base_model = self.load_base_model()
        config = IA3Config(
            init_ia3_weights=False,
            target_modules=["conv2d"],
            feedforward_modules=["conv2d"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "ia3_conv2d")

    def test_loha(self):
        # TODO
        self.skipTest("Skipping LoHa for now because init is not seedable")
        base_model = self.load_base_model()
        config = LoHaConfig(
            r=8,
            init_weights=False,
            target_modules=["conv2d"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "loha_conv2d")

    def test_lokr(self):
        # TODO
        self.skipTest("Skipping LoKr for now because init is not seedable")
        base_model = self.load_base_model()
        config = LoKrConfig(
            r=8,
            init_weights=False,
            target_modules=["conv2d"],
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lokr_conv2d")


class TestOpt(RegressionTester):
    def get_output(self, model):
        input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(self.torch_device)
        with torch.inference_mode():
            output = model(input).logits
        return output

    def load_base_model(self):
        self.fix_seed()
        return AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to(self.torch_device)

    def test_lora(self):
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_opt-350m")

    def test_adalora(self):
        base_model = self.load_base_model()
        config = AdaLoraConfig(
            r=8,
            init_lora_weights=False,
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "adalora_opt-350m")

    def test_ia3(self):
        base_model = self.load_base_model()
        config = IA3Config(init_ia3_weights=False)
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "ia3_opt-350m")


@require_torch_gpu
@require_bitsandbytes
class TestOpt8bitBnb(RegressionTester):
    def get_output(self, model):
        input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(self.torch_device)
        with torch.inference_mode():
            output = model(input).logits
        return output

    def load_base_model(self):
        self.fix_seed()
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-350m",
            load_in_8bit=True,
        )
        return model

    def test_lora_8bit(self):
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_opt-350m_bnb_8bit")

    def test_adalora(self):
        # TODO
        self.skipTest(
            "Skipping AdaLora for now, getting TypeError: unsupported operand type(s) for +=: 'dict' and 'Tensor'"
        )
        base_model = self.load_base_model()
        config = AdaLoraConfig(
            init_r=6,
            target_r=4,
            tinit=50,
            tfinal=100,
            deltaT=5,
            beta1=0.3,
            beta2=0.3,
            orth_reg_weight=0.2,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "adalora_opt-350m_8bit")


@require_torch_gpu
@require_bitsandbytes
class TestOpt4bitBnb(RegressionTester):
    def get_output(self, model):
        input = torch.LongTensor([[1, 0, 1, 0, 1, 2]]).to(self.torch_device)
        with torch.inference_mode():
            output = model(input).logits
        return output

    def load_base_model(self):
        self.fix_seed()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_type=torch.float32,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-350m",
            quantization_config=bnb_config,
            torch_dtype=torch.float32,
        )
        return model

    def test_lora_4bit(self):
        base_model = self.load_base_model()
        config = LoraConfig(
            r=8,
            init_lora_weights=False,
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "lora_opt-350m_bnb_4bit")

    def test_adalora(self):
        # TODO
        self.skipTest("Skipping AdaLora for now because of a bug, see #1113")
        base_model = self.load_base_model()
        config = AdaLoraConfig(
            init_r=6,
            target_r=4,
            tinit=50,
            tfinal=100,
            deltaT=5,
            beta1=0.3,
            beta2=0.3,
            orth_reg_weight=0.2,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, config)
        self.assert_results_equal_or_store(model, "adalora_opt-350m_4bit")
