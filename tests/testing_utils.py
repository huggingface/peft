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
import unittest
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from peft.import_utils import is_aqlm_available, is_auto_awq_available, is_auto_gptq_available, is_optimum_available


def require_torch_gpu(test_case):
    """
    Decorator marking a test that requires a GPU. Will be skipped when no GPU is available.
    """
    if not torch.cuda.is_available():
        return unittest.skip("test requires GPU")(test_case)
    else:
        return test_case


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires multiple GPUs. Will be skipped when less than 2 GPUs are available.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return unittest.skip("test requires multiple GPUs")(test_case)
    else:
        return test_case


def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires the bitsandbytes library. Will be skipped when the library is not installed.
    """
    try:
        import bitsandbytes  # noqa: F401

        test_case = pytest.mark.bitsandbytes(test_case)
    except ImportError:
        test_case = pytest.mark.skip(reason="test requires bitsandbytes")(test_case)
    return test_case


def require_auto_gptq(test_case):
    """
    Decorator marking a test that requires auto-gptq. These tests are skipped when auto-gptq isn't installed.
    """
    return unittest.skipUnless(is_auto_gptq_available(), "test requires auto-gptq")(test_case)


def require_aqlm(test_case):
    """
    Decorator marking a test that requires aqlm. These tests are skipped when aqlm isn't installed.
    """
    return unittest.skipUnless(is_aqlm_available(), "test requires aqlm")(test_case)


def require_auto_awq(test_case):
    """
    Decorator marking a test that requires auto-awq. These tests are skipped when auto-awq isn't installed.
    """
    return unittest.skipUnless(is_auto_awq_available(), "test requires auto-awq")(test_case)


def require_optimum(test_case):
    """
    Decorator marking a test that requires optimum. These tests are skipped when optimum isn't installed.
    """
    return unittest.skipUnless(is_optimum_available(), "test requires optimum")(test_case)


@contextmanager
def temp_seed(seed: int):
    """Temporarily set the random seed. This works for python numpy, pytorch."""

    np_state = np.random.get_state()
    np.random.seed(seed)

    torch_state = torch.random.get_rng_state()
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch_cuda_states = torch.cuda.get_rng_state_all()
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        np.random.set_state(np_state)

        torch.random.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_states)


def get_state_dict(model, unwrap_compiled=True):
    """
    Get the state dict of a model. If the model is compiled, unwrap it first.
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)
    return model.state_dict()
