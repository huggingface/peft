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
import unittest

import torch


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
    except ImportError:
        return unittest.skip("test requires bitsandbytes")(test_case)
    else:
        return test_case
