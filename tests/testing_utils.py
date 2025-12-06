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
import os
from contextlib import contextmanager
from functools import lru_cache, wraps
from unittest import mock

import numpy as np
import pytest
import torch
from accelerate.test_utils.testing import get_backend
from datasets import load_dataset

from peft import (
    AdaLoraConfig,
    IA3Config,
    LNTuningConfig,
    LoraConfig,
    PromptLearningConfig,
    VBLoRAConfig,
)
from peft.import_utils import (
    is_aqlm_available,
    is_auto_awq_available,
    is_auto_gptq_available,
    is_eetq_available,
    is_gptqmodel_available,
    is_hqq_available,
    is_optimum_available,
    is_torchao_available,
)
from peft.utils import is_transformers_ge_v5


# Globally shared model cache used by `hub_online_once`.
_HUB_MODEL_ACCESSES = {}


torch_device, device_count, memory_allocated_func = get_backend()


def require_non_cpu(test_case):
    """
    Decorator marking a test that requires a hardware accelerator backend. These tests are skipped when there are no
    hardware accelerator available.
    """
    return pytest.mark.skipif(torch_device == "cpu", reason="test requires a hardware accelerator")(test_case)


def require_non_xpu(test_case):
    """
    Decorator marking a test that should be skipped for XPU.
    """
    return pytest.mark.skipif(torch_device == "xpu", reason="test requires a non-XPU")(test_case)


def require_torch_gpu(test_case):
    """
    Decorator marking a test that requires a GPU. Will be skipped when no GPU is available.
    """
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")(test_case)


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires multiple GPUs. Will be skipped when less than 2 GPUs are available.
    """
    multi_cuda_unavailable = not torch.cuda.is_available() or (device_count < 2)
    return pytest.mark.skipif(multi_cuda_unavailable, reason="test requires multiple GPUs")(test_case)


def require_torch_multi_accelerator(test_case):
    """
    Decorator marking a test that requires multiple hardware accelerators. These tests are skipped on a machine without
    multiple accelerators.
    """
    multi_device_unavailable = (torch_device == "cpu") or (device_count < 2)
    return pytest.mark.skipif(multi_device_unavailable, reason="test requires multiple hardware accelerators")(
        test_case
    )


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
    is_gptq_avaiable = is_gptqmodel_available() or is_auto_gptq_available()
    return pytest.mark.skipif(not is_gptq_avaiable, reason="test requires auto-gptq")(test_case)


def require_gptqmodel(test_case):
    """
    Decorator marking a test that requires gptqmodel. These tests are skipped when gptqmodel isn't installed.
    """
    return pytest.mark.skipif(not is_gptqmodel_available(), reason="test requires gptqmodel")(test_case)


def require_aqlm(test_case):
    """
    Decorator marking a test that requires aqlm. These tests are skipped when aqlm isn't installed.
    """
    return pytest.mark.skipif(not is_aqlm_available(), reason="test requires aqlm")(test_case)


def require_hqq(test_case):
    """
    Decorator marking a test that requires aqlm. These tests are skipped when aqlm isn't installed.
    """
    return pytest.mark.skipif(not is_hqq_available(), reason="test requires hqq")(test_case)


def require_auto_awq(test_case):
    """
    Decorator marking a test that requires auto-awq. These tests are skipped when auto-awq isn't installed.
    """
    return pytest.mark.skipif(not is_auto_awq_available(), reason="test requires auto-awq")(test_case)


def require_eetq(test_case):
    """
    Decorator marking a test that requires eetq. These tests are skipped when eetq isn't installed.
    """
    return pytest.mark.skipif(not is_eetq_available(), reason="test requires eetq")(test_case)


def require_optimum(test_case):
    """
    Decorator marking a test that requires optimum. These tests are skipped when optimum isn't installed.
    """
    return pytest.mark.skipif(not is_optimum_available(), reason="test requires optimum")(test_case)


def require_torchao(test_case):
    """
    Decorator marking a test that requires torchao. These tests are skipped when torchao isn't installed.
    """
    return pytest.mark.skipif(not is_torchao_available(), reason="test requires torchao")(test_case)


def require_deterministic_for_xpu(test_case):
    @wraps(test_case)
    def wrapper(*args, **kwargs):
        if torch_device == "xpu":
            original_state = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                return test_case(*args, **kwargs)
            finally:
                torch.use_deterministic_algorithms(original_state)
        else:
            return test_case(*args, **kwargs)

    return wrapper


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


@lru_cache
def load_dataset_english_quotes():
    # can't use pytest fixtures for now because of unittest style tests
    data = load_dataset("ybelkada/english_quotes_copy")
    return data


@lru_cache
def load_cat_image():
    # can't use pytest fixtures for now because of unittest style tests
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]
    return image


def set_init_weights_false(config_cls, kwargs):
    # helper function that sets the config kwargs such that the model is *not* initialized as an identity transform
    kwargs = kwargs.copy()

    if issubclass(config_cls, PromptLearningConfig):
        return kwargs
    if config_cls in (LNTuningConfig, VBLoRAConfig):
        return kwargs

    if config_cls in (LoraConfig, AdaLoraConfig):
        kwargs["init_lora_weights"] = False
    elif config_cls == IA3Config:
        kwargs["init_ia3_weights"] = False
    else:
        kwargs["init_weights"] = False
    return kwargs


@contextmanager
def hub_online_once(model_id: str):
    """Set env[HF_HUB_OFFLINE]=1 (and patch transformers/hugging_face_hub to think that it was always that way)
    for model ids that were already to avoid contacting the hub twice for the same model id in the context. The global
    variable `_HUB_MODEL_ACCESSES` tracks the number of hits per model id between `hub_online_once` calls.

    The reason for doing a context manager and not patching specific methods (e.g., `from_pretrained`) is that there
    are a lot of places (`PeftConfig.from_pretrained`, `get_peft_state_dict`, `load_adapter`, ...) that possibly
    communicate with the hub to download files / check versions / etc.

    Note that using this context manager can cause problems when used in code sections that access different resources.
    Example:

    ```
    def test_something(model_id, config_kwargs):
        with hub_online_once(model_id):
            model = ...from_pretrained(model_id)
            self.do_something_specific_with_model(model)
    ```
    It is assumed that `do_something_specific_with_model` is an absract method that is implement by several tests.
    Imagine the first test simply does `model.generate([1,2,3])`. The second call from another test suite however uses
    a tokenizer (`AutoTokenizer.from_pretrained(model_id)`) - this will fail since the first pass was online but didn't
    use the tokenizer and we're now in offline mode and cannot fetch the tokenizer. The recommended workaround is to
    extend the cache key (`model_id` passed to `hub_online_once` in this case) by something in case the tokenizer is
    used, so that these tests don't share a cache pool with the tests that don't use a tokenizer.

    It is best to avoid using this context manager in *yield* fixtures (normal fixtures are fine) as this is equivalent
    to wrapping the whole test in the context manager without explicitly writing it out, leading to unexpected
    `HF_HUB_OFFLINE` behavior in the test body.
    """
    global _HUB_MODEL_ACCESSES
    override = {}

    try:
        if model_id in _HUB_MODEL_ACCESSES:
            override = {"HF_HUB_OFFLINE": "1"}
            _HUB_MODEL_ACCESSES[model_id] += 1
        elif model_id not in _HUB_MODEL_ACCESSES:
            _HUB_MODEL_ACCESSES[model_id] = 0
        is_offline = override.get("HF_HUB_OFFLINE", False) == "1"

        with (
            # strictly speaking it is not necessary to set the environment variable since most code that's out there
            # is evaluating it at import time and we'd have to reload the modules for it to take effect. It's
            # probably still a good idea to have it if there's some dynamic code that checks it.
            mock.patch.dict(os.environ, override),
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", is_offline),
        ):
            if is_transformers_ge_v5:
                with mock.patch("transformers.utils.hub.is_offline_mode", lambda: is_offline):
                    yield
            else:  # TODO remove if transformers <= 4 no longer supported
                with mock.patch("transformers.utils.hub._is_offline_mode", is_offline):
                    yield
    except Exception:
        # in case of an error we have to assume that we didn't access the model properly from the hub
        # for the first time, so the next call cannot be considered cached.
        if _HUB_MODEL_ACCESSES.get(model_id) == 0:
            del _HUB_MODEL_ACCESSES[model_id]
        raise
