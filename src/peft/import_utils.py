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
import importlib
import importlib.metadata as importlib_metadata
import platform
from functools import lru_cache

import packaging.version
import torch


@lru_cache
def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


@lru_cache
def is_bnb_4bit_available() -> bool:
    if not is_bnb_available():
        return False

    import bitsandbytes as bnb

    return hasattr(bnb.nn, "Linear4bit")


@lru_cache
def is_auto_gptq_available():
    if importlib.util.find_spec("auto_gptq") is not None:
        AUTOGPTQ_MINIMUM_VERSION = packaging.version.parse("0.5.0")
        version_autogptq = packaging.version.parse(importlib_metadata.version("auto_gptq"))
        if AUTOGPTQ_MINIMUM_VERSION <= version_autogptq:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq}, "
                f"but only versions above {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )


@lru_cache
def is_gptqmodel_available():
    if importlib.util.find_spec("gptqmodel") is not None:
        GPTQMODEL_MINIMUM_VERSION = packaging.version.parse("2.0.0")
        OPTIMUM_MINIMUM_VERSION = packaging.version.parse("1.24.0")
        version_gptqmodel = packaging.version.parse(importlib_metadata.version("gptqmodel"))
        if GPTQMODEL_MINIMUM_VERSION <= version_gptqmodel:
            if is_optimum_available():
                version_optimum = packaging.version.parse(importlib_metadata.version("optimum"))
                if OPTIMUM_MINIMUM_VERSION <= version_optimum:
                    return True
                else:
                    raise ImportError(
                        f"gptqmodel requires optimum version `{OPTIMUM_MINIMUM_VERSION}` or higher. Found version `{version_optimum}`, "
                        f"but only versions above `{OPTIMUM_MINIMUM_VERSION}` are supported"
                    )
            else:
                raise ImportError(
                    f"gptqmodel requires optimum version `{OPTIMUM_MINIMUM_VERSION}` or higher to be installed."
                )
        else:
            raise ImportError(
                f"Found an incompatible version of gptqmodel. Found version `{version_gptqmodel}`, "
                f"but only versions above `{GPTQMODEL_MINIMUM_VERSION}` are supported"
            )


@lru_cache
def is_optimum_available() -> bool:
    return importlib.util.find_spec("optimum") is not None


@lru_cache
def is_torch_tpu_available(check_device=True):
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # We need to check if `xla_device` can be found, will raise a RuntimeError if not
            try:
                import torch_xla.core.xla_model as xm

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


@lru_cache
def is_aqlm_available():
    return importlib.util.find_spec("aqlm") is not None


@lru_cache
def is_auto_awq_available():
    return importlib.util.find_spec("awq") is not None


@lru_cache
def is_eetq_available():
    return importlib.util.find_spec("eetq") is not None


@lru_cache
def is_hqq_available():
    return importlib.util.find_spec("hqq") is not None


@lru_cache
def is_inc_available():
    return importlib.util.find_spec("neural_compressor") is not None


@lru_cache
def is_torchao_available():
    if importlib.util.find_spec("torchao") is None:
        return False

    TORCHAO_MINIMUM_VERSION = packaging.version.parse("0.4.0")
    try:
        torchao_version = packaging.version.parse(importlib_metadata.version("torchao"))
    except importlib_metadata.PackageNotFoundError:
        # Same idea as in diffusers:
        # https://github.com/huggingface/diffusers/blob/9f06a0d1a4a998ac6a463c5be728c892f95320a8/src/diffusers/utils/import_utils.py#L351-L357
        # It's not clear under what circumstances `importlib_metadata.version("torchao")` can raise an error even
        # though `importlib.util.find_spec("torchao") is not None` but it has been observed, so adding this for
        # precaution.
        return False

    if torchao_version < TORCHAO_MINIMUM_VERSION:
        raise ImportError(
            f"Found an incompatible version of torchao. Found version {torchao_version}, "
            f"but only versions above {TORCHAO_MINIMUM_VERSION} are supported"
        )
    return True


@lru_cache
def is_xpu_available(check_device=False):
    """
    Checks if XPU acceleration is available and potentially if a XPU is in the environment
    """

    system = platform.system()
    if system == "Darwin":
        return False
    else:
        if check_device:
            try:
                # Will raise a RuntimeError if no XPU is found
                _ = torch.xpu.device_count()
                return torch.xpu.is_available()
            except RuntimeError:
                return False
        return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache
def is_diffusers_available():
    return importlib.util.find_spec("diffusers") is not None
