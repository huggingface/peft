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
import importlib
import sys


if sys.version_info[0] < 3.8:
    _is_python_greater_3_8 = False
else:
    _is_python_greater_3_8 = True


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_bnb_4bit_available():
    if _is_python_greater_3_8:
        from importlib.metadata import version

        bnb_version = version("bitsandbytes")
    else:
        from pkg_resources import get_distribution

        bnb_version = get_distribution("bitsandbytes").version
    return bnb_version >= "0.39.0"
