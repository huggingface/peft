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

from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError


def hub_file_exists(repo_id: str, filename: str, revision: str = None, repo_type: str = None) -> bool:
    r"""
    Checks if a file exists in a remote Hub repository.
    """
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
    try:
        get_hf_file_metadata(url)
        return True
    except EntryNotFoundError:
        return False
