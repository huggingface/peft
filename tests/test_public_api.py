# Copyright 2026-present the HuggingFace Inc. team.
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
"""Tests for the integrity of the top-level ``peft`` public API."""

from collections import Counter

import peft


class TestPublicApi:
    def test_all_has_no_duplicates(self):
        """``peft.__all__`` must not contain duplicate names.

        Duplicate entries silently confuse documentation tooling, IDE
        auto-import, and linters that rely on ``__all__`` to determine the
        package's public surface.
        """
        duplicates = {name: count for name, count in Counter(peft.__all__).items() if count > 1}
        assert not duplicates, f"Duplicate names in peft.__all__: {duplicates}"

    def test_all_names_are_importable(self):
        """Every name in ``peft.__all__`` must be accessible on the package.

        Catches stale or mistyped entries that would break
        ``from peft import *``.
        """
        missing = [name for name in peft.__all__ if not hasattr(peft, name)]
        assert not missing, f"Names in peft.__all__ but not on the package: {missing}"
