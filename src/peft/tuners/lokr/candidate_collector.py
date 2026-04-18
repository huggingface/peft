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

"""LoKr candidate enumerator. STUB. Returns empty until implemented.

TODO: implement candidate enumeration for LoKr. The function should yield
full module names that LoKr could validly adapt. See lora/candidate_collector.py
for the implemented LoRA pattern.
"""

from peft.tuners.target_suggester import register_candidate_collector


@register_candidate_collector("LOKR")
def lokr_candidates(model):
    """Yield full module names that LoKr could validly adapt. STUB."""
    return iter(())
