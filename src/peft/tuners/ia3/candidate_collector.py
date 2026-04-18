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

"""IA3 candidate enumerator. STUB. Returns empty until implemented.

TODO: implement candidate enumeration for IA3. The function should yield
full module names that IA3 could validly adapt. IA3 has a feedforward vs
attention distinction (see PR #3154 comment 4250868072) which is out of
scope for the initial draft and should be addressed in a follow-up PR.
"""

from peft.tuners.target_suggester import register_candidate_collector


@register_candidate_collector("IA3")
def ia3_candidates(model):
    """Yield full module names that IA3 could validly adapt. STUB."""
    return iter(())
