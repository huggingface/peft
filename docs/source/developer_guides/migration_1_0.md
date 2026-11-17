<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Migration guide for PEFT v1.0

This guide describes API changes in PEFT v1.0 and how to update existing code.

## Active and merged adapters

Active adapters now determine which adapters participate in the `forward` pass, including through merged weights. Merging and unmerging preserve the active adapter selection. The fact that a weight is merged is thus a 'technical detail' of how the forward pass is computed, the active adapters are the 'source of truth' of which adapters contribute to the output. Earlier PEFT versions silently ignored unmerged active adapters on layers that had any merged adapters, and permitted inactive adapters to affect predictions through merged weights.

Each affected layer must merge all of its applicable active adapters in their active order. Requests that include inactive adapters, only part of a layer's active selection, or a different order now raise `ValueError` before any layer is merged. Update calls relying on the previous behavior as follows:

- To merge an inactive adapter into a model, select it first with [`~PeftModel.set_adapter`], then call [`PeftModel.merge_adapter`].
- To merge multiple active adapters, pass all of them in their active order to the [`PeftModel.merge_adapter`] call, or callit without a selection, which defaults to all active adapters. Sequential partial merges of the same layer are rejected.
- If `["default", "other"]` is active, change `model.merge_adapter(["other", "default"])` to `model.merge_adapter(["default", "other"])`. To use the reversed composition, change the active order first with `model.base_model.set_adapter(["other", "default"])`.

The restriction of requiring merged adapters to be equal the active adapters does *not* apply to `merge_and_unload`.

Only adapters actually present in a layer count. If adapters target completely disjoint layers, either adapter can still be merged separately and their requested order does not matter.

Omitting `adapter_names` or passing `None` selects the active adapters for both merging and unloading. An empty list is a no-op for `merge_adapter`, while `merge_and_unload(adapter_names=[])` exports the base model without adapter contributions. Other PEFT method-specific merging restrictions still apply.

### Duplicate adapter names

Adapter name lists passed to `set_adapter`, `merge_adapter`, and `merge_and_unload` must contain unique names. Repeated names now raise `ValueError`. PEFT < v1.0 could apply an adapter repeatedly when its name appeared more than once.

To migrate, replace `model.base_model.set_adapter(["foo", "bar", "foo"])` with `model.base_model.set_adapter(["foo", "bar"])`. If `["foo", "bar"]` is active, replace `model.merge_adapter(["foo", "bar", "bar"])` with `model.merge_adapter(["foo", "bar"])`.
