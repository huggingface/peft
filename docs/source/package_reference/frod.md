<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# FRoD: Full-Rank Efficient Fine-Tuning with Rotational Degrees

FRoD is a parameter-efficient fine-tuning method that combines a shared full-rank basis with sparse learnable
rotational degrees. The adapter update is expressed through fixed projection tensors and trainable coefficients, which
allows FRoD to apply full-rank updates while keeping the number of trained parameters small.

When saving the adapter parameters, it is possible to avoid storing the projection tensors by setting
`save_projection=False` on the `FRODConfig`. In that case, the projections are restored from the base model weights and
the fixed random seed from `projection_prng_key`. This reduces checkpoint size, but the default is
`save_projection=True` to make checkpoint loading independent of regeneration details.

FRoD currently has the following constraint:

- Only `nn.Linear` and `transformers.pytorch_utils.Conv1D` layers are supported.

## FRODConfig

[[autodoc]] tuners.frod.config.FRODConfig

## FRODModel

[[autodoc]] tuners.frod.model.FRODModel
