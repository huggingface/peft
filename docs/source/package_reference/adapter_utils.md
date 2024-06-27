<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LyCORIS

[LyCORIS](https://hf.co/papers/2309.14859) (Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion) are LoRA-like matrix decomposition adapters that modify the cross-attention layer of the UNet. The [LoHa](loha) and [LoKr](lokr) methods inherit from the `Lycoris` classes here.

## LycorisConfig

[[autodoc]] tuners.lycoris_utils.LycorisConfig

## LycorisLayer

[[autodoc]] tuners.lycoris_utils.LycorisLayer

## LycorisTuner

[[autodoc]] tuners.lycoris_utils.LycorisTuner