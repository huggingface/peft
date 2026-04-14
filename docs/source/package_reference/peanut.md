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

# PEANuT: Parameter-Efficient Adaptation with Weight-aware Neural Tweakers

[PEANuT](https://arxiv.org/abs/2410.01870) is a parameter-efficient fine-tuning technique that introduces
weight-aware neural tweakers to generate adapter updates from the frozen pretrained weights themselves. Instead of
learning a purely linear low-rank update as in LoRA, PEANuT conditions the adapter transformation on the base weight,
which makes the update rule more expressive while keeping the number of trainable parameters small.

PEANuT uses an input projection `A`, an output projection `B`, and optional intermediate residual encoder/decoder
pairs with non-linear activations. This makes it possible to model more complex update patterns than weight-agnostic
linear adapters while still remaining within the PEFT setting.

PEANuT currently has the following tradeoffs:

Pros:
- Higher theoretical expressiveness than linear low-rank updates.
- Better performance than LoRA on a range of tasks under similar budgets.
- Works well in very low-parameter regimes, for example around `0.2M` trainable parameters.

Cons:
- Higher memory usage than LoRA, because `ΔW` is explicitly constructed before being applied.
- Slower training and inference than LoRA, and deeper intermediate layers increase the overhead further.
- The non-linearity can require more careful hyperparameter tuning, especially learning rate and related optimization settings.

If these tradeoffs do not fit your use case, consider other PEFT methods such as LoRA.

The abstract from the paper is:

> Fine-tuning large pre-trained foundation models often yields excellent downstream performance but is prohibitively expensive when updating all parameters. Parameter-efficient fine-tuning (PEFT) methods such as LoRA alleviate this by introducing lightweight update modules, yet they commonly rely on weight-agnostic linear approximations, limiting their expressiveness. In this work, we propose PEANuT, a novel PEFT framework that introduces weight-aware neural tweakers, compact neural modules that generate task-adaptive updates conditioned on frozen pre-trained weights. PEANuT provides a flexible yet efficient way to capture complex update patterns without full model tuning. We theoretically show that PEANuT achieves equivalent or greater expressivity than existing linear PEFT methods with comparable or fewer parameters. Extensive experiments across four benchmarks with over twenty datasets demonstrate that PEANuT consistently outperforms strong baselines in both NLP and vision tasks, while maintaining low computational overhead.

## PeanutConfig

[[autodoc]] tuners.peanut.config.PeanutConfig

## PeanutModel

[[autodoc]] tuners.peanut.model.PeanutModel

