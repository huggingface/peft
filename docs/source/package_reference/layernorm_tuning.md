<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LayerNorm Tuning

LayerNorm Tuning ([LN Tuning](https://huggingface.co/papers/2312.11420)) is a PEFT method that only fine-tunes the parameters of the LayerNorm layers in a model.
The paper has tested the performance of this method on large language models and has shown that it can achieve strong performance with a significant reduction in the number of trainable parameters and GPU memory usage.
However, the method is not limited to language models and can be applied to any model that uses LayerNorm layers.
In this implementation, the default is that all layernorm layers inside a model is finetuned, but it could be used to target other layer types such as `MLP` or `Attention` layers, this can be done by specifying the `target_modules` in the `LNTuningConfig`.

The abstract from the paper is:

*This paper introduces an efficient strategy to transform Large Language Models (LLMs) into Multi-Modal Large Language Models (MLLMs). By conceptualizing this transformation as a domain adaptation process, i.e., transitioning from text understanding to embracing multiple modalities, we intriguingly note that, within each attention block, tuning LayerNorm suffices to yield strong performance. Moreover, when benchmarked against other tuning approaches like full parameter finetuning or LoRA, its benefits on efficiency are substantial. For example, when compared to LoRA on a 13B model scale, performance can be enhanced by an average of over 20% across five multi-modal tasks, and meanwhile, results in a significant reduction of trainable parameters by 41.9% and a decrease in GPU memory usage by 17.6%. On top of this LayerNorm strategy, we showcase that selectively tuning only with conversational data can improve efficiency further. Beyond these empirical outcomes, we provide a comprehensive analysis to explore the role of LayerNorm in adapting LLMs to the multi-modal domain and improving the expressive power of the model.*

## LNTuningConfig

[[autodoc]] tuners.ln_tuning.config.LNTuningConfig

## LNTuningModel

[[autodoc]] tuners.ln_tuning.model.LNTuningModel