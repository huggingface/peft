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
The paper introduces it as a way to turn a pre-trained language model into a multi-modal model, and reports that tuning only the LayerNorm parameters can be sufficient for that task while reducing the number of trainable parameters and the GPU memory usage compared to full fine-tuning and LoRA.
However, the method is not limited to language models and can be applied to any model that uses LayerNorm layers.
In this implementation, the default is that all layernorm layers inside a model is finetuned, but it could be used to target other layer types such as `MLP` or `Attention` layers, this can be done by specifying the `target_modules` in the `LNTuningConfig`.

The abstract from the paper is:

*This paper introduces an efficient strategy to transform Large Language Models (LLMs) into Multi-Modal Large Language Models (MLLMs). By conceptualizing this transformation as a domain adaptation process, i.e., transitioning from text understanding to embracing multiple modalities, we intriguingly note that, within each attention block, tuning LayerNorm suffices to yield strong performance. Moreover, when benchmarked against other tuning approaches like full parameter finetuning or LoRA, its benefits on efficiency are substantial. For example, when compared to LoRA on a 13B model scale, performance can be enhanced by an average of over 20% across five multi-modal tasks, and meanwhile, results in a significant reduction of trainable parameters by 41.9% and a decrease in GPU memory usage by 17.6%. On top of this LayerNorm strategy, we showcase that selectively tuning only with conversational data can improve efficiency further. Beyond these empirical outcomes, we provide a comprehensive analysis to explore the role of LayerNorm in adapting LLMs to the multi-modal domain and improving the expressive power of the model.*

## Usage

For the task of sequence classification, one can initialize the LN Tuning config as follows:

```py
from transformers import AutoModelForSequenceClassification
from peft import LNTuningConfig, TaskType, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased",
    num_labels=2,
)
peft_config = LNTuningConfig(
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

### When to use LayerNorm Tuning

The paper evaluates LayerNorm Tuning on multi-modal adaptation, where adding a new modality to a pre-trained LLM is the task at hand, and compares it against LoRA and full fine-tuning in that setting.
If your task is adding a modality to an existing LLM, that is the scenario the paper measured.
For text-only fine-tuning the results are much less clear-cut: the benchmark comparison below shows a wide spread of outcomes across methods and tasks, so benchmark LayerNorm Tuning against LoRA on your own data before committing to it.
LayerNorm Tuning is also not the smallest adapter available — a LoRA adapter with `r=1` has fewer trainable parameters, and [IA3](./ia3) or [prompt tuning](./prompt_tuning) are smaller still. Its appeal is that it touches the LayerNorm parameters of every layer, which is a very small and stable set of parameters to adapt.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=LN_TUNING"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## LNTuningConfig

[[autodoc]] tuners.ln_tuning.config.LNTuningConfig

## LNTuningModel

[[autodoc]] tuners.ln_tuning.model.LNTuningModel
