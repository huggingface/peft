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

# Prompt tuning

[Prompt tuning](https://hf.co/papers/2104.08691) adds a task-specific, virtual prompt to the input that consists of trainable vectors in the embedding space. The virtual token parameters are updated independently of the pretrained model parameters which are frozen.

The abstract from the paper is:

*In this work, we explore "prompt tuning", a simple yet effective mechanism for learning "soft prompts" to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's "few-shot" learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method "closes the gap" and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant in that large models are costly to share and serve, and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. Our method can be seen as a simplification of the recently proposed "prefix tuning" of Li and Liang (2021), and we provide a comparison to this and other similar approaches. Finally, we show that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer, as compared to full model tuning*.

In contrast to [prefix tuning](../package_reference/prefix_tuning), only the
input of the first layer receives the virtual tokens.

## Usage

There are two decisions to take: how many virtual tokens are added to the
input of the model (`num_virtual_tokens`) - this will define how many
trainable parameters there will be - and how these tokens are initialized.

Create a [`PromptTuningConfig`] with the task type, the initial prompt tuning text to train the model with, the number of virtual tokens to add and learn, and a tokenizer.

```py
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358"
```

## Benchmark overview

<iframe
	src="https://hubnemo-peft-method-comparison-individual.hf.space/?highlight[type]=PROMPT_TUNING"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## PromptTuningConfig

[[autodoc]] tuners.prompt_tuning.config.PromptTuningConfig

## PromptEmbedding

[[autodoc]] tuners.prompt_tuning.model.PromptEmbedding
