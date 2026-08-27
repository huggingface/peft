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

# P-tuning

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/p-tuning.png"/>
</div>
<small>Prompt tokens can be inserted anywhere in the input sequence, and they are optimized by a prompt encoder <a href="https://hf.co/papers/2103.10385">(image source)</a>.</small>

[P-tuning](https://hf.co/papers/2103.10385) is designed for natural language understanding (NLU) tasks and all language models.

The abstract from the paper is:

*While GPTs with traditional fine-tuning fail to achieve strong results on natural language understanding (NLU), we show that GPTs can be better than or comparable to similar-sized BERTs on NLU tasks with a novel method P-tuning -- which employs trainable continuous prompt embeddings. On the knowledge probing (LAMA) benchmark, the best GPT recovers 64\% (P@1) of world knowledge without any additional text provided during test time, which substantially improves the previous best by 20+ percentage points. On the SuperGlue benchmark, GPTs achieve comparable and sometimes better performance to similar-sized BERTs in supervised learning. Importantly, we find that P-tuning also improves BERTs' performance in both few-shot and supervised settings while largely reducing the need for prompt engineering. Consequently, P-tuning outperforms the state-of-the-art approaches on the few-shot SuperGlue benchmark.*.

The method adds trainable prompt embeddings to the input that is optimized by a prompt encoder to find a better prompt, eliminating the need to manually design prompts. The prompt tokens can be added anywhere in the input sequence, and p-tuning also introduces anchor tokens for improving performance. A prompt encoder (a bidirectional long-short term memory network or LSTM) is used to optimize the prompt parameters. Unlike prefix tuning:

- the prompt tokens can be inserted anywhere in the input sequence, and it isn't restricted to only the beginning
- the prompt tokens are only added to the input instead of adding them to every layer of the model
- introducing *anchor* tokens can improve performance because they indicate characteristics of a component in the input sequence

The paper's results suggest that P-tuning is more efficient than manually crafting prompts, and it enables GPT-like models to compete with BERT-like models on NLU tasks.

## Usage

Create a [`PromptEncoderConfig`] with the task type, the number of virtual tokens to add and learn, and the hidden size of the encoder for learning the prompt parameters.

```py
from peft import PromptEncoderConfig, get_peft_model

peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"
```

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=P_TUNING"
	frameborder="0"
	width="850"
	height="1000"
></iframe>


# API

## PromptEncoderConfig

[[autodoc]] tuners.p_tuning.config.PromptEncoderConfig

## PromptEncoder

[[autodoc]] tuners.p_tuning.model.PromptEncoder

