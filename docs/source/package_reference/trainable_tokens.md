<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainable Tokens

The Trainable Tokens method provides a way to target specific token embeddings for fine-tuning without resorting to
training the full embedding matrix or using an adapter on the embedding matrix. It is based on the initial implementation from
[here](https://github.com/huggingface/peft/pull/1541).

The method only targets specific tokens and selectively trains the token indices you specify. Consequently the
required RAM will be lower and disk memory is also significantly lower than storing the full fine-tuned embedding matrix.

Some preliminary benchmarks acquired with [this script](https://github.com/huggingface/peft/blob/main/scripts/train_memory.py)
suggest that for `gemma-2-2b` (which has a rather large embedding matrix) you can save ~4 GiB VRAM with Trainable Tokens
over fully fine-tuning the embedding matrix. While LoRA will use comparable amounts of VRAM it might also target
tokens you don't want to be changed. Note that these are just indications and varying embedding matrix sizes might skew
these numbers a bit.

Note that this method does not add tokens for you, you have to add tokens to the tokenizer yourself and resize the
embedding matrix of the model accordingly. This method will only re-train the embeddings for the tokens you specify.
This method can also be used in conjunction with LoRA layers! See [the LoRA developer guide](../developer_guides/lora#efficiently-train-tokens-alongside-lora).

> [!TIP]
> Saving the model with [`~PeftModel.save_pretrained`] or retrieving the state dict using
> [`get_peft_model_state_dict`] when adding new tokens may save the full embedding matrix instead of only the difference
> as a precaution because the embedding matrix was resized. To save space you can disable this behavior by setting
> `save_embedding_layers=False` when calling `save_pretrained`. This is safe to do as long as you don't modify the
> embedding matrix through other means as well, as such changes will be not tracked by trainable tokens.

## TrainableTokensConfig

[[autodoc]] tuners.trainable_tokens.config.TrainableTokensConfig

## TrainableTokensModel

[[autodoc]] tuners.trainable_tokens.model.TrainableTokensModel

