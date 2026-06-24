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

# Multitask prompt tuning

[Multitask prompt tuning](https://huggingface.co/papers/2303.02861)  decomposes the soft prompts of each task into a single learned transferable prompt instead of a separate prompt for each task. The single learned prompt can be adapted for each task by multiplicative low rank updates.

The abstract from the paper is:

*Prompt tuning, in which a base pretrained model is adapted to each task via conditioning on learned prompt vectors, has emerged as a promising approach for efficiently adapting large language models to multiple downstream tasks. However, existing methods typically learn soft prompt vectors from scratch, and it has not been clear how to exploit the rich cross-task knowledge with prompt vectors in a multitask learning setting. We propose multitask prompt tuning (MPT), which first learns a single transferable prompt by distilling knowledge from multiple task-specific source prompts. We then learn multiplicative low rank updates to this shared prompt to efficiently adapt it to each downstream target task. Extensive experiments on 23 NLP datasets demonstrate that our proposed approach outperforms the state-of-the-art methods, including the full finetuning baseline in some cases, despite only tuning 0.035% as many task-specific parameters*.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt.png"/>
</div>
<small><a href="https://hf.co/papers/2303.02861">Multitask prompt tuning enables parameter-efficient transfer learning</a>.</small>

MPT consists of two stages:

1. source training - for each task, its soft prompt is decomposed into task-specific vectors. The task-specific vectors are multiplied together to form another matrix W, and the Hadamard product is used between W and a shared prompt matrix P to generate a task-specific prompt matrix. The task-specific prompts are distilled into a single prompt matrix that is shared across all tasks. This prompt is trained with multitask training.
2. target adaptation - to adapt the single prompt for a target task, a target prompt is initialized and expressed as the Hadamard product of the shared prompt matrix and the task-specific low-rank prompt matrix.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt-decomposition.png"/>
</div>
<small><a href="https://hf.co/papers/2103.10385">Prompt decomposition</a>.</small>

## Benchmark overview

There is no benchmark for this method yet. Feel free to contribute an experiment
configuration but make sure to first create an issue
[here](https://github.com/huggingface/peft/issues).


# API

## MultitaskPromptTuningConfig

[[autodoc]] tuners.multitask_prompt_tuning.config.MultitaskPromptTuningConfig

## MultitaskPromptEmbedding

[[autodoc]] tuners.multitask_prompt_tuning.model.MultitaskPromptEmbedding
