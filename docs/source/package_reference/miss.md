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

# MiSS

[MiSS (Matrix Shard Sharing)](https://huggingface.co/papers/2409.15371) is a PEFT method that 
achieves an excellent balance between model performance and computational efficiency. Unlike LoRA 
which uses two separate low-rank matrices (A and B), MiSS decomposes the weight matrix into 
multiple fragment matrices and uses a single shared trainable matrix across all fragments. This 
shard-sharing mechanism reduces the number of trainable parameters while maintaining strong 
adaptability, and significantly improves initialization speed and training throughput compared 
to LoRA and its variants.

The abstract from the paper is:

*Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), 
effectively reduce the number of trainable parameters in Large Language Models (LLMs). However, 
as model scales continue to grow, the demand for computational resources remains a significant 
challenge. Existing LoRA variants often struggle to strike an optimal balance between adaptability 
(model performance and convergence speed) and efficiency (computational overhead, memory usage, 
and initialization time). This paper introduces MiSS (Matrix Shard Sharing), a novel PEFT approach 
that addresses this trade-off through a simple shard-sharing mechanism. MiSS leverages the insight 
that a low-rank adaptation can be achieved by decomposing the weight matrix into multiple fragment 
matrices and utilizing a shared, trainable common fragment. This method constructs the low-rank 
update matrix through the replication of these shared, partitioned shards. We also propose a 
hardware-efficient and broadly applicable implementation for MiSS. Extensive experiments conducted 
on a range of tasks, alongside a systematic analysis of computational performance, demonstrate 
MiSS's superiority. The results show that MiSS significantly outperforms standard LoRA and its 
prominent variants in both model performance metrics and computational efficiency, including 
initialization speed and training throughput. By effectively balancing expressive power and resource 
utilization, MiSS offers a compelling solution for efficiently adapting large-scale models.*

## When to use MiSS

MiSS is a good choice when:

- You want faster initialization and higher training throughput than LoRA
- You need to reduce memory usage while maintaining model performance
- You are fine-tuning large language models where computational efficiency matters
- You want a drop-in alternative to LoRA with minimal configuration changes

If you need stronger expressiveness at the cost of some efficiency, consider the `bat` 
initialization variant (see below).

## init_weights modes

MiSS supports three initialization modes via the `init_weights` parameter:

- `True` (default): Standard MiSS initialization. Best starting point for most use cases.
- `"bat"`: Enables nonlinear updates across different shards. Produces better results than 
  standard MiSS but uses more memory and is approximately twice as slow. Use this when 
  performance is the priority over efficiency.
- `"mini"`: Uses a smaller rank along the `out_features` dimension, controlled by `mini_r`. 
  This reduces trainable parameters further. When using this mode, `mini_r` must be set and 
  `out_features` must be divisible by `mini_r`.

## Quick start

```python
import torch
from peft import MissConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Standard MiSS
config = MissConfig(
    r=64,
    miss_dropout=0.01,
    task_type="CAUSAL_LM"
)

# BAT variant — better performance, more memory
# config = MissConfig(
#     r=64,
#     init_weights="bat",
#     task_type="CAUSAL_LM"
# )

# Mini variant — fewer trainable parameters
# config = MissConfig(
#     r=64,
#     init_weights="mini",
#     mini_r=8,
#     task_type="CAUSAL_LM"
# )

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

For a full fine-tuning example including training and inference, see the 
[MiSS fine-tuning example](https://github.com/huggingface/peft/tree/main/examples/miss_finetuning).

## MissConfig

[[autodoc]] tuners.miss.config.MissConfig

## MissModel

[[autodoc]] tuners.miss.model.MissModel