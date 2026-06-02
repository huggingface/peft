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

# Differentially Private Fine-Tuning

Differential privacy (DP) is a training technique that limits the influence of individual training examples on a model's learned parameters. This can help reduce memorization of sensitive information while still allowing a model to learn useful patterns from data.

PEFT methods are a natural fit for differentially private training because only a small subset of model parameters are updated during fine-tuning. This reduces memory requirements and can make privacy-preserving training significantly more practical for large language models.

This tutorial demonstrates how to combine PEFT with [FastDP](https://github.com/awslabs/fast-differential-privacy) to perform differentially private fine-tuning of Transformer models. The same workflow can be applied to many PEFT methods, including LoRA, Prompt Tuning, Prefix Tuning, IA3, Vera, LoHa, LoKr, OFT, and BOFT.

## Setup

Install PEFT, Transformers, and FastDP.

```py
pip install peft transformers fastDP
```

Next, load a pretrained model and tokenizer.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-350m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

## Create a PEFT model

Configure a PEFT method and attach it to the base model. This example uses LoRA.

```py
from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
```

You can verify that only adapter parameters will be updated during training.

```py
model.print_trainable_parameters()
```

## Configure differential privacy

Create an optimizer as usual.

```py
import torch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)
```

Then initialize a FastDP privacy engine and attach it to the optimizer.

```py
import fastDP

privacy_engine = fastDP.PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(train_dataset),
    epochs=num_epochs,
    target_epsilon=8.0,
)

privacy_engine.attach(optimizer)
```

After attaching the privacy engine, FastDP automatically performs gradient clipping and noise injection during optimization.

## Training

Training proceeds normally after the privacy engine is attached.

```py
model.train()

for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```

FastDP manages the privacy-preserving modifications internally, allowing the PEFT model to be trained with a standard PyTorch training loop.

## Using other PEFT methods

The same procedure can be applied to other PEFT configurations. Replace the LoRA configuration with any supported PEFT method before calling [`get_peft_model`](https://github.com/huggingface/peft/blob/main/src/peft/mapping_func.py).

For example:

```py
from peft import PromptTuningConfig

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=64,
)

model = get_peft_model(model, peft_config)
```

or:

```py
from peft import PrefixTuningConfig

peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30,
)

model = get_peft_model(model, peft_config)
```

Since FastDP is attached to the final PEFT model and optimizer, no additional changes are required when switching between supported PEFT methods.

## Saving the adapter

After training, save the adapter weights as you would for any other PEFT workflow.

```py
model.save_pretrained("dp-lora-adapter")
```

The adapter can later be loaded into a compatible base model using the standard PEFT loading APIs.

## Next steps

This tutorial demonstrated how to combine PEFT with FastDP to perform differentially private fine-tuning. For more information about available PEFT methods, see the PEFT method guides. For details about privacy accounting, clipping strategies, and privacy budgets, refer to the FastDP documentation.
