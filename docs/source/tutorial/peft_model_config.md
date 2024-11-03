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

# PEFT configurations and models

The sheer size of today's large pretrained models - which commonly have billions of parameters - present a significant training challenge because they require more storage space and more computational power to crunch all those calculations. You'll need access to powerful GPUs or TPUs to train these large pretrained models which is expensive, not widely accessible to everyone, not environmentally friendly, and not very practical. PEFT methods address many of these challenges. There are several types of PEFT methods (soft prompting, matrix decomposition, adapters), but they all focus on the same thing, reduce the number of trainable parameters. This makes it more accessible to train and store large models on consumer hardware.

The PEFT library is designed to help you quickly train large models on free or low-cost GPUs, and in this tutorial, you'll learn how to setup a configuration to apply a PEFT method to a pretrained base model for training. Once the PEFT configuration is setup, you can use any training framework you like (Transformer's [`~transformers.Trainer`] class, [Accelerate](https://hf.co/docs/accelerate), a custom PyTorch training loop).

## PEFT configurations

<Tip>

Learn more about the parameters you can configure for each PEFT method in their respective API reference page.

</Tip>

A configuration stores important parameters that specify how a particular PEFT method should be applied.

For example, take a look at the following [`LoraConfig`](https://huggingface.co/ybelkada/opt-350m-lora/blob/main/adapter_config.json) for applying LoRA and [`PromptEncoderConfig`](https://huggingface.co/smangrul/roberta-large-peft-p-tuning/blob/main/adapter_config.json) for applying p-tuning (these configuration files are already JSON-serialized). Whenever you load a PEFT adapter, it is a good idea to check whether it has an associated adapter_config.json file which is required.

<hfoptions id="config">
<hfoption id="LoraConfig">

```json
{
  "base_model_name_or_path": "facebook/opt-350m", #base model to apply LoRA to
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA", #PEFT method type
  "r": 16,
  "revision": null,
  "target_modules": [
    "q_proj", #model modules to apply LoRA to (query and value projection layers)
    "v_proj"
  ],
  "task_type": "CAUSAL_LM" #type of task to train model on
}
```

You can create your own configuration for training by initializing a [`LoraConfig`].

```py
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
```

</hfoption>
<hfoption id="PromptEncoderConfig">

```json
{
  "base_model_name_or_path": "roberta-large", #base model to apply p-tuning to
  "encoder_dropout": 0.0,
  "encoder_hidden_size": 128,
  "encoder_num_layers": 2,
  "encoder_reparameterization_type": "MLP",
  "inference_mode": true,
  "num_attention_heads": 16,
  "num_layers": 24,
  "num_transformer_submodules": 1,
  "num_virtual_tokens": 20,
  "peft_type": "P_TUNING", #PEFT method type
  "task_type": "SEQ_CLS", #type of task to train model on
  "token_dim": 1024
}
```

You can create your own configuration for training by initializing a [`PromptEncoderConfig`].

```py
from peft import PromptEncoderConfig, TaskType

p_tuning_config = PromptEncoderConfig(
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=128,
    num_attention_heads=16,
    num_layers=24,
    num_transformer_submodules=1,
    num_virtual_tokens=20,
    token_dim=1024,
    task_type=TaskType.SEQ_CLS
)
```

</hfoption>
</hfoptions>

## PEFT models

With a PEFT configuration in hand, you can now apply it to any pretrained model to create a [`PeftModel`]. Choose from any of the state-of-the-art models from the [Transformers](https://hf.co/docs/transformers) library, a custom model, and even new and unsupported transformer architectures.

For this tutorial, load a base [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) model to finetune.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```

Use the [`get_peft_model`] function to create a [`PeftModel`] from the base facebook/opt-350m model and the `lora_config` you created earlier.

```py
from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
"trainable params: 1,572,864 || all params: 332,769,280 || trainable%: 0.472659014678278"
```

Now you can train the [`PeftModel`] with your preferred training framework! After training, you can save your model locally with [`~PeftModel.save_pretrained`] or upload it to the Hub with the [`~transformers.PreTrainedModel.push_to_hub`] method.

```py
# save locally
lora_model.save_pretrained("your-name/opt-350m-lora")

# push to Hub
lora_model.push_to_hub("your-name/opt-350m-lora")
```

To load a [`PeftModel`] for inference, you'll need to provide the [`PeftConfig`] used to create it and the base model it was trained from.

```py
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```

<Tip>

By default, the [`PeftModel`] is set for inference, but if you'd like to train the adapter some more you can set `is_trainable=True`.

```py
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora", is_trainable=True)
```

</Tip>

The [`PeftModel.from_pretrained`] method is the most flexible way to load a [`PeftModel`] because it doesn't matter what model framework was used (Transformers, timm, a generic PyTorch model). Other classes, like [`AutoPeftModel`], are just a convenient wrapper around the base [`PeftModel`], and makes it easier to load PEFT models directly from the Hub or locally where the PEFT weights are stored.

```py
from peft import AutoPeftModelForCausalLM

lora_model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
```

Take a look at the [AutoPeftModel](package_reference/auto_class) API reference to learn more about the [`AutoPeftModel`] classes.

## Next steps

With the appropriate [`PeftConfig`], you can apply it to any pretrained model to create a [`PeftModel`] and train large powerful models faster on freely available GPUs! To learn more about PEFT configurations and models, the following guide may be helpful:

* Learn how to configure a PEFT method for models that aren't from Transformers in the [Working with custom models](../developer_guides/custom_models) guide.
