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

The sheer size of today's large pretrained models - which commonly have billions of parameters - present a significant training challenge because they require more computational power to crunch all those calculations and storage space. You'll need access to powerful GPUs or TPUs to train these large pretrained models which is expensive, not widely accessible to everyone, not environmentally friendly, and not very practical. PEFT methods address many of these challenges. There are several types of PEFT methods (soft prompting, matrix decomposition, adapters), but they all focus on the same thing, reduce the number of trainable parameters. This makes it more accessible to train and store large language models (LLMs) on consumer hardware.

The PEFT library is designed to help you quickly train LLMs on free or low-cost GPUs, and in this tutorial, you'll learn how to setup a configuration to apply a PEFT method to a pretrained base model for training. Once the PEFT configuration is setup, you can use any training framework you like (Transformer's [`~transformers.Trainer`] class, Accelerate, PyTorch).

## PEFT configurations

A configuration stores important parameters that specify how a specific PEFT method should be applied. There are two main types of PEFT configurations:

* [`PeftConfig`] is the base configuration class for non-prompt based methods
    * [`LoraConfig`] inherits from [`PeftConfig`] and is for the LoRA method
        * [`LycorisConfig`] inherits from [`LoraConfig`] and is for LoRA-variants like LoHa and LoKr
* [`PromptLearningConfig`] is base configuration class for prompt-based methods like p-tuning or prefix tuning

For example, take a look at this [`LoraConfig`](https://huggingface.co/ybelkada/opt-350m-lora/blob/main/adapter_config.json) for applying LoRA to the facebook/opt-350m model and [`PromptEncoderConfig`](https://huggingface.co/smangrul/roberta-large-peft-p-tuning/blob/main/adapter_config.json) for applying p-tuning to the roberta-large model.

<Tip>

Learn more about the parameters you can configure for each PEFT method in their respective API reference page.

</Tip>

<hfoptions id="config">
<hfoption id="LoRA">

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
  "target_modules": [ #model modules to apply LoRA to (query and value projection layers)
    "q_proj",
    "v_proj"
  ],
  "task_type": "CAUSAL_LM" #type of task to train model on
}
```

You can create your own configuration for training by initializing a [`LoraConfig`].

```py
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    lora_alpha=32,
    lora_dropout=0.05
)
```

</hfoption>
<hfoption id="p-tuning">

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
from peft import PromptEncoderConfig

p_tuning_config = PromptEncoderConfig(
    encoder_reprameterization_type="MLP",
    encoder_hidden_size=128,
    num_attention_heads=16,
    num_layers=24,
    num_transformer_submodules=1,
    num_virtual_tokens=20,
    token_dim=1024,
    task_type="SEQ_CLS"
)
```

</hfoption>
</hfoptions>

## PEFT models

With a configuration for a PEFT method in hand, you can now apply it to any pretrained model to create a [`PeftModel`]. You can choose from any of the state-of-the-art models from the Transformers library, a custom model, and even new and unsupported transformer architectures.

Load a base model to finetune.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m"
)
```

Create a [`PeftModel`] from the base model and a configuration (for this tutorial, you'll use the [`LoraConfig`] you created earlier). The [`get_peft_model`] method returns a trainable [`PeftModel`].

```py
from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
"trainable params: 1,572,864 || all params: 332,769,280 || trainable%: 0.472659014678278"
```

Now you can train the [`PeftModel`] with your preferred training framework!

After you've trained a [`PeftModel`], you can save it locally or upload it to the Hub. To load a [`PeftModel`], you'll need to provide the [`PeftConfig`] used to create it and the base model it was trained from.

```py
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```

By default, the [`PeftModel`] can only be used for inference, but if you'd like to train the adapter some more you can set `is_trainable=True`.

```py
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora", is_trainable=True)
```

The [`PeftModel.from_pretrained`] method is the most flexible way to load a [`PeftModel`] because it doesn't matter what model framework was used (Transformers, timm, a generic PyTorch model). Other classes, like [`AutoPeftModel`], are just a convenient wrapper around the base [`PeftModel`].

## Next steps

With a correctly configured PEFT method, you can apply it to any pretrained model to create a [`PeftModel`] and train it with fewer parameters! To learn more about PEFT configurations and models, the following guide may be helpful:

* Learn how to configure a PEFT method for models that aren't from Transformers in the [Working with custom models](../developer_guides/custom_models) guide.
