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

# Quicktour

PEFT offers parameter-efficient methods for finetuning large pretrained models. The traditional paradigm is to finetune all of a model's parameters for each downstream task, but this is becoming exceedingly costly and impractical because of the enormous number of parameters in models today. Instead, it is more efficient to train a smaller number of prompt parameters or use a reparametrization method like low-rank adaptation (LoRA) to reduce the number of trainable parameters.

This quicktour will show you PEFT's main features and how you can train or run inference on large models that would typically be inaccessible on consumer devices.

## Train

Each PEFT method is defined by a [`PeftConfig`] class that stores all the important parameters for building a [`PeftModel`]. For example, to train with LoRA, load and create a [`LoraConfig`] class and specify the following parameters:

- `task_type`: the task to train for (sequence-to-sequence language modeling in this case)
- `inference_mode`: whether you're using the model for inference or not
- `r`: the dimension of the low-rank matrices
- `lora_alpha`: the scaling factor for the low-rank matrices
- `lora_dropout`: the dropout probability of the LoRA layers

```python
from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
```

<Tip>

See the [`LoraConfig`] reference for more details about other parameters you can adjust, such as the modules to target or the bias type.

</Tip>

Once the [`LoraConfig`] is setup, create a [`PeftModel`] with the [`get_peft_model`] function. It takes a base model - which you can load from the Transformers library - and the [`LoraConfig`] containing the parameters for how to configure a model for training with LoRA.

Load the base model you want to finetune.

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
```

Wrap the base model and `peft_config` with the [`get_peft_model`] function to create a [`PeftModel`]. To get a sense of the number of trainable parameters in your model, use the [`print_trainable_parameters`] method.

```python
from peft import get_peft_model

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
```

Out of [bigscience/mt0-large's](https://huggingface.co/bigscience/mt0-large) 1.2B parameters, you're only training 0.19% of them!

That is it 🎉! Now you can train the model with the Transformers [`~transformers.Trainer`], Accelerate, or any custom PyTorch training loop.

For example, to train with the [`~transformers.Trainer`] class, setup a [`~transformers.TrainingArguments`] class with some training hyperparameters.

```py
training_args = TrainingArguments(
    output_dir="your-name/bigscience/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

Pass the model, training arguments, dataset, tokenizer, and any other necessary component to the [`~transformers.Trainer`], and call [`~transformers.Trainer.train`] to start training.

```py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Save model

After your model is finished training, you can save your model to a directory using the [`~transformers.PreTrainedModel.save_pretrained`] function.

```py
model.save_pretrained("output_dir")
```

You can also save your model to the Hub (make sure you're logged in to your Hugging Face account first) with the [`~transformers.PreTrainedModel.push_to_hub`] function.

```python
from huggingface_hub import notebook_login

notebook_login()
model.push_to_hub("your-name/bigscience/mt0-large-lora")
```

Both methods only save the extra PEFT weights that were trained, meaning it is super efficient to store, transfer, and load. For example, this [facebook/opt-350m](https://huggingface.co/ybelkada/opt-350m-lora) model trained with LoRA only contains two files: `adapter_config.json` and `adapter_model.safetensors`. The `adapter_model.safetensors` file is just 6.3MB!

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">The adapter weights for a opt-350m model stored on the Hub are only ~6MB compared to the full size of the model weights, which can be ~700MB.</figcaption>
</div>

## Inference

<Tip>

Take a look at the [AutoPeftModel](package_reference/auto_class) API reference for a complete list of available `AutoPeftModel` classes.

</Tip>

Easily load any PEFT-trained model for inference with the [`AutoPeftModel`] class and the [`~transformers.PreTrainedModel.from_pretrained`] method:

```py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model = model.to("cuda")
model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

"Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
```

For other tasks that aren't explicitly supported with an `AutoPeftModelFor` class - such as automatic speech recognition - you can still use the base [`AutoPeftModel`] class to load a model for the task.

```py
from peft import AutoPeftModel

model = AutoPeftModel.from_pretrained("smangrul/openai-whisper-large-v2-LORA-colab")
```

## Next steps

Now that you've seen how to train a model with one of the PEFT methods, we encourage you to try out some of the other methods like prompt tuning. The steps are very similar to the ones shown in the quicktour:

1. prepare a [`PeftConfig`] for a PEFT method
2. use the [`get_peft_model`] method to create a [`PeftModel`] from the configuration and base model

Then you can train it however you like! To load a PEFT model for inference, you can use the [`AutoPeftModel`] class.

Feel free to also take a look at the task guides if you're interested in training a model with another PEFT method for a specific task such as semantic segmentation, multilingual automatic speech recognition, DreamBooth, token classification, and more.
