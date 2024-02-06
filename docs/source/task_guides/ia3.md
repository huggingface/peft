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

# IA3

[IA3](../conceptual_guides/ia3) multiplies the model's activations (the keys and values in the self-attention and encoder-decoder attention blocks, and the intermediate activation of the position-wise feedforward network) by three learned vectors. This PEFT method introduces an even smaller number of trainable parameters than LoRA which introduces weight matrices instead of vectors. The original model's parameters are kept frozen and only these vectors are updated. As a result, it is faster, cheaper and more efficient to finetune for a new downstream task.

This guide will show you how to train a sequence-to-sequence model with IA3 to *generate a sentiment* given some financial news.

<Tip>

Some familiarity with the general process of training a sequence-to-sequence would be really helpful and allow you to focus on how to apply IA3. If you’re new, we recommend taking a look at the [Translation](https://huggingface.co/docs/transformers/tasks/translation) and [Summarization](https://huggingface.co/docs/transformers/tasks/summarization) guides first from the Transformers documentation. When you’re ready, come back and see how easy it is to drop PEFT in to your training!

</Tip>

## Dataset

You'll use the sentences_allagree subset of the [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) dataset. This subset contains financial news with 100% annotator agreement on the sentiment label. Take a look at the [dataset viewer](https://huggingface.co/datasets/financial_phrasebank/viewer/sentences_allagree) for a better idea of the data and sentences you'll be working with.

Load the dataset with the [`~datasets.load_dataset`] function. This subset of the dataset only contains a train split, so use the [`~datasets.train_test_split`] function to create a train and validation split. Create a new `text_label` column so it is easier to understand what the `label` values `0`, `1`, and `2` mean.

```py
from datasets import load_dataset

ds = load_dataset("financial_phrasebank", "sentences_allagree")
ds = ds["train"].train_test_split(test_size=0.1)
ds["validation"] = ds["test"]
del ds["test"]

classes = ds["train"].features["label"].names
ds = ds.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

ds["train"][0]
{'sentence': 'It will be operated by Nokia , and supported by its Nokia NetAct network and service management system .',
 'label': 1,
 'text_label': 'neutral'}
```

Load a tokenizer and create a preprocessing function that:

1. tokenizes the inputs, pads and truncates the sequence to the `max_length`
2. apply the same tokenizer to the labels but with a shorter `max_length` that corresponds to the label
3. mask the padding tokens

```py
from transformers import AutoTokenizer

text_column = "sentence"
label_column = "text_label"
max_length = 128

tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs
```

Use the [`~datasets.Dataset.map`] function to apply the preprocessing function to the entire dataset.

```py
processed_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```

Create a training and evaluation [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), and set `pin_memory=True` to speed up data transfer to the GPU during training if your dataset samples are on a CPU.

```py
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_ds = processed_ds["train"]
eval_ds = processed_ds["validation"]

batch_size = 8

train_dataloader = DataLoader(
    train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```

## Model

Now you can load a pretrained model to use as the base model for IA3. This guide uses the [bigscience/mt0-large](https://huggingface.co/bigscience/mt0-large) model, but you can use any sequence-to-sequence model you like.

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
```

### PEFT configuration and model

All PEFT methods need a configuration that contains and specifies all the parameters for how the PEFT method should be applied. Create an [`IA3Config`] with the task type and set the inference mode to `False`. You can find additional parameters for this configuration in the [API reference](../package_reference/ia3#ia3config).

<Tip>

Call the [`~PeftModel.print_trainable_parameters`] method to compare the number of trainable parameters of [`PeftModel`] versus the number of parameters in the base model!

</Tip>

Once the configuration is setup, pass it to the [`get_peft_model`] function along with the base model to create a trainable [`PeftModel`].

```py
from peft import IA3Config, get_peft_model

peft_config = IA3Config(task_type="SEQ_2_SEQ_LM")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 282,624 || all params: 1,229,863,936 || trainable%: 0.022980103060766553"
```

### Training

Set up an optimizer and learning rate scheduler.

```py
import torch
from transformers import get_linear_schedule_with_warmup

lr = 8e-3
num_epochs = 3

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```

Move the model to the GPU and create a training loop that reports the loss and perplexity for each epoch.

```py
from tqdm import tqdm

device = "cuda"
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

## Share your model

After training is complete, you can upload your model to the Hub with the [`~transformers.PreTrainedModel.push_to_hub`] method. You'll need to login to your Hugging Face account first and enter your token when prompted.

```py
from huggingface_hub import notebook_login

account = <your-hf-account-name>
peft_model_id = f"{account}/mt0-large-ia3"
model.push_to_hub(peft_model_id)
```

## Inference

To load the model for inference, use the [`~AutoPeftModelForSeq2SeqLM.from_pretrained`] method. Let's also load a sentence of financial news from the dataset to generate a sentiment for.

```py
from peft import AutoPeftModelForSeq2SeqLM

model = AutoPeftModelForSeq2SeqLM.from_pretrained("<your-hf-account-name>/mt0-large-ia3").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

i = 15
inputs = tokenizer(ds["validation"][text_column][i], return_tensors="pt")
print(ds["validation"][text_column][i])
"The robust growth was the result of the inclusion of clothing chain Lindex in the Group in December 2007 ."
```

Call the [`~transformers.GenerationMixin.generate`] method to generate the predicted sentiment label.

```py
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
['positive']
```
