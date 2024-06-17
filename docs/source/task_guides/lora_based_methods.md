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

# LoRA methods

A popular way to efficiently train large models is to insert (typically in the attention blocks) smaller trainable matrices that are a low-rank decomposition of the delta weight matrix to be learnt during finetuning. The pretrained model's original weight matrix is frozen and only the smaller matrices are updated during training. This reduces the number of trainable parameters, reducing memory usage and training time which can be very expensive for large models.

There are several different ways to express the weight matrix as a low-rank decomposition, but [Low-Rank Adaptation (LoRA)](../conceptual_guides/adapter#low-rank-adaptation-lora) is the most common method. The PEFT library supports several other LoRA variants, such as [Low-Rank Hadamard Product (LoHa)](../conceptual_guides/adapter#low-rank-hadamard-product-loha), [Low-Rank Kronecker Product (LoKr)](../conceptual_guides/adapter#low-rank-kronecker-product-lokr), and [Adaptive Low-Rank Adaptation (AdaLoRA)](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora). You can learn more about how these methods work conceptually in the [Adapters](../conceptual_guides/adapter) guide. If you're interested in applying these methods to other tasks and use cases like semantic segmentation, token classification, take a look at our [notebook collection](https://huggingface.co/collections/PEFT/notebooks-6573b28b33e5a4bf5b157fc1)!

This guide will show you how to quickly train an image classification model - with a low-rank decomposition method - to identify the class of food shown in an image.

<Tip>

Some familiarity with the general process of training an image classification model would be really helpful and allow you to focus on the low-rank decomposition methods. If you're new, we recommend taking a look at the [Image classification](https://huggingface.co/docs/transformers/tasks/image_classification) guide first from the Transformers documentation. When you're ready, come back and see how easy it is to drop PEFT in to your training!

</Tip>

Before you begin, make sure you have all the necessary libraries installed.

```bash
pip install -q peft transformers datasets
```

## Dataset

In this guide, you'll use the [Food-101](https://huggingface.co/datasets/food101) dataset which contains images of 101 food classes (take a look at the [dataset viewer](https://huggingface.co/datasets/food101/viewer/default/train) to get a better idea of what the dataset looks like).

Load the dataset with the [`~datasets.load_dataset`] function.

```py
from datasets import load_dataset

ds = load_dataset("food101")
```

Each food class is labeled with an integer, so to make it easier to understand what these integers represent, you'll create a `label2id` and `id2label` dictionary to map the integer to its class label.

```py
labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]
"baklava"
```

Load an image processor to properly resize and normalize the pixel values of the training and evaluation images.

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
```

You can also use the image processor to prepare some transformation functions for data augmentation and pixel scaling.

```py
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
```

Define the training and validation datasets, and use the [`~datasets.Dataset.set_transform`] function to apply the transformations on-the-fly.

```py
train_ds = ds["train"]
val_ds = ds["validation"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```

Finally, you'll need a data collator to create a batch of training and evaluation data and convert the labels to `torch.tensor` objects.

```py
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

## Model

Now let's load a pretrained model to use as the base model. This guide uses the [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) model, but you can use any image classification model you want. Pass the `label2id` and `id2label` dictionaries to the model so it knows how to map the integer labels to their class labels, and you can optionally pass the `ignore_mismatched_sizes=True` parameter if you're finetuning a checkpoint that has already been finetuned.

```py
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
```

### PEFT configuration and model

Every PEFT method requires a configuration that holds all the parameters specifying how the PEFT method should be applied. Once the configuration is setup, pass it to the [`~peft.get_peft_model`] function along with the base model to create a trainable [`PeftModel`].

<Tip>

Call the [`~PeftModel.print_trainable_parameters`] method to compare the number of parameters of [`PeftModel`] versus the number of parameters in the base model!

</Tip>

<hfoptions id="loras">
<hfoption id="LoRA">

[LoRA](../conceptual_guides/adapter#low-rank-adaptation-lora) decomposes the weight update matrix into *two* smaller matrices. The size of these low-rank matrices is determined by its *rank* or `r`. A higher rank means the model has more parameters to train, but it also means the model has more learning capacity. You'll also want to specify the `target_modules` which determine where the smaller matrices are inserted. For this guide, you'll target the *query* and *value* matrices of the attention blocks. Other important parameters to set are `lora_alpha` (scaling factor), `bias` (whether `none`, `all` or only the LoRA bias parameters should be trained), and `modules_to_save` (the modules apart from the LoRA layers to be trained and saved). All of these parameters - and more - are found in the [`LoraConfig`].

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 667,493 || all params: 86,543,818 || trainable%: 0.7712775047664294"
```

</hfoption>
<hfoption id="LoHa">

[LoHa](../conceptual_guides/adapter#low-rank-hadamard-product-loha) decomposes the weight update matrix into *four* smaller matrices and each pair of smaller matrices is combined with the Hadamard product. This allows the weight update matrix to keep the same number of trainable parameters when compared to LoRA, but with a higher rank (`r^2` for LoHA when compared to `2*r` for LoRA). The size of the smaller matrices is determined by its *rank* or `r`. You'll also want to specify the `target_modules` which determines where the smaller matrices are inserted. For this guide, you'll target the *query* and *value* matrices of the attention blocks. Other important parameters to set are `alpha` (scaling factor), and `modules_to_save` (the modules apart from the LoHa layers to be trained and saved). All of these parameters - and more - are found in the [`LoHaConfig`].

```py
from peft import LoHaConfig, get_peft_model

config = LoHaConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 1,257,317 || all params: 87,133,642 || trainable%: 1.4429753779831676"
```

</hfoption>
<hfoption id="LoKr">

[LoKr](../conceptual_guides/adapter#low-rank-kronecker-product-lokr) expresses the weight update matrix as a decomposition of a Kronecker product, creating a block matrix that is able to preserve the rank of the original weight matrix. The size of the smaller matrices are determined by its *rank* or `r`. You'll also want to specify the `target_modules` which determines where the smaller matrices are inserted. For this guide, you'll target the *query* and *value* matrices of the attention blocks. Other important parameters to set are `alpha` (scaling factor), and `modules_to_save` (the modules apart from the LoKr layers to be trained and saved). All of these parameters - and more - are found in the [`LoKrConfig`].

```py
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 116,069 || all params: 87,172,042 || trainable%: 0.13314934162033282"
```

</hfoption>
<hfoption id="AdaLoRA">

[AdaLoRA](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora) efficiently manages the LoRA parameter budget by assigning important weight matrices more parameters and pruning less important ones. In contrast, LoRA evenly distributes parameters across all modules. You can control the average desired *rank* or `r` of the matrices, and which modules to apply AdaLoRA to with `target_modules`. Other important parameters to set are `lora_alpha` (scaling factor), and `modules_to_save` (the modules apart from the AdaLoRA layers to be trained and saved). All of these parameters - and more - are found in the [`AdaLoraConfig`].

```py
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 520,325 || all params: 87,614,722 || trainable%: 0.5938785036606062"
```

</hfoption>
</hfoptions>

### Training

For training, let's use the [`~transformers.Trainer`] class from Transformers. The [`Trainer`] contains a PyTorch training loop, and when you're ready, call [`~transformers.Trainer.train`] to start training. To customize the training run, configure the training hyperparameters in the [`~transformers.TrainingArguments`] class. With LoRA-like methods, you can afford to use a higher batch size and learning rate.

> [!WARNING]
> AdaLoRA has an [`~AdaLoraModel.update_and_allocate`] method that should be called at each training step to update the parameter budget and mask, otherwise the adaptation step is not performed. This requires writing a custom training loop or subclassing the [`~transformers.Trainer`] to incorporate this method. As an example, take a look at this [custom training loop](https://github.com/huggingface/peft/blob/912ad41e96e03652cabf47522cd876076f7a0c4f/examples/conditional_generation/peft_adalora_seq2seq.py#L120).

```py
from transformers import TrainingArguments, Trainer

account = "stevhliu"
peft_model_id = f"{account}/google/vit-base-patch16-224-in21k-lora"
batch_size = 128

args = TrainingArguments(
    peft_model_id,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    label_names=["labels"],
)
```

Begin training with [`~transformers.Trainer.train`].

```py
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    data_collator=collate_fn,
)
trainer.train()
```

## Share your model

Once training is complete, you can upload your model to the Hub with the [`~transformers.PreTrainedModel.push_to_hub`] method. You’ll need to login to your Hugging Face account first and enter your token when prompted.

```py
from huggingface_hub import notebook_login

notebook_login()
```

Call [`~transformers.PreTrainedModel.push_to_hub`] to save your model to your repositoy.

```py
model.push_to_hub(peft_model_id)
```

## Inference

Let's load the model from the Hub and test it out on a food image.

```py
from peft import PeftConfig, PeftModel
from transformers import AutoImageProcessor
from PIL import Image
import requests

config = PeftConfig.from_pretrained("stevhliu/vit-base-patch16-224-in21k-lora")
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(model, "stevhliu/vit-base-patch16-224-in21k-lora")

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg">
</div>

Convert the image to RGB and return the underlying PyTorch tensors.

```py
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
```

Now run the model and return the predicted class!

```py
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
"Predicted class: beignets"
```
