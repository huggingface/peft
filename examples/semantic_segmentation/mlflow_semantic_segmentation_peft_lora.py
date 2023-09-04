import argparse
from datasets import load_dataset
import json
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ColorJitter
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer
import evaluate
from peft import LoraConfig, get_peft_model
from huggingface_hub import cached_download, hf_hub_url
import mlflow
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="nvidia/mit-b0", help="Checkpoint for the model")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation")
parser.add_argument("--output_dir", type=str, default="./outputs", help="Output dir")
args = parser.parse_args()

# Load dataset
ds = load_dataset("scene_parse_150", split="train[:150]")
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

# Load label mapping
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# Image processing
image_processor = AutoImageProcessor.from_pretrained(args.checkpoint, reduce_labels=True)
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)


def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    else:
        return Image.fromarray(np_image)


def train_transforms(example_batch):
    images = [jitter(handle_grayscale_image(x)) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

# Metrics and model setup
metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=image_processor.reduce_labels,
        )

        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


model = AutoModelForSemanticSegmentation.from_pretrained(
    args.checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
print_trainable_parameters(model)

config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

model_name = args.checkpoint.split("/")[-1]

# mlflow initial
experiment_id = mlflow.create_experiment('semantic_segmentation-{}'.format(model_name))
experiment = mlflow.get_experiment(experiment_id)
mlflow_runner = mlflow.start_run(run_name=model_name, experiment_id=experiment.experiment_id)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    save_total_limit=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
)
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

mlflow.start_run()
train_results =  trainer.train()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_model()
trainer.save_state()

mlflow.log_metric('loss', train_results.metrics["train_loss"])
mlflow.log_metric('runtime', train_results.metrics["train_runtime"])
mlflow.log_metric('samples_per_second', train_results.metrics["train_samples_per_second"])
mlflow.log_metric('steps_per_second', train_results.metrics["train_steps_per_second"])
mlflow.end_run()