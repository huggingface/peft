import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import evaluate
from peft import LoraConfig, get_peft_model
import mlflow
import os
from transformers.optimization import Adafactor, AdafactorSchedule

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image Classification with LoRA")
parser.add_argument("--model_checkpoint", type=str, default="microsoft/swin-base-patch4-window7-224",
                    help="The model checkpoint to use")
parser.add_argument('--dataset_name', type=str, default='food101', help='The name of the Dataset (from the HuggingFace hub) to train on.')
parser.add_argument("--batch_size", type=int, default=152,
                    help="Batch size for training and evaluation")
parser.add_argument("--learning_rate", type=float, default=5e-3,
                    help="Learning rate for training")
parser.add_argument("--num_train_epochs", type=int, default=1,
                    help="Number of training epochs")
parser.add_argument("--logging_steps", type=int, default=1,
                    help="Log metrics every X steps")
parser.add_argument('--cache_dir', type=str, default=None, help='Directory to read/write data.')
parser.add_argument("--amp", type=str, choices=["bf16", "fp16", "no"], default="no", help="Choose AMP mode")
parser.add_argument('--checkpoint_dir', type=str, default="/nas/kimng/repo_clean/peft/examples/image_classification/swin-base-patch4-window7-224/10epoch.pt", help='Directory to save checkpoints.')
parser.add_argument('--load_checkpoint', type=str, default=False, help='Load checkpoint or not.')
parser.add_argument('--save_checkpoint', type=str, default=False, help='Save checkpoint or not.')
args = parser.parse_args()


# Load dataset
dataset = load_dataset(args.dataset_name, split="train[:5000]", cache_dir=args.cache_dir)
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Load image processor
image_processor = AutoImageProcessor.from_pretrained(args.model_checkpoint)

# Define data transforms
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose([
    RandomResizedCrop(image_processor.size["height"]),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])
val_transforms = Compose([
    Resize(image_processor.size["height"]),
    CenterCrop(image_processor.size["height"]),
    ToTensor(),
    normalize,
])

def get_num_parameters(model):
  num_params = 0
  for param in model.parameters():
    num_params += param.numel()
  # in million
  num_params /= 10**6
  return num_params

# Preprocess functions for train and val datasets
def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# Split the dataset into train and validation sets
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# Print trainable parameters of a model
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")

# Load the base image classification model
model = AutoModelForImageClassification.from_pretrained(
    args.model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
print_trainable_parameters(model)

# Apply LoRA to the model
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# Define training arguments
model_name = args.model_checkpoint.split("/")[-1]

# optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
# lr_scheduler = AdafactorSchedule(optimizer)
# optimizers = (optimizer, lr_scheduler)

if args.save_checkpoint=="True":
    training_args = TrainingArguments(
    output_dir=args.checkpoint_dir,
    overwrite_output_dir=True,
    save_total_limit=1,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=args.batch_size,
    fp16=args.amp,
    num_train_epochs=args.num_train_epochs,
    logging_steps=args.logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
)
else:
    training_args = TrainingArguments(
        f"{model_name}-finetuned-lora-food101",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        fp16=args.amp,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        label_names=["labels"],
    )


# Compute evaluation metrics
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# Collate function for data loading
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Create the Trainer
trainer = Trainer(
    lora_model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
    # optimizers=optimizers
)

# Train the model
mlflow.start_run()
if args.load_checkpoint and os.path.exists(args.checkpoint_dir):
    try:
        train_results = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    except ValueError as e:
        print(f"Error: {e}")
        print("No valid checkpoint found, training from scratch.")
        train_results = trainer.train()
else:
    train_results = trainer.train()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_model()
trainer.save_state()

num_params = get_num_parameters(lora_model)
mlflow.log_param('num_params', num_params)

for metric_dict in trainer.state.log_history:
    if 'loss' in metric_dict:
        mlflow.log_metric('loss', metric_dict['loss'], step=metric_dict['step'])
        mlflow.log_metric('lr', metric_dict['learning_rate'], step=metric_dict['step'])

mlflow.log_metric('epoch', train_results.metrics["epoch"])
mlflow.log_metric('train_loss', train_results.metrics["train_loss"])
mlflow.log_metric('train_runtime', train_results.metrics["train_runtime"])
mlflow.log_metric('throughput', train_results.metrics["train_samples_per_second"])
mlflow.log_metric('steps_per_second', train_results.metrics["train_steps_per_second"])
epoch_time = train_results.metrics["train_runtime"] / train_results.metrics["epoch"]
mlflow.log_metric('epoch_time', epoch_time)
mlflow.end_run()

# Evaluate the model on the validation set
trainer.evaluate(val_ds)

