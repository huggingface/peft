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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image Classification with LoRA")
parser.add_argument("--model_checkpoint", type=str, default="google/vit-base-patch16-224-in21k",
                    help="The model checkpoint to use")
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for training and evaluation")
parser.add_argument("--learning_rate", type=float, default=5e-3,
                    help="Learning rate for training")
parser.add_argument("--num_train_epochs", type=int, default=5,
                    help="Number of training epochs")
parser.add_argument("--output_dir", type=str, default="./outputs", help="Output dir")

args = parser.parse_args()

# mlflow init
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load dataset
dataset = load_dataset("food101", split="train[:5000]")
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
training_args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=args.batch_size,
    fp16=True,
    num_train_epochs=args.num_train_epochs,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
)

# mlflow initial
experiment_id = mlflow.create_experiment('image_classification-{}'.format(model_name))
experiment = mlflow.get_experiment(experiment_id)
mlflow_runner = mlflow.start_run(run_name=model_name, experiment_id=experiment.experiment_id)

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
    data_collator=collate_fn,
)

# Train the model
with mlflow_runner:
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_model()
    trainer.save_state()

    mlflow.log_metric('loss', train_results.metrics["train_loss"])
    mlflow.log_metric('runtime', train_results.metrics["train_runtime"])
    mlflow.log_metric('samples_per_second', train_results.metrics["train_samples_per_second"])
    mlflow.log_metric('steps_per_second', train_results.metrics["train_steps_per_second"])
    mlflow.end_run()

# Evaluate the model on the validation set
trainer.evaluate(val_ds)
