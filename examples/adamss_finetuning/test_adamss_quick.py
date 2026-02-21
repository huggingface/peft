"""
Quick test for AdaMSS example - runs 1 epoch on small subset
"""

import sys


sys.path.insert(0, "/Users/onelong/Documents/WorkSpace/CodeSpace/AdaMSS-main/peft-main/src")

import evaluate
import torch
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

from peft import AdaMSSConfig, ASACallback, get_peft_model


print("=" * 80)
print("🧪 AdaMSS Quick Test")
print("=" * 80)

# Load small subset
print("\n📦 Loading CIFAR-10 (small subset for testing)...")
dataset = load_dataset("cifar10")
train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_ds = train_val["train"].select(range(100))  # Only 100 samples
val_ds = train_val["test"].select(range(50))

print(f"✅ Dataset: {len(train_ds)} train, {len(val_ds)} val")

# Prepare data
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
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


def preprocess_train(examples):
    examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["img"]]
    return examples


def preprocess_val(examples):
    examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["img"]]
    return examples


train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# Load model
print("\n🤖 Loading ViT model...")
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10,
    ignore_mismatched_sizes=True,
)

# Configure AdaMSS
print("\n⚙️  Applying AdaMSS...")
config = AdaMSSConfig(
    r=100,
    num_subspaces=10,
    subspace_rank=3,
    target_modules=["query", "value"],
    use_asa=True,
    target_kk=5,
    modules_to_save=["classifier"],
)

model = get_peft_model(model, config)
print("\n📊 Parameter statistics:")
model.print_trainable_parameters()

# Setup ASA callback
print("\n🔥 Setting up ASA callback...")
asa_callback = ASACallback(
    target_kk=5,
    init_warmup=5,
    final_warmup=20,
    mask_interval=10,
)

# Metrics
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


# Training arguments
training_args = TrainingArguments(
    output_dir="./test_adamss_output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=0.01,
    weight_decay=0.0005,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    remove_unused_columns=False,
    report_to="none",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[asa_callback],
)

# Train
print("\n" + "=" * 80)
print("🚀 Starting training (1 epoch on 100 samples)...")
print("=" * 80 + "\n")

try:
    trainer.train()
    print("\n✅ Training completed successfully!")

    # Evaluate
    metrics = trainer.evaluate()
    print(f"\n📊 Validation Accuracy: {metrics['eval_accuracy']:.2%}")

    print("\n" + "=" * 80)
    print("✅ Test PASSED - AdaMSS example works correctly!")
    print("=" * 80)

except Exception as e:  # noqa: BLE001
    print("\n" + "=" * 80)
    print(f"❌ Test FAILED: {e}")
    print("=" * 80)
    import traceback

    traceback.print_exc()
    sys.exit(1)
