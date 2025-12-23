"""
Image Classification with AdaMSS and ASA Callback

This script demonstrates how to fine-tune a Vision Transformer (ViT) model
using AdaMSS (Adaptive Matrix Decomposition with Subspace Selection) and
ASA (Adaptive Subspace Allocation) callback from PEFT.

Example usage:
    python image_classification_adamss_asa.py \\
        --model_name_or_path google/vit-base-patch16-224-in21k \\
        --dataset_name cifar10 \\
        --adamss_r 100 \\
        --adamss_k 10 \\
        --adamss_ri 3 \\
        --use_asa \\
        --target_kk 5 \\
        --num_epochs 10 \\
        --output_dir ./output

Requirements:
    pip install peft transformers datasets torch torchvision evaluate
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from peft import AdaMSSConfig, get_peft_model, ASACallback


# Global preprocessing functions (to avoid closure issues with set_transform)
def _preprocess_images(examples, img_col, transforms):
    """Apply image transformations."""
    examples["pixel_values"] = [
        transforms(img.convert("RGB")) for img in examples[img_col]
    ]
    return examples


def _collate_batch(examples, label_col):
    """Collate examples into a batch."""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex[label_col] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


@dataclass
class AdaMSSArguments:
    """Arguments for AdaMSS configuration."""
    
    adamss_r: int = field(
        default=100,
        metadata={"help": "SVD decomposition rank (R in paper)."}
    )
    adamss_k: int = field(
        default=10,
        metadata={"help": "Number of subspaces (K in paper)."}
    )
    adamss_ri: int = field(
        default=3,
        metadata={"help": "Subspace rank (rk in paper), typically 1-3."}
    )
    use_asa: bool = field(
        default=False,
        metadata={"help": "Enable Adaptive Subspace Allocation."}
    )
    target_kk: int = field(
        default=5,
        metadata={"help": "Target number of active subspaces when ASA is enabled."}
    )
    asa_init_warmup: int = field(
        default=50,
        metadata={"help": "ASA warmup steps before starting masking."}
    )
    asa_final_warmup: int = field(
        default=1000,
        metadata={"help": "ASA steps to reach target active subspaces."}
    )
    asa_mask_interval: int = field(
        default=100,
        metadata={"help": "ASA interval between subspace selection updates."}
    )


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models from huggingface.co."}
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""
    
    dataset_name: str = field(
        default="cifar10",
        metadata={"help": "Name of dataset to use (cifar10, cifar100, etc.)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples (for debugging)."}
    )


def prepare_transforms(image_processor):
    """Prepare image transformations."""
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = image_processor.size["height"]
    
    train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, AdaMSSArguments, TrainingArguments))
    model_args, data_args, adamss_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load dataset
    print(f"📦 Loading {data_args.dataset_name} dataset...")
    dataset = load_dataset(data_args.dataset_name)
    
    # Auto-detect column names for flexibility
    column_names = dataset["train"].column_names
    
    # Find image column (usually 'img' or 'image')
    if "img" in column_names:
        img_name = "img"
    elif "image" in column_names:
        img_name = "image"
    else:
        raise ValueError(f"Could not find image column in {column_names}")
    
    # Find label column (usually 'label', 'labels', or 'fine_label')
    if "label" in column_names:
        label_name = "label"
    elif "labels" in column_names:
        label_name = "labels"
    elif "fine_label" in column_names:
        label_name = "fine_label"
    else:
        raise ValueError(f"Could not find label column in {column_names}")
    
    print(f"ℹ️  Detected columns - Image: '{img_name}', Label: '{label_name}'")
    
    # Split into train/val/test
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = train_val["train"]
    val_ds = train_val["test"]
    test_ds = dataset["test"]
    
    # Limit train samples if specified (for quick testing)
    if data_args.max_train_samples:
        train_ds = train_ds.select(range(min(data_args.max_train_samples, len(train_ds))))
        # Also limit validation for faster testing
        val_ds = val_ds.select(range(min(5000, len(val_ds))))
    
    labels = train_ds.features[label_name].names
    num_classes = len(labels)
    print(f"✅ Dataset loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"   Number of classes: {num_classes}")
    
    # Create label mappings
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    # Load image processor
    print("\n📥 Loading image processor...")
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    # Prepare transforms
    train_transforms, val_transforms = prepare_transforms(image_processor)
    
    # Use partial to bind parameters at module level (avoid set_transform closure issues)
    train_ds.set_transform(partial(_preprocess_images, img_col=img_name, transforms=train_transforms))
    val_ds.set_transform(partial(_preprocess_images, img_col=img_name, transforms=val_transforms))
    test_ds.set_transform(partial(_preprocess_images, img_col=img_name, transforms=val_transforms))
    
    # Data collator
    collate_fn = partial(_collate_batch, label_col=label_name)
    
    # Load base model
    print("\n🤖 Loading base model...")
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=model_args.cache_dir,
    )
    
    # Configure AdaMSS
    print("\n⚙️  Configuring AdaMSS...")
    print(f"   r={adamss_args.adamss_r}, K={adamss_args.adamss_k}, ri={adamss_args.adamss_ri}")
    if adamss_args.use_asa:
        print(f"   ASA enabled: target {adamss_args.target_kk}/{adamss_args.adamss_k} active subspaces")
    
    config = AdaMSSConfig(
        r=adamss_args.adamss_r,
        num_subspaces=adamss_args.adamss_k,
        subspace_rank=adamss_args.adamss_ri,
        target_modules=["query", "value"],
        use_asa=adamss_args.use_asa,
        target_kk=adamss_args.target_kk if adamss_args.use_asa else None,
        modules_to_save=["classifier"],
    )
    
    # Apply PEFT
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Setup ASA callback if enabled
    callbacks = []
    if adamss_args.use_asa:
        print("\n🔥 Setting up ASA callback...")
        asa_callback = ASACallback(
            target_kk=adamss_args.target_kk,
            init_warmup=adamss_args.asa_init_warmup,
            final_warmup=adamss_args.asa_final_warmup,
            mask_interval=adamss_args.asa_mask_interval,
        )
        callbacks.append(asa_callback)
    
    # Metrics
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        # Handle tuple outputs (logits, hidden_states)
        if isinstance(preds, tuple):
            preds = preds[0]
        predictions = preds.argmax(axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    # Override remove_unused_columns for set_transform compatibility
    # When using set_transform (lazy loading), original columns must be kept
    training_args.remove_unused_columns = False
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")
    
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"\n🎯 Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    
    # Save model
    trainer.save_model()
    print(f"\n✅ Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
