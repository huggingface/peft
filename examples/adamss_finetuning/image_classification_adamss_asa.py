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


# Hyperparameters from Table 18 in the paper
HYPERPARAMS = {
    "vit-large-patch16-224-in21k": {
        "pets": {"lr": 0.001, "head_lr": 0.0005, "wd": 0.0005},
        "cars": {"lr": 0.01, "head_lr": 0.005, "wd": 0.1},
        "cifar10": {"lr": 0.01, "head_lr": 0.05, "wd": 0.1},
        "cifar100": {"lr": 0.01, "head_lr": 0.05, "wd": 0.05},
        "eurosat": {"lr": 0.01, "head_lr": 0.0005, "wd": 0.01},
        "fgvc": {"lr": 0.01, "head_lr": 0.0005, "wd": 0.0005},
        "resisc": {"lr": 0.01, "head_lr": 0.0005, "wd": 0.1},
    },
    "vit-base-patch16-224-in21k": {
        "pets": {"lr": 0.005, "head_lr": 0.005, "wd": 0.0005},
        "cars": {"lr": 0.01, "head_lr": 0.005, "wd": 0.0},
        "cifar10": {"lr": 0.01, "head_lr": 0.005, "wd": 0.05},
        "cifar100": {"lr": 0.01, "head_lr": 0.005, "wd": 0.05},
        "eurosat": {"lr": 0.01, "head_lr": 0.0005, "wd": 0.05},
        "fgvc": {"lr": 0.01, "head_lr": 0.005, "wd": 0.0005},
        "resisc": {"lr": 0.01, "head_lr": 0.005, "wd": 0.0005},
    },
}

# Model-specific K values (number of subspaces)
MODEL_K_VALUES = {
    "vit-large-patch16-224-in21k": 16,
    "vit-base-patch16-224-in21k": 10,
}
# Dataset configurations (matching exec_adamss_peft.py)
DATASET_CONFIGS = {
    "cars": {
        "train": "Multimodal-Fatima/StanfordCars_train",
        "test": "Multimodal-Fatima/StanfordCars_test",
        "img_col": "image",
        "label_col": "label",
    },
    "cifar10": {
        "train": "Multimodal-Fatima/CIFAR10_train",
        "test": "Multimodal-Fatima/CIFAR10_test",
        "img_col": "image",
        "label_col": "label",
    },
    "cifar100": {
        "train": "cifar100",
        "test": "cifar100",
        "img_col": "img",
        "label_col": "fine_label",
    },
    "eurosat": {
        "dataset": "timm/eurosat-rgb",
        "img_col": "image",
        "label_col": "label",
    },
    "pets": {
        "train": "timm/oxford-iiit-pet",
        "test": "timm/oxford-iiit-pet",
        "img_col": "image",
        "label_col": "label",
    },
}


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
class ImageClassificationArguments:
    """Arguments for image classification with AdaMSS and ASA."""
    
    # Model configuration
    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Model identifier: vit-base or vit-large"}
    )
    dataset_name: str = field(
        default="cifar10",
        metadata={"help": "Dataset: cifar10, cifar100, pets, cars, eurosat, fgvc, resisc"}
    )
    
    # AdaMSS Configuration
    adamss_r: int = field(default=100, metadata={"help": "SVD rank"})
    adamss_k: int = field(default=10, metadata={"help": "Number of subspaces (K), auto-set based on model"})
    adamss_ri: int = field(default=3, metadata={"help": "Subspace rank (rk), use 3 for vision"})
    
    # ASA Configuration
    use_asa: bool = field(default=False, metadata={"help": "Enable Adaptive Subspace Allocation"})
    target_kk: int = field(default=5, metadata={"help": "Target active subspaces for ASA"})
    asa_init_warmup: int = field(default=50, metadata={"help": "ASA init warmup in STEPS"})
    asa_final_warmup: int = field(default=1000, metadata={"help": "ASA final warmup in STEPS"})
    asa_mask_interval: int = field(default=100, metadata={"help": "ASA mask interval in STEPS"})
    asa_beta1: float = field(default=0.85, metadata={"help": "EMA coefficient for importance"})
    asa_beta2: float = field(default=0.85, metadata={"help": "EMA coefficient for uncertainty"})
    asa_tt: float = field(default=3.0, metadata={"help": "ASA schedule exponent"})
    
    # Training Configuration
    num_epochs: int = field(default=10, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=32, metadata={"help": "Batch size per device"})
    warmup_ratio: float = field(default=0.0, metadata={"help": "Warmup ratio"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Max training samples (for debug)"})
    
    # Other
    seed: int = field(default=0, metadata={"help": "Random seed"})
    output_dir: str = field(default="./output", metadata={"help": "Output directory"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory"})


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
    parser = HfArgumentParser(ImageClassificationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Auto-detect model type and set K value
    model_name = args.model_name_or_path
    model_type = None
    for key in MODEL_K_VALUES.keys():
        if key in model_name:
            model_type = key
            break
    
    if model_type is None:
        # Default to base model
        model_type = "vit-base-patch16-224-in21k"
        print(f"Warning: Model type not recognized, defaulting to {model_type}")
    
    # Override K value based on model type
    args.adamss_k = MODEL_K_VALUES[model_type]
    
    # Get hyperparameters from Table 18
    if model_type in HYPERPARAMS and args.dataset_name in HYPERPARAMS[model_type]:
        hp = HYPERPARAMS[model_type][args.dataset_name]
        print(f"Using Table 18 hyperparameters for {model_type} + {args.dataset_name}")
        print(f"   lr={hp['lr']}, head_lr={hp['head_lr']}, wd={hp['wd']}")
    else:
        hp = {"lr": 0.01, "head_lr": 0.005, "wd": 0.0005}
        print(f"Warning: No Table 18 hyperparameters found, using defaults: {hp}")
    
    print("\n" + "="*80)
    print(f"AdaMSS {'with ASA' if args.use_asa else 'without ASA'} - {args.dataset_name.upper()}")
    print("="*80)
    print(f"Model: {model_type}")
    print(f"AdaMSS: r={args.adamss_r}, K={args.adamss_k}, ri={args.adamss_ri}")
    if args.use_asa:
        print(f"ASA: Target {args.target_kk}/{args.adamss_k} subspaces")
        print(f"     Warmup steps {args.asa_init_warmup} → {args.asa_final_warmup}")
    print(f"Training: {args.num_epochs} epochs, batch_size={args.batch_size}, seed={args.seed}")
    print("="*80 + "\n")
    
    # Get dataset configuration
    if args.dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[args.dataset_name]
    img_name = config["img_col"]
    label_name = config["label_col"]
    
    # Load dataset
    print(f"Loading {args.dataset_name} dataset...")
    if "dataset" in config:
        # Single dataset with train/val/test splits (e.g., eurosat)
        dataset = load_dataset(config["dataset"], cache_dir=args.cache_dir)
        train_val = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        train_ds = train_val["train"]
        val_ds = train_val["test"]
        # Try 'test' split, fall back to 'val' if not available
        if "test" in dataset:
            test_ds = dataset["test"]
        elif "val" in dataset:
            test_ds = dataset["val"]
        else:
            print(f"Warning: No test/val split found, using validation set as test")
            test_ds = val_ds
    else:
        # Separate train and test datasets (e.g., cars, cifar10)
        train_val_ds = load_dataset(config["train"], split="train", cache_dir=args.cache_dir)
        test_ds = load_dataset(config["test"], split="test", cache_dir=args.cache_dir)
        
        # Split train into train and validation
        train_val = train_val_ds.train_test_split(test_size=0.1, seed=args.seed)
        train_ds = train_val["train"]
        val_ds = train_val["test"]
    
    print(f"Detected columns - Image: '{img_name}', Label: '{label_name}'")
    
    # Limit train samples if specified (for quick testing)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        # Also limit validation for faster testing
        val_ds = val_ds.select(range(min(5000, len(val_ds))))
    
    labels = train_ds.features[label_name].names
    num_classes = len(labels)
    print(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"   Number of classes: {num_classes}")
    
    # Create label mappings
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    # Load image processor
    print("\nLoading image processor...")
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
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
    print("\nLoading base model...")
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir,
    )
    
    # Configure AdaMSS
    print("\nApplying AdaMSS...")
    config = AdaMSSConfig(
        r=args.adamss_r,
        num_subspaces=args.adamss_k,
        subspace_rank=args.adamss_ri,
        target_modules=["query", "value"],
        use_asa=args.use_asa,
        target_kk=args.target_kk if args.use_asa else None,
        modules_to_save=["classifier"],
    )
    
    # Apply PEFT
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Print detailed parameter breakdown (same logic as exec_adamss_peft.py)
    print("\n[Detailed Parameter Breakdown]")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad)
    adamss_params = trainable_params - head_params
    print(f"Classifier Head Params: {head_params:,}")
    print(f"AdaMSS Adapter Params:  {adamss_params:,}")
    print(f"Total Trainable Params: {trainable_params:,}")
    
    # Setup ASA callback if enabled
    callbacks = []
    if args.use_asa:
        print("\nSetting up ASA callback...")
        asa_callback = ASACallback(
            target_kk=args.target_kk,
            init_warmup=args.asa_init_warmup,
            final_warmup=args.asa_final_warmup,
            mask_interval=args.asa_mask_interval,
            beta1=args.asa_beta1,
            beta2=args.asa_beta2,
            tt=args.asa_tt,
            verbose=True,  # Enable verbose output for ASA monitoring
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
    
    # Create TrainingArguments manually (not parsed to avoid conflicts)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=hp['lr'],
        weight_decay=hp['wd'],
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=100,
        logging_strategy="steps",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,  # Required for set_transform compatibility
    )
    
    # Create custom optimizer with different LR for head
    from torch.optim import AdamW
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad],
            "lr": hp['head_lr'],
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad],
            "lr": hp['lr'],
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=hp['wd'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )
    
    # GPU memory monitoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n[GPU Memory - Before Training]")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    # GPU memory stats
    if torch.cuda.is_available():
        print(f"\n[GPU Memory - Peak During Training]")
        print(f"Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Peak Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    
    # Print best metric
    if trainer.state.best_metric is not None:
        print(f"\n[Best Model Info]")
        print(f"Best accuracy: {trainer.state.best_metric:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")
    
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"\nTest Accuracy: {test_metrics['test_accuracy']:.4f}")
    
    # Save model
    trainer.save_model()
    print(f"\nModel saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
