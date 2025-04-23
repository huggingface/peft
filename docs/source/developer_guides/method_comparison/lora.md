# LoRA (Low-Rank Adaptation)

## Overview
LoRA is a parameter-efficient fine-tuning method that introduces trainable low-rank matrices into transformer layers. It's particularly effective for large language models and offers a good balance between performance and resource efficiency.

## Key Features
- Memory efficient (~0.1% of base model parameters)
- Minimal impact on inference speed
- Easy to implement and use
- Compatible with most transformer architectures

## Performance Characteristics

### Memory Efficiency
| Model Size | LoRA Parameters | Memory Usage |
|------------|----------------|--------------|
| 1B         | ~1M            | ~4MB         |
| 7B         | ~7M            | ~28MB        |
| 13B        | ~13M           | ~52MB        |
| 70B        | ~70M           | ~280MB       |

### Training Performance
| Metric | Value |
|--------|-------|
| Training Speed | Fast (similar to full fine-tuning) |
| Convergence | Quick (typically 1-2 epochs) |
| Inference Overhead | < 5% |

## Use Cases

### Best For
- General fine-tuning tasks
- Large language models
- Multi-task learning
- Resource-constrained environments

### Not Recommended For
- Tasks requiring extensive model modifications
- Very small models (< 100M parameters)
- Real-time applications with strict latency requirements

## Implementation

### Basic Usage
```python
from peft import LoraConfig, get_peft_model

# Define LoRA configuration
config = LoraConfig(
    r=8,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Create PEFT model
model = get_peft_model(model, config)
```

### Advanced Configuration
```python
# Custom LoRA configuration for specific needs
config = LoraConfig(
    r=16,  # higher rank for better performance
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["classifier"],
)
```

## Hyperparameter Tuning

### Recommended Ranges
| Parameter | Recommended Range | Impact |
|-----------|------------------|--------|
| rank (r) | 4-32 | Higher = better performance, more parameters |
| alpha | 8-64 | Controls scaling of LoRA weights |
| dropout | 0.0-0.1 | Regularization, prevent overfitting |

### Optimal Settings by Model Size
| Model Size | Rank | Alpha | Dropout |
|------------|------|-------|---------|
| < 1B      | 4-8  | 16-32 | 0.05    |
| 1B-7B     | 8-16 | 32-64 | 0.05    |
| 7B-13B    | 16-32| 64    | 0.1     |
| > 13B     | 32   | 64    | 0.1     |

## Comparison with Other Methods

### Performance Comparison
| Method | Memory Efficiency | Training Speed | Use Case Flexibility |
|--------|------------------|----------------|----------------------|
| LoRA   | High            | Fast          | High                |
| Full FT | Low            | Slow          | High                |
| Adapter | Medium         | Medium        | Medium              |
| Prompt | Very High      | Very Fast     | Low                 |

### Memory Usage Comparison
| Method | Parameters (% of base) | Memory Overhead |
|--------|----------------------|-----------------|
| LoRA   | 0.1%                | Low            |
| Full FT | 100%               | High           |
| Adapter | 0.5%               | Medium         |
| Prompt | 0.01%              | Very Low       |

## Best Practices

1. **Rank Selection**
   - Start with rank 8 for most cases
   - Increase rank for better performance if needed
   - Consider model size when choosing rank

2. **Target Modules**
   - Include attention layers (q_proj, v_proj)
   - Add more layers for complex tasks
   - Consider model architecture

3. **Training Tips**
   - Use learning rate 1e-4 to 5e-4
   - Apply gradient clipping
   - Monitor loss convergence

## Common Issues and Solutions

### Problem: Slow Training
**Solution:**
```python
# Optimize training speed
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Focus on key layers
    lora_dropout=0.0,  # Remove dropout for speed
)
```

### Problem: High Memory Usage
**Solution:**
```python
# Reduce memory usage
config = LoraConfig(
    r=4,  # Lower rank
    lora_alpha=16,
    target_modules=["q_proj"],  # Fewer target modules
)
```

## Examples

### Text Classification
```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Configure LoRA
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
)

# Create PEFT model
model = get_peft_model(model, config)
```

### Language Model Fine-tuning
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["c_attn"],
    lora_dropout=0.1,
)

# Create PEFT model
model = get_peft_model(model, config)
```

## References
1. [LoRA Paper](https://arxiv.org/abs/2106.09685)
2. [PEFT Documentation](https://huggingface.co/docs/peft/index)
3. [Implementation Guide](https://github.com/huggingface/peft) 