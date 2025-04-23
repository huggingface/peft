# Bone (Bottleneck Orthogonal Network)

## Overview
Bone is a parameter-efficient fine-tuning method that uses orthogonal transformations in bottleneck layers. It's particularly effective for small to medium-sized models and offers excellent memory efficiency.

## Key Features
- Extremely memory efficient (~0.05% of base model parameters)
- Fast inference speed
- Good for small to medium models
- Simple implementation

## Performance Characteristics

### Memory Efficiency
| Model Size | Bone Parameters | Memory Usage |
|------------|----------------|--------------|
| 100M       | ~50K           | ~200KB       |
| 1B         | ~500K          | ~2MB         |
| 7B         | ~3.5M          | ~14MB        |
| 13B        | ~6.5M          | ~26MB        |

### Training Performance
| Metric | Value |
|--------|-------|
| Training Speed | Fast |
| Convergence | Quick (typically 1-2 epochs) |
| Inference Overhead | < 2% |

## Use Cases

### Best For
- Small to medium models
- Resource-constrained devices
- Classification tasks
- Quick experiments

### Not Recommended For
- Large language models (>13B parameters)
- Complex generation tasks
- Tasks requiring extensive adaptation

## Implementation

### Basic Usage
```python
from peft import BoneConfig, get_peft_model

# Define Bone configuration
config = BoneConfig(
    bottleneck_size=64,  # size of bottleneck layer
    target_modules=["attention.output"],
    dropout=0.1,
)

# Create PEFT model
model = get_peft_model(model, config)
```

### Advanced Configuration
```python
# Custom Bone configuration
config = BoneConfig(
    bottleneck_size=128,  # larger bottleneck
    target_modules=["attention.output", "intermediate"],
    dropout=0.2,
    use_orthogonal=True,  # enable orthogonal transformations
    orthogonal_eps=1e-6,  # epsilon for numerical stability
)
```

## Hyperparameter Tuning

### Recommended Ranges
| Parameter | Recommended Range | Impact |
|-----------|------------------|--------|
| bottleneck_size | 32-256 | Larger = better performance, more parameters |
| dropout | 0.0-0.3 | Regularization |
| orthogonal_eps | 1e-8 to 1e-4 | Numerical stability |

### Optimal Settings by Model Size
| Model Size | Bottleneck Size | Dropout | Orthogonal Eps |
|------------|----------------|---------|----------------|
| < 100M    | 32            | 0.1     | 1e-6          |
| 100M-1B   | 64            | 0.15    | 1e-6          |
| 1B-7B     | 128           | 0.2     | 1e-5          |
| 7B-13B    | 256           | 0.25    | 1e-5          |

## Comparison with Other Methods

### Performance Comparison
| Method | Memory Efficiency | Training Speed | Model Size Suitability |
|--------|------------------|----------------|-----------------------|
| Bone   | Very High       | Fast          | Small-Medium         |
| LoRA   | High            | Fast          | All                  |
| Adapter | Medium         | Medium        | All                  |
| Prompt | Very High      | Very Fast     | All                  |

### Memory Usage Comparison
| Method | Parameters (% of base) | Training Memory | Inference Memory |
|--------|----------------------|-----------------|------------------|
| Bone   | 0.05%               | Very Low       | Very Low         |
| LoRA   | 0.1%                | Low            | Low              |
| Adapter | 0.5%                | Medium         | Medium           |
| Prompt | 0.01%               | Very Low       | Very Low         |

## Best Practices

1. **Bottleneck Size Selection**
   - Start with size 64 for most cases
   - Increase for better performance
   - Consider model size and task complexity

2. **Target Modules**
   - Focus on attention outputs
   - Add intermediate layers for complex tasks
   - Consider model architecture

3. **Training Tips**
   - Use learning rate 5e-5 to 2e-4
   - Monitor orthogonal condition
   - Use gradient clipping

## Common Issues and Solutions

### Problem: Orthogonal Instability
**Solution:**
```python
# Improve numerical stability
config = BoneConfig(
    bottleneck_size=64,
    target_modules=["attention.output"],
    dropout=0.1,
    use_orthogonal=True,
    orthogonal_eps=1e-4,  # Increase epsilon
)
```

### Problem: Limited Adaptation
**Solution:**
```python
# Increase adaptation capacity
config = BoneConfig(
    bottleneck_size=128,  # Larger bottleneck
    target_modules=["attention.output", "intermediate"],  # More target modules
    dropout=0.1,
    use_orthogonal=True,
)
```

## Examples

### Text Classification
```python
from transformers import AutoModelForSequenceClassification
from peft import BoneConfig, get_peft_model

# Load base model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Configure Bone
config = BoneConfig(
    bottleneck_size=64,
    target_modules=["attention.output"],
    dropout=0.1,
    use_orthogonal=True,
)

# Create PEFT model
model = get_peft_model(model, config)
```

### Small Model Fine-tuning
```python
from transformers import AutoModelForCausalLM
from peft import BoneConfig, get_peft_model

# Load small base model
model = AutoModelForCausalLM.from_pretrained("gpt2-small")

# Configure Bone
config = BoneConfig(
    bottleneck_size=32,
    target_modules=["attention.output"],
    dropout=0.1,
    use_orthogonal=True,
)

# Create PEFT model
model = get_peft_model(model, config)
```

## References
1. [Bone Paper](https://arxiv.org/abs/your-paper-url)
2. [PEFT Documentation](https://huggingface.co/docs/peft/index)
3. [Implementation Guide](https://github.com/huggingface/peft) 