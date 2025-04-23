# Method Comparison Guide

This guide provides a comprehensive comparison of different Parameter-Efficient Fine-Tuning (PEFT) methods available in the PEFT library. Each method has its own strengths and is suited for different use cases.

## Available Methods

- [LoRA (Low-Rank Adaptation)](lora.md) - A versatile method that works well across different model sizes
- [LoRA-FA (LoRA with Fast Adaptation)](lora_fa.md) - An enhanced version of LoRA optimized for quick adaptation
- [Bone (Bottleneck Orthogonal Network)](bone.md) - A memory-efficient method particularly suited for small to medium models

## Quick Comparison

| Method | Memory Efficiency | Training Speed | Best For |
|--------|------------------|----------------|----------|
| LoRA | High | Fast | General fine-tuning, large models |
| LoRA-FA | High | Very Fast | Quick adaptation, resource-constrained environments |
| Bone | Very High | Fast | Small to medium models, classification tasks |

## Choosing the Right Method

When selecting a PEFT method, consider the following factors:

1. **Model Size**
   - Small models (<1B parameters): Consider Bone
   - Medium to large models: Consider LoRA or LoRA-FA

2. **Resource Constraints**
   - Limited memory: Bone or LoRA-FA
   - Limited training time: LoRA-FA

3. **Task Type**
   - Classification: Bone
   - Generation: LoRA or LoRA-FA
   - Multi-task learning: LoRA

4. **Performance Requirements**
   - Fast adaptation: LoRA-FA
   - Maximum performance: LoRA
   - Memory efficiency: Bone

## Implementation Details

Each method has its own configuration and implementation details. Please refer to the individual method documentation for specific implementation guides:

- [LoRA Implementation Guide](lora.md#implementation)
- [LoRA-FA Implementation Guide](lora_fa.md#implementation)
- [Bone Implementation Guide](bone.md#implementation)

## Performance Metrics

For detailed performance metrics and comparisons, please refer to the individual method documentation. Each method's documentation includes:

- Memory efficiency metrics
- Training performance characteristics
- Use case recommendations
- Hyperparameter tuning guides

## Best Practices

1. Start with LoRA for general use cases
2. Use LoRA-FA when quick adaptation is required
3. Consider Bone for small models or memory-constrained environments
4. Always benchmark performance before committing to a method

## References

- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [Implementation Guide](https://github.com/huggingface/peft) 