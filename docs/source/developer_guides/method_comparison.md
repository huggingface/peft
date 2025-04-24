# Method Comparison Guide

This guide provides a comprehensive comparison of different Parameter-Efficient Fine-Tuning (PEFT) methods available in the PEFT library. Each method has its own strengths and is suited for different use cases.

## Available Methods

- [LoRA (Low-Rank Adaptation)](method_comparison/lora.md) - A versatile method that works well across different model sizes
- [LoRA-FA (LoRA with Fast Adaptation)](method_comparison/lora_fa.md) - An enhanced version of LoRA optimized for quick adaptation
- [Bone (Bottleneck Network)](method_comparison/bone.md) - A method with unique merged inference capabilities

## Quick Comparison

| Method | Memory Efficiency | Training Speed | Parameter Efficiency |
|--------|------------------|----------------|----------------------|
| LoRA | High (0.96-1.90%) | Fast | 0.96-1.90% of parameters |
| LoRA-FA | Very High (0.24-0.47%) | Fast | 0.24-0.47% of parameters |
| Bone | Medium (15.30-30.39%) | Fast | 15.30-30.39% of parameters |

## Choosing the Right Method

When selecting a PEFT method, consider the following factors:

1. **Model Size**
   - Small models (<1B parameters): All methods work well
   - Medium to large models (>1B parameters): LoRA and LoRA-FA have proven efficiency with parameter ratio decreasing as models grow larger
   - Bone's parameter efficiency improves with larger models (15.30% for 1.3B vs 30.39% for 350M)

2. **Resource Constraints**
   - Limited memory: LoRA shows excellent memory efficiency (9-48MB for models 125M-1.3B)
   - Very limited memory: LoRA-FA shows superior memory efficiency (1.12-6.00MB for models 125M-1.3B)
   - Fast inference priority: Bone offers superior merged inference (43-51% speedup)

3. **Task Type**
   - Consider benchmarks specific to your task type
   - Different methods may excel at different tasks

4. **Performance Requirements**
   - Inference efficiency: Bone offers significantly faster merged inference (-43.10% to -51.49% overhead)
   - Lowest parameter count: LoRA-FA requires fewest parameters (0.24-0.47%)
   - Memory efficiency: All methods offer significant memory savings compared to full fine-tuning

## Tradeoffs

Each method has its own tradeoffs that should be considered:

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| LoRA | Well-established, minimal inference overhead | Requires more parameters than LoRA-FA |
| LoRA-FA | Superior parameter efficiency, faster convergence | May have higher inference overhead in some configurations |
| Bone | Excellent merged inference speed, good performance | Higher parameter count (15.30-30.39%) |

## Implementation Details

Each method has its own configuration and implementation details. Please refer to the individual method documentation for specific implementation guides:

- [LoRA Implementation Guide](method_comparison/lora.md#implementation)
- [LoRA-FA Implementation Guide](method_comparison/lora_fa.md#implementation)
- [Bone Implementation Guide](method_comparison/bone.md#implementation)

## Performance Metrics

For detailed performance metrics and comparisons, please refer to the individual method documentation. Each method's documentation includes:

- Memory efficiency metrics
- Training performance characteristics
- Use case recommendations
- Hyperparameter tuning guides

## Best Practices

1. Start with benchmarking each method on your specific task
2. Consider the trade-offs between memory efficiency, training speed, and adaptation quality
3. Larger models benefit more from parameter-efficient methods (lower relative parameter count)
4. If inference speed is critical, consider Bone's merge capability (43-51% speedup)
5. For maximum parameter efficiency, LoRA-FA offers the lowest parameter count

## References

- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [Implementation Guide](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [LoRA-FA Paper](https://arxiv.org/abs/2308.03303) (Lin et al., 2023)