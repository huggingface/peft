# Block-Diagonal LoRA for Eliminating Communication Overhead in Tensor Parallel LoRA Serving

Block-Diagonal LoRA (BD-LoRA) is a LoRA variant in which some LoRA factors are constrained to be block-diagonal. This allows faster serving by eliminating communication overheads
when running inference on multiple GPUs. Despite the block-diagonal constraint, BD-LoRA is similarly performant to vanilla LoRA at similar parameter counts.

BD-LoRA is designed to be used with tensor parallelism, which means sharding the weights of a model among multiple GPUs. A popular sharding strategy is the [Megatron Sharding Strategy](https://arxiv.org/abs/1909.08053). For two linear layers $W_1$, $W_2$ that follow each other (for example the up and down projections in a transformer MLP module), we will shard the first layer in a column-parallel way (which requires LoRA B to be block-diagonal) and the second layer in a row-parallel way (which requires LoRA A to be block-diagonal). For the attention module, this can be similarly achieved by taking the Q, K and V projections together as $W_1$ and the out projection as $W_2$, sharding accordingly. This sharding allows a compatible inference engine to distribute each block-diagonal shard over a a different GPU, cutting the need to communicate partial results among GPUs. In the image below, you can see the exact sharding strategy and how this saves computational efforts.

Paper: https://hf.co/papers/2510.23346

<div>
<img src="https://github.com/huggingface/peft/blob/main/examples/bdlora_finetuning/bdlora-sharding.png?raw=true" width="800"/>
</div>

### Performance, rank and parameter count
BD-LoRA achieves similar performance to LoRA (see image below, or the `method_comparison` folder in the peft repository root) at the same parameter count. However, as every other factor in BD-LoRA is block-diagonal, a BD-LoRA adapter will have less parameters than a LoRA adapter at the same rank. The performance of BD-LoRA is only competitive when the rank is then increased accordingly. We provide example code for rank-matching at the end of this example notebook.

<div>
<img src="https://github.com/huggingface/peft/blob/main/examples/bdlora_finetuning/bdlora-performance.png?raw=true" width="600"/>
</div>

