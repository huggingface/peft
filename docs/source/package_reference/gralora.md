# GraLoRA

[**Granular Low-Rank Adaptation (GraLoRA)**](https://huggingface.co/papers/2505.20355) is a PEFT method designed to enhance the **expressivity** of low-rank adaptation while improving **robustness to outlier** activations, based on insights from well-known issues in quantization.

![GraLoRA Overview](https://github.com/SqueezeBits/GraLoRA/raw/main/figure/gralora_overview.png)

Unlike standard LoRA, which applies a single low-rank adapter across the entire feature space, GraLoRA introduces a structured and fine-grained adaptation scheme. It divides the adaptation space into a grid of $𝑘^2$ smaller, independent adapter pairs, each responsible for a localized subset of the input and output dimensions. As a result, each adapter operates on a subspace that is $k$ times smaller in both dimensions than the original LoRA adapter.

This granular decomposition enables spatially localized and context-aware updates, effectively increasing representational capacity without additional parameters or computational cost. By isolating the influence of extreme activations within smaller subspaces, GraLoRA mitigates gradient distortion and preserves inter-channel balance during adaptation.

---

The abstract from the paper is:

*Low-Rank Adaptation (LoRA) is a popular method for parameter-efficient fine-
tuning (PEFT) of generative models, valued for its simplicity and effectiveness.
Despite recent enhancements, LoRA still suffers from a fundamental limitation:
overfitting when the bottleneck is widened. It performs best at ranks 32–64, yet its
accuracy stagnates or declines at higher ranks, still falling short of full fine-tuning
(FFT) performance. We identify the root cause as LoRA’s structural bottleneck,
which introduces gradient entanglement to the unrelated input channels and distorts
gradient propagation. To address this, we introduce a novel structure, Granular
Low-Rank Adaptation (GraLoRA) that partitions weight matrices into sub-blocks,
each with its own low-rank adapter. With negligible computational or storage cost,
GraLoRA overcomes LoRA’s limitations, effectively increases the representational
capacity, and more closely approximates FFT behavior. Experiments on code
generation, commonsense reasoning, mathematical reasoning, general language
understanding, and image generation benchmarks show that GraLoRA consistently
outperforms LoRA and other baselines, achieving up to +8.5% absolute gain in
Pass@1 on HumanEval+. These improvements hold across model sizes and rank
settings, making GraLoRA a scalable and robust solution for PEFT.*

