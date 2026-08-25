<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Distilling Mixture-of-Expert layers into dense layers

Mixture-of-Experts (MoE) models have the advantage that despite having a high parameter count, they are relatively quick at inference because only a sparse subset of "expert" weights (usually, a small subset of weights is chosen on a per-token level). However, they usually still require a lot of memory at inference time, as all experts are held in memory for quick access. The idea of distilling MoE into dense layers is that for specific tasks, we may only need a dense subset of those experts. Think about using a generalist LLM in a very specific domain or a multi-lingual LLM only in one language. In an ideal scenario, after distillation, the model is not only more parameter-efficient, but also quicker at inference.

MoE-to-dense ([Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://huggingface.co/papers/2605.28207), [original source repository](https://github.com/krafton-ai/moe-to-dense/)) is a specific set of techniques to convert MoE models into a dense models. For each MoE layer, the experts are scored by importance using the routing statistics of the model, the most important experts are concatenated into a single dense feed-forward network (FFN), and the resulting dense student is refined by distilling the logits of the original MoE model (the teacher) into it.

> [!NOTE]
> Strictly speaking, this is not a parameter-*efficient* fine-tuning method: the trainable "adapter" consists of the full dense FFNs, whose size by default corresponds to the number of *active* parameters of the MoE model. However, after training the adapter on a use-case specific dataset, it allows to create a more parameter efficient version of the original model. With PEFT, this approach also allows to create multiple adapters, switch between or disable them, use `modules_to_save` to fully fine-tune some layers, save checkpoints with reduced size, etc. During training, the adapter can be disabled to retrieve the teacher model without requiring extra parameters. This is why adding it to PEFT is still a good fit.

The abstract from the paper is:

*Mixture-of-Experts (MoE) is now the dominant architecture for frontier language models, yet it requires all expert parameters to be loaded in memory, making it less preferable for memory-constrained deployment. Existing compression methods reduce the number of experts but the output remains an MoE model with the same fundamental limitation. We present the first systematic framework for converting a trained MoE into a standard fully dense architecture: experts are scored, selected, and grouped, then concatenated into a dense FFN and refined by knowledge distillation from the MoE teacher. We evaluate 7 scoring, 5 grouping, and 2 magnitude scaling methods across a range of selected expert counts on Qwen3-30B-A3B, yielding 350 configurations. We find that the choice of scoring method is the most impactful, with our novel diversity-aware scoring consistently outperforming prior methods on Qwen3-30B-A3B, DeepSeek-V2-Lite, and GPT-OSS-20B. Under a controlled comparison at matched parameter count, MoE-to-dense outperforms dense-to-dense pruning by +6.3 pp in average downstream accuracy after ~4B-token distillation at 1.6x faster training wall-clock speed.*

The current PEFT implementation only supports a small subset of options presented in the paper. If you would like to see more features supported, please let us know by creating an [issue on our repository](https://github.com/huggingface/peft/issues).

## How it works

A complete training run typically consists of these steps:

1. Score and select: During forward passes of the MoE model, a hook on the router of each MoE layer collects routing statistics. Experts are scored by their *conditional probability* (CP, Section 3.2, Eq. 3 of the paper): the average routing probability of the expert over the tokens for which it was selected, which favors specialist experts. Among the paper's scoring methods that can be computed from the router outputs alone, CP performed best (Section 4.2, Figure 4). The `num_experts_to_keep` highest scoring experts are kept; by default this is the number of experts the router activates per token (top-k), so that the dense model has exactly as many parameters as the MoE model has active parameters. This is the paper's *pure pruning* configuration (`K = k`), which it found to outperform selecting more experts and merging them (Section 4.2).
2. Concatenate: The kept experts are concatenated into a single dense FFN per layer (Section 3.1, Eq. 2 of the paper), with the down projections (and their biases) scaled uniformly by `1/num_experts_to_keep` (the *uniform* down projection scaling of Section 3.4 and Appendix H, which matches the average routing weight when the router normalizes its top-k weights to sum to 1). The concatenation preserves the intermediate activations of the kept experts exactly; only the token-dependent routing weights are replaced by these constants (Appendix A of the paper). Technically, the dense FFN is an instance of the model's own experts class configured with a single expert, which means that the activation function, the tensor layout, and the kernels of the original implementation are reused.
3. Distill: The dense student is trained to match the logits of the MoE teacher by minimizing the forward KL divergence `KL(p_teacher || p_student)` (Section 3.5, Eq. 5 of the paper). The teacher forward pass is the forward pass with disabled adapters.
4. Export: [`MoeToDenseModel.compress_and_unload`] replaces the MoE layers by the dense FFNs and adjusts the model config, resulting in a regular transformers model that can be saved and loaded without PEFT.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM
from peft import MoeToDenseConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, device_map="cuda")
# the experts modules of all MoE layers are detected automatically
model = get_peft_model(model, MoeToDenseConfig())

# 1. collect routing statistics on calibration data (a few hundred batches of the training data are sufficient)
model.eval()
with torch.no_grad():
    for batch in calibration_batches:
        model(**batch)
# 2. score the experts and build the dense FFNs; afterwards, the model uses the dense FFNs instead of the MoE experts
model.update_and_allocate()

# 3. distill the MoE teacher (= adapters disabled) into the dense student
model.train()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
for batch in train_batches:
    loss = model.get_distillation_loss(**batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 4. export a standalone dense model
dense_model = model.compress_and_unload()
dense_model.save_pretrained("qwen3-30b-a3b-dense")
```

Some notes on this workflow:

- Before [`MoeToDenseModel.update_and_allocate`] is called, the model behaves like the original MoE model. Routing statistics are collected during *all* forward passes until the dense FFNs are allocated, so any forward pass counts as calibration, including the teacher forward passes of [`MoeToDenseModel.get_distillation_loss`]. Calling `update_and_allocate()` a second time has no effect.
- [`MoeToDenseModel.get_distillation_loss`] accepts the same arguments as the forward call of the model. It runs the teacher forward pass without gradients, then the student forward pass, and returns the forward KL divergence averaged over the tokens (Section 3.5, Eq. 5 of the paper; the paper found this logit-only objective to clearly beat the reverse KL divergence and an additional hidden-state loss, Section 4.3, Table 5). If `labels` are passed, tokens with a label of `-100` are ignored, otherwise, if an `attention_mask` is passed, padding tokens are ignored. Pass precomputed `teacher_logits` to take control of the teacher forward pass yourself (e.g. to distill from expanded teacher routing, Section 3.5, Eq. 6 of the paper, or to cache teacher logits). The KL divergence is computed in chunks of tokens (`chunk_size`) with activation checkpointing to limit the peak memory, which matters for large vocabularies.
- The adapter, i.e. the dense FFNs, can be saved with `save_pretrained` and loaded with [`PeftModel.from_pretrained`] like any other adapter. A loaded adapter does not need calibration. Be aware that the adapter checkpoint is large relatively large compared to a typical PEFT adapter, as it contains large linear layer weights.
- The dense FFNs keep the dtype of the base model, they are *not* cast to float32 like most PEFT adapters (`autocast_adapter_dtype` has no effect). To train them in float32 with mixed precision, cast the trainable parameters after `update_and_allocate()` and wrap the loss computation in `torch.autocast`.
- `modules_to_save` fully fine-tunes additional modules together with the dense FFNs. Typical targets would be the attention projections and the norms, which are typically small compared to the MoE layers. Trainable copies are created and saved with the adapter, while the teacher uses the originals. This lets the rest of the network adapt to the replaced FFNs, which matters because the residual stream changes when an MoE layer is replaced and the error may compound over the layers. Keep the embeddings and the language modeling head out of it, they are shared between teacher and student.
- `num_experts_to_keep` can be set to a value other than the router's top-k, resulting in a wider (more capacity, more parameters) or narrower dense FFN. Note that this is *not* what the paper does for `K > k` (see [Deviations from the paper](#deviations-from-the-paper)) and has not been evaluated.

A complete example including distillation on MetaMathQA and evaluation on GSM8K can be found in [`examples/moe_to_dense`](https://github.com/huggingface/peft/tree/main/examples/moe_to_dense).

## Supported architectures

The implementation targets the *experts* module of the MoE layers, i.e. the module that holds the weights of all experts as 3D tensors (e.g. `Qwen3MoeExperts`), and leaves the router module, which must be a sibling of the experts module, in place. Some details differ between architectures (attribute names, config field names, whether the router returns logits or probabilities, per-expert output scales); these are covered by a small per-architecture registry (`peft.tuners.moe_to_dense.arch.ARCH_SPECS`) with a duck-typing fallback for unknown architectures.

- Tested: Qwen3-MoE, GPT-OSS, Gemma 4 (text). These have registry entries and are covered by the test suite.
- Work through the generic fallback: Mixtral, Qwen2-MoE, OLMoE, GraniteMoE, GraniteMoE-shared.
- Requirements for other architectures: the experts module must be decorated with Transformers' `use_experts_implementation` (which provides the tensor layout information), its router and experts modules must be constructible from the config alone, and the router must return a tuple containing the selected expert indices and the logits or probabilities over all experts. Older transformers MoE implementations (especially pre v4) with one `nn.Module` per expert, custom code models, and e.g. Llama 4 are not supported.

## Export and inference

[`MoeToDenseModel.compress_and_unload`] uses one of two strategies, chosen per architecture:

- Dense MLP (Qwen3-MoE): the whole MoE block (router and experts) is replaced by the architecture's dense MLP class and the converted layers are registered as dense layers in the config (`mlp_only_layers`). The result is a true dense model without any MoE machinery or overhead at inference time.
- One expert (generic fallback, used for GPT-OSS, Gemma 4, and all other architectures): the MoE block is kept, but with a single expert (the dense FFN) and a router with zeroed projections that always selects this expert with weight 1. The config is patched accordingly (one expert, top-1 routing, larger intermediate size). This is necessary for architectures without a dense sibling or with MoE layers that cannot be expressed as a single dense MLP (Gemma 4 runs its experts in parallel to a dense MLP with separate norms). The exported model is exact, but it still pays for the routing (a tiny router projection and the gather/scatter of the experts implementation). This overhead is negligible for prefill and mostly consists of kernel launches for decoding (tip: try `torch.compile` for better performance). Prefer `experts_implementation="grouped_mm"` or `"eager"` over `"batched_mm"` for the exported model.

In both cases, the exported model can be saved with `save_pretrained` and loaded with `from_pretrained` without PEFT. Inference engines other than transformers should handle the dense MLP export as a regular dense model; for the one expert export, they would treat it as an MoE model with a single expert, which has not been tested.

`merge_and_unload()`, `merge_adapter()`, and `unmerge_adapter()` are not supported, as the dense FFNs replace the MoE layers instead of being added to them.

## Deviations from the paper

The implementation follows the recipe that the paper found to work best, but leaves out most of the design space that the paper explores:

- Pure pruning only: The paper selects `K >= k` experts and, for `K > k`, merges them into `k` groups by score-weighted averaging using one of five grouping strategies, so that the dense FFN always has the width of the active path. Only the `K = k` case (each kept expert forms its own group, no merging) is implemented, which the paper found to be the best configuration for its diversity-aware scoring. Setting `num_experts_to_keep` to a value other than `k` changes the width of the dense FFN instead of merging experts.
- Conditional probability scoring only: The paper's best scoring methods, ACP (activation-weighted conditional probability) and DO-ACP (D-optimal selection on ACP), additionally require the output norms of all experts and the Gram matrix of the expert outputs on calibration data. Computing these requires a dedicated calibration pass that evaluates *all* experts for every token, which is not implemented. CP, the best method that can be computed from the router outputs alone, is used instead.
- Uniform scaling only: The down projections are scaled uniformly by `1/num_experts_to_keep` (the paper's default, Section 3.4); the paper's alternative, scaling proportional to the expert scores (Appendix H), is not implemented. Uniform scaling matches the average routing weight when the router normalizes the top-k weights to sum to 1; per-expert output scales as in Gemma 4 are taken into account. For routers that do not normalize the weights (e.g. OLMoE with `norm_topk_prob=False`, or the DeepSeek routers with their scaling factor), the initialization is mis-scaled, which distillation compensates for but which makes the student before distillation worse than necessary.
- Routing probabilities: The scores are computed from the softmax over the router logits, as in the paper. For routers that do not use a softmax (e.g. the sigmoid routers of DeepSeek-V3), the scores are still computed from a softmax over the router outputs, which is a different ranking than the paper's definition would give.
- Shared experts are not concatenated into the dense FFN as in the paper's DeepSeek-V2-Lite experiments; they are left in place as separate (dense) modules. The resulting model is functionally equivalent, only structured differently.
- Expanded teacher routing (`k' > k`, Section 4.3), which gave the paper a small improvement, is not built in, but can be done by computing the teacher logits yourself and passing them via `teacher_logits`.
- Distillation setup: Only the paper's best objective, forward KL on the logits, is provided (reverse KL and intermediate hidden-state losses, which the paper found to hurt in Section 4.3, are not). The training loop, optimizer, and data pipeline are left to the user; the paper's hyperparameters (Section 4.1: Training) should be a good starting point for general-purpose distillation.

## Limitations

- Quantized experts (e.g. bitsandbytes or the MXFP4 checkpoints of GPT-OSS) are not supported, as the expert weights have to be sliced and concatenated. Load the model with dequantized experts.
- Multiple adapters can be added or loaded (e.g. with different `num_experts_to_keep`), but only one adapter can be active at a time.
- Hub kernels that replace the forward of the whole MoE block (e.g. `use_kernels=True` for GPT-OSS) read the expert weights directly and bypass the dense FFNs; keep them disabled while the adapter is attached. The `experts_implementation` setting of Transformers is supported, but changing it after `get_peft_model` via `set_experts_implementation` only affects the teacher, as the dense FFNs hold a copy of the config until they are exported.
- The quality of the dense model depends on the size of the training dataset. In the paper, the dense student of Qwen3-30B-A3B reaches 58% average downstream accuracy after ~4B tokens versus 80% for the teacher, and stays below Qwen's pretrained dense models of similar size (Section 4.4). The method is most useful when no small dense sibling of the MoE model exists or when the model is distilled for a specialized use case.

## MoeToDenseConfig

[[autodoc]] tuners.moe_to_dense.config.MoeToDenseConfig

## MoeToDenseModel

[[autodoc]] tuners.moe_to_dense.model.MoeToDenseModel
    - update_and_allocate
    - get_distillation_loss
    - compress_and_unload
    - unload
