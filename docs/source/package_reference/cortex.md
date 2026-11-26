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

# Cortex

Cortex adds trainable channels to the frozen feed-forward networks (FFNs) of a pretrained model. Each added channel works like a native FFN channel: it has its own gate, up, and down weights and uses the activation function of the FFN it extends. This is the Cortex-Structural parameterization from [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399).

The abstract from the paper is:

> Context can improve large language model performance, but these gains may be lost after context removal. Inspired by biological systems consolidation, we formulate Memory Consolidation (MC) as internalizing these gains while preserving non-target capabilities, connecting context-specified editing, skill learning, and behavioral steering under a common objective. We introduce a five-task-family testbed measuring consolidation and retention under single, batch, and sequential updates. We propose Cortex, which generates prospective queries to anticipate uses of context and distills context-conditioned outputs into mergeable low-rank or structural feed-forward network updates. Across 4B–120B dense and mixture-of-experts backbones, including both text-only and multimodal models, Cortex lies on the empirical consolidation–retention Pareto frontier in most evaluated settings. After 50 sequential factual edits on a 14B model, Cortex-Structural recovers 88.9% of the context-induced gain while retaining 99.8% of original non-target performance.

[LoRA](lora) learns low-rank updates of existing projections. Cortex instead adds nonlinear FFN capacity. Both methods support merging, but while merging LoRA keeps the shapes of the weights, merging Cortex makes the FFNs wider, so the merged model keeps the added parameters and computation.

The width `k` is the number of channels added to each FFN. For an FFN with hidden size `d`, an adapter adds `3 * d * k` parameters, or `2 * d * k` if the FFN has no gate, plus biases if the FFN has biases (see [Biases](#biases)). For example, Llama-3.2-3B has 28 FFNs with `d=3072`. With `k=16`, Cortex adds 147,456 parameters per FFN, or 4,128,768 parameters in total, which is about 0.129% of the parameters of the base model. Since `k` and the LoRA rank `r` result in different numbers of parameters, compare the two methods at the same number of trainable parameters.

## Quickstart

```python
from transformers import AutoModelForCausalLM
from peft import CortexConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
config = CortexConfig(k=16, init_weights="sample", task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

The adapter can be trained with any loss, e.g. the usual cross-entropy loss or a distillation loss. See the [fine-tuning example](https://github.com/huggingface/peft/tree/main/examples/cortex) for training with the Transformers `Trainer`.

## Supported FFNs

Cortex adapts whole FFNs, so `target_modules` should select FFN modules like `mlp` or `experts`, not their projections. Two kinds of FFNs are supported:

- **Dense FFNs** computing `down_proj(act_fn(gate_proj(x)) * up_proj(x))`, where the projections are `nn.Linear` layers. This is the MLP of Llama, Mistral, Qwen, Gemma, and many other Transformers models, and also the shared experts of most MoE models. Without `gate_proj`, the FFN must compute `down_proj(act_fn(up_proj(x)))`. The gate and up projections may also be fused into a single `gate_up_proj`, whose output consists of the gate half followed by the up half. Variants of the clamped SwiGLU of GPT-OSS, like those of DeepSeek-V4, GLM5 Next, and MiniMax-M3-VL, are supported as well. Custom FFNs following these conventions work, too.
- **Packed MoE experts** that follow the Transformers experts interface (classes decorated with `use_experts_implementation`), e.g. the `experts` modules of Qwen3-MoE, Mixtral, GPT-OSS, DeepSeek, GLM, MiniMax, and Kimi models. The added channels use the gating of the experts, e.g. the clamped SwiGLU of GPT-OSS. This requires Transformers v5.

Cortex relies on these formulas when merging, so it raises an error for FFNs that compute something else: FFNs with submodules other than the projections and `act_fn`, e.g. a normalization layer between the projections, FFNs with parameters or buffers of their own, e.g. an output scale, and FFNs with an `activation_sparsity`, like some layers of Gemma3n, whose activations depend on statistics over all channels.

If `target_modules` is not set, the following modules are targeted with the default `moe_target="shared"`, see [MoE models](#moe-models) for the other values:

- the FFNs of Llama, Mistral, Qwen2, Qwen3, Qwen3.5, and Gemma3 text models, and the dense FFNs of the layers of Gemma4 text models;
- the experts of MoE models without shared experts: Qwen3-MoE, Mixtral, OLMoE, GPT-OSS, MiniMax, and MiniMax-M2;
- the shared experts of MoE models with shared experts: DeepSeek-V2, V3, V3.2, and V4, GLM4-MoE, GLM4-MoE-Lite, GLM-MoE-DSA, GLM5 Next, Kimi Linear, MiniMax-M3-VL, NemotronH, Qwen3-Next, and Qwen3.5-MoE.

For other models, set `target_modules` explicitly. In models whose first layers are dense, like the DeepSeek models, the dense FFNs and the MoE blocks are both called `mlp`, so `target_modules=["mlp"]` would select the MoE blocks as well. To also adapt the dense FFNs, select them by their layer index, e.g. with `target_modules=r".*\.layers\.[0-2]\.mlp|.*\.shared_experts"` for the first three layers of DeepSeek-V3.

## MoE models

For MoE models, `moe_target` determines where channels are added:

| `moe_target` | Added channels |
| --- | --- |
| `"shared"` (default) | `k` channels in the native shared experts, like the `shared_experts` of DeepSeek-V3. In MoE blocks without native shared experts, `k` channels common to all experts instead, which are applied to every token and weighted by the sum of its routing weights |
| `"routed"` | `k_routed` channels per expert, applied when the router selects that expert and weighted by its routing weight, just like the native channels of the expert |
| `"both"` | both |

With `target_modules`, only the targeted modules are adapted, and `moe_target` determines which channels they get. Cortex raises an error if a targeted module would get no channels, e.g. packed experts next to native shared experts with `moe_target="shared"`, and warns if `moe_target="both"` is only applied to the packed experts or only to the native shared experts of a MoE block.

Native shared experts are recognized as a dense FFN, or a module whose name contains `shared`, next to the packed experts or next to the module that holds them. So the dense FFN of each layer of Gemma4 counts as shared experts as well. Other dense FFNs are extended regardless of `moe_target`.

Unlike the output of native shared experts, the output of the experts is weighted by the routing weights. Since the channels common to all experts are added to every expert when merging, their output is multiplied by the sum of the selected routing weights of each token, so that merging is exact for every router. For routers whose routing weights sum to one, like those of Mixtral, GPT-OSS, or Qwen3-MoE with `norm_topk_prob=True`, this is the same as adding the output as is.

`k_routed` defaults to `k`. Since routed channels are added to every expert, their number of parameters is multiplied by the number of experts. Before merging, they are computed with a loop over the experts, like the eager experts implementation, which is slower than the other channels, especially for models with many experts.

## Initialization

With `init_weights="random"` (default), the gate and up weights use the default initialization of `nn.Linear`. With `init_weights="sample"`, they are copies of randomly sampled channels of the base FFN, with Gaussian noise with standard deviation `init_noise_std` added. Routed channels are sampled from their own expert, and channels common to all experts from all experts. In both cases, the down weights and biases are initialized to zero, so the adapter initially does not change the output of the model.

## Biases

With `cortex_bias="auto"` (default), each projection of the added channels has a bias if the corresponding projection of the base FFN has one, e.g. for GPT-OSS, so that merging does not change the architecture of the model. With `cortex_bias="none"`, they have no biases. Biases can't be added to FFNs without biases, since they couldn't be merged.

The gate and up biases belong to the added channels, and with `init_weights="sample"`, they are copied from the sampled channels. The down bias, on the other hand, is added to the output of the FFN like the down bias of the base FFN, so it is shared by all added channels. Merging adds it to the down bias of the base FFN, and unmerging subtracts it again, which restores the base FFN up to floating point errors.

## Merging and saving

`merge_adapter()` appends the adapter channels to the FFNs, and `unmerge_adapter()` removes them again. `merge_and_unload()` returns the base model with the wider FFNs and updates their widths in the model config, e.g. `intermediate_size`, so that the merged model can be saved and loaded like any other Transformers model. Since it removes the adapter layers from the PEFT model, save the adapter first:

```python
model.save_pretrained("cortex-adapter")

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged-model")

merged_model = AutoModelForCausalLM.from_pretrained("merged-model")
```

Following the naming convention of Transformers, Cortex updates the integer config fields whose names contain `intermediate`, including those of nested configs. If a field determines the widths of several FFNs, it is increased to fit the widest merged FFN, and the other FFNs are padded with zero channels, which don't change their output. This increases the size of the merged model: for example, `moe_intermediate_size` determines the widths of both the routed and the shared experts of DeepSeek-V3, so extending only the shared experts, as with the default `moe_target="shared"`, pads all routed experts, which adds about 5.1B zero parameters with `k=16`, or about 0.8% of the model. To use a merged model in memory without saving it, `merge_adapter()` suffices, which does not update the config and so does not pad any FFN.

If Cortex cannot infer or verify a config for the merged model, `merge_and_unload()` warns with the reason and leaves the config unchanged, so the merged model cannot be loaded with `from_pretrained`. This happens e.g. if the width of an FFN is not determined by a single integer config field whose name contains `intermediate`, or if such a field also determines the shapes of other parameters. Then, load the saved adapter and merge it whenever a merged model is needed:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
merged_model = PeftModel.from_pretrained(base_model, "cortex-adapter").merge_and_unload()
```

Merged MoE models have wider experts. If the default `grouped_mm` experts implementation fails with these widths, e.g. because the kernel requires aligned widths, load the merged model with `experts_implementation="eager"`.

## Limitations

- Quantized base weights and tensor or expert parallelism are not supported.
- For MoE models, the MoE blocks must call their `experts` module. Kernels that replace the forward of the whole MoE block, like the hub kernel used by GPT-OSS with `use_kernels=True`, bypass the adapter.
- Merging requires the full weights in memory, e.g. it does not work with DeepSpeed ZeRO-3 or with weights that are offloaded to the CPU or disk with Accelerate.
- `init_weights="sample"` copies weights of the base model, so they must be loaded, i.e. not offloaded to the CPU or disk with Accelerate.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=CORTEX"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

## Citation

If you use Cortex in your research, please cite [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399).

```bibtex
@misc{zhang2026cortex,
  title = {Cortex: Consolidating Context into Parametric Memory},
  author = {Jintian Zhang and Feng He and Xinnian Liang and Ziwen Xu and Haoming Xu and Haojing Huang and Shuangzhi Wu and Ningyu Zhang},
  year = {2026},
  eprint = {2610.20399},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2610.20399}
}
```

# API

## CortexConfig

[[autodoc]] tuners.cortex.config.CortexConfig

## CortexModel

[[autodoc]] tuners.cortex.model.CortexModel
