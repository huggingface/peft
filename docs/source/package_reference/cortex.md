<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Cortex

Cortex adds trainable nonlinear channels to the frozen feed-forward network (FFN) blocks of a pretrained model. This is the Cortex-Structural parameterization from [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399).

[LoRA](lora) learns low-rank updates to existing projections, including attention and FFN projections. Both methods support merging: standard LoRA keeps the original weight shapes and parameter count, while Cortex widens the native FFN and retains the added parameters and computation.

Choose Cortex when you want to add nonlinear FFN capacity; choose LoRA when preserving the merged model's original size or adapting attention projections is a priority. Cortex width `k` and LoRA rank `r` imply different parameter budgets, so compare methods using matched trainable parameter counts and the same training and evaluation settings.

The width `k` controls the added FFN capacity. For each standard bias-free gated FFN with input/output dimension `d`, an adapter adds `3*d*k` parameters; a non-gated FFN adds `2*d*k`.

For example, Llama-3.2-3B has 28 bias-free gated FFNs with `d=3072`. With `k=16`, Cortex adds 147,456 parameters per FFN. Adapting all 28 adds 4,128,768 parameters (about 4.1M), matching the configuration in the MetaMathQA benchmark below. This is about **0.129% of the original model's 3,212,749,824 parameters**, calculated as `adapter parameters / original model parameters * 100`.

Train the adapter with a supervised task loss or teacher distillation. The [consolidation example](https://github.com/huggingface/peft/tree/main/examples/cortex/consolidate) demonstrates distilling documentation into an adapter; the [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA/experiments/cortex) provides supervised training configurations and benchmark commands.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM

from peft import CortexConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
config = CortexConfig(k=16, init_weights="sample", task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

The [fine-tuning example](https://github.com/huggingface/peft/tree/main/examples/cortex/finetune) trains from a local JSONL dataset and saves the adapter and tokenizer. Its README shows how to reload them.

## Supported architectures

Automatic discovery recognizes native Llama, Qwen2, Qwen3, Qwen3.5, Gemma3, and Gemma4 dense FFNs, together with the dense/shared FFNs registered for the MoE families below. The following table lists the registered native MoE layouts by Transformers model type.

| Family | Registered MoE model types |
| --- | --- |
| GPT-OSS | `gpt_oss` |
| Qwen | `qwen3_moe`, `qwen3_next`, `qwen3_5_moe` |
| Mixtral | `mixtral` |
| OLMoE | `olmoe` |
| DeepSeek | `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `deepseek_v4` |
| GLM | `glm4_moe`, `glm4_moe_lite`, `glm_moe_dsa`, `glm5_next_text` |
| MiniMax | `minimax`, `minimax_m2`, `minimax_m3_vl_text` |
| Kimi | `kimi_linear` |
| Nemotron | `nemotron_h` |
| Gemma | `gemma4_text` with MoE enabled |

For multimodal models, Cortex adapts the text FFN and MoE blocks listed here.

Packed MoE layouts are validated with **Transformers 5.17.0**. Compatible dense FFNs can also run on older Transformers releases. Older `ModuleList`-based expert implementations are rejected. Packed expert parameters must be floating-point tensors.

## Target selection

`target_modules` selects complete FFNs or MoE blocks. Lists match exact module names or dot-delimited suffixes; strings use full regular expression matching. For example, `target_modules=["mlp"]` matches `mlp` and `model.layers.0.mlp`, but not `some_mlp`. Leave it unset for automatic discovery, or use `exclude_modules` to exclude blocks or resolved FFNs. Select whole FFNs rather than their individual linear projections.

For MoE, select the parent block, such as `model.layers.0.mlp`, to control branches with `moe_target`. Explicitly selecting a packed experts leaf, such as `model.layers.0.mlp.experts`, selects routed branches. Dense FFNs ignore `moe_target`.

Use `modules_to_save` to train and save a full module, such as a classification head. Cortex adapts the FFNs outside those modules and existing auxiliary wrappers. Choose saved modules that do not contain Cortex layers.

For a custom FFN, set `target_modules` explicitly and use the following structure:

- **Computation:** **`down(act(gate(x)) * up(x))`**, or `down(act(up(x)))` without a gate. Use ordinary `nn.Linear` projections named `up_proj`, `down_proj`, and optionally `gate_proj`, with a supported deterministic elementwise activation exposed as `act_fn`, such as SiLU or GELU.
- **Other structures:** FFNs with extra normalization, residuals, stochastic operations, or custom activation implementations need dedicated structure support.

The [custom FFN classification example](https://github.com/huggingface/peft/tree/main/examples/cortex/finetune#custom-ffn-classification) shows a PyTorch training loop with `task_type` unset.

## Initialization and biases

| Option | Behavior |
| --- | --- |
| `init_weights="random"` (default) | Initialize gate/up weights with Kaiming-uniform initialization. |
| `init_weights="sample"` | Copy paired gate/up channels and available biases from the base FFN. Routed branches sample their own expert; public branches sample from the expert pool. |
| `init_noise_std=0.01` | Add Gaussian noise to sampled weights. Use `0` for exact copying. |
| `adapter_bias=None` (default) | Follow bias presence in each corresponding base projection. Use `True` to create all adapter biases, or `False` to create none. |
| `adapter_bias_init="uniform"` (default) | Initialize gate/up biases uniformly during random initialization. Use `"zeros"` for zero initialization. Sample initialization copies available native biases and zeros missing ones. |

Both initialization strategies zero the down weights and bias, so the adapter starts with the base model's output. Loading a trained adapter restores its saved weights. See [`CortexConfig`] for the boolean aliases and testing options.

Initialize new adapters on a model with weights loaded in memory (not meta tensors). Use `low_cpu_mem_usage=True` when loading trained adapters, rather than when initializing a new one.

For BF16 training, load the base with `dtype=torch.bfloat16`. PEFT promotes adapter weights to FP32 by default; set `autocast_adapter_dtype=False` in `get_peft_model` to keep them in BF16.

## MoE branches

| `moe_target` | Native shared FFN present | No native shared FFN |
| --- | --- | --- |
| `"shared"` (default) | Extend the native shared FFN | Add a public branch evaluated for every token |
| `"routed"` | Extend each routed expert | Extend each routed expert |
| `["shared", "routed"]` | Extend both | Add public and routed branches |

`k_routed` sets the added width **per routed expert**. `k_shared` sets the width per physical shared FFN or public branch. Both default to `k`; dense FFNs always use `k`. Native shared branches retain their original gates and normalization. Routed branches reuse the native router's expert selections and weights.

For a model without native shared experts, `moe_target="shared"` creates the shared-only extension. Its default `public_weighting="unweighted"` adds the public output independently of routing weights. Set `public_weighting="routing"` to multiply that output by the sum of the selected router weights. This enables public-branch merging even when the selected weights are not normalized to sum to one.

```python
# Replace this path with a supported native MoE checkpoint.
# Use eager experts for this example; keep the native MoE parent forward.
base_moe_model = AutoModelForCausalLM.from_pretrained(
    "path/to/moe-base", experts_implementation="eager"
)
config = CortexConfig(
    k=16,
    moe_target="shared",
    public_weighting="routing",
    init_weights="sample",
    task_type="CAUSAL_LM",
)
moe_model = get_peft_model(base_moe_model, config)
```

`public_weighting` is part of the public branch's trained behavior, so choose it before training. To merge an unweighted public branch, the selected router weights must have a constant, nonzero sum. Use `public_weighting="routing"` for merging public branches with unnormalized Qwen3-MoE or OLMoE routing. Routed-only branches can merge normally.

For one MoE block with gated FFNs of input/output dimension `d`, routed branches add `3*d*E*k_routed` matrix parameters across `E` experts; a public branch adds `3*d*k_shared`. Account for both total parameters and active branches when comparing budgets.

## Task adapters

Create named adapters with `get_peft_model(..., adapter_name="task1")` and `add_adapter("task2", config)`. `model.set_adapter("task2")` selects one task. Adapters on the same physical FFN must use the same branch role, while widths and initialization may differ. Native shared and routed FFNs are separate targets; public and routed branches in a model without native shared FFNs share a physical target.

For cumulative training, activate both adapters through the tuner, then freeze the earlier one:

```python
model.base_model.set_adapter(["task1", "task2"])
model.set_requires_grad("task1", requires_grad=False)
model.set_requires_grad("task2", requires_grad=True)
model.zero_grad(set_to_none=True)
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=1e-3,
)
```

- After adding or switching adapters, reapply the freeze settings and rebuild the optimizer for the trainable parameters.
- Save all required adapters with `selected_adapters`, including earlier adapters whose `modules_to_save` copies are part of the model.
- To resume training, save the optimizer state and activation policy alongside the adapter checkpoints.
- Freezing an earlier adapter keeps its parameters fixed; an active new adapter still contributes to the model's outputs.

## Saving and structural merging

Use the standard PEFT checkpoint APIs and reload the same base model:

```python
from peft import PeftModel

model.save_pretrained("structural-adapter")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
restored = PeftModel.from_pretrained(base_model, "structural-adapter").eval()

merged = restored.merge_and_unload(safe_merge=True)
merged.save_pretrained("structural-native")
native = AutoModelForCausalLM.from_pretrained("structural-native")
```

Saved configurations retain resolved target paths and roles. Create a fresh `CortexConfig` for a different model, and retain the saved configuration when loading an adapter; `key_mapping` alone does not remap its target roles.

`merge_adapter` expands the native FFN with the selected adapter channels; `unmerge_adapter` restores the original FFN. `merge_and_unload` returns the expanded native model and removes all adapter wrappers, including unselected adapters. Already-merged channels remain in the exported model. To export only a selected adapter after earlier merges:

```python
model.unmerge_adapter()
exported = model.merge_and_unload(adapter_names=["task2"], safe_merge=True)
```

Keep adapter checkpoints if you need task switching. Unmerge before adding another adapter to an existing merged wrapper. For sequential training on an exported model, inject a fresh adapter into the returned native model. Small floating-point differences between merged and unmerged outputs are expected.

Native export updates the configuration widths needed for reloading registered models. Custom model constructors must handle their expanded dimensions. Exporting a public branch copies its added channels into every routed expert, increasing storage for those channels by the expert count.

## Limitations

- **Weights and layouts:** use floating-point weights. Quantized projections, tensor/expert-parallel shards, unknown remote model implementations, and expert-internal output normalization (`has_post_expert_norm=True`) are unsupported.
- **MoE kernels:** keep the native MoE parent forward; kernels limited to attention or normalization can still be used. Reload exported MoE models with `experts_implementation="eager"` to accommodate expanded widths. Cortex does not include a fused routed-adapter kernel.
- **Offload:** CPU placement and `low_cpu_mem_usage=True` adapter loading are supported. Active Accelerate CPU/disk offload hooks are unsupported.
- **Merging:** merge with full weights available on an unpartitioned model, outside the distributed training engine. Adapter biases require corresponding biases in the native FFN.
- **Failure recovery:** merge, unmerge, and export update layers sequentially and are not atomic. `safe_merge=True` checks for non-finite values. If an operation fails partway through, reload the base model and saved adapter checkpoint.
- **Distributed and compiled execution:** DDP and DeepSpeed ZeRO training have been tested with dense and MoE models, including GPT-OSS-20B. Multi-rank FSDP and compiled execution have not been validated.

## CortexConfig

[[autodoc]] tuners.cortex.config.CortexConfig

## CortexModel

[[autodoc]] tuners.cortex.model.CortexModel
    - merge_adapter
    - unmerge_adapter
    - merge_and_unload
    - unload
