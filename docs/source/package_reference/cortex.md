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

On multimodal models, automatic discovery targets the text backbone. To adapt a registered FFN in a speech decoder, include its path in `target_modules`.

Packed MoE layouts are validated with **Transformers 5.17.0**. Compatible dense FFNs can also run on older Transformers releases. Older `ModuleList`-based expert implementations are rejected. Packed expert parameters must be floating-point tensors.

The supported MoE blocks call their experts with flattened hidden states, integer expert indices, and floating-point routing weights, in that order. The first argument must be named `hidden_states`, and all three must accept positional arguments. The indices and weights both have shape `[num_tokens, top_k]`. A `CortexMoeSpec` can rename the indices and weights arguments while retaining these requirements.

For other native Transformers models, Cortex can infer a standard dense FFN or a packed expert layout. For example, Mistral's MLP uses the same gated FFN formula as Llama. Cortex checks the projection layout and activation; for an unfamiliar dense MLP, it also checks the operations in its forward. Inference requires inspectable, unmodified FFN forwards and may reject Accelerate device-placement hooks even when offloading is disabled. For MoE, Cortex first identifies the parent block so that a request for routed experts cannot select its shared FFN instead. An incompatible target raises an error before adapters are installed.

An inferred MoE can have shared experts outside the router's immediate parent. If Cortex cannot locate the shared FFN or establish that the architecture has none, it rejects automatic `shared` selection. You can still select a supported routed expert. Saved adapter configurations retain the roles of their physical modules.

`get_peft_model()` automatically performs structure and interface checks, but does not run numerical equivalence checks. When integrating a new model, run `validate_layout.py` separately to compare outputs with nonzero adapter parameters before merge, after merge, and after native reload.

When Cortex infers a layout, the warning lists representative module paths and links to this section. The [layout check](https://github.com/huggingface/peft/tree/main/examples/cortex#check-a-transformers-layout) explores dense FFNs or the shared, routed, and combined MoE selections at different widths. The script is not installed by pip; obtain it from a PEFT source checkout matching your installed release or commit, and run it from the repository root.

Use `--cortex-config` to validate a new configuration with custom specifications before injecting or training an adapter. To check **one saved adapter configuration**, use `--adapter-config`; this preserves its paths, roles, and widths and rejects probe overrides. The check uses a small random model and validates the selected layout on the reduced configuration, not the training quality of a saved checkpoint.

## Target selection

`target_modules` selects complete FFNs or MoE blocks. Lists match exact module names or dot-delimited suffixes; strings use full regular expression matching. For example, `target_modules=["mlp"]` matches `mlp` and `model.layers.0.mlp`, but not `some_mlp`. Leave it unset for automatic discovery, or use `exclude_modules` to exclude blocks or resolved FFNs. Select whole FFNs rather than their individual linear projections.

For MoE, select the parent block, such as `model.layers.0.mlp`, to control branches with `moe_target`. Explicitly selecting a packed experts leaf, such as `model.layers.0.mlp.experts`, selects routed branches. Dense FFNs ignore `moe_target`.

Use `modules_to_save` to train and save a full module, such as a classification head. Cortex adapts the FFNs outside those modules and existing auxiliary wrappers. Choose saved modules that do not contain Cortex layers.

For a custom FFN using the conventional names below, set `target_modules` explicitly:

- **Computation:** **`down(act(gate(x)) * up(x))`**, or `down(act(up(x)))` without a gate. Use ordinary `nn.Linear` projections named `up_proj`, `down_proj`, and optionally `gate_proj`, with a supported deterministic elementwise activation exposed as `act_fn`, such as SiLU or GELU.
- **Other structures:** FFNs with extra normalization, residuals, stochastic operations, or custom activation implementations need dedicated structure support.

The [custom FFN classification example](https://github.com/huggingface/peft/tree/main/examples/cortex/finetune#custom-ffn-classification) shows a PyTorch training loop with `task_type` unset.

## Custom structure specifications

Use [`CortexFfnSpec`] and [`CortexMoeSpec`] when a model uses supported FFN computations but different field names. The built-in registry and user configuration use these same classes to describe existing FFN and router structures.

### Configure field mappings

`ffn_specs` and `moe_specs` map a class name to its specification. A fully qualified name, such as `my_package.modeling.MyFfn`, takes precedence over the short class name. Each specification includes its own defaults and replaces the class's complete built-in entry.

Providing a specification enables automatic discovery for that class. Use `target_modules` to restrict which instances are adapted.

For an FFN computing `w2(activation(w1(x)) * w3(x))`, with its constructor width stored in `config.ffn_channels`:

```python
from peft import CortexConfig, CortexFfnSpec, CortexMoeSpec

config = CortexConfig(
    k=16,
    ffn_specs={
        "MyFfn": CortexFfnSpec(
            gate_proj="w1",
            up_proj="w3",
            down_proj="w2",
            act_fn="activation",
            width_field="ffn_channels",
        ),
    },
)
```

Names refer to direct attributes of the FFN, not dotted paths. Set `has_gate=False` for `down(act(up(x)))`. Use `gate_up_proj` to name a fused gate/up projection, and `packed=True` for packed expert parameters. Packed experts still need the supported layout metadata.

If the FFN stores its current width under a nonstandard attribute, set `width_attributes`, for example `width_attributes=("channels",)`, so merge and unmerge update it.

For an MoE whose routed experts are stored in `bank` and shared FFN in `common`, add:

```python
config.moe_specs = {
    "MyMoe": CortexMoeSpec(
        experts="bank",
        shared_modules=("common",),
        routing_indices="selected_ids",
        routing_weights="selected_weights",
    ),
}
```

The routing names describe the packed experts' `forward` arguments. The example requires `forward(hidden_states, selected_ids, selected_weights)`: the first argument must be named `hidden_states`, and the first three arguments must accept positional calls. Omit both routing names to use the supported native signatures. Also provide an `ffn_specs` entry for the experts class if its projection names differ.

Set `shared_modules=()` only when the MoE has no native shared FFN. `None` discovers conventional shared-FFN names; if none are found, it leaves their absence unresolved.

### Validate and reuse specifications

Cortex validates the referenced projections, shapes, activation and expert signature before injection. Providing a specification declares that the module follows the corresponding supported computation; checking names and shapes does not prove equivalence for an arbitrary custom `forward`. Extra normalization, residual operations or other unsupported arithmetic still require dedicated implementation support. Public-branch merging follows the routing requirements in [MoE branches](#moe-branches).

Specifications are saved in the adapter configuration and restored on load. The width field must describe the constructor's actual FFN width. Export uses the same width binding for untargeted FFNs of that class, padding them when needed to keep native reload consistent.

Structure specifications describe the base model, independently of an adapter's selected targets. Include descriptions of untargeted custom FFNs that share its export width fields, and reuse the complete `ffn_specs` and `moe_specs` mappings for every adapter on that model. Keep these mappings unchanged after injection so each saved adapter remains self-contained.

Cortex rejects different or missing descriptions across adapters. Short and fully qualified keys are equivalent when they resolve to the same class and specification.

To run the [numerical layout check](#supported-architectures) on a new configuration, put its fields in a JSON file without resolved `target_roles`, then run:

```sh
python examples/cortex/validate_layout.py --model-name path/to/config-directory --cortex-config cortex.json --k 2 4
```

Command-line probe options override the JSON fields. An explicit `k` or `moe_target` in the JSON selects those widths or MoE branches; when omitted, the probe uses widths 2 and 4 and explores the applicable dense or MoE modes, including parent blocks described by `moe_specs`. As in training, JSON `moe_target` has no effect on a dense-only selection.

A complete JSON saved by `CortexConfig.save_pretrained` includes its defaults, so those values also constrain the probe. The [example README](https://github.com/huggingface/peft/tree/main/examples/cortex#check-a-transformers-layout) shows a minimal configuration for exploration. The script constructs a native Transformers model; specifications do not import custom model code or enable `trust_remote_code`.

Mode sweeps apply to selected MoE parent blocks. Explicit shared-FFN targets always exercise shared branches, and explicit expert-module targets always exercise routed branches, regardless of a JSON or CLI `moe_target` value. Selecting both leaves checks their combination once per width and preserves any selected dense FFNs. When parents and fixed leaves from different blocks are selected together, `moe_target` changes only the parent selections.

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

For BF16 training, load the base with `dtype=torch.bfloat16`. PEFT promotes BF16 and FP16 adapter weights to FP32 by default. Set `autocast_adapter_dtype=False` in `get_peft_model` to keep each adapter projection in the dtype of its corresponding base projection. If the base fuses gate and up, both adapter projections inherit the fused weight's dtype.

## MoE branches

| `moe_target` | Native shared FFN present | Architecture known to have no native shared FFN |
| --- | --- | --- |
| `"shared"` (default) | Extend the native shared FFN | Add a public branch evaluated for every token |
| `"routed"` | Extend each routed expert | Extend each routed expert |
| `["shared", "routed"]` | Extend both | Add public and routed branches |

`k_routed` sets the added width **per routed expert**. `k_shared` sets the width per physical shared FFN or public branch. Both default to `k`; dense FFNs always use `k`. Native shared branches retain their original gates and normalization. Routed branches reuse the native router's expert selections and weights.

For a registered layout known to lack native shared experts, `moe_target="shared"` creates the shared-only extension. Its default `public_weighting="unweighted"` adds the public output independently of routing weights. Set `public_weighting="routing"` to multiply that output by the sum of the selected router weights. This enables public-branch merging even when the selected weights are not normalized to sum to one.

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

For each `modules_to_save` module, export retains the copy belonging to the selected adapter, even if another adapter is currently active. If none of the selected adapters has a copy, export retains the original module. Selecting multiple copies of the same module raises an error before merging; different saved modules can belong to different selected adapters.

Keep adapter checkpoints if you need task switching. Unmerge before adding another adapter to an existing merged wrapper. For sequential training on an exported model, inject a fresh adapter into the returned native model. Small floating-point differences between merged and unmerged outputs are expected.

Native export updates the configuration widths needed for reloading the model. For inferred layouts, Cortex uses conventional width fields such as `intermediate_size`, `moe_intermediate_size`, and `shared_expert_intermediate_size`. If the field cannot account for the original FFN width, keep the adapter checkpoint or add a dedicated export rule. A custom model must handle any additional modules that read the same width: for example, a classifier sized from `intermediate_size` also needs to match the updated config on reload. Exporting a public branch copies its added channels into every routed expert, increasing storage for those channels by the expert count.

## Limitations

- **Weights and layouts:** use floating-point weights. Quantized projections, tensor/expert-parallel shards, unsupported FFN computations, and expert-internal output normalization (`has_post_expert_norm=True`) are unsupported.
- **MoE kernels:** keep the native MoE parent forward; kernels limited to attention or normalization can still be used. Reload exported MoE models with `experts_implementation="eager"` to accommodate expanded widths. Cortex does not include a fused routed-adapter kernel.
- **Offload:** CPU placement and `low_cpu_mem_usage=True` adapter loading are supported. Active Accelerate CPU/disk offload hooks are unsupported.
- **Merging:** merge with full weights available on an unpartitioned model, outside the distributed training engine. Adapter biases require corresponding biases in the native FFN.
- **Failure recovery:** merge, unmerge, and export update layers sequentially and are not atomic. `safe_merge=True` checks for non-finite values. If an operation fails partway through, reload the base model and saved adapter checkpoint.
- **Distributed and compiled execution:** DDP and DeepSpeed ZeRO training have been tested with dense and MoE models, including GPT-OSS-20B. Multi-rank FSDP and compiled execution have not been validated.

## CortexConfig

[[autodoc]] tuners.cortex.config.CortexConfig

## CortexFfnSpec

[[autodoc]] tuners.cortex.config.CortexFfnSpec

## CortexMoeSpec

[[autodoc]] tuners.cortex.config.CortexMoeSpec

## CortexModel

[[autodoc]] tuners.cortex.model.CortexModel
    - merge_adapter
    - unmerge_adapter
    - merge_and_unload
    - unload
