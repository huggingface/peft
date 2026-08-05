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

# Super-Tuning

[Super-Tuning](https://huggingface.co/papers/2607.09287) is a sparse fine-tuning method that freezes the base weight and trains only a sparse support of scalar entries selected by weight magnitude — a distinct point in the trainable-parameter Pareto vs LoRA / IA³. Setting `r` additionally allocates a LoRA-style low-rank adapter composed additively on top of the sparse support (the paper's "Supra" hybrid).

Default scoring is magnitude-only and data-free. The paper's 8B ablation reports `magnitude-topk` at 79.02% average outperforming Wanda-weighted saliency at 78.66% while requiring no calibration pass. Users wanting a different scoring rule (Wanda, activation-aware, task-conditioned, hand-crafted) can compute indices externally and pass them via [`SupertuningModel.set_precomputed_indices`], which accepts a `{module_name: LongTensor}` mapping.

Super-Tuning currently has the following constraint:

- Only `nn.Linear` layers are supported.

The abstract from the paper is:

> Fine-tuning large language models with parameter-efficient methods has become standard practice, but existing approaches like LoRA restrict the trainable subspace to a low-rank decomposition. We introduce Super-Tuning, a sparse fine-tuning approach that instead selects a small support of individual scalar weight entries — an unrestricted-rank trainable set at a fixed parameter budget. Selection is guided by pruning-inspired saliency signals: magnitude-only scoring (PaFi-style) or activation-weighted scoring (Wanda-style). We show that on Llama-3.2-1B and Meta-Llama-3-8B fine-tunes evaluated on Math17K, magnitude-based Super-Tuning matches or exceeds LoRA at comparable parameter budgets, and that a hybrid variant "Supra" — combining sparse support with a low-rank component — further improves downstream accuracy.

## Usage

**Pure Super (magnitude scoring, data-free):**

```python
from peft import SupertuningConfig, get_peft_model

config = SupertuningConfig(target_modules=["q_proj", "v_proj"], sparsity=0.9)
model = get_peft_model(base_model, config)
```

**Supra hybrid** (sparse support + LoRA composed additively):

```python
config = SupertuningConfig(
    target_modules=["q_proj", "v_proj"], sparsity=0.9,
    r=8, lora_alpha=16,   # lora_alpha defaults to 2 * r when omitted
)
model = get_peft_model(base_model, config)
```

**Custom scoring** (e.g. Wanda computed externally):

```python
config = SupertuningConfig(target_modules=["q_proj", "v_proj"], sparsity=0.9)
model = get_peft_model(base_model, config)

# Compute indices in any framework, pass a {module_name: LongTensor} dict keyed by
# module names relative to the inner base model (model.base_model.model):
model.base_model.set_precomputed_indices({
    "model.layers.0.self_attn.q_proj": custom_indices_0,
    "model.layers.0.self_attn.v_proj": custom_indices_1,
    # ...
})
```

## SupertuningConfig

[[autodoc]] tuners.supertuning.config.SupertuningConfig

## SupertuningModel

[[autodoc]] tuners.supertuning.model.SupertuningModel
