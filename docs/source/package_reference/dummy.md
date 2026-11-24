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

# Dummy

The Dummy method is not a research method and is not intended to be used for fine-tuning. It is a minimal, deliberately trivial PEFT method that serves two purposes:

1. It is a template for contributors who want to add a new PEFT method to PEFT. Reading its implementation (`src/peft/tuners/dummy/`) alongside the [contributing guide](https://huggingface.co/docs/peft/main/en/developer_guides/contributing) shows all the pieces a new tuner needs: a config, tuner layers, the tuner model, and the registration via `register_peft_method`.
2. It is a test vehicle for PEFT features that every tuner should support, such as saving and loading, merging, quantization, and mixed adapter types.

## How Dummy works

The implementation is essentially a plain LoRA without any of the optimizations or extras that LoRA offers: each adapted layer gets a down-projection `A` and an up-projection `B`, and the effective weight of the layer becomes `W + B @ A * (lora_alpha / r)`. The effective weight is therefore easy to predict and reason about.

By default (`init_weights=True`), `B` is initialized with zeros, so the adapter starts out as an exact identity transform: the adapted layer behaves exactly like the base layer until the adapter has been trained. This is the convention that most PEFT methods follow. When the config sets `init_weights=False`, both `A` and `B` are randomly initialized, so the adapter has a non-zero effect right away.

The method supports `nn.Linear` layers, the `Conv1D` layer used by some Transformers models (like GPT-2), `nn.Conv1d`, `nn.Conv2d`, and `nn.Conv3d` layers, including grouped convolutions, which can be merged as long as `r` is divisible by `groups`, and `nn.Embedding` layers. Linear layers additionally support quantization via the generic PEFT quantization backend.

## Example usage

```python
from transformers import AutoModelForCausalLM
from peft import DummyConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = DummyConfig(target_modules=["q_proj", "v_proj"], r=8, lora_alpha=16)
model = get_peft_model(base_model, config)
```

## Benchmark overview

<iframe
    src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=DUMMY"
	frameborder="0"
	width="850"
	height="1000"
></iframe>


# API

## DummyConfig

[[autodoc]] tuners.dummy.config.DummyConfig

## DummyModel

[[autodoc]] tuners.dummy.model.DummyModel
