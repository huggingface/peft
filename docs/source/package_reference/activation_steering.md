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

# Activation Steering

Activation steering is a method to directly influence the activations of individual neurons of a pretrained model. If you already know which neurons to influence and how, PEFT is a convenient interface to inject the intervention into the model and share your checkpoint with others. If you don't, PEFT can also find the neurons for a behavior by the gradient of a loss and fit their doses, see [Finding neurons](#finding-neurons).

The original inspiration for this came from [this tweet](https://x.com/_lyraaaa_/status/2113833330429083662). Repositories used for reference are [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) and [this reproduction](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering).

The reason why activation steering is a good fit for inclusion in the PEFT package is that PEFT already has all the machinery to handle such interventions:

- Share calibrated checkpoints with others.
- Easy to declaratively target specific layers.
- Adapter can be disabled and re-enabled.
- Multiple adapters can be loaded at the same time.

A neuron is an input channel of a linear (or `Conv1D`) projection, typically an MLP's down projection, whose input channels are the MLP's hidden units. Steering adds a dose to those channels before the projection. The dose is measured in units of the median L2 norm of the stream the addition lands in: by default the output of the layer containing the projection, i.e. the residual stream. Models that normalize the MLP output before adding it to the residual stream, such as Gemma 2/3 and OLMo 2, should use `dose_unit="module_output"`. Merging of the adapter is not supported.

## Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import ActivationSteeringConfig, TaskType, get_peft_model
from peft.tuners.activation_steering import calibrate_activation_steering


model_id = "Qwen/Qwen3.5-4B"
config = ActivationSteeringConfig(
    base_model_name_or_path=model_id,
    task_type=TaskType.CAUSAL_LM,
    steering_pattern={"layers.15.mlp.down_proj": {"indices": [3920], "init_val": 0.5}},
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = get_peft_model(base_model, config).eval()

# calibrating and saving, no actual training is required
calibration_text = (
    "The river runs past the village and through the fields. "
    "At the market, people trade fruit, bread, and stories from their travels. "
    "A programmer writes a function to sort a list of numbers."
)
inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
calibrate_activation_steering(model, {**inputs, "use_cache": False})
model.save_pretrained("path/to/checkpoint")

# sharing the adapter
repo_id = "your-username/qwen3.5-4b-my-intervention"
model.push_to_hub(repo_id)
```

Others can then use this adapter like this:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.utils import logging

from peft import PeftConfig, PeftModel

set_seed(42)
model_id = "Qwen/Qwen3.5-4B"
adapter_id = "your-username/qwen3.5-4b-my-intervention"
config = PeftConfig.from_pretrained(adapter_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id).eval()

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Tell me a short story."}],
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(model.device)
prompt_length = inputs.input_ids.shape[1]

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_end|>"]),
        pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True))
```

For a complete script using Qwen 3.5 4B, check [this example](https://github.com/huggingface/peft/tree/main/examples/activation_steering).

## Finding neurons

Without a `steering_pattern`, every input channel of the targeted projections gets a dose of zero, which leaves the model unchanged. [`~tuners.activation_steering.select_neurons`] then scores all doses at once by the gradient of a loss, keeps the best ones, and removes all others in place, so that the saved adapter stays small:

```python
from peft.tuners.activation_steering import calibrate_activation_steering, select_neurons

config = ActivationSteeringConfig(
    base_model_name_or_path=model_id,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["down_proj"],
    # e.g. leave out the last 2 of the 32 layers
    layers_to_transform=list(range(30)),
)
model = get_peft_model(base_model, config)
calibrate_activation_steering(model, {**inputs, "use_cache": False})


def train_step():
    # examples of the desired behavior, e.g. replies in the target style, with the prompt tokens masked in the labels
    for batch in behavior_batches:
        model(**batch).loss.backward()


select_neurons(model, train_step, num_neurons=8, init_val=0.5)
print(model.peft_config["default"].steering_pattern)
model.save_pretrained("path/to/checkpoint")
```

Each kept neuron starts at a dose of `init_val`, with the sign that lowers the loss. The selection is first-order: each kept neuron helps on its own, but the effect of several neurons together is not the sum of their effects. The kept doses are regular trainable parameters, so they can be refined with any training loop, e.g. [`~transformers.Trainer`], before saving. A loss term that keeps the outputs on unrelated prompts close to those of the base model, computed under `model.disable_adapter()`, can limit side effects. With gradient checkpointing, run the backward pass of each steered forward pass before the next one, since the recomputation uses the attention mask of the most recent model call. Call `select_neurons` before creating the optimizer.

# API

## ActivationSteeringConfig

[[autodoc]] tuners.activation_steering.config.ActivationSteeringConfig

## ActivationSteeringModel

[[autodoc]] tuners.activation_steering.model.ActivationSteeringModel

## Utilities

[[autodoc]] tuners.activation_steering.calibration.calibrate_activation_steering

[[autodoc]] tuners.activation_steering.selection.select_neurons
