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

A neuron is an input channel of a linear (or `Conv1D`) projection, typically an MLP's down projection, whose input channels are the MLP's hidden units. Steering adds a dose to those channels before the projection. By default, the dose is measured in units of the median L2 norm of the projection's output. Doses found with neuron-explorer are in units of the residual stream, i.e. the output of the layer containing the projection; for these, pass `dose_unit="residual_stream"`, as in the example below. This unit is not suited to models that normalize the MLP output before adding it to the residual stream, such as Gemma 2/3 and OLMo 2. Merging of the adapter is not supported.

## Example

The following example steers a single neuron of Qwen3.5-4B, the input channel 3920 of `layers.15.mlp.down_proj`, with a dose of 0.5 in units of the residual stream. Calibration measures this unit on a short text, after which the adapter is saved and shared on the Hub:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import ActivationSteeringConfig, TaskType, get_peft_model
from peft.tuners.activation_steering import calibrate_activation_steering


model_id = "Qwen/Qwen3.5-4B"
config = ActivationSteeringConfig(
    task_type=TaskType.CAUSAL_LM,
    steering_pattern={"layers.15.mlp.down_proj": {"indices": [3920], "init_val": 0.5}},
    dose_unit="residual_stream",
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

from peft import PeftModel

set_seed(42)
model_id = "Qwen/Qwen3.5-4B"
adapter_id = "your-username/qwen3.5-4b-my-intervention"
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

Without a `steering_pattern`, every input channel of the targeted projections gets a dose of zero, which leaves the model unchanged. [`~tuners.activation_steering.select_neurons`] then scores all doses at once by the gradient of a loss, keeps the best ones, and removes all others in place, so that the saved adapter stays small. The following example looks for neurons that make Qwen3.5-0.8B reply in German. The examples of this behavior are the model's own replies when a system prompt asks for it, so that the selected neurons take the place of the system prompt:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import ActivationSteeringConfig, TaskType, get_peft_model
from peft.tuners.activation_steering import calibrate_activation_steering, select_neurons


model_id = "Qwen/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
eos_token_id = tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_end|>"])


def tokenize_chat(messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(base_model.device)


# examples of the desired behavior: the model's own replies when a system prompt asks for it
instruction = {"role": "system", "content": "Always reply in German, whatever language the user writes in."}
prompts = [
    "Tell me about your weekend.",
    "How do I reverse a list in Python?",
    "What is the weather like today?",
    "Can you recommend a book?",
    "What should I cook for dinner?",
    "Explain what a black hole is.",
    "Give me a tip for staying focused.",
    "What is your favorite season and why?",
]
batches = []
for prompt in prompts:
    question = {"role": "user", "content": prompt}
    instructed = tokenize_chat([instruction, question])
    with torch.no_grad():
        output = base_model.generate(
            **instructed,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    reply_ids = output[:, instructed.input_ids.shape[1] :]
    # the same reply without the system prompt; only the reply tokens count in the loss
    prompt_ids = tokenize_chat([question]).input_ids
    batches.append(
        {
            "input_ids": torch.cat([prompt_ids, reply_ids], dim=1),
            "labels": torch.cat([torch.full_like(prompt_ids, -100), reply_ids], dim=1),
        }
    )

num_layers = base_model.config.get_text_config().num_hidden_layers
config = ActivationSteeringConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["down_proj"],
    # leave out the last 2 layers, whose neurons mostly push single tokens
    layers_to_transform=list(range(num_layers - 2)),
)
model = get_peft_model(base_model, config)

calibration_text = (
    "The river runs past the village and through the fields. "
    "At the market, people trade fruit, bread, and stories from their travels. "
    "A programmer writes a function to sort a list of numbers."
)
inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
calibrate_activation_steering(model, {**inputs, "use_cache": False})


def train_step():
    for batch in batches:
        model(**batch, use_cache=False).loss.backward()


# keep 4 neurons, each starting at a dose of 0.5 with the sign that lowers the loss
select_neurons(model, train_step, num_neurons=4, init_val=0.5)
print(model.peft_config["default"].steering_pattern)
model.save_pretrained("path/to/checkpoint")
```

The saved adapter loads like any other PEFT adapter. Generating with and without it shows the effect of the selected neurons:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base_model, "path/to/checkpoint").eval()

inputs = tokenize_chat([{"role": "user", "content": "How do I make a cup of tea?"}])
prompt_length = inputs.input_ids.shape[1]
generation_kwargs = {
    "max_new_tokens": 64,
    "do_sample": False,
    "eos_token_id": eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

with torch.inference_mode():
    steered = model.generate(**inputs, **generation_kwargs)
    with model.disable_adapter():
        unsteered = model.generate(**inputs, **generation_kwargs)
print("steered:", tokenizer.decode(steered[0, prompt_length:], skip_special_tokens=True))
print("unsteered:", tokenizer.decode(unsteered[0, prompt_length:], skip_special_tokens=True))
```

Each kept neuron starts at a dose of `init_val`, with the sign that lowers the loss. The selection is first-order: each kept neuron helps on its own, but the effect of several neurons together is not the sum of their effects. The gradient ranks the neurons, but it shows neither the dose at which the behavior appears nor the side effects, so choosing among the candidates is largely trial and error. Compare generations with and without the adapter, as above, on prompts for the behavior and on unrelated prompts, and try other values of `num_neurons` and `init_val`. The useful doses lie in a narrow range: in this example, a dose of 0.3 turns about 7 in 10 replies into German, and a dose of 0.7 nearly all of them, but makes some replies repeat themselves. Examples in the model's own words, as above, tend to work better than hand-written ones, since the loss also rewards the other ways in which hand-written replies differ from the model's, such as their length and wording, and the selected neurons may change those instead. To try a subset of the selected neurons, create an adapter whose `steering_pattern` contains only their entries of the printed pattern, with the same `dose_unit`, and calibrate it.

The kept doses are regular trainable parameters, so they can also be refined with any training loop, e.g. [`~transformers.Trainer`], before saving. A loss term that keeps the outputs on unrelated prompts close to those of the base model, computed under `model.disable_adapter()`, can limit side effects. With gradient checkpointing, run the backward pass of each steered forward pass before the next one, since the recomputation uses the attention mask of the most recent model call. Call `select_neurons` before creating the optimizer.

# API

## ActivationSteeringConfig

[[autodoc]] tuners.activation_steering.config.ActivationSteeringConfig

## ActivationSteeringModel

[[autodoc]] tuners.activation_steering.model.ActivationSteeringModel

## Utilities

[[autodoc]] tuners.activation_steering.calibration.calibrate_activation_steering

[[autodoc]] tuners.activation_steering.selection.select_neurons
