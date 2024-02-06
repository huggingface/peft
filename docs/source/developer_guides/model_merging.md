<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Model merging

Training a model for each task can be costly, take up storage space, and the models aren't able to learn new information to improve their performance. Multitask learning can overcome some of these limitations by training a model to learn several tasks, but this is expensive to train and designing a dataset for it can be challenging. *Model merging* offers a solution to these challenges by combining multiple pretrained models into one model, giving it the combined abilities of each individual model, without any additional training.

PEFT provides two methods for model merging:

* [TIES](https://hf.co/papers/2306.01708) - TrIm, Elect, and Merge (TIES) is a three-step method for merging models. First, redundant parameters are trimmed, then conflicting signs are resolved into an aggregated vector, and finally the parameters whose signs are the same as the aggregate sign are averaged. This method takes into account that some values (redundant and sign disagreement) can degrade performance in the merged model.
* [DARE](https://hf.co/papers/2311.03099) - Drop And REscale is a method that can be used to prepare for model merging methods like TIES. It works by randomly dropping parameters according to a drop rate and rescaling the remaining parameters. This helps to reduce the number of redundant and potentially interfering parameters among multiple models.

Models are merged with the [`~LoraModel.add_weighted_adapter`] method, and the specific model merging method is specified in the `combination_type` parameter. This guide will show you how to merge models with TIES, DARE, and a combination of both TIES and DARE.

## TIES

The [`~utils.ties`] method uses a [`~utils.magnitude_based_pruning`] approach to trim redundant parameters such that only the top-k percent of values are kept from each task vector. The number of values to keep are specified by the `density` parameter. The task tensors are weighted, and the [`~utils.calculate_majority_sign_mask`] *elects* the sign vector. This means calculating the total magnitude for each parameter across all models. Lastly, the [`~utils.disjoint_merge`] function calculates the mean of the parameter values whose sign is the same as the *elected sign vector*.

With PEFT, TIES merging is enabled by setting `combination_type="ties"` and setting `ties_density` to a value of the weights to keep from the individual models. For example, let's merge three finetuned [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) models: [tinyllama_lora_nobots](https://huggingface.co/smangrul/tinyllama_lora_norobots), [tinyllama_lora_sql](https://huggingface.co/smangrul/tinyllama_lora_sql), and [tinyllama_lora_adcopy](https://huggingface.co/smangrul/tinyllama_lora_adcopy).

Load the base model and use the [`~transformers.PreTrainedModel.resize_token_embeddings`] method to account for special tokens added to the embedding layer for each finetuned model. This method ensures the special tokens and initialization of the embedding layers are consistent.

Then you can use the [`~PeftModel.load_adapter`] method to load and assign each adapter a name:

```py
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

config = PeftConfig.from_pretrained("smangrul/tinyllama_lora_norobots")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("smangrul/tinyllama_lora_norobots")

model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, "smangrul/tinyllama_lora_norobots", adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")
```

Set the adapters, weights, `adapter_name`, `combination_type`, and `ties_density` with the [`~LoraModel.add_weighted_adapter`] method.

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", ties_density=density)
```

Set the newly merged model as the active model with the [`~LoraModel.set_adapter`] method.

```py
model.eval()
model.set_adapter("merge")
```

Now you can use the merged model as an instruction-tuned model to write ad copy or SQL queries!

<hfoptions id="ties">
<hfoption id="instruct">

```py
messages = [
    {"role": "user", "content": "Write an essay about Generative AI."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
<hfoption id="ad copy">

```py
messages = [
    {"role": "system", "content": "Create a text ad given the following product and description."},
    {"role": "user", "content": "Product: Sony PS5 PlayStation Console\nDescription: The PS5 console unleashes new gaming possibilities that you never anticipated."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
<hfoption id="SQL">

```py
text = """Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""

inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer("</s>").input_ids[-1])
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## DARE

The DARE method uses the [`~utils.random_pruning`] approach to randomly drop parameters, and only preserve a percentage of the parameters set in the `density` parameter. The remaining tensors are rescaled to keep the expected output unchanged.

With PEFT, DARE is enabled by setting `combination_type="dare_ties"` and setting `density` to a value of the weights to keep from the individual models.

> [!TIP]
> DARE is a super useful method for preparing models for merging which means it can be combined with other methods like TIES, `linear` (a weighted average of the task tensors) or `svd` (calculated from the *delta weights*, the model parameters before and after finetuning) or a combination of all of the above like `dare_ties_svd`.

Let's merge three diffusion models to generate a variety of images in different styles using only one model. The models you'll use are (feel free to choose your own): [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl) and [ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora).

Load the base model and then use the [`~diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] method to load and assign each adapter a name:

```py
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

pipeline.load_lora_weights(
    "nerijs/pixel-art-xl", 
    weight_name="pixel-art-xl.safetensors", 
    adapter_name="pixel"
)
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora", 
    weight_name="super-cereal-sdxl-lora.safetensors", 
    adapter_name="cereal"
)
```

Use [`~LoraModel.add_weighted_adapter`] on the pipeline's UNet to set the weights for each adapter, the `adapter_name`, the `combination_type`, and `density`.

```py
adapters = ["pixel", "cereal"]
weights = [1.0, 1.0]
adapter_name = "merge"
density = 0.5
pipeline.unet.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
```

Make the newly merged model the active model with the [`~LoraModel.set_adapter`] method.

```py
pipeline.unet.set_adapter("merge")
```

Now you can use the merged model to generate images in two different styles!

```py
prompt = "soft fluffy pancakes shaped like kawaii bear faces"
generator = [torch.Generator(device="cuda").manual_seed(0)]
image = pipeline(prompt, generator=generator).images[0]
image
```
