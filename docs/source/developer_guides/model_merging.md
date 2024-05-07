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

Training a model for each task can be costly, take up storage space, and the models aren't able to learn new information to improve their performance. Multitask learning can overcome some of these limitations by training a model to learn several tasks, but it is expensive to train and designing a dataset for it is challenging. *Model merging* offers a solution to these challenges by combining multiple pretrained models into one model, giving it the combined abilities of each individual model without any additional training.

PEFT provides several methods for merging models like a linear or SVD combination. This guide focuses on two methods that are more efficient for merging LoRA adapters by eliminating redundant parameters:

* [TIES](https://hf.co/papers/2306.01708) - TrIm, Elect, and Merge (TIES) is a three-step method for merging models. First, redundant parameters are trimmed, then conflicting signs are resolved into an aggregated vector, and finally the parameters whose signs are the same as the aggregate sign are averaged. This method takes into account that some values (redundant and sign disagreement) can degrade performance in the merged model.
* [DARE](https://hf.co/papers/2311.03099) - Drop And REscale is a method that can be used to prepare for other model merging methods like TIES. It works by randomly dropping parameters according to a drop rate and rescaling the remaining parameters. This helps to reduce the number of redundant and potentially interfering parameters among multiple models.

Models are merged with the [`~LoraModel.add_weighted_adapter`] method, and the specific model merging method is specified in the `combination_type` parameter.

## Merge method

With TIES and DARE, merging is enabled by setting `combination_type` and `density` to a value of the weights to keep from the individual models. For example, let's merge three finetuned [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) models: [tinyllama_lora_nobots](https://huggingface.co/smangrul/tinyllama_lora_norobots), [tinyllama_lora_sql](https://huggingface.co/smangrul/tinyllama_lora_sql), and [tinyllama_lora_adcopy](https://huggingface.co/smangrul/tinyllama_lora_adcopy).

<Tip warninig={true}>

When you're attempting to merge fully trained models with TIES, you should be aware of any special tokens each model may have added to the embedding layer which are not a part of the original checkpoint's vocabulary. This may cause an issue because each model may have added a special token to the same embedding position. If this is the case, you should use the [`~transformers.PreTrainedModel.resize_token_embeddings`] method to avoid merging the special tokens at the same embedding index.

<br>

This shouldn't be an issue if you're only merging LoRA adapters trained from the same base model.

</Tip>

Load a base model and can use the [`~PeftModel.load_adapter`] method to load and assign each adapter a name:

```py
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

config = PeftConfig.from_pretrained("smangrul/tinyllama_lora_norobots")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("smangrul/tinyllama_lora_norobots")

model = PeftModel.from_pretrained(model, "smangrul/tinyllama_lora_norobots", adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")
```

Set the adapters, weights, `adapter_name`, `combination_type`, and `density` with the [`~LoraModel.add_weighted_adapter`] method.

<hfoptions id="merge-method">
<hfoption id="TIES">

Weight values greater than `1.0` typically produce better results because they preserve the correct scale. A good default starting value for the weights is to set all values to `1.0`.

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 1.0, 1.0]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
```

</hfoption>
<hfoption id="DARE">

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
```

</hfoption>
</hfoptions>

Set the newly merged model as the active model with the [`~LoraModel.set_adapter`] method.

```py
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


## Merging (IA)³ Models
The (IA)³ models facilitate linear merging of adapters. To merge adapters in an (IA)³ model, utilize the `add_weighted_adapter` method from the `IA3Model` class. This method is analogous to the `add_weighted_adapter` method used in `LoraModel`, with the key difference being the absence of the `combination_type` parameter. For example, to merge three (IA)³ adapters into a PEFT model, you would proceed as follows:

```py
adapters = ["adapter1", "adapter2", "adapter3"]
weights = [0.4, 0.3, 0.3]
adapter_name = "merge"
model.add_weighted_adapter(adapters, weights, adapter_name)
```

It is recommended that the weights sum to 1.0 to preserve the scale of the model. The merged model can then be set as the active model using the `set_adapter` method:

```py
model.set_adapter("merge")
```
