<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# LoRA

LoRA is low-rank decomposition method to reduce the number of trainable parameters which speeds up finetuning large models and uses less memory. In PEFT, using LoRA is as easy as setting up a [`LoraConfig`] and wrapping it with [`get_peft_model`] to create a trainable [`PeftModel`].

This guide explores in more detail other options and features for using LoRA.

## Initialization

The initialization of LoRA weights is controlled by the parameter `init_lora_weights` in [`LoraConfig`]. By default, PEFT initializes LoRA weights with Kaiming-uniform for weight A and zeros for weight B resulting in an identity transform (same as the reference [implementation](https://github.com/microsoft/LoRA)).

It is also possible to pass `init_lora_weights="gaussian"`. As the name suggests, this initializes weight A with a Gaussian distribution and zeros for weight B (this is how [Diffusers](https://huggingface.co/docs/diffusers/index) initializes LoRA weights).

```py
from peft import LoraConfig

config = LoraConfig(init_lora_weights="gaussian", ...)
```

There is also an option to set `init_lora_weights=False` which is useful for debugging and testing. This should be the only time you use this option. When choosing this option, the LoRA weights are initialized such that they do *not* result in an identity transform.

```py
from peft import LoraConfig

config = LoraConfig(init_lora_weights=False, ...)
```

### PiSSA
[PiSSA](https://huggingface.co/papers/2404.02948) initializes the LoRA adapter using the principal singular values and singular vectors. This straightforward modification allows PiSSA to converge more rapidly than LoRA and ultimately attain superior performance. Moreover, PiSSA reduces the quantization error compared to QLoRA, leading to further enhancements.

Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model:
```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="pissa", ...)
```
Alternatively, execute fast SVD, which takes only a few seconds. The number of iterations determines the trade-off between the error and computation time:
```python
lora_config = LoraConfig(init_lora_weights="pissa_niter_[number of iters]", ...)
```
For detailed instruction on using PiSSA, please follow [these instructions](https://github.com/huggingface/peft/tree/main/examples/pissa_finetuning).

### CorDA

[CorDA](https://huggingface.co/papers/2406.05223) builds task-aware LoRA adapters from weight decomposition oriented by the context of downstream task to learn (instruction-previewed mode, IPM) or world knowledge to maintain (knowledge-preserved mode, KPM).
The KPM not only achieves better performance than LoRA on fine-tuning tasks, but also mitigates the catastrophic forgetting of pre-trained world knowledge.
When preserving pre-trained knowledge is not a concern,
the IPM is favored because it can further accelerate convergence and enhance the fine-tuning performance.

You need to configure the initialization method to "corda", and specify the mode of IPM or KPM and the dataset to collect covariance matrices.

```py
@torch.no_grad()
def run_model():
    # Assume `model` and `dataset` is in context...
    model.eval()
    for batch in dataset:
        model(**batch)


corda_config = CordaConfig(
    corda_method="kpm",
)
lora_config = LoraConfig(
    init_lora_weights="corda",
    corda_config=corda_config,
)
preprocess_corda(model, lora_config, run_model=run_model)
peft_model = get_peft_model(model, lora_config)
```

For detailed instruction on using CorDA, please follow [these instructions](https://github.com/huggingface/peft/tree/main/examples/corda_finetuning).

### OLoRA
[OLoRA](https://huggingface.co/papers/2406.01775) utilizes QR decomposition to initialize the LoRA adapters. OLoRA translates the base weights of the model by a factor of their QR decompositions, i.e., it mutates the weights before performing any training on them. This approach significantly improves stability, accelerates convergence speed, and ultimately achieves superior performance.

You just need to pass a single additional option to use OLoRA:
```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="olora", ...)
```
For more advanced usage, please refer to our [documentation](https://github.com/huggingface/peft/tree/main/examples/olora_finetuning).

### EVA
[EVA](https://huggingface.co/papers/2410.07170) performs SVD on the input activations of each layer and uses the right-singular vectors to initialize LoRA weights. It is therefore a data-driven initialization scheme. Furthermore EVA adaptively allocates ranks across layers based on their "explained variance ratio" - a metric derived from the SVD analysis.

You can use EVA by setting `init_lora_weights="eva"` and defining [`EvaConfig`] in [`LoraConfig`]:
```python
from peft import LoraConfig, EvaConfig
peft_config = LoraConfig(
    init_lora_weights = "eva",
    eva_config = EvaConfig(rho = 2.0),
    ...
)
```
The parameter `rho` (≥ 1.0) determines how much redistribution is allowed. When `rho=1.0` and `r=16`, LoRA adapters are limited to exactly 16 ranks, preventing any redistribution from occurring. A recommended value for EVA with redistribution is 2.0, meaning the maximum rank allowed for a layer is 2r.

It is recommended to perform EVA initialization on a GPU as it is much faster. To optimize the amount of available memory for EVA, you can use the `low_cpu_mem_usage` flag in [`get_peft_model`]:
```python
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)
```
Then, call [`initialize_lora_eva_weights`] to initialize the EVA weights (in most cases the dataloader used for eva initialization can be the same as the one used for finetuning):
```python
initialize_lora_eva_weights(peft_model, dataloader)
```
EVA works out of the box with bitsandbytes. Simply initialize the model with `quantization_config` and call [`initialize_lora_eva_weights`] as usual.

<Tip>

For further instructions on using EVA, please refer to our [documentation](https://github.com/huggingface/peft/tree/main/examples/eva_finetuning).

</Tip>

### LoftQ

#### Standard approach

When quantizing the base model for QLoRA training, consider using the [LoftQ initialization](https://huggingface.co/papers/2310.08659), which has been shown to improve performance when training quantized models. The idea is that the LoRA weights are initialized such that the quantization error is minimized. To use LoftQ, follow [these instructions](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning).

In general, for LoftQ to work best, it is recommended to target as many layers with LoRA as possible, since those not targeted cannot have LoftQ applied. This means that passing `LoraConfig(..., target_modules="all-linear")` will most likely give the best results. Also, you should use `nf4` as quant type in your quantization config when using 4bit quantization, i.e. `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.

#### A more convenient way

An easier but more limited way to apply LoftQ initialization is to use the convenience function `replace_lora_weights_loftq`. This takes the quantized PEFT model as input and replaces the LoRA weights in-place with their LoftQ-initialized counterparts.

```python
from peft import replace_lora_weights_loftq
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
base_model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
# note: don't pass init_lora_weights="loftq" or loftq_config!
lora_config = LoraConfig(task_type="CAUSAL_LM")
peft_model = get_peft_model(base_model, lora_config)
replace_lora_weights_loftq(peft_model)
```

`replace_lora_weights_loftq` also allows you to pass a `callback` argument to give you more control over which layers should be modified or not, which empirically can improve the results quite a lot. To see a more elaborate example of this, check out [this notebook](https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/LoftQ_weight_replacement.ipynb).

`replace_lora_weights_loftq` implements only one iteration step of LoftQ. This means that only the LoRA weights are updated, instead of iteratively updating LoRA weights and quantized base model weights. This may lead to lower performance but has the advantage that we can use the original quantized weights derived from the base model, instead of having to keep an extra copy of modified quantized weights. Whether this tradeoff is worthwhile depends on the use case.

At the moment, `replace_lora_weights_loftq` has these additional limitations:

- Model files must be stored as a `safetensors` file.
- Only bitsandbytes 4bit quantization is supported.

<Tip>

Learn more about how PEFT works with quantization in the [Quantization](quantization) guide.

</Tip>

### Rank-stabilized LoRA

Another way to initialize [`LoraConfig`] is with the [rank-stabilized LoRA (rsLoRA)](https://huggingface.co/papers/2312.03732) method. The LoRA architecture scales each adapter during every forward pass by a fixed scalar which is set at initialization and depends on the rank `r`. The scalar is given by `lora_alpha/r` in the original implementation, but rsLoRA uses `lora_alpha/math.sqrt(r)` which stabilizes the adapters and increases the performance potential from using a higher `r`.

```py
from peft import LoraConfig

config = LoraConfig(use_rslora=True, ...)
```

### Weight-Decomposed Low-Rank Adaptation (DoRA)

This technique decomposes the updates of the weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, especially at low ranks. For more information on DoRA, see  https://huggingface.co/papers/2402.09353.

```py
from peft import LoraConfig

config = LoraConfig(use_dora=True, ...)
```

If parts of the model or the DoRA adapter are offloaded to CPU you can get a significant speedup at the cost of some temporary (ephemeral) VRAM overhead by using `ephemeral_gpu_offload=True` in `config.runtime_config`.

```py
from peft import LoraConfig, LoraRuntimeConfig

config = LoraConfig(use_dora=True, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True), ...)
```

A `PeftModel` with a DoRA adapter can also be loaded with `ephemeral_gpu_offload=True` flag using the `from_pretrained` method as well as the `load_adapter` method.

```py
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, peft_model_id, ephemeral_gpu_offload=True)
```

DoRA is optimized (computes faster and takes less memory) for models in the evaluation mode, or when dropout is set to 0. We reuse the
base result at those times to get the speedup.
Running [dora finetuning](https://github.com/huggingface/peft/blob/main/examples/dora_finetuning/dora_finetuning.py)
with `CUDA_VISIBLE_DEVICES=0 time python examples/dora_finetuning/dora_finetuning.py --quantize --lora_dropout 0 --batch_size 16 --eval_step 2 --use_dora`
on a 4090 with gradient accumulation set to 2 and max step to 20 resulted with the following observations:

| | Without Optimization | With Optimization |
| :--: | :--: | :--: |
| train_runtime | 359.7298 | **279.2676** |
| train_samples_per_second | 1.779 | **2.292** |
| train_steps_per_second | 0.056 | **0.072** |

#### Caveats

- DoRA only supports embedding, linear, and Conv2d layers at the moment.
- DoRA introduces a bigger overhead than pure LoRA, so it is recommended to merge weights for inference, see [`LoraModel.merge_and_unload`].
- DoRA should work with weights quantized with bitsandbytes ("QDoRA"). However, issues have been reported when using QDoRA with DeepSpeed Zero2.

### QLoRA-style training

The default LoRA settings in PEFT add trainable weights to the query and value layers of each attention block. But [QLoRA](https://hf.co/papers/2305.14314), which adds trainable weights to all the linear layers of a transformer model, can provide performance equal to a fully finetuned model. To apply LoRA to all the linear layers, like in QLoRA, set `target_modules="all-linear"` (easier than specifying individual modules by name which can vary depending on the architecture).

```py
config = LoraConfig(target_modules="all-linear", ...)
```

### Memory efficient Layer Replication with LoRA

An approach used to improve the performance of models is to expand a model by duplicating layers in the model to build a larger model from a pretrained model of a given size. For example increasing a 7B model to a 10B model as described in the [SOLAR](https://huggingface.co/papers/2312.15166) paper. PEFT LoRA supports this kind of expansion in a memory efficient manner that supports further fine-tuning using LoRA adapters attached to the layers post replication of the layers. The replicated layers do not take additional memory as they share the underlying weights so the only additional memory required is the memory for the adapter weights. To use this feature you would create a config with the `layer_replication` argument.

```py
config = LoraConfig(layer_replication=[[0,4], [2,5]], ...)
```

Assuming the original model had 5 layers `[0, 1, 2 ,3, 4]`, this would create a model with 7 layers arranged as `[0, 1, 2, 3, 2, 3, 4]`. This follows the [mergekit](https://github.com/arcee-ai/mergekit) pass through merge convention where sequences of layers specified as start inclusive and end exclusive tuples are stacked to build the final model. Each layer in the final model gets its own distinct set of LoRA adapters.

[Fewshot-Metamath-OrcaVicuna-Mistral-10B](https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B) is an example of a model trained using this method on Mistral-7B expanded to 10B. The
[adapter_config.json](https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B/blob/main/adapter_config.json) shows a sample LoRA adapter config applying this method for fine-tuning.

### Fine grained control over ranks and alpha (scaling)

By default, all layers targeted with LoRA will have the same rank `r` and the same `lora_alpha` (which determines the LoRA scaling), depending on what was specified in the [`LoraConfig`]. In some cases, however, you may want to indicate different values for different layers. This is possible by passing the `rank_pattern` and `alpha_pattern` arguments to [`LoraConfig`]. These arguments should be dictionaries with the key being the layer name and the value being the rank/alpha value. The keys can be [regular expressions](https://docs.python.org/3/library/re.html) (regex). All LoRA layers that are not explicitly mentioned in `rank_pattern` and `alpha_pattern` will take the default `r` and `lora_alpha` values.

To give an example, let's assume that we have a model with the following structure:

```python
>>> print(model)
Outer(
  (foo): Linear(...)
  (module): Middle(
    (foo): Linear(...)
    (foobar): Linear(...)
    (module): Inner(
      (foo): Linear(...)
      (barfoo): Linear(...)
    )
  )
)
```

- `rank_pattern={"foo": 42}` will match all 3 `foo` layers. Neither `foobar` nor `barfoo` are matched.
- `rank_pattern={"^foo": 42}` will only match the `foo` layer of the model, but neither `module.foo` nor `module.module.foo`. This is because the `^` means "start of string" when using regular expressions, and only `foo` starts with `"foo"`, the other layer names have prefixes.
- `rank_pattern={"^module.foo": 42}` matches only `module.foo`, but not `module.module.foo`, for the same reason.
- `rank_pattern={"module.foo": 42}` matches both `module.foo` and `module.module.foo`, but not `foo`.
- `rank_pattern={"^foo": 42, "^module.module.foo": 55}` matches `foo` and `module.module.foo`, respectively, but not `module.foo`.
- There is no need to indicate `$` to mark the end of the match, as this is added automatically by PEFT.

The same logic applies to `alpha_pattern`. If you're in doubt, don't try to get fancy with regular expressions -- just pass the full name for each module with a different rank/alpha, preceded by the `^` prefix, and you should be good.

## Optimizers

LoRA training can optionally include special purpose optimizers. Currently PEFT supports LoRA-FA and LoRA+.

### LoRA-FA Optimizer

LoRA training can be more effective and efficient using LoRA-FA, as described in [LoRA-FA](https://huggingface.co/papers/2308.03303). LoRA-FA reduces activation memory consumption by fixing the matrix A and only tuning the matrix B. During training, the gradient of B is optimized to approximate the full parameter fine-tuning gradient. Moreover, the memory consumption of LoRA-FA is not sensitive to the rank (since it erases the activation of $A$), therefore it can improve performance by enlarging lora rank without increasing memory consumption.

```py
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer
from transformers import Trainer, get_cosine_schedule_with_warmup

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

config = LoraConfig(...)
model = get_peft_model(base_model, config)

optimizer = create_lorafa_optimizer(
    model=model,
    r=128,
    lora_alpha=32,
    lr=7e-5,
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
)

trainer = Trainer(
    ...,
    optimizers=(optimizer, scheduler),
)
```

### LoRA+ optimized LoRA

LoRA training can be optimized using [LoRA+](https://huggingface.co/papers/2402.12354), which uses different learning rates for the adapter matrices A and B, shown to increase finetuning speed by up to 2x and performance by 1-2%.

```py
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import Trainer
import bitsandbytes as bnb

base_model = ...
config = LoraConfig(...)
model = get_peft_model(base_model, config)

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=bnb.optim.Adam8bit,
    lr=5e-5,
    loraplus_lr_ratio=16,
)
scheduler = None

...
trainer = Trainer(
    ...,
    optimizers=(optimizer, scheduler),
)
```

## Efficiently train tokens alongside LoRA

Sometimes it is necessary to not only change some layer's weights but to add new tokens as well. With larger models this can be a memory-costly endeavour. PEFT LoRA adapters support the `trainable_token_indices` parameter which allows tuning of other tokens alongside fine-tuning of specific layers with LoRA. This method only trains the tokens you specify and leaves all other tokens untouched. This saves memory and doesn't throw away learned context of existing token embeddings in contrast to when training the whole embedding matrix. Under the hood this method uses the layer of [`TrainableTokensModel`].

```py
# for layer 'embed_tokens'
config = LoraConfig(trainable_token_indices=[idx_1, idx_2, ...], ...)

# specific embedding layer
config = LoraConfig(trainable_token_indices={'emb_tokens': [idx_1, idx_2, ...]}, ...)
```

In the snippet below we show how to add new tokens to the model and how to train it alongside the other layers in the model.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# we define our new tokens and add them to the tokenizer as special tokens
special_tokens = ['<|start_think|>', '<|stop_think|>']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# make room for new tokens in the embedding matrix if it isn't big enough already
base_model.resize_token_embeddings(max(len(tokenizer), base_model.model.embed_tokens.num_embeddings)

# typical LoRA config with `trainable_token_indices` targeting embedding layer `embed_tokens`
# and specifically our new tokens we just added
lora_config = LoraConfig(
    target_modules='all-linear',
    trainable_token_indices={'embed_tokens': tokenizer.convert_tokens_to_ids(special_tokens)},
)
peft_model = get_peft_model(base_model, lora_config)

# proceed to train the model like normal
[...]
```

The token weights are part of your adapter state dict and saved alongside the LoRA weights.
If we would have used full fine-tuning with `modules_to_save=['embed_tokens']` we would have stored the full embedding matrix in the checkpoint, leading to a much bigger file.

To give a bit of an indication how much VRAM can be saved, a rudimentary comparison of the above example was made between training the embedding matrix fully (`modules_to_save=["embed_tokens"]`), using a LoRA for the embedding matrix (`target_modules=[..., "embed_tokens"]`, rank 32) and trainable tokens (`trainable_token_indices=[...]`, 6 tokens). Trainable tokens used about as much VRAM (15,562MB vs. 15,581MB) as LoRA while being specific to the tokens and saved ~1GB of VRAM over fully training the embedding matrix.


## Merge LoRA weights into the base model

While LoRA is significantly smaller and faster to train, you may encounter latency issues during inference due to separately loading the base model and the LoRA adapter. To eliminate latency, use the [`~LoraModel.merge_and_unload`] function to merge the adapter weights with the base model. This allows you to use the newly merged model as a standalone model. The [`~LoraModel.merge_and_unload`] function doesn't keep the adapter weights in memory.

Below is a diagram that explains the intuition of LoRA adapter merging:

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png"/>
</div>

We show in the snippets below how to run that using PEFT.

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_and_unload()
```

If you need to keep a copy of the weights so you can unmerge the adapter later or delete and load different ones, you should use the [`~LoraModel.merge_adapter`] function instead. Now you have the option to use [`~LoraModel.unmerge_adapter`] to return the base model.

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_adapter()

# unmerge the LoRA layers from the base model
model.unmerge_adapter()
```

The [`~LoraModel.add_weighted_adapter`] function is useful for merging multiple LoRAs into a new adapter based on a user provided weighting scheme in the `weights` parameter. Below is an end-to-end example.

First load the base model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, device_map="auto"
)
```

Then we load the first adapter:

```python
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id, adapter_name="sft")
```

Then load a different adapter and merge it with the first one:

```python
weighted_adapter_name = "sft-dpo"
model.load_adapter("alignment-handbook/zephyr-7b-dpo-lora", adapter_name="dpo")
model.add_weighted_adapter(
    adapters=["sft", "dpo"],
    weights=[0.7, 0.3],
    adapter_name=weighted_adapter_name,
    combination_type="linear"
)
model.set_adapter(weighted_adapter_name)
```

<Tip>

There are several supported methods for `combination_type`. Refer to the [documentation](../package_reference/lora#peft.LoraModel.add_weighted_adapter) for more details. Note that "svd" as the `combination_type` is not supported when using `torch.float16` or `torch.bfloat16` as the datatype.

</Tip>

Now, perform inference:

```python
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    generate_ids = model.generate(**inputs, max_length=30)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(outputs)
```

## Load adapters

Adapters can be loaded onto a pretrained model with [`~PeftModel.load_adapter`], which is useful for trying out different adapters whose weights aren't merged. Set the active adapter weights with the [`~LoraModel.set_adapter`] function.

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# load different adapter
model.load_adapter("alignment-handbook/zephyr-7b-dpo-lora", adapter_name="dpo")

# set adapter as active
model.set_adapter("dpo")
```

To return the base model, you could use [`~LoraModel.unload`] to unload all of the LoRA modules or [`~LoraModel.delete_adapter`] to delete the adapter entirely.

```py
# unload adapter
model.unload()

# delete adapter
model.delete_adapter("dpo")
```

## Inference with different LoRA adapters in the same batch

Normally, each inference batch has to use the same adapter(s) in PEFT. This can sometimes be annoying, because we may have batches that contain samples intended to be used with different LoRA adapters. For example, we could have a base model that works well in English and two more LoRA adapters, one for French and one for German. Usually, we would have to split our batches such that each batch only contains samples of one of the languages, we cannot combine different languages in the same batch.

Thankfully, it is possible to mix different LoRA adapters in the same batch using the `adapter_name` argument. Below, we show an example of how this works in practice. First, let's load the base model, English, and the two adapters, French and German, like this:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = ...
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
# load the LoRA adapter for French
peft_model = PeftModel.from_pretrained(model, <path>, adapter_name="adapter_fr")
# next, load the LoRA adapter for German
peft_model.load_adapter(<path>, adapter_name="adapter_de")
```

Now, we want to generate text on a sample that contains all three languages: The first three samples are in English, the next three are in French, and the last three are in German. We can use the `adapter_names` argument to specify which adapter to use for each sample. Since our base model is used for English, we use the special string `"__base__"` for these samples. For the next three samples, we indicate the adapter name of the French LoRA fine-tune, in this case `"adapter_fr"`. For the last three samples, we indicate the adapter name of the German LoRA fine-tune, in this case `"adapter_de"`. This way, we can use the base model and the two adapters in a single batch.

```python
inputs = tokenizer(
    [
        "Hello, my dog is cute",
        "Hello, my cat is awesome",
        "Hello, my fish is great",
        "Salut, mon chien est mignon",
        "Salut, mon chat est génial",
        "Salut, mon poisson est super",
        "Hallo, mein Hund ist süß",
        "Hallo, meine Katze ist toll",
        "Hallo, mein Fisch ist großartig",
    ],
    return_tensors="pt",
    padding=True,
)

adapter_names = [
    "__base__", "__base__", "__base__",
    "adapter_fr", "adapter_fr", "adapter_fr",
    "adapter_de", "adapter_de", "adapter_de",
]
output = peft_model.generate(**inputs, adapter_names=adapter_names, max_new_tokens=20)
```

Note that the order does not matter here, i.e. the samples in the batch don't need to be grouped by adapter as in the example above. We just need to ensure that the `adapter_names` argument is aligned correctly with the samples.

Additionally, the same approach also works with the `modules_to_save` feature, which allows for saving and reusing specific neural network layers, such as custom heads for classification tasks, across different LoRA adapters.

### Caveats

Using this feature has some drawbacks, namely:

- It only works for inference, not for training.
- Disabling adapters using the `with model.disable_adapter()` context takes precedence over `adapter_names`.
- You cannot pass `adapter_names` when some adapter weights were merged with base weight using the `merge_adapter` method. Please unmerge all adapters first by calling `model.unmerge_adapter()`.
- For obvious reasons, this cannot be used after calling `merge_and_unload()`, since all the LoRA adapters will be merged into the base weights in this case.
- This feature does not currently work with DoRA, so set `use_dora=False` in your `LoraConfig` if you want to use it.
- The `modules_to_save` feature is currently only supported for the layers of types `Linear`, `Embedding`, `Conv2d` and `Conv1d`.
- There is an expected overhead for inference with `adapter_names`, especially if the amount of different adapters in the batch is high. This is because the batch size is effectively reduced to the number of samples per adapter. If runtime performance is your top priority, try the following:
  - Increase the batch size.
  - Try to avoid having a large number of different adapters in the same batch, prefer homogeneous batches. This can be achieved by buffering samples with the same adapter and only perform inference with a small handful of different adapters.
  - Take a look at alternative implementations such as [LoRAX](https://github.com/predibase/lorax), [punica](https://github.com/punica-ai/punica), or [S-LoRA](https://github.com/S-LoRA/S-LoRA), which are specialized to work with a large number of different adapters.
