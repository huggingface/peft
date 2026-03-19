<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# LoRA conversion

Functions that allow to convert non-LoRA PEFT models to LoRA models.

## Description

PEFT supports dozens of different parameter effficient fine-tuning techniques. The most popular one by far is LoRA. This means that many other packages support LoRA too. For example, [Diffusers](https://huggingface.co/docs/diffusers/main/en/api/loaders/lora) allows to load LoRA adapters to change the capabilities of diffusion models. [vLLM](https://docs.vllm.ai/en/stable/features/lora/) allows serving models with LoRA adapters. This is nice but unfortunately, all the other, non-LoRA PEFT methods are rarely supported. Therefore, even if another PEFT method would work better for your specific use case, you may be prevented from using it because downstream packages offer no support.

Here we present a potential solution. PEFT offers two functions, [`save_as_lora`] and [`convert_to_lora`], which allow to convert a PEFT adapter into a LoRA adapter. Not all PEFT methods support this for now, but if they do, it means you can start with the PEFT method that works best for you and then later use it as if it were a LoRA adapter.

## Example

The LoRA rank for the converted adapter can either be set to a fixed rank by passing an int > 0 to the `rank` argument, or a dynamic rank, which adapts to each layer, by passing a float between 0 and 1 to the `rank` argument. Dynamic ranks can potentially be more efficient (same performance with fewer parameters).

### Fixed LoRA rank

The usage of [`save_as_lora`] is relatively straightforward:

```python
from peft import get_peft_model, save_as_lora

# first load and train your non-LoRA PEFT model as normal
base_model = ...
non_lora_config = ...
model = get_peft_model(base_model, non_lora_config)
# check that this PEFT method can indeed be converted to LoRA
assert model.supports_lora_conversion()
...  # train the model

# the rank of the LoRA adapter that you want to convert to
target_rank = 64
# save as a LoRA checkpoint
save_as_lora(output_path, model, rank=target_rank)
```

This will create a LoRA checkpoint at `output_path` that you can load like any other LoRA adapter, or use in downstream packages such as Diffusers or vLLM.

The [`convert_to_lora`] function is useful if you don't want to save the converted LoRA adapter but instead want to use the converted weights right away, for example to perform evaluations:

```python
from peft import convert_to_lora, get_peft_model, set_peft_model_state_dict

base_model = ...
non_lora_config = ...
model = get_peft_model(base_model, non_lora_config)
...  # train the model

# get the lora config and state dict of the converted lora model
lora_config, lora_state_dict = convert_to_lora(model, rank=target_rank)
# reload the base model, or use model.unload()
base_model = ...
# apply the lora config to the base model
lora_model = get_peft_model(base_model, lora_config)
# load the LoRA weights onto the base model
set_peft_model_state_dict(lora_model, state_dict)
```

### Dynamic LoRA rank

In the examples above, we used a fixed LoRA rank for conversion. However, it is conceivable that some layers don't require a high rank to be accurately converted, while other layers require a higher rank. To accomodate this, PEFT offers the option to pass a float between 0 and 1 as the `rank` argument. Let's say you pass `rank=0.5`. This means that for each layer, the rank for the LoRA adapter is chosen such that the LoRA adapter explains 50% of the variance in weight introduced by original adapter. In more technical terms, under the hood we perform a [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) on the weight contribution of the adapter and then take the top singular values that, when normalized, sum up to the passed value.

```python
# set a dynamic rank by passing a float
threshold = 0.7
# save as a LoRA checkpoint
save_as_lora(output_path, model, rank=threshold)
# get the lora config and state dict directly:
lora_config, lora_state_dict = convert_to_lora(model, rank=threshold)
# inspect the different ranks per layer:
print(lora_config.rank_pattern)
```

Using this type of dynamic LoRA rank can be useful if the contribution of the different layers varies a lot. The disadvantage is that it could mean that some layers will have a very high LoRA rank, which can lead to memory spikes. Please test what works best for your use case.

### LoRA to LoRA conversion

It is also possible to convert a LoRA adapter into another LoRA adapter. Why would you want to do that? There is one reason, namely if you want to reduce the rank of the LoRA adapter. If, after training, you want to shrink the LoRA adapter, use [`save_as_lora`] or [`convert_to_lora`] and pass a smaller rank. This will give you a new LoRA adapter that has a smaller memory and storage footprint.

## Metrics

### Non-LoRA to LoRA conversion

Of course, converting one PEFT adapter into another adapter is a lossy process. The new adapter will most likely not perform as well as the initial adapter. Therefore, it is highly advised to **evaluate the converted LoRA adapter**. This way, you can make sure that the converted adapter performs well enough for your use case. The general rule applies that the higher the rank of the LoRA adaper, the better it will approximate your initial adapter. This means that the converted LoRA adapter may require more parameters than the original adapter to achieve a similar performanace.

To give an example, here are some numbers that were derived on the [PEFT MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA). For this, a [LoHa](https://huggingface.co/docs/peft/package_reference/loha) was used to fine-tune `meta-llama/Llama-3.2-3B` on MetaMathQA and evaluated on GSM8K. The initial LoKr adapter had rank 32, resulting in 18,350,080 trainable parameters, and a test accuracy of 41.85%. Evaluation required 12.25 GB of memory. The checkpoint was converted into LoRA with different values for the `rank`. The resulting outcome is:

| rank | trainable parameters | test accuracy (%) | accuracy change | memory reserved (max, GB) | memory increase |
|------|---------------------:|------------------:|----------------:|--------------------------:|----------------:|
| 8    |              2293760 |             37.60 |           -4.25 |                     12.41 |            0.16 |
| 16   |              4587520 |             38.89 |           -2.96 |                     12.15 |           -0.10 |
| 32   |              9175040 |             40.11 |           -1.74 |                     12.41 |            0.16 |
| 64   |             18350080 |             39.20 |           -2.65 |                     12.18 |           -0.07 |
|      |                      |                   |                 |                           |                 |
| 0.4  |              2428928 |             37.60 |           -4.25 |                     12.41 |            0.16 |
| 0.5  |              4761600 |             40.18 |           -1.67 |                     12.41 |            0.16 |
| 0.6  |              8857600 |             39.42 |           -2.43 |                     12.41 |            0.16 |
| 0.7  |             16230400 |             39.04 |           -2.81 |                     12.15 |           -0.10 |

As you can see, we can attain a test accuracy that comes close to the original LoHa adapter if the rank is sufficiently high. Choosing the right rank is a tradeoff between model performance and model efficiency. To reproduce this experiment, follow the script at https://github.com/huggingface/peft/tree/main/scripts/evaluate-lora-conversion.py.

Note that the number of trainable parameters cannot be translated one to one into memory usage. Some PEFT methods require more, some less memory, even with the same number of trainable parameters. Therefore, even if after conversion, the LoRA adapter has more parameters than the original one, it could still be more memory efficient when serving.

### LoRA to LoRA conversion

Similar to the experiment above, we can also evaluate LoRA to LoRA conversion (i.e. LoRA compression). Here, we start with a LoRA adapter of rank 64 trained on the same setup as above with RS-LoRA. The initial adapter has 18,350,080 trainable parameters, a test accuracy of 52.92%, and requires 12.58 GB of memory for evaluation. The following table shows the results of converting this adapter to LoRA adapters of smaller rank:

| rank | trainable parameters | test accuracy (%) | accuracy change | memory reserved (max, GB) | memory increase |
|------|---------------------:|------------------:|----------------:|--------------------------:|----------------:|
| 8    |              2293760 |             43.37 |           -9.55 |                     12.38 |           -0.20 |
| 16   |              4587520 |             48.90 |           -4.02 |                     12.38 |           -0.20 |
| 32   |              9175040 |             51.48 |           -1.44 |                     12.49 |           -0.09 |
| 48   |             13762560 |             52.01 |           -0.91 |                     12.38 |           -0.20 |
|      |                      |                   |                 |                           |                 |
| 0.5  |              2150400 |             44.12 |           -8.80 |                     12.37 |           -0.21 |
| 0.6  |              3082240 |             47.54 |           -5.38 |                     12.37 |           -0.21 |
| 0.7  |              4448256 |             50.49 |           -2.43 |                     12.37 |           -0.21 |
| 0.8  |              6510592 |             50.11 |           -2.81 |                     12.37 |           -0.21 |
| 0.9  |             10022912 |             51.55 |           -1.37 |                     12.38 |           -0.20 |
| 0.95 |             12976128 |             52.62 |           -0.30 |                     12.39 |           -0.19 |

So for instance for rank 0.95, we can close the accuracy gap to just 0.3 percentage points while reducing the number of parameters by 30%. Also note that these compressed LoRAs can be better than directly training them on the lower rank -- e.g. for rank 32, training directly results in a test accuracy of 48.22%, while conversion from rank 64 results in 51.48%.

## Caveats

There are some limitations to the LoRA conversion. As mentioned above, a reduction in performance is expected and the converted LoRA will most likely be less parameter efficient than the original adapter. Morever, LoRA conversion has these limitations:

- Right now, only adapters applied to linear layers can be converted.
- Not all PEFT methods currently support LoRA conversion.

If there is a lot of demand to extend LoRA conversion, please let us know and we will make it work with more layer types and PEFT methods.

## API

### Convert a non-LoRA model to a LoRA model, return the `LoraConfig` and `state_dict`

[[autodoc]] tuners.lora.conversion.convert_to_lora
    - all

### Convert a non-LoRA model to a LoRA model, save the adapter checkpoint and config at the given path

[[autodoc]] tuners.lora.conversion.save_as_lora
    - all
