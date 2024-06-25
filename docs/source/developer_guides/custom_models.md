<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Custom models

Some fine-tuning techniques, such as prompt tuning, are specific to language models. That means in 🤗 PEFT, it is
assumed a 🤗 Transformers model is being used. However, other fine-tuning techniques - like
[LoRA](../conceptual_guides/lora) - are not restricted to specific model types.

In this guide, we will see how LoRA can be applied to a multilayer perceptron, a computer vision model from the [timm](https://huggingface.co/docs/timm/index) library, or a new 🤗 Transformers architecture.

## Multilayer perceptron

Let's assume that we want to fine-tune a multilayer perceptron with LoRA. Here is the definition:

```python
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_units_hidden=2000):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(20, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        return self.seq(X)
```

This is a straightforward multilayer perceptron with an input layer, a hidden layer, and an output layer.

<Tip>

For this toy example, we choose an exceedingly large number of hidden units to highlight the efficiency gains
from PEFT, but those gains are in line with more realistic examples.

</Tip>

There are a few linear layers in this model that could be tuned with LoRA. When working with common 🤗 Transformers
models, PEFT will know which layers to apply LoRA to, but in this case, it is up to us as a user to choose the layers.
To determine the names of the layers to tune:

```python
print([(n, type(m)) for n, m in MLP().named_modules()])
```

This should print:

```
[('', __main__.MLP),
 ('seq', torch.nn.modules.container.Sequential),
 ('seq.0', torch.nn.modules.linear.Linear),
 ('seq.1', torch.nn.modules.activation.ReLU),
 ('seq.2', torch.nn.modules.linear.Linear),
 ('seq.3', torch.nn.modules.activation.ReLU),
 ('seq.4', torch.nn.modules.linear.Linear),
 ('seq.5', torch.nn.modules.activation.LogSoftmax)]
```

Let's say we want to apply LoRA to the input layer and to the hidden layer, those are `'seq.0'` and `'seq.2'`. Moreover,
let's assume we want to update the output layer without LoRA, that would be `'seq.4'`. The corresponding config would
be:

```python
from peft import LoraConfig

config = LoraConfig(
    target_modules=["seq.0", "seq.2"],
    modules_to_save=["seq.4"],
)
```

With that, we can create our PEFT model and check the fraction of parameters trained:

```python
from peft import get_peft_model

model = MLP()
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# prints trainable params: 56,164 || all params: 4,100,164 || trainable%: 1.369798866581922
```

Finally, we can use any training framework we like, or write our own fit loop, to train the `peft_model`.

For a complete example, check out [this notebook](https://github.com/huggingface/peft/blob/main/examples/multilayer_perceptron/multilayer_perceptron_lora.ipynb).

## timm models

The [timm](https://huggingface.co/docs/timm/index) library contains a large number of pretrained computer vision models.
Those can also be fine-tuned with PEFT. Let's check out how this works in practice.

To start, ensure that timm is installed in the Python environment:

```bash
python -m pip install -U timm
```

Next we load a timm model for an image classification task:

```python
import timm

num_classes = ...
model_id = "timm/poolformer_m36.sail_in1k"
model = timm.create_model(model_id, pretrained=True, num_classes=num_classes)
```

Again, we need to make a decision about what layers to apply LoRA to. Since LoRA supports 2D conv layers, and since
those are a major building block of this model, we should apply LoRA to the 2D conv layers. To identify the names of
those layers, let's look at all the layer names:

```python
print([(n, type(m)) for n, m in model.named_modules()])
```

This will print a very long list, we'll only show the first few:

```
[('', timm.models.metaformer.MetaFormer),
 ('stem', timm.models.metaformer.Stem),
 ('stem.conv', torch.nn.modules.conv.Conv2d),
 ('stem.norm', torch.nn.modules.linear.Identity),
 ('stages', torch.nn.modules.container.Sequential),
 ('stages.0', timm.models.metaformer.MetaFormerStage),
 ('stages.0.downsample', torch.nn.modules.linear.Identity),
 ('stages.0.blocks', torch.nn.modules.container.Sequential),
 ('stages.0.blocks.0', timm.models.metaformer.MetaFormerBlock),
 ('stages.0.blocks.0.norm1', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.0.token_mixer', timm.models.metaformer.Pooling),
 ('stages.0.blocks.0.token_mixer.pool', torch.nn.modules.pooling.AvgPool2d),
 ('stages.0.blocks.0.drop_path1', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.layer_scale1', timm.models.metaformer.Scale),
 ('stages.0.blocks.0.res_scale1', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.norm2', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.0.mlp', timm.layers.mlp.Mlp),
 ('stages.0.blocks.0.mlp.fc1', torch.nn.modules.conv.Conv2d),
 ('stages.0.blocks.0.mlp.act', torch.nn.modules.activation.GELU),
 ('stages.0.blocks.0.mlp.drop1', torch.nn.modules.dropout.Dropout),
 ('stages.0.blocks.0.mlp.norm', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.mlp.fc2', torch.nn.modules.conv.Conv2d),
 ('stages.0.blocks.0.mlp.drop2', torch.nn.modules.dropout.Dropout),
 ('stages.0.blocks.0.drop_path2', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.layer_scale2', timm.models.metaformer.Scale),
 ('stages.0.blocks.0.res_scale2', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.1', timm.models.metaformer.MetaFormerBlock),
 ('stages.0.blocks.1.norm1', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.1.token_mixer', timm.models.metaformer.Pooling),
 ('stages.0.blocks.1.token_mixer.pool', torch.nn.modules.pooling.AvgPool2d),
 ...
 ('head.global_pool.flatten', torch.nn.modules.linear.Identity),
 ('head.norm', timm.layers.norm.LayerNorm2d),
 ('head.flatten', torch.nn.modules.flatten.Flatten),
 ('head.drop', torch.nn.modules.linear.Identity),
 ('head.fc', torch.nn.modules.linear.Linear)]
 ]
```

Upon closer inspection, we see that the 2D conv layers have names such as `"stages.0.blocks.0.mlp.fc1"` and
`"stages.0.blocks.0.mlp.fc2"`. How can we match those layer names specifically? You can write a [regular
expressions](https://docs.python.org/3/library/re.html) to match the layer names. For our case, the regex
`r".*\.mlp\.fc\d"` should do the job.

Furthermore, as in the first example, we should ensure that the output layer, in this case the classification head, is
also updated. Looking at the end of the list printed above, we can see that it's named `'head.fc'`. With that in mind,
here is our LoRA config:

```python
config = LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=["head.fc"])
```

Then we only need to create the PEFT model by passing our base model and the config to `get_peft_model`:

```python
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# prints trainable params: 1,064,454 || all params: 56,467,974 || trainable%: 1.88505789139876
```

This shows us that we only need to train less than 2% of all parameters, which is a huge efficiency gain.

For a complete example, check out [this notebook](https://github.com/huggingface/peft/blob/main/examples/image_classification/image_classification_timm_peft_lora.ipynb).

## New transformers architectures

When new popular transformers architectures are released, we do our best to quickly add them to PEFT. If you come across a transformers model that is not supported out of the box, don't worry, it will most likely still work if the config is set correctly. Specifically, you have to identify the layers that should be adapted and set them correctly when initializing the corresponding config class, e.g. `LoraConfig`. Here are some tips to help with this.

As a first step, it is a good idea is to check the existing models for inspiration. You can find them inside of [constants.py](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py) in the PEFT repository. Often, you'll find a similar architecture that uses the same names. For example, if the new model architecture is a variation of the "mistral" model and you want to apply LoRA, you can see that the entry for "mistral" in `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` contains `["q_proj", "v_proj"]`. This tells you that for "mistral" models, the `target_modules` for LoRA should be `["q_proj", "v_proj"]`:

```python
from peft import LoraConfig, get_peft_model

my_mistral_model = ...
config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    ...,  # other LoRA arguments
)
peft_model = get_peft_model(my_mistral_model, config)
```

If that doesn't help, check the existing modules in your model architecture with the `named_modules` method and try to identify the attention layers, especially the key, query, and value layers. Those will often have names such as `c_attn`, `query`, `q_proj`, etc. The key layer is not always adapted, and ideally, you should check whether including it results in better performance.

Additionally, linear layers are common targets to be adapted (e.g. in [QLoRA paper](https://arxiv.org/abs/2305.14314), authors suggest to adapt them as well). Their names will often contain the strings `fc` or `dense`.

If you want to add a new model to PEFT, please create an entry in [constants.py](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py) and open a pull request on the [repository](https://github.com/huggingface/peft/pulls). Don't forget to update the [README](https://github.com/huggingface/peft#models-support-matrix) as well.

## Verify parameters and layers

You can verify whether you've correctly applied a PEFT method to your model in a few ways.

* Check the fraction of parameters that are trainable with the [`~PeftModel.print_trainable_parameters`] method. If this number is lower or higher than expected, check the model `repr` by printing the model. This shows the names of all the layer types in the model. Ensure that only the intended target layers are replaced by the adapter layers. For example, if LoRA is applied to `nn.Linear` layers, then you should only see `lora.Linear` layers being used.

```py
peft_model.print_trainable_parameters()
```

* Another way you can view the adapted layers is to use the `targeted_module_names` attribute to list the name of each module that was adapted.

```python
print(peft_model.targeted_module_names)
```

## Unsupported module types

Methods like LoRA only work if the target modules are supported by PEFT. For example, it's possible to apply LoRA to `nn.Linear` and `nn.Conv2d` layers, but not, for instance, to `nn.LSTM`. If you find a layer class you want to apply PEFT to is not supported, you can:

 - define a custom mapping to dynamically dispatch custom modules in LoRA
 -  open an [issue](https://github.com/huggingface/peft/issues) and request the feature where maintainers will implement it or guide you on how to implement it yourself if demand for this module type is sufficiently high

### Experimental support for dynamic dispatch of custom modules in LoRA

> [!WARNING]
> This feature is experimental and subject to change, depending on its reception by the community. We will introduce a public and stable API if there is significant demand for it.

PEFT supports an experimental API for custom module types for LoRA. Let's assume you have a LoRA implementation for LSTMs. Normally, you would not be able to tell PEFT to use it, even if it would theoretically work with PEFT. However, this is possible with dynamic dispatch of custom layers.

The experimental API currently looks like this:

```python
class MyLoraLSTMLayer:
    ...

base_model = ...  # load the base model that uses LSTMs

# add the LSTM layer names to target_modules
config = LoraConfig(..., target_modules=["lstm"])
# define a mapping from base layer type to LoRA layer type
custom_module_mapping = {nn.LSTM: MyLoraLSTMLayer}
# register the new mapping
config._register_custom_module(custom_module_mapping)
# after registration, create the PEFT model
peft_model = get_peft_model(base_model, config)
# do training
```

<Tip>

When you call [`get_peft_model`], you will see a warning because PEFT does not recognize the targeted module type. In this case, you can ignore this warning.

</Tip>

By supplying a custom mapping, PEFT first checks the base model's layers against the custom mapping and dispatches to the custom LoRA layer type if there is a match. If there is no match, PEFT checks the built-in LoRA layer types for a match.

Therefore, this feature can also be used to override existing dispatch logic, e.g. if you want to use your own LoRA layer for `nn.Linear` instead of using the one provided by PEFT.

When creating your custom LoRA module, please follow the same rules as the [existing LoRA modules](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py). Some important constraints to consider:

- The custom module should inherit from `nn.Module` and `peft.tuners.lora.layer.LoraLayer`.
- The `__init__` method of the custom module should have the positional arguments `base_layer` and `adapter_name`. After this, there are additional `**kwargs` that you are free to use or ignore.
- The learnable parameters should be stored in an `nn.ModuleDict` or `nn.ParameterDict`, where the key corresponds to the name of the specific adapter (remember that a model can have more than one adapter at a time).
- The name of these learnable parameter attributes should start with `"lora_"`, e.g. `self.lora_new_param = ...`.
- Some methods are optional, e.g. you only need to implement `merge` and `unmerge` if you want to support weight merging.

Currently, the information about the custom module does not persist when you save the model. When loading the model, you have to register the custom modules again.

```python
# saving works as always and includes the parameters of the custom modules
peft_model.save_pretrained(<model-path>)

# loading the model later:
base_model = ...
# load the LoRA config that you saved earlier
config = LoraConfig.from_pretrained(<model-path>)
# register the custom module again, the same way as the first time
custom_module_mapping = {nn.LSTM: MyLoraLSTMLayer}
config._register_custom_module(custom_module_mapping)
# pass the config instance to from_pretrained:
peft_model = PeftModel.from_pretrained(model, tmp_path / "lora-custom-module", config=config)
```

If you use this feature and find it useful, or if you encounter problems, let us know by creating an issue or a discussion on GitHub. This allows us to estimate the demand for this feature and add a public API if it is sufficiently high.
