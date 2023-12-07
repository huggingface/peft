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

# Working with custom models

Some fine-tuning techniques, such as prompt tuning, are specific to language models. That means in 🤗 PEFT, it is
assumed a 🤗 Transformers model is being used. However, other fine-tuning techniques - like
[LoRA](../conceptual_guides/lora) - are not restricted to specific model types.

In this guide, we will see how LoRA can be applied to a multilayer perceptron and a computer vision model from the [timm](https://huggingface.co/docs/timm/index) library.

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
print([(n, type(m)) for n, m in MLP().named_modules()])
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
