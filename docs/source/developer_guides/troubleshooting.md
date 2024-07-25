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

# Troubleshooting

If you encounter any issue when using PEFT, please check the following list of common issues and their solutions.

## Examples don't work

Examples often rely on the most recent package versions, so please ensure they're up-to-date. In particular, check the following package versions:

- `peft`
- `transformers`
- `accelerate`
- `torch`

In general, you can update the package version by running this command inside your Python environment:

```bash
python -m pip install -U <package_name>
```

Installing PEFT from source is useful for keeping up with the latest developments:

```bash
python -m pip install git+https://github.com/huggingface/peft
```

## ValueError: Attempting to unscale FP16 gradients

This error probably occurred because the model was loaded with `torch_dtype=torch.float16` and then used in an automatic mixed precision (AMP) context, e.g. by setting `fp16=True` in the [`~transformers.Trainer`] class from 🤗 Transformers. The reason is that when using AMP, trainable weights should never use fp16. To make this work without loading the whole model in fp32, add the following to your code:

```python
peft_model = get_peft_model(...)

# add this:
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()

# proceed as usual
trainer = Trainer(model=peft_model, fp16=True, ...)
trainer.train()
```

Alternatively, you can use the [`~utils.cast_mixed_precision_params`] function to correctly cast the weights:

```python
from peft import cast_mixed_precision_params

peft_model = get_peft_model(...)
cast_mixed_precision_params(peft_model, dtype=torch.float16)

# proceed as usual
trainer = Trainer(model=peft_model, fp16=True, ...)
trainer.train()
```

<Tip>

Starting from PEFT verion v0.12.0, PEFT automatically promotes the dtype of adapter weights from `torch.float16` and `torch.bfloat16` to `torch.float32` where appropriate. To _prevent_ this behavior, you can pass `autocast_adapter_dtype=False` to [`~get_peft_model`], to [`~PeftModel.from_pretrained`], and to [`~PeftModel.load_adapter`].

</Tip>

## Bad results from a loaded PEFT model

There can be several reasons for getting a poor result from a loaded PEFT model which are listed below. If you're still unable to troubleshoot the problem, see if anyone else had a similar [issue](https://github.com/huggingface/peft/issues) on GitHub, and if you can't find any, open a new issue.

When opening an issue, it helps a lot if you provide a minimal code example that reproduces the issue. Also, please report if the loaded model performs at the same level as the model did before fine-tuning, if it performs at a random level, or if it is only slightly worse than expected. This information helps us identify the problem more quickly.

### Random deviations

If your model outputs are not exactly the same as previous runs, there could be an issue with random elements. For example:

1. please ensure it is in `.eval()` mode, which is important, for instance, if the model uses dropout
2. if you use [`~transformers.GenerationMixin.generate`] on a language model, there could be random sampling, so obtaining the same result requires setting a random seed
3. if you used quantization and merged the weights, small deviations are expected due to rounding errors

### Incorrectly loaded model

Please ensure that you load the model correctly. A common error is trying to load a _trained_ model with [`get_peft_model`] which is incorrect. Instead, the loading code should look like this:

```python
from peft import PeftModel, PeftConfig

base_model = ...  # to load the base model, use the same code as when you trained it
config = PeftConfig.from_pretrained(peft_model_id)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
```

### Randomly initialized layers

For some tasks, it is important to correctly configure `modules_to_save` in the config to account for randomly initialized layers. 

As an example, this is necessary if you use LoRA to fine-tune a language model for sequence classification because 🤗 Transformers adds a randomly initialized classification head on top of the model. If you do not add this layer to `modules_to_save`, the classification head won't be saved. The next time you load the model, you'll get a _different_ randomly initialized classification head, resulting in completely different results.

PEFT tries to correctly guess the `modules_to_save` if you provide the `task_type` argument in the config. This should work for transformers models that follow the standard naming scheme. It is always a good idea to double check though because we can't guarantee all models follow the naming scheme.

When you load a transformers model that has randomly initialized layers, you should see a warning along the lines of:

```
Some weights of <MODEL> were not initialized from the model checkpoint at <ID> and are newly initialized: [<LAYER_NAMES>].
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

The mentioned layers should be added to `modules_to_save` in the config to avoid the described problem.

### Extending the vocabulary

For many language fine-tuning tasks, extending the model's vocabulary is necessary since new tokens are being introduced. This requires extending the embedding layer to account for the new tokens and also storing the embedding layer in addition to the adapter weights when saving the adapter.

Save the embedding layer by adding it to the `target_modules` of the config. The embedding layer name must follow the standard naming scheme from Transformers. For example, the Mistral config could look like this:

```python
config = LoraConfig(..., target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"])
```

Once added to `target_modules`, PEFT automatically stores the embedding layer when saving the adapter if the model has the [`~transformers.PreTrainedModel.get_input_embeddings`] and [`~transformers.PreTrainedModel.get_output_embeddings`]. This is generally the case for Transformers models.

If the model's embedding layer doesn't follow the Transformer's naming scheme, you can still save it by manually passing `save_embedding_layers=True` when saving the adapter:

```python
model = get_peft_model(...)
# train the model
model.save_pretrained("my_adapter", save_embedding_layers=True)
```

For inference, load the base model first and resize it the same way you did before you trained the model. After you've resized the base model, you can load the PEFT checkpoint.

For a complete example, please check out [this notebook](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_with_additional_tokens.ipynb).

### Check layer and model status

Sometimes a PEFT model can end up in a bad state, especially when handling multiple adapters. There can be some confusion around what adapters exist, which one is active, which one is merged, etc. To help investigate this issue, call the [`~peft.PeftModel.get_layer_status`] and the [`~peft.PeftModel.get_model_status`] methods. 

The [`~peft.PeftModel.get_layer_status`] method gives you a detailed overview of each targeted layer's active, merged, and available adapters.

```python
>>> from transformers import AutoModel
>>> from peft import get_peft_model, LoraConfig

>>> model_id = "google/flan-t5-small"
>>> model = AutoModel.from_pretrained(model_id)
>>> model = get_peft_model(model, LoraConfig())

>>> model.get_layer_status()
[TunerLayerStatus(name='model.encoder.block.0.layer.0.SelfAttention.q',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['default'],
                  merged_adapters=[],
                  requires_grad={'default': True},
                  available_adapters=['default']),
 TunerLayerStatus(name='model.encoder.block.0.layer.0.SelfAttention.v',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['default'],
                  merged_adapters=[],
                  requires_grad={'default': True},
                  available_adapters=['default']),
...]

>>> model.get_model_status()
TunerModelStatus(
    base_model_type='T5Model',
    adapter_model_type='LoraModel',
    peft_types={'default': 'LORA'},
    trainable_params=344064,
    total_params=60855680,
    num_adapter_layers=48,
    enabled=True,
    active_adapters=['default'],
    merged_adapters=[],
    requires_grad={'default': True},
    available_adapters=['default'],
)
```

In the model state output, you should look out for entries that say `"irregular"`. This means PEFT detected an inconsistent state in the model. For instance, if `merged_adapters="irregular"`, it means that for at least one adapter, it was merged on some target modules but not on others. The inference results will most likely be incorrect as a result.

The best way to resolve this issue is to reload the whole model and adapter checkpoint(s). Ensure that you don't perform any incorrect operations on the model, e.g. manually merging adapters on some modules but not others.

Convert the layer status into a pandas `DataFrame` for an easier visual inspection.

```python
from dataclasses import asdict
import pandas as pd

df = pd.DataFrame(asdict(layer) for layer in model.get_layer_status())
```

It is possible to get this information for non-PEFT models if they are using PEFT layers under the hood, but some information like the `base_model_type` or the `peft_types` cannot be determined in that case. As an example, you can call this on a [diffusers](https://huggingface.co/docs/diffusers/index) model like so:

```python
>>> import torch
>>> from diffusers import StableDiffusionPipeline
>>> from peft import get_model_status, get_layer_status

>>> path = "runwayml/stable-diffusion-v1-5"
>>> lora_id = "takuma104/lora-test-text-encoder-lora-target"
>>> pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
>>> pipe.load_lora_weights(lora_id, adapter_name="adapter-1")
>>> pipe.load_lora_weights(lora_id, adapter_name="adapter-2")
>>> pipe.set_lora_device(["adapter-2"], "cuda")
>>> get_layer_status(pipe.text_encoder)
[TunerLayerStatus(name='text_model.encoder.layers.0.self_attn.k_proj',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['adapter-2'],
                  merged_adapters=[],
                  requires_grad={'adapter-1': False, 'adapter-2': True},
                  available_adapters=['adapter-1', 'adapter-2'],
                  devices={'adapter-1': ['cpu'], 'adapter-2': ['cuda']}),
 TunerLayerStatus(name='text_model.encoder.layers.0.self_attn.v_proj',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['adapter-2'],
                  merged_adapters=[],
                  requires_grad={'adapter-1': False, 'adapter-2': True},
                  devices={'adapter-1': ['cpu'], 'adapter-2': ['cuda']}),
...]

>>> get_model_status(pipe.unet)
TunerModelStatus(
    base_model_type='other',
    adapter_model_type='None',
    peft_types={},
    trainable_params=797184,
    total_params=861115332,
    num_adapter_layers=128,
    enabled=True,
    active_adapters=['adapter-2'],
    merged_adapters=[],
    requires_grad={'adapter-1': False, 'adapter-2': True},
    available_adapters=['adapter-1', 'adapter-2'],
    devices={'adapter-1': ['cpu'], 'adapter-2': ['cuda']},
)
```

## Reproducibility

### Models using batch norm

When loading a trained PEFT model where the base model uses batch norm (e.g. `torch.nn.BatchNorm1d` or `torch.nn.BatchNorm2d`), you may find that you cannot reproduce the exact same outputs. This is because the batch norm layers keep track of running stats during training, but these stats are not part of the PEFT checkpoint. Therefore, when you load the PEFT model, the running stats of the base model will be used (i.e. from before training with PEFT).

Depending on your use case, this may not be a big deal. If, however, you need your outputs to be 100% reproducible, you can achieve this by adding the batch norm layers to `modules_to_save`. Below is an example of this using resnet and LoRA. Notice that we set `modules_to_save=["classifier", "normalization"]`. We need the `"classifier"` argument because our task is image classification, and we add the `"normalization"` argument to ensure that the batch norm layers are saved in the PEFT checkpoint.

```python
from transformers import AutoModelForImageClassification
from peft import LoraConfig, get_peft_model

model_id = "microsoft/resnet-18"
base_model = AutoModelForImageClassification.from_pretrained(self.model_id)
config = LoraConfig(
    target_modules=["convolution"],
    modules_to_save=["classifier", "normalization"],
),
```

Depending on the type of model you use, the batch norm layers could have different names than `"normalization"`, so please ensure that the name matches your model architecture.
