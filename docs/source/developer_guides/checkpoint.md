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

# PEFT checkpoint format

This document describes how PEFT's checkpoint files are structured and how to convert between the PEFT format and other formats.

## PEFT files

PEFT (parameter-efficient fine-tuning) methods only update a small subset of a model's parameters rather than all of them. This is nice because checkpoint files can generally be much smaller than the original model files and are easier to store and share. However, this also means that to load a PEFT model, you need to have the original model available as well.

When you call [`~PeftModel.save_pretrained`] on a PEFT model, the PEFT model saves three files, described below:

1. `adapter_model.safetensors` or `adapter_model.bin`

By default, the model is saved in the `safetensors` format, a secure alternative to the `bin` format, which is known to be susceptible to [security vulnerabilities](https://huggingface.co/docs/hub/security-pickle) because it uses the pickle utility under the hood. Both formats store the same `state_dict` though, and are interchangeable.

The `state_dict` only contains the parameters of the adapter module, not the base model. To illustrate the difference in size, a normal BERT model requires ~420MB of disk space, whereas an IA³ adapter on top of this BERT model only requires ~260KB.

2. `adapter_config.json`

The `adapter_config.json` file contains the configuration of the adapter module, which is necessary to load the model. Below is an example of an `adapter_config.json` for an IA³ adapter with standard settings applied to a BERT model:

```json
{
  "auto_mapping": {
    "base_model_class": "BertModel",
    "parent_library": "transformers.models.bert.modeling_bert"
  },
  "base_model_name_or_path": "bert-base-uncased",
  "fan_in_fan_out": false,
  "feedforward_modules": [
    "output.dense"
  ],
  "inference_mode": true,
  "init_ia3_weights": true,
  "modules_to_save": null,
  "peft_type": "IA3",
  "revision": null,
  "target_modules": [
    "key",
    "value",
    "output.dense"
  ],
  "task_type": null
}
```

The configuration file contains:

- the adapter module type stored, `"peft_type": "IA3"`
- information about the base model like `"base_model_name_or_path": "bert-base-uncased"`
- the revision of the model (if any), `"revision": null`

If the base model is not a pretrained Transformers model, the latter two entries will be `null`. Other than that, the settings are all related to the specific IA³ adapter that was used to fine-tune the model.

3. `README.md`

The generated `README.md` is the model card of a PEFT model and contains a few pre-filled entries. The intent of this is to make it easier to share the model with others and to provide some basic information about the model. This file is not needed to load the model.

## Convert to PEFT format

When converting from another format to the PEFT format, we require both the `adapter_model.safetensors` (or `adapter_model.bin`) file and the `adapter_config.json` file.

### adapter_model

For the model weights, it is important to use the correct mapping from parameter name to value for PEFT to load the file. Getting this mapping right is an exercise in checking the implementation details, as there is no generally agreed upon format for PEFT adapters.

Fortunately, figuring out this mapping is not overly complicated for common base cases. Let's look at a concrete example, the [`LoraLayer`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py):

```python
# showing only part of the code

class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs
```

In the `__init__` code used by all `LoraLayer` classes in PEFT, there are a bunch of parameters used to initialize the model, but only a few are relevant for the checkpoint file: `lora_A`, `lora_B`, `lora_embedding_A`, and `lora_embedding_B`. These parameters are listed in the class attribute `adapter_layer_names` and contain the learnable parameters, so they must be included in the checkpoint file. All the other parameters, like the rank `r`, are derived from the `adapter_config.json` and must be included there (unless the default value is used).

Let's check the `state_dict` of a PEFT LoRA model applied to BERT. When printing the first five keys using the default LoRA settings (the remaining keys are the same, just with different layer numbers), we get:

- `base_model.model.encoder.layer.0.attention.self.query.lora_A.weight` 
- `base_model.model.encoder.layer.0.attention.self.query.lora_B.weight` 
- `base_model.model.encoder.layer.0.attention.self.value.lora_A.weight` 
- `base_model.model.encoder.layer.0.attention.self.value.lora_B.weight` 
- `base_model.model.encoder.layer.1.attention.self.query.lora_A.weight`
- etc.

Let's break this down:

- By default, for BERT models, LoRA is applied to the `query` and `value` layers of the attention module. This is why you see `attention.self.query` and `attention.self.value` in the key names for each layer.
- LoRA decomposes the weights into two low-rank matrices, `lora_A` and `lora_B`. This is where `lora_A` and `lora_B` come from in the key names.
- These LoRA matrices are implemented as `nn.Linear` layers, so the parameters are stored in the `.weight` attribute (`lora_A.weight`, `lora_B.weight`).
- By default, LoRA isn't applied to BERT's embedding layer, so there are _no entries_ for `lora_A_embedding` and `lora_B_embedding`.
- The keys of the `state_dict` always start with `"base_model.model."`. The reason is that, in PEFT, we wrap the base model inside a tuner-specific model (`LoraModel` in this case), which itself is wrapped in a general PEFT model (`PeftModel`). For this reason, these two prefixes are added to the keys. When converting to the PEFT format, it is required to add these prefixes.

<Tip>

This last point is not true for prefix tuning techniques like prompt tuning. There, the extra embeddings are directly stored in the `state_dict` without any prefixes added to the keys.

</Tip>

When inspecting the parameter names in the loaded model, you might be surprised to find that they look a bit different, e.g. `base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight`. The difference is the *`.default`* part in the second to last segment. This part exists because PEFT generally allows the addition of multiple adapters at once (using an `nn.ModuleDict` or `nn.ParameterDict` to store them). For example, if you add another adapter called "other", the key for that adapter would be `base_model.model.encoder.layer.0.attention.self.query.lora_A.other.weight`.

When you call [`~PeftModel.save_pretrained`], the adapter name is stripped from the keys. The reason is that the adapter name is not an important part of the model architecture; it is just an arbitrary name. When loading the adapter, you could choose a totally different name, and the model would still work the same way. This is why the adapter name is not stored in the checkpoint file.

<Tip>

If you call `save_pretrained("some/path")` and the adapter name is not `"default"`, the adapter is stored in a sub-directory with the same name as the adapter. So if the name is "other", it would be stored inside of `some/path/other`.

</Tip>

In some circumstances, deciding which values to add to the checkpoint file can become a bit more complicated. For example, in PEFT, DoRA is implemented as a special case of LoRA. If you want to convert a DoRA model to PEFT, you should create a LoRA checkpoint with extra entries for DoRA. You can see this in the `__init__` of the previous `LoraLayer` code:

```python
self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
```

This indicates that there is an optional extra parameter per layer for DoRA.

### adapter_config

All the other information needed to load a PEFT model is contained in the `adapter_config.json` file. Let's check this file for a LoRA model applied to BERT:

```json
{
  "alpha_pattern": {},
  "auto_mapping": {
    "base_model_class": "BertModel",
    "parent_library": "transformers.models.bert.modeling_bert"
  },
  "base_model_name_or_path": "bert-base-uncased",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 8,
  "lora_dropout": 0.0,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 8,
  "rank_pattern": {},
  "revision": null,
  "target_modules": [
    "query",
    "value"
  ],
  "task_type": null,
  "use_dora": false,
  "use_rslora": false
}
```

This contains a lot of entries, and at first glance, it could feel overwhelming to figure out all the right values to put in there. However, most of the entries are not necessary to load the model. This is either because they use the default values and don't need to be added or because they only affect the initialization of the LoRA weights, which is irrelevant when it comes to loading the model. If you find that you don't know what a specific parameter does, e.g., `"use_rslora",` don't add it, and you should be fine. Also note that as more options are added, this file will get more entries in the future, but it should be backward compatible.

At the minimum, you should include the following entries:

```json
{
  "target_modules": ["query", "value"],
  "peft_type": "LORA"
}
```

However, adding as many entries as possible, like the rank `r` or the `base_model_name_or_path` (if it's a Transformers model) is recommended. This information can help others understand the model better and share it more easily. To check which keys and values are expected, check out the [config.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py) file (as an example, this is the config file for LoRA) in the PEFT source code.

## Model storage

In some circumstances, you might want to store the whole PEFT model, including the base weights. This can be necessary if, for instance, the base model is not available to the users trying to load the PEFT model. You can merge the weights first or convert it into a Transformer model.

### Merge the weights

The most straightforward way to store the whole PEFT model is to merge the adapter weights into the base weights:

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained(...)
```

There are some disadvantages to this approach, though:

- Once [`~LoraModel.merge_and_unload`] is called, you get a basic model without any PEFT-specific functionality. This means you can't use any of the PEFT-specific methods anymore.
- You cannot unmerge the weights, load multiple adapters at once, disable the adapter, etc.
- Not all PEFT methods support merging weights.
- Some PEFT methods may generally allow merging, but not with specific settings (e.g. when using certain quantization techniques).
- The whole model will be much larger than the PEFT model, as it will contain all the base weights as well.

But inference with a merged model should be a bit faster.

### Convert to a Transformers model

Another way to save the whole model, assuming the base model is a Transformers model, is to use this hacky approach to directly insert the PEFT weights into the base model and save it, which only works if you "trick" Transformers into believing the PEFT model is not a PEFT model. This only works with LoRA because other adapters are not implemented in Transformers.

```python
model = ...  # the PEFT model
...
# after you finish training the model, save it in a temporary location
model.save_pretrained(<temp_location>)
# now load this model directly into a transformers model, without the PEFT wrapper
# the PEFT weights are directly injected into the base model
model_loaded = AutoModel.from_pretrained(<temp_location>)
# now make the loaded model believe that it is _not_ a PEFT model
model_loaded._hf_peft_config_loaded = False
# now when we save it, it will save the whole model
model_loaded.save_pretrained(<final_location>)
# or upload to Hugging Face Hub
model_loaded.push_to_hub(<final_location>)
```

