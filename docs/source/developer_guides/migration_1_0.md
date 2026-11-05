# Migrating to PEFT 1.0

## Adapter selection and configuration

### Activating multiple adapters

For PEFT methods that support composition, activate several adapters directly on [`PeftModel`]:

```python
# Before
model.base_model.set_adapter(["adapter0", "adapter1"])

# PEFT 1.0
model.set_adapter(["adapter0", "adapter1"])
```

This only works for PEFT methods that support multiple active adapters, thus prompt learning methods etc. reject this. Auxiliary modules impose their own restrictions: two selected adapters cannot use the same `modules_to_save` module or the same trainable-token wrapper. Adapters targeting different auxiliary modules can be active together.

### Inspecting active adapters

Use `model.active_adapters` for the complete selection. For ordinary tuners, `model.active_adapter` now consistently returns the first active adapter, including after calling the underlying tuner's setter or deleting an active adapter. This can differ from earlier PEFT versions, where it could retain an inactive or deleted name. Moreover, `model.active_adapter` is now read-only; don't use it to change the active adapter, use `model.set_adapter` instead.

### Replacing `active_peft_config`

`model.active_peft_config` is deprecated and will be removed in PEFT v1.2. It returns one representative configuration, which cannot describe the case of multiple active adapters.

Select configurations explicitly:

```python
config = model.peft_config["adapter0"]
active_configs = {name: model.peft_config[name] for name in model.active_adapters}
```

Use the model-level properties for decisions about the PEFT method:

```python
# Before
if model.active_peft_config.is_prompt_learning:
    ...

# PEFT 1.0
if model.is_prompt_learning:
    ...
```
