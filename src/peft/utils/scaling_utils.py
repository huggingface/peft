from contextlib import contextmanager
from peft import LoraModel
from peft.tuners.lora.layer import LoraLayer

@contextmanager
def set_adapter_scale(model, alpha):
    # 1. TODO: Check whether scaling is prohibited on model
    print("Checking ...")

    # 2. Modify scaling values
    original_scaling = {}
    print("Scaling ...")
    for module in model.modules():
        if isinstance(module, LoraLayer):
            original_scaling[module] = module.scaling.copy()
            module.scaling = dict((k, v * alpha) for k, v in module.scaling.items())
    yield

    # 3. Restore original scaling values after exiting the context
    print("Restoring ...")
    for module, scaling in original_scaling.items():
        module.scaling = scaling
