from loralib import lora_state_dict

from .config import PETType


def get_pet_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the PET model.

    Args:
        model (:obj:`PETModel`): The PET model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be teh underlying model/unwrapped model (i.e. model.module).
        state_dict (:
            obj:`dict`, `optional`): The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    if state_dict is None:
        state_dict = model.state_dict()
    if model.pet_config.pet_type == PETType.LORA:
        to_return = lora_state_dict(model, bias=model.pet_config.bias)
    else:
        to_return = {}
        prompt_embeddings = model.get_prompt_embedding_to_save()
        to_return["prompt_embeddings"] = prompt_embeddings
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                to_return[key] = value
    return to_return


def set_pet_model_state_dict(model, pet_model_state_dict):
    """
    Set the state dict of the PET model.

    Args:
        model (:obj:`PETModel`): The PET model.
        pet_model_state_dict (:obj:`dict`): The state dict of the PET model.
    """

    model.load_state_dict(pet_model_state_dict, strict=False)
    if model.pet_config.pet_type != PETType.LORA:
        model.prompt_encoder.embedding.load_state_dict(
            {"weight": pet_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return model


def pet_model_load_and_dispatch(model, pet_model_state_dict, pet_config, max_memory=None):
    """
    Load the PET model state dict and dispatch the model to the correct device.

    Args:
        model (:obj:`PETModel`): The Pre-trained base model which has already been sharded and dispatched
        using `accelerate` functionalities.
        pet_model_state_dict (:obj:`dict`): The state dict of the PET model.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
    """
    from accelerate import infer_auto_device_map, dispatch_model
    from accelerate.hooks import remove_hook_from_submodules, AlignDevicesHook, add_hook_to_module
    from ..mapping import get_pet_model

    remove_hook_from_submodules(model)
    model = get_pet_model(model, pet_config)
    model.print_trainable_parameters()
    set_pet_model_state_dict(model, pet_model_state_dict)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=model._no_split_modules)
    model = dispatch_model(model, device_map=device_map)
    hook = AlignDevicesHook(io_same_device=True)
    if model.pet_config.pet_type == PETType.LORA:
        add_hook_to_module(model.base_model.model, hook)
    else:
        add_hook_to_module(model.base_model, hook)
    return model
