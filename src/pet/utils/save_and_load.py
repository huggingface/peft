from loralib import lora_state_dict

from .config import PETType


def get_pet_model_state_dict(model):
    """
    Get the state dict of the PET model.

    Args:
        model (:obj:`PETModel`): The PET model.
    """

    if model.pet_config.pet_type == PETType.LORA:
        return lora_state_dict(model)
    else:
        to_return = {}
        state_dict = model.state_dict()
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
