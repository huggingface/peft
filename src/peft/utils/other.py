# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple

import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size

from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
    CONFIG_NAME,
    EMBEDDING_LAYER_NAMES,
    INCLUDE_LINEAR_LAYERS_SHORTHAND,
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    bloom_model_postprocess_past_key_value,
    starcoder_model_postprocess_past_key_value,
)


__all__ = [
    "CONFIG_NAME",
    "EMBEDDING_LAYER_NAMES",
    "SAFETENSORS_WEIGHTS_NAME",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING",
    "WEIGHTS_NAME",
    "INCLUDE_LINEAR_LAYERS_SHORTHAND",
    "bloom_model_postprocess_past_key_value",
    "starcoder_model_postprocess_past_key_value",
]


# Get current device name based on available devices
def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    r"""
    Note this method only works for `transformers` models.

    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
        use_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments to pass to the gradient checkpointing function, please refer to the documentation of
            `torch.utils.checkpoint.checkpoint` for more details about the arguments that you can pass to that method.
            Note this is only available in the latest transformers versions (> 4.34.1).
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    is_aqlm_quantized = getattr(model, "quantization_method", None) == "aqlm"
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if not is_gptq_quantized and not is_aqlm_quantized:
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized or is_aqlm_quantized) and use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model


# For backward compatibility
def prepare_model_for_int8_training(*args, **kwargs):
    warnings.warn(
        "prepare_model_for_int8_training is deprecated and will be removed in a future version. Use prepare_model_for_kbit_training instead.",
        FutureWarning,
    )
    return prepare_model_for_kbit_training(*args, **kwargs)


# copied from transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class ModulesToSaveWrapper(torch.nn.Module):
    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = torch.nn.ModuleDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)
        self.check_module()

    def check_module(self):
        """Perform some sanity checks on the module to ensure that it works"""
        # Try to anticipate some modules that users could try to target that would not work.
        # Note: It's not possible to check hasattr(module, "forward"), since that returns True for ModuleDict and
        # ModuleList, even though their forward methods cannot be called
        forbidden_classes = (torch.nn.ModuleDict, torch.nn.ModuleList, torch.nn.ParameterDict, torch.nn.ParameterList)
        if isinstance(self.original_module, forbidden_classes):
            cls_name = self.original_module.__class__.__name__
            raise TypeError(f"modules_to_save cannot be applied to modules of type {cls_name}")

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def weight(self):
        if self.active_adapter not in self.modules_to_save:
            return self.original_module.weight
        return self.modules_to_save[self.active_adapter].weight

    def update(self, adapter_name):
        context_manager = nullcontext()
        for _, param in self.original_module.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                import deepspeed

                context_manager = deepspeed.zero.GatheredParameters(self.original_module.parameters(), modifier_rank=0)
                break
        with context_manager:
            self.modules_to_save.update(torch.nn.ModuleDict({adapter_name: copy.deepcopy(self.original_module)}))

        if hasattr(self.modules_to_save[adapter_name], "_hf_hook"):
            old_hook = self.modules_to_save[adapter_name]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[adapter_name])
            add_hook_to_module(self.modules_to_save[adapter_name], new_hook)

        self.original_module.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.modules_to_save[adapter_name].requires_grad_(True)

    def _create_new_hook(self, old_hook):
        r"""
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[k] = old_hook_attr[k]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def forward(self, *args, **kwargs):
        if self.disable_adapters or (self.active_adapter not in self.modules_to_save):
            return self.original_module(*args, **kwargs)
        return self.modules_to_save[self.active_adapter](*args, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            # already in the desired state, do nothing
            return

        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: str):
        """Set the active adapter

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(f"Adapter {adapter_name} not found in {self.modules_to_save.keys()}")

        self.modules_to_save[self.active_adapter].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def _set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
                target.set_adapter(target.active_adapter)
            else:
                new_module = ModulesToSaveWrapper(target, adapter_name)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)


def _set_adapter(model, adapter_name):
    def check_adapter_name(adapter_name):
        if isinstance(adapter_name, str):
            return adapter_name

        # adapter_name is a list of str
        if len(adapter_name) > 1:
            raise ValueError("Only one adapter can be set at a time for modules_to_save")
        elif len(adapter_name) == 0:
            raise ValueError("Please specify at least one adapter to set")
        adapter_name = adapter_name[0]
        return adapter_name

    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            # only check the adapter_name if we actually encounter a ModulesToSaveWrapper, otherwise we don't care
            adapter_name = check_adapter_name(adapter_name)
            module.set_adapter(adapter_name)


def _prepare_prompt_learning_config(peft_config, model_config):
    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)

    return peft_config


def fsdp_auto_wrap_policy(model):
    import functools
    import os

    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from ..tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    default_transformer_cls_names_to_wrap = (
        ",".join(model._no_split_modules) if getattr(model, "_no_split_modules", None) is not None else ""
    )
    transformer_cls_names_to_wrap = os.environ.get(
        "FSDP_TRANSFORMER_CLS_TO_WRAP", default_transformer_cls_names_to_wrap
    ).split(",")
    transformer_cls_to_wrap = {PrefixEncoder, PromptEncoder, PromptEmbedding}
    for layer_class in transformer_cls_names_to_wrap:
        transformer_cls = FullyShardedDataParallelPlugin.get_module_class_from_name(model, layer_class)
        if transformer_cls is None:
            raise Exception("Could not find the transformer layer class to wrap in the model.")
        else:
            transformer_cls_to_wrap.add(transformer_cls)

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


def _is_valid_match(key: str, target_key: str):
    """
    Helper function to match module names target_key and key. Makes sure that either the key is exactly the target_key
    or the target_key is a submodule of key
    """
    if key.endswith(target_key):
        if len(key) > len(target_key):
            return key.endswith("." + target_key)  # must be a sub module
        return True
    return False


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


def get_quantization_config(model: torch.nn.Module, method: str):
    """
    Get the quantization config of the related quantization method
    """
    if (
        hasattr(model, "config")
        and hasattr(model.config, "quantization_config")
        and (getattr(model, "quantization_method", None) == method)
    ):
        return model.config.quantization_config
    return None


def get_auto_gptq_quant_linear(gptq_quantization_config):
    """
    Get the right AutoGPTQQuantLinear class based on the quantization config file
    """
    if gptq_quantization_config is not None and is_auto_gptq_available():
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

        desc_act = gptq_quantization_config.desc_act
        group_size = gptq_quantization_config.group_size
        bits = gptq_quantization_config.bits
        if hasattr(gptq_quantization_config, "use_exllama"):
            use_exllama = gptq_quantization_config.use_exllama
        else:
            use_exllama = not gptq_quantization_config.disable_exllama
        if hasattr(gptq_quantization_config, "exllama_config"):
            exllama_version = gptq_quantization_config.exllama_config["version"]
        else:
            exllama_version = 1
        AutoGPTQQuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=desc_act,
            group_size=group_size,
            bits=bits,
            disable_exllama=not (use_exllama and exllama_version == 1),
            disable_exllamav2=not (use_exllama and exllama_version == 2),
        )
        return AutoGPTQQuantLinear
    return None


def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.

    This method is the exact same copy of
    https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L282C1-L300C58 but we added
    it here manually to avoid import issue with old versions of transformers.
    """
    if tensor.device.type == "xla" and is_torch_tpu_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)


def cast_mixed_precision_params(model, dtype):
    """
    Cast all non-trainable parameters of the model to the given `dtype`. The `dtype` can be `torch.float16` or
    `torch.bfloat16` as per the mixed-precision training you are performing. The trainable parameters are cast to full
    precision. This is meant to reduce the GPU memory usage when using PEFT methods by using half-precision dtype for
    non-trainable parameters. Having the trainable parameters in full-precision preserves training stability when using
    automatic mixed-precision training.

    Args:
        model (`torch.nn.Module`):
            The model to cast the non-trainable parameters of.
        dtype (`torch.dtype`):
            The dtype to cast the non-trainable parameters to. The `dtype` can be `torch.float16` or
    `torch.bfloat16` as per the mixed-precision training you are performing.
    """
    for p in model.parameters():
        if not p.requires_grad:
            p.data = p.to(dtype)
        else:
            p.data = p.to(torch.float32)


def str_to_bool(value: str) -> int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    # same as function as in accelerate.utils, which replaces the deprecated distutils.util.strtobool
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {value}")


def check_file_exists_on_hf_hub(repo_id: str, filename: str, **kwargs) -> Optional[bool]:
    """Check if a file exists on HF Hub, if check was not successful returns None instead of erroring.

    Respect offline mode if set.

    """
    exists: Optional[bool] = None
    if str_to_bool(os.environ.get("HF_HUB_OFFLINE", "0")):
        # user set offline mode, cannot check
        return exists

    try:
        exists = file_exists(repo_id, filename, **kwargs)
    except (HFValidationError, EntryNotFoundError):
        # error, exists stays None
        pass
    except Exception as e:
        warnings.warn(
            f"Unable to fetch remote file due to the following error {e} - silently ignoring the lookup"
            f" for the file {filename} in {repo_id}."
        )

    return exists
