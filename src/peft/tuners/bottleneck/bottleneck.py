import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.config import PeftConfig
from peft.utils import PeftType
from transformers.activations import ACT2FN


TRANSFORMERS_MODELS_TO_ADAPTER_TYPE_MAPPING = {
    "bloom": {"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
    "gptj": {"fc_in":"mh_adapter", "fc_out":"output_adapter"},
    "gpt_neo": {"c_fc":"mh_adapter", "c_proj":"output_adapter"},
    "llama": {"gate_proj": "mh_adapter", "up_proj":"mh_adapter", "down_proj":"output_adapter"},
    "qwen": {"gate_proj": "mh_adapter", "up_proj":"mh_adapter", "down_proj":"output_adapter"},
    "qwen2": {"gate_proj": "mh_adapter", "up_proj":"mh_adapter", "down_proj":"output_adapter"},
    "opt": {"fc1":"mh_adapter", "fc2":"output_adapter"},
    "chatglm": {"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
}

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class BottleneckConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Bottleneck`].

    Args:
        bottleneck_size (`int`): The size of the bottleneck.
        non_linearity (`str`): The non-linearity to apply to the bottleneck.
        dropout (`float`, optional): The dropout probability of the bottleneck. Default to 0.0
        bias ('str'): Bias type for Bottleneck. Can be 'none', 'all' or 'adapter_only'. Default to 'none'.
        use_parallel_adapter (:obj:`bool`, optional): Whether to use parallel adapter. Defaults to False.
        scaling (:obj:`float` or :obj:`str`, optional):
            Scaling factor to use for scaled addition of adapter outputs as done by He et al. (2021). Can be either a
            constant factor (float) or the string "learned", in which case the scaling factor is learned. Defaults to
            1.0.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Adapter to.
        init_weights (:obj:`str`, optional): Initialization method for the weights of the adapter modules.
            Currently, this can be either "bert" (default) or "mam_adapter".
        modules_to_save (`List[str]`):List of modules apart from Bottleneck adapter layers to be set as trainable
            and saved in the final checkpoint.
    """
    bottleneck_size : int = field(default=256, metadata={"help": "The size of the bottleneck"})
    non_linearity : str = field(default="tanh", metadata={"help": "The non-linearity to apply to the bottleneck"})
    adapter_dropout : float = field(default=0.0, metadata={"help": "The dropout probability of the bottleneck, default to 0.0"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Adapter."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    use_parallel_adapter: bool = field(default=False, metadata={"help": "Whether to use parallel adapter"})
    use_adapterp: bool = field(default=False, metadata={"help": "Whether to use adapterp"})
    scaling: Union[float, str] = 1.0
    bias: str = field(default="none", metadata={"help": "Bias type for Bottleneck. Can be 'none', 'all' or 'adapter_only'"})
    init_weights: str = field(default="bert", metadata={"help": "Initialization method for the weights of the adapter modules."})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Adapter layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.BOTTLENECK


class BottleneckModel(torch.nn.Module):
    """
    Creates Bottleneck adapter model for a pretrained trainsformers model.

    Args:
        model ('transformers.PreTrainedModel'): The pretrained model to be adapted.
        config (`BottleneckConfig`): The configuration of the Bottleneck adapter.
    
    Returns:
        `torch.nn.Module`: The Bottleneck adapter model.
    
    Example::

        >>> from transformers import AutoModelForCausalLM, BottleneckConfig
        >>> from peft import BottleneckModel, BottleneckConfig
        >>> config = BottleneckConfig(
            peft_type="BOTTLNECK", task="CAUSAL_LM", target_modules=["gate_proj", "up_proj", "down_proj"],
            bottleneck_size=256, non_linearity="tanh",
        )
        >>> model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf") 
        >>> bottleneck_model = BottleneckModel(config, model)

    **Attribute**:
        - **model** (`transformers.PreTrainedModel`): The pretrained model to be adapted.
        - **peft_config** (`BottleneckConfig`): The configuration of the Bottleneck adapter.
    """

    def __init__(self, model, config, adapter_name="default"):
        super().__init__()
        self.model = model
        print(f"config: =========\n{config}\n================")
        self.peft_config = config
        self.adapter_name = adapter_name
        self._find_and_replace()
        mark_only_adapter_as_trainable(self.model, self.peft_config[self.adapter_name].bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Adapter with 4-bit or 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "bottleneck_size": self.peft_config[self.adapter_name].bottleneck_size,
            "non_linearity": self.peft_config[self.adapter_name].non_linearity,
            "adapter_dropout": self.peft_config[self.adapter_name].adapter_dropout,
            "scaling": self.peft_config[self.adapter_name].scaling,
            "init_weights": self.peft_config[self.adapter_name].init_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config[self.adapter_name].target_modules, str):
                target_module_found = re.fullmatch(self.peft_config[self.adapter_name].target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config[self.adapter_name].target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                # determine the type of adapter to be used, this will effect the forward pass
                if self.peft_config[self.adapter_name].use_parallel_adapter:
                    adapter_type = "parallel_adapter"
                else:
                    adapter_type = TRANSFORMERS_MODELS_TO_ADAPTER_TYPE_MAPPING[self.model.config.model_type][target_name]
                kwargs.update({"adapter_type": adapter_type})
                    
                bias = target.bias is not None
                if loaded_in_4bit and isinstance(target, bnb.nn.Linear4bit):
                    kwargs.update(
                        {
                            "compute_dtype": target.compute_dtype,
                            "compress_statistics": target.weight.compress_statistics,
                            "quant_type": target.weight.quant_type,
                        }
                    )
                    print(f"Linear4bit.")
                    if adapter_type == "mh_adapter":
                        new_module = Linear4bit(target.in_features, target.in_features, bias=bias, **kwargs)
                    elif adapter_type == "output_adapter":
                        new_module = Linear4bit(target.out_features, target.out_features, bias=bias, **kwargs)
                    elif adapter_type == "parallel_adapter":
                        new_module = Linear4bit(target.in_features, target.out_features, bias=bias, **kwargs)
                elif loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    print(f"Linear8bitLt.")
                    if adapter_type == "mh_adapter":
                        new_module = Linear8bitLt(target.in_features, target.in_features, bias=bias, **kwargs)
                    elif adapter_type == "output_adapter":
                        new_module = Linear8bitLt(target.out_features, target.out_features, bias=bias, **kwargs)
                    elif adapter_type == "parallel_adapter":
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear):
                    if adapter_type == "mh_adapter":
                        new_module = Linear(target.in_features, target.in_features, bias=bias, **kwargs)
                    elif adapter_type == "output_adapter":
                        new_module = Linear(target.out_features, target.out_features, bias=bias, **kwargs)
                    elif adapter_type == "parallel_adapter":
                        new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config[self.adapter_name].target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
    
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "adapter_" in name:
                module.to(old_module.weight.device)
        
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config[self.adapter_name]).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, AdapterLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/adapter-hub/adapter-transformers/blob/master/src/transformers/adapters/modeling.py and lora.py from huggingfance PEFT
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# Copy from lora.py
# had to adapt it for `lora_only` to work 
def mark_only_adapter_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "adapter_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "adapter_only":
        for m in model.modules():
            if isinstance(m, AdapterLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class AdapterLayer:
    def __init__(
        self,
        bottleneck_size: int,
        non_linearity: str,
        adapter_dropout: float,
        scaling: Union[float, str],
    ):
        self.bottleneck_size = bottleneck_size
        self.non_linearity = non_linearity
        self.scaling = scaling
        #optional dropout
        if adapter_dropout > 0.0:
            self.adapter_dropout = nn.Dropout(p=adapter_dropout)
        else:
            self.adapter_dropout = lambda x: x
        self.disable_adapters = False


class Linear(nn.Linear, AdapterLayer):
    """
    Bottleneck adapter in a dense layer. The adapter can be applied after the multi-head attention layer and/or
    after the feed-forward layer.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        adapter_type: str,
        bottleneck_size: int,
        non_linearity: str,
        adapter_dropout: float,
        scaling: Union[float, str],
        init_weights: str,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        AdapterLayer.__init__(self, bottleneck_size=bottleneck_size,
                                non_linearity=non_linearity,
                                adapter_dropout=adapter_dropout,
                                scaling=scaling)

        self.init_weights = init_weights
        self.adapter_type = adapter_type
        if isinstance(scaling, float):
            self.adapter_scaling = scaling
        elif scaling == "learned":
            self.adapter_scaling = nn.Parameter(torch.ones(1))
        # Actual trainable parameters
        self.adapter_down = nn.Linear(in_features, bottleneck_size, bias=False)
        self.adapter_up = nn.Linear(bottleneck_size, out_features, bias=False)
        self.act_fn = ACT2FN[self.non_linearity]
        #Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if hasattr(self, "adapter_down"):
            if self.init_weights == "bert":
                self.adapter_down.apply(self.init_bert_weights)
                self.adapter_up.apply(self.init_bert_weights)
            elif self.init_weights == "mam_adapter":
                nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.adapter_up.weight)
            else:
                raise ValueError("Unknown init_weights type: {}".format(self.config["init_weights"]))

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.adapter_down.train(mode)
        self.adapter_up.train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.adapter_down.eval()
        self.adapter_up.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            return F.linear(x, self.weight, bias=self.bias)
        else:
            if self.adapter_type == "mh_adapter":
                # for mh_adapter, x will pass the adapter first and then the linear layer
                expected_dtype = x.dtype
                residual = x

                if x.dtype != torch.float32:
                    x = x.float()
                output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))).to(expected_dtype) * self.adapter_scaling

                output = output + residual
                
                result = F.linear(output, self.weight, bias=self.bias)
            elif self.adapter_type == "output_adapter":
                # for output_adapter, x will pass the linear layer first and then the adapter
                x = F.linear(x, self.weight, bias=self.bias)
                expected_dtype = x.dtype
                residual = x

                if x.dtype != torch.float32:
                    x = x.float()

                output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))).to(expected_dtype) * self.adapter_scaling

                result = output + residual
            elif self.adapter_type == "parallel_adapter":
                # for parallel_adapter, x will pass the linear layer first and the adapter layer parallelly. 
                # The output of the adapter layer will be added to the output of the linear layer
                result = F.linear(x, self.weight, bias=self.bias)
                expected_dtype = result.dtype

                if x.dtype != torch.float32:
                    x = x.float()
                output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))).to(expected_dtype) * self.adapter_scaling

                result = result + output
            return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, AdapterLayer):
        # Aadapter layer for 8bit linear layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            adapter_type: str,
            bottleneck_size: int,
            non_linearity: str,
            adapter_dropout: float,
            scaling: Union[float, str],
            init_weights: str,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            AdapterLayer.__init__(
                self, 
                bottleneck_size=bottleneck_size, 
                non_linearity=non_linearity, 
                adapter_dropout=adapter_dropout,
                scaling=scaling,)
            
            self.init_weights = init_weights
            self.adapter_type = adapter_type
            if isinstance(scaling, float):
                self.adapter_scaling = scaling
            elif scaling == "learned":
                self.adapter_scaling = nn.Parameter(torch.ones(1))
            # Actual trainable parameters
            self.adapter_down = nn.Linear(in_features, bottleneck_size, bias=False)
            self.adapter_up = nn.Linear(bottleneck_size, out_features, bias=False)
            self.act_fn = ACT2FN[self.non_linearity]
            #Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.reset_parameters()
        
        def reset_parameters(self):
            nn.Linear.reset_parameters(self)
            # if we want to initialize with the bert strategy then this function is called for all the linear layers
            if hasattr(self, "adapter_down"):
                if self.init_weights == "bert":
                    self.adapter_down.apply(self.init_bert_weights)
                    self.adapter_up.apply(self.init_bert_weights)
                elif self.init_weights == "mam_adapter":
                    nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
                    nn.init.zeros_(self.adapter_up.weight)
                else:
                    raise ValueError("Unknown init_weights type: {}".format(config["init_weights"]))

        # This is copied from the BertPreTrainedModel class to make this a self containing class.
        @staticmethod
        def init_bert_weights(module):
            """Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # std defaults to 0.02, this might need to be changed
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        def forward(self, x: torch.Tensor):
            result_pre_forward = super().forward(x)

            if self.disable_adapters:
                return result_pre_forward
            else:
                if self.adapter_type == "mh_adapter":
                    if not torch.is_autocast_enabled():
                        expected_dtype = x.dtype

                        if x.dtype != torch.float32:
                            x = x.float()
                        
                        residual = x
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))).to(expected_dtype) * self.adapter_scaling
                        output = (output + residual).to(expected_dtype)

                        result = super().forward(output)
                    else:
                        residual = x
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))) * self.adapter_scaling
                        output = output + residual

                        result = super().forward(output)
                elif self.adapter_type == "output_adapter":
                    if not torch.is_autocast_enabled():
                        expected_dtype = result_pre_forward.dtype

                        if result_pre_forward.dtype != torch.float32:
                            result_pre_forward = result_pre_forward.float()

                        residual = result_pre_forward
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(result_pre_forward)))).to(expected_dtype) * self.adapter_scaling
                        result = (output + residual).to(expected_dtype)
                    else:
                        residual = result_pre_forward
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(result_pre_forward)))) * self.adapter_scaling
                        result = output + residual
                elif self.adapter_type == "parallel_adapter":
                    if not torch.is_autocast_enabled():
                        expected_dtype = result_pre_forward.dtype

                        if x.dtype != torch.float32:
                            x = x.float()
                        
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))).to(expected_dtype) * self.adapter_scaling
                        result = result_pre_forward + output
                    else:
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))) * self.adapter_scaling
                        result = result_pre_forward + output

                return result

    class Linear4bit(bnb.nn.Linear4bit, AdapterLayer):
        # Adapter layer for 4bit linear layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            adapter_type: str,
            bottleneck_size: int,
            non_linearity: str,
            adapter_dropout: float,
            scaling: Union[float, str],
            init_weights: str,
            **kwargs,
        ):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float16),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            AdapterLayer.__init__(
                self, 
                bottleneck_size=bottleneck_size, 
                non_linearity=non_linearity, 
                adapter_dropout=adapter_dropout,
                scaling=scaling,)
            
            self.init_weights = init_weights
            self.adapter_type = adapter_type
            if isinstance(scaling, float):
                self.adapter_scaling = scaling
            elif scaling == "learned":
                self.adapter_scaling = nn.Parameter(torch.ones(1))
            # Actual trainable parameters
            self.adapter_down = nn.Linear(in_features, bottleneck_size, bias=False)
            self.adapter_up = nn.Linear(bottleneck_size, out_features, bias=False)
            self.act_fn = ACT2FN[self.non_linearity]
            #Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.reset_parameters()
        
        def reset_parameters(self):
            nn.Linear.reset_parameters(self)
            # if we want to initialize with the bert strategy then this function is called for all the linear layers
            if hasattr(self, "adapter_down"):
                if self.init_weights == "bert":
                    self.adapter_down.apply(self.init_bert_weights)
                    self.adapter_up.apply(self.init_bert_weights)
                elif self.init_weights == "mam_adapter":
                    nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
                    nn.init.zeros_(self.adapter_up.weight)
                else:
                    raise ValueError("Unknown init_weights type: {}".format(config["init_weights"]))

        # This is copied from the BertPreTrainedModel class to make this a self containing class.
        @staticmethod
        def init_bert_weights(module):
            """Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # std defaults to 0.02, this might need to be changed
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        def forward(self, x: torch.Tensor):
            result_pre_forward = super().forward(x)

            if self.disable_adapters:
                return result_pre_forward
            else:
                if self.adapter_type == "mh_adapter":
                    if not torch.is_autocast_enabled():
                        expected_dtype = x.dtype

                        if x.dtype != torch.float32:
                            x = x.float()
                        
                        residual = x
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))).to(expected_dtype) * self.adapter_scaling
                        output = (output + residual).to(expected_dtype)

                        result = super().forward(output)
                    else:
                        residual = x
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))) * self.adapter_scaling
                        output = output + residual

                        result = super().forward(output)
                elif self.adapter_type == "output_adapter":
                    if not torch.is_autocast_enabled():
                        expected_dtype = result_pre_forward.dtype

                        if result_pre_forward.dtype != torch.float32:
                            result_pre_forward = result_pre_forward.float()

                        residual = result_pre_forward
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(result_pre_forward)))).to(expected_dtype) * self.adapter_scaling
                        result = (output + residual).to(expected_dtype)
                    else:
                        residual = result_pre_forward
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(result_pre_forward)))) * self.adapter_scaling
                        result = output + residual
                elif self.adapter_type == "parallel_adapter":
                    if not torch.is_autocast_enabled():
                        expected_dtype = result_pre_forward.dtype

                        if x.dtype != torch.float32:
                            x = x.float()
                        
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))).to(expected_dtype) * self.adapter_scaling
                        result = result_pre_forward + output
                    else:
                        output = self.adapter_up(self.act_fn(self.adapter_down(self.adapter_dropout(x)))) * self.adapter_scaling
                        result = result_pre_forward + output

                return result
                        

                        



            







        