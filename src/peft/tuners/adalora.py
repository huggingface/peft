import importlib
import math
import re
import warnings
import numpy as np 
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose
from .lora import LoraConfig, LoraModel, LoraLayer, mark_only_lora_as_trainable


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class AdaLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        target_r (`int`): The target average rank of incremental matrix. 
        init_r (`int`): The initial rank for each incremental matrix. 
        tinit (`int`): The steps of initial fine-tuning warmup. 
        tfinal (`int`): The step of final fine-tuning. 
        deltaT (`int`): The time internval between two budget allocations. 
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification. 
        orth_reg_weight (`float`): The coefficient of orthogonal regularization. 
        total_step (`int`): The total training steps that should be specified before training. 
    """
    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Intial Lora matrix dimension."})
    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    deltaT: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."}) 
    orth_reg_weight: float = field(
        default=0.5, 
        metadata={"help": "The orthogonal regularization coefficient."}
    )
    total_step: Optional[int] = field(
        default=None, 
        metadata={"help": "The total training steps."}
    )
    rank_pattern: Optional[dict] = field(
        default=None, 
        metadata={"help":"The saved rank pattern."}
    )

    def __post_init__(self):
        self.peft_type = PeftType.ADALORA



class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model.
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY 

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.

    Returns:
        `torch.nn.Module`: The AdaLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import AdaLoraModel, AdaLoraConfig 
        >>> config = AdaLoraConfig(
                peft_type="ADALORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") 
        >>> model = AdaLoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLora model.
    """

    def __init__(self, config, model):
        nn.Module.__init__(self)
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.rankallocator = RankAllocator(config, self.model)

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": self.peft_config.init_r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": self.peft_config.merge_weights or self.peft_config.inference_mode,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt) and self.peft_config.enable_lora is None:
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    new_module = SVDLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = SVDLinear(target.in_features, target.out_features, bias=bias, **kwargs)
                # TODO: Implement the MergedLinear of SVD Adapattion 
                # elif self.peft_config.enable_lora is not None:
                #     kwargs.update({"enable_lora": self.peft_config.enable_lora})
                #     if isinstance(target, Conv1D):
                #         in_features, out_features = target.weight.shape
                #     else:
                #         in_features, out_features = target.in_features, target.out_features
                #         if kwargs["fan_in_fan_out"]:
                #             warnings.warn(
                #                 "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                #                 "Setting fan_in_fan_out to False."
                #             )
                #             kwargs["fan_in_fan_out"] = False
                #     new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs) 

        # Calculate the orthogonal regularization 
        orth_reg_weight = self.peft_config.orth_reg_weight
        assert orth_reg_weight > 0 

        if hasattr(outputs, "loss"):
            regu_loss = 0 
            num_param = 0 
            for n,p in self.model.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    para_cov = p @ p.T if "lora_A" in n else p.T @ p 
                    I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
                    I.requires_grad = False
                    num_param += 1
                    regu_loss += torch.norm(para_cov-I, p="fro")
            regu_loss = regu_loss / num_param 
            outputs.loss += orth_reg_weight * regu_loss  
        return outputs 


    def _prepare_new_module(self, target, rank_idx):
        if isinstance(rank_idx, list):
            rank = sum(rank_idx)
        elif isinstance(rank_idx, torch.Tensor):
            rank_idx = rank_idx.view(-1)
            rank = rank_idx.sum().item()
        else:
            raise ValueError("Unexcepted type of rank_idx")
        kwargs = {
            "r": rank,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": self.peft_config.merge_weights or self.peft_config.inference_mode,
        }
        bias = target.bias is not None
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt) and self.peft_config.enable_lora is None:
            kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = SVDLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
        elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
            new_module = SVDLinear(target.in_features, target.out_features, bias=bias, **kwargs)
        new_module = new_module.to(target.weight.device)

        with torch.no_grad():
            new_module.weight.copy_(target.weight)
            if bias:
                new_module.bias.copy_(target.bias)
            if rank > 0:
                new_module.lora_E.copy_(target.lora_E[rank_idx])
                new_module.lora_A.copy_(target.lora_A[rank_idx])
                new_module.lora_B.copy_(target.lora_B[:,rank_idx])
                # The scaling is exactly as the previous 
                new_module.ranknum.copy_(target.ranknum)
        return new_module


    def resize_modules_by_rank_pattern(self, rank_pattern):
        for name,rank_idx in rank_pattern.items():
            key = ".".join(name.split(".")[0:-1]) 
            parent, target, target_name = self._get_submodules(key) 
            new_module = self._prepare_new_module(target, rank_idx)
            self._replace_module(parent, target_name, new_module, target)


    def update_and_allocate(self, global_step):
        # Update the importance score and allocate the budget 
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            budget, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step)
            if rank_pattern:
                self.peft_config.rank_pattern = rank_pattern 
        # Finalize the budget allocation 
        elif global_step == self.peft_config.total_step - self.peft_config.tfinal: 
            budget, rank_pattern = self.rankallocator.update_and_allocate(
                self.model, global_step, force_mask=True
            )
            self.resize_modules_by_rank_pattern(rank_pattern)
            self.peft_config.rank_pattern = rank_pattern
            self.rankallocator.reset_ipt() 
            print("Finalize the rank pattern.")
        # Pass the function and do forward propagation 
        else: 
            return None



class SVDLinear(nn.Linear, LoraLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # Right singular vectors
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # Singular values 
            self.lora_E = nn.Parameter(self.weight.new_zeros(r, 1)) 
            # Left singular vectors
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # The current rank
            self.ranknum = nn.Parameter(self.weight.new_zeros(1), requires_grad=False)
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= transpose(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling/(self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += transpose(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling/(self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                ) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)


if is_bnb_available():
    class SVDLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Low-rank matrix for SVD-based adaptation 
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            # Actual trainable parameters
            if r > 0:
                # Right singular vectors
                self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
                # Singular values 
                self.lora_E = nn.Parameter(self.weight.new_zeros(r, 1)) 
                # Left singular vectors
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
                # The current rank
                self.ranknum = nn.Parameter(self.weight.new_zeros(1), requires_grad=False)
                self.ranknum.data.fill_(float(self.r))
                self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                self.ranknum.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.zeros_(self.lora_E)
                nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
                nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_dropout(x) @ (self.lora_A*self.lora_E).T @ self.lora_B.T /(self.ranknum+1e-5)
                    ).to(expected_dtype) * self.scaling 
                    result += output
                else:
                    output = (
                        self.lora_dropout(x) @ (self.lora_A*self.lora_E).T @ self.lora_B.T /(self.ranknum+1e-5)
                    ) * self.scaling 
                    result += output
            return result



class RankAllocator(object):
    """
    The RankAllocator for AdaLoraModel. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY 

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.  

    """
    def __init__(self, peft_config, model):
        self.peft_config = peft_config
        self.beta1 = peft_config.beta1 
        self.beta2 = peft_config.beta2 
        assert (self.beta1>0 and self.beta1<1)
        assert (self.beta2>0 and self.beta2<1)

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0 
        self.name_set = set() 
        for n,p in model.named_parameters():
            if "lora_A" in n: 
                self.init_bgt += p.size(0) 
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = list(sorted(self.name_set)) 
        # The total final rank budget 
        self.target_bgt = self.peft_config.target_r * len(self.name_set) 

    def budget_schedule(self, step:int):
        tinit = self.peft_config.tinit 
        tfinal = self.peft_config.tfinal 
        total_step = self.peft_config.total_step 
        # Initial warmup 
        if step <= tinit: 
            budget = self.init_bgt 
            mask_ind = False 
        # Final fine-tuning 
        elif step > total_step - tfinal: 
            budget = self.target_bgt 
            mask_ind = True 
        else: 
            # Budget decreasing with a cubic scheduler 
            mul_coeff = 1 - (step-tinit) / (total_step-tfinal-tinit)
            budget = int(
                (self.init_bgt-self.target_bgt)*(mul_coeff**3)+self.target_bgt
            )
            mask_ind = True if step % self.peft_config.deltaT == 0 else False 
        return budget, mask_ind 

    def update_ipt(self, model): 
        # Update the sensitivity and uncertainty for every weight 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Sensitivity smoothing 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                        (1 - self.beta1)*self.ipt[n]
                    # Uncertainty quantification 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1) 
        return sum_ipt 

    def mask_to_budget(self, model, budget): 
        value_ipt = {}
        vector_ipt = {} 
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n,p in model.named_parameters(): 
            if "lora_A" in n: 
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt: 
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if "lora_B" in n: 
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                if name_m not in vector_ipt: 
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if "lora_E" in n:
                entry_ipt = self._element_score(n)                 
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet 
        for name_m in vector_ipt: 
            ipt_E = value_ipt[name_m] 
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m%"lora_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = torch.kthvalue(
            torch.cat(all_score), 
            k = self.init_bgt - budget,
        )[0].item()

        rank_pattern = {}
        # Mask the unimportant triplets 
        with torch.no_grad():
            for n,p in model.named_parameters():
                if "lora_E" in n: 
                    p.masked_fill_(triplet_ipt[n]<=mask_threshold, 0.0)
                    rank_pattern[n] = (~(triplet_ipt[n]<=mask_threshold)).view(-1).tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        # # Update the importance score and allocate the budget 
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        print("budget:", budget) 
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None 
        return budget, rank_pattern


