# Copyright 2024-present the HuggingFace Inc. team.
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
# Reference paper: https://arxiv.org/abs/2405.16833


import copy
import os
from dataclasses import dataclass, field

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from peft import PeftConfig

from .loftq_utils import _SafetensorLoader as SafetensorLoader


@dataclass
class SafeLoraConfig:
    """
    This is the configuration class to store the configuration of a safeLora.


    Args:

    base_model_path (`str`): The path of the base model for obtaining the aligned matrix.

    aligned_model_path (`str`): The path of the aligned model for obtaining the aligned matrix.

    peft_model_path (`str`): The path of the LoRA wieghts and configs.

    select_layers_type (`str`): How to select projection layers? options: [threshold, number]

    threshold (`float`): The threshold of cosine similarity for selecting projected layers.

    num_proj_layers (`int`): The number of projected layers.

    devices (`str`): Devices are used in SafeLoRA (cuda or cpu).

    save_weights (`bool`): Replacing and saving SafeLoRA weights to the original LoRA file.

    local_files_only (`bool`): Using for snapshot_download.

    dtype (`torch.dtype`): Data type for model weights, e.g., torch.float32 or torch.bfloat16. If your device is CPU, you should use torch.float32.

    """

    base_model_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "The path of the base model for obtaining the aligned matrix."},
    )

    aligned_model_path: str = field(
        default="TheBloke/Llama-2-7B-Chat-fp16",
        metadata={"help": "The path of the aligned model for obtaining the aligned matrix."},
    )

    peft_model_path: str = field(
        default="LisaSchunke/llama-2-7b-peft-finetuned-20000-dataset",
        metadata={"help": "The path of the LoRA wieghts and configs."},
    )

    select_layers_type: str = field(
        default="number",
        metadata={"help": "How to select projection layers? options: [threshold, number]."},
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity for selecting projected layers."},
    )

    num_proj_layers: int = field(
        default=10,
        metadata={"help": "The number of projected layers."},
    )

    device: str = field(
        default="cuda",
        metadata={"help": "Devices are used in SafeLoRA. (cuda or cpu)"},
    )

    save_weights: bool = field(
        default=True,
        metadata={"help": "Replacing and saving SafeLoRA weights to the original LoRA file."},
    )
    local_files_only: bool = field(
        default=False,
        metadata={"help": "Using for snapshot_download."},
    )

    dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata={"help": "Data type for model weights, e.g., torch.float32 or torch.bfloat16. If your device is CPU, you should use torch.float32."},
    )

    def __post_init__(self):
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None.")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None.")
        if self.peft_model_path is None:
            raise ValueError("peft_model_path cannot be None.")
        if self.devices != 'cuda' and self.dtype != torch.float32:
            raise ValueError("When using a CPU, please set dtype to torch.float32, as CPUs do not support torch.float16.")


def get_aligned_matrix(base_model_path, aligned_model_path, devices, peft_config, configs):
    """
    Get projected matrix by following the config (target_modules) from the peft model.
    The dimensions between the base model's weights and the aligned model's weights should be the same.
    """
    sl_align = SafetensorLoader(aligned_model_path, local_files_only=configs.local_files_only)
    sl_base = SafetensorLoader(base_model_path, local_files_only=configs.local_files_only)

    base_model_parameters = [
        name for name in sl_base.weight_map.keys() if any(v in name for v in list(peft_config.target_modules))
    ]
    align_model_parameters = [
        name for name in sl_align.weight_map.keys() if any(v in name for v in list(peft_config.target_modules))
    ]
    safety_vector = []
    for name_base, name_align in zip(base_model_parameters, align_model_parameters):
        if sl_base.get_tensor(name_base).shape != sl_align.get_tensor(name_align).shape:
            raise ValueError(
                "The dimensions of the base model's weight should be the same with the aligned model's weight."
            )
        vec = sl_base.get_tensor(name_base) - sl_align.get_tensor(name_align)
        vec = vec.to(devices)
        if devices == "cpu":
            vec = vec.to(torch.float32)
        vec = torch.mm(vec, vec.t()) / torch.norm(vec)
        safety_vector.append((vec).detach().cpu())
    return safety_vector


def project_weights(configs, peft_weights, v):
    ori_peft_weights = copy.deepcopy(peft_weights)
    vars_names_LoRA_A = [name for name in peft_weights.keys() if "lora_A" in name]
    vars_names_LoRA_B = [name for name in peft_weights.keys() if "lora_B" in name]
    num_projected_layers = 0
    dis = []
    cos_total = []
    for idx, (name_A, name_B) in enumerate(zip(vars_names_LoRA_A, vars_names_LoRA_B)):
        A = ori_peft_weights[name_A]
        if configs.devices != "cpu":
            P = v[idx].to(torch.bfloat16).to(configs.devices)
        else:
            P = v[idx].to("cpu")
        W = torch.mm(P, ori_peft_weights[name_B])
        fW = torch.mm(W, A)
        ori = torch.mm(ori_peft_weights[name_B], A)
        cos = torch.round(torch.nn.functional.cosine_similarity(fW.reshape(1, -1), ori.reshape(1, -1)) * 10**5) / 10**5
        cos_total.append(cos.item())
        if cos <= configs.threshold:
            num_projected_layers += 1
            peft_weights[name_B] = W
        else:
            peft_weights[name_B] = ori_peft_weights[name_B]

        dist = 1 / (1 + torch.norm(peft_weights[name_B].reshape(1, -1) - W.reshape(1, -1)))

        dis.append(dist.item())
    return peft_weights, cos_total


def apply_safelora(safelora_config: SafeLoraConfig):
    """

    The official code of Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning Large Language Models: https://arxiv.org/abs/2405.16833

    After fine-tuning large language models (LLMs) using LoRA, the alignment of the resulting models may decrease.
    Therefore, applying `apply_safelora()` is intended to help preserve the alignment of the final models.

    It is important to note that the model weights of the aligned model and the base model must be of the same size.


    Example:

    from peft.utils.safelora import SafeLoraConfig, apply_safelora

    config = SafeLoraConfig(base_model_path='../LLM_Models/llama-2-7b-hf/',\
                            aligned_model_path='../LLM_Models/llama-2-7b-chat-fp16/',
                            peft_model_path = '../finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42',
                            devices='cuda',
                            select_layers_type='threshold',
                            save_weights=True)

    final_lora_weight = apply_safelora(config)

    """

    peft_config = PeftConfig.from_pretrained(safelora_config.peft_model_path)

    projected_matrix = get_aligned_matrix(
        safelora_config.base_model_path, safelora_config.aligned_model_path, safelora_config.devices, peft_config, safelora_config
    )

    with safe_open(
        f"{os.path.join(safelora_config.peft_model_path, 'adapter_model.safetensors')}", framework="pt", device=safelora_config.devices
    ) as f:
        if (safelora_config.devices).lower() == "cpu":
            peft_weights = {name: f.get_tensor(name).to(safelora_config.dtype) for name in f.keys()}
        else:
            peft_weights = {name: f.get_tensor(name).to(safelora_config.dtype) for name in f.keys()}
    if safelora_config.select_layers_type == "threshold":
        final_weights, _ = project_weights(safelora_config, peft_weights, projected_matrix)
    elif safelora_config.select_layers_type == "number":
        _, cos = project_weights(safelora_config, peft_weights, projected_matrix)
        thrs = torch.sort(torch.Tensor(cos))[0][: safelora_config.num_proj_layers][-1]
        safelora_config.threshold = thrs
        final_weights, _ = project_weights(safelora_config, peft_weights, projected_matrix)

    if safelora_config.save_weights:
        save_file(final_weights, f"{os.path.join(safelora_config.peft_model_path, 'adapter_model.safetensors')}")

    return final_weights
