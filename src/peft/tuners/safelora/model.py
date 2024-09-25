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

import numpy
import torch
from transformers import AutoModelForCausalLM


class SafeLoRA:
    def __init__(self, peft_model:torch.nn.Module, config):
        """
        Please use safelora.model to get the projected model.

        How to use SafeLoRA:
        path = './LLM_Models/llama-2-7b-chat-fp16/' # load your base model of the peft model
        model = AutoModelForCausalLM.from_pretrained(path)
        pmodel = PeftModel.from_pretrained(model, 'finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42/',torch_dtype=torch.float16) #load peft model

        SafeLoRAConfig.base_model_path = './LLM_Models/llama-2-7b-hf/'  #you should modify the path
        SafeLoRAConfig.aligned_model_path = './LLM_Models/llama-2-7b-chat-fp16/' #you should modify the path

        safelora = SafeLoRA(pmodel, SafeLoRAConfig)

        Finally, you can get the projected model by "safelora.model".
        """
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config["default"]
        self.model_ori = copy.deepcopy(peft_model)
        project_matrix = self.get_aligned_matrix()
        if self.config.select_layers_type == 'threshold':
            self.model, _ = self.projected_weighted(project_matrix, self.config.threshold, show_info=True)
        elif self.config.select_layers_type == 'number':
            model, cos = self.projected_weighted(project_matrix, 0.3, show_info=False)
            thrs = numpy.sort(cos)[:self.config.num_proj_layers][-1]
            self.model, _ = self.projected_weighted(project_matrix, thrs, show_info=True)
        else:
            raise ValueError("The method of select_layer_type should be threshold or number.")

    def get_aligned_matrix(self):
        """
        Get projected matrix by following the config (target_modules) from the peft model.
        The dimensions between the base model's weights and the aligned model's weights should be the same.
        """
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        v = []
        proj_modules = list(self.peft_config.target_modules)
        for (b_name, b_param) , (a_name, a_param) in zip (base_model.named_parameters(), aligned_model.named_parameters()):
            if any(module in a_name for module in proj_modules):
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
                vec = a_param - b_param
                vec = vec.to(self.config.devices)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append((vec).detach().cpu())
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        v = project_matrix
        idx = 0
        i = 0
        dis = []
        cos_total = []
        for (name, param),(name_ori, param_ori) in zip(self.peft_model.named_parameters(), self.model_ori.named_parameters()):
            if 'lora' in name:
                if param.shape[0] == self.peft_config.r:
                    B = copy.deepcopy(param_ori)
                if param.shape[0] != self.peft_config.r:
                    P = v[idx].to(param.device)
                    W = torch.mm(P, param_ori.data)
                    fW = torch.mm(W, B)
                    ori = torch.mm(param_ori, B)
                    W_new = torch.mm(P, param_ori.data)
                    cos = numpy.round(torch.nn.functional.cosine_similarity(fW.reshape(1,-1), ori.reshape(1,-1)).item(),5)
                    cos_total.append(cos)

                    if cos <=  thrs_cos:
                        i+=1
                        param.data =  W_new
                    else:
                        param.data = param_ori
                    dist = 1 / (1+torch.norm(param.data.reshape(1,-1)-W.reshape(1,-1)))

                    dis.append(dist.item())
                    idx += 1
        if show_info:
            print(f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {numpy.mean(dis)} (> 0.8 is better).")
        return self.peft_model, cos_total

