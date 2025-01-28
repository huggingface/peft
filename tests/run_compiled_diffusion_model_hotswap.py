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
"""This is a standalone script that checks that we can hotswap a LoRA adapter on a compiled model

By itself, this script is not super interesting but when we collect the compile logs, we can check that hotswapping
does not trigger recompilation. This is done in the TestLoraHotSwapping class in test_pipelines.py.

Running this script with `check_hotswap(False)` will load the LoRA adapter without hotswapping, which will result in
recompilation.

There is an equivalent test in diffusers, see https://github.com/huggingface/diffusers/pull/9453.

"""

import os
import sys
import tempfile

import torch
from diffusers import UNet2DConditionModel
from diffusers.utils.testing_utils import floats_tensor

from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.hotswap import prepare_model_for_compiled_hotswap


torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_small_unet():
    # from diffusers UNet2DConditionModelTests
    # TODO: This appears not to work yet in full pipeline context, see:
    # https://github.com/huggingface/diffusers/pull/9453#issuecomment-2418508871
    torch.manual_seed(0)
    init_dict = {
        "block_out_channels": (4, 8),
        "norm_num_groups": 4,
        "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
        "cross_attention_dim": 8,
        "attention_head_dim": 2,
        "out_channels": 4,
        "in_channels": 4,
        "layers_per_block": 1,
        "sample_size": 16,
    }
    model = UNet2DConditionModel(**init_dict)
    return model.to(torch_device)


def get_unet_lora_config(lora_rank, lora_alpha):
    # from diffusers test_models_unet_2d_condition.py
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights=False,
        use_dora=False,
    )
    return unet_lora_config


def get_dummy_input():
    # from UNet2DConditionModelTests
    batch_size = 4
    num_channels = 4
    sizes = (16, 16)

    noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor([10]).to(torch_device)
    encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(torch_device)

    return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}


def set_lora_device(model, adapter_names, device):
    # copied from diffusers LoraBaseMixin.set_lora_device
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            for adapter_name in adapter_names:
                module.lora_A[adapter_name].to(device)
                module.lora_B[adapter_name].to(device)
                # this is a param, not a module, so device placement is not in-place -> re-assign
                if hasattr(module, "lora_magnitude_vector") and module.lora_magnitude_vector is not None:
                    if adapter_name in module.lora_magnitude_vector:
                        module.lora_magnitude_vector[adapter_name] = module.lora_magnitude_vector[adapter_name].to(
                            device
                        )


def check_hotswap(do_hotswap, ranks=(8, 8), alpha_scalings=(16, 16)):
    dummy_input = get_dummy_input()
    unet = get_small_unet()
    rank0, rank1 = ranks
    alpha0, alpha1 = alpha_scalings
    lora_config0 = get_unet_lora_config(rank0, alpha0)
    lora_config1 = get_unet_lora_config(rank1, alpha1)
    unet.add_adapter(lora_config0, adapter_name="adapter0")
    unet.add_adapter(lora_config1, adapter_name="adapter1")

    with tempfile.TemporaryDirectory() as tmp_dirname:
        unet.save_lora_adapter(os.path.join(tmp_dirname, "0"), safe_serialization=True, adapter_name="adapter0")
        unet.save_lora_adapter(os.path.join(tmp_dirname, "1"), safe_serialization=True, adapter_name="adapter1")
        del unet

        unet = get_small_unet()
        file_name0 = os.path.join(os.path.join(tmp_dirname, "0"), "pytorch_lora_weights.safetensors")
        file_name1 = os.path.join(os.path.join(tmp_dirname, "1"), "pytorch_lora_weights.safetensors")
        unet.load_lora_adapter(file_name0, safe_serialization=True, adapter_name="adapter0")

        prepare_model_for_compiled_hotswap(
            unet, config={"adapter0": lora_config0, "adapter1": lora_config1}, target_rank=max(ranks)
        )
        unet = torch.compile(unet, mode="reduce-overhead")
        unet(**dummy_input)["sample"]

        if do_hotswap:
            unet.load_lora_adapter(file_name1, adapter_name="default_0", hotswap=True)
        else:
            # offloading the old and loading the new adapter will result in recompilation
            set_lora_device(unet, adapter_names=["default_0"], device="cpu")
            unet.load_lora_adapter(file_name1, adapter_name="other_name", hotswap=False)

        # we need to call forward to potentially trigger recompilation
        unet(**dummy_input)["sample"]


if __name__ == "__main__":
    # check_hotswap(False) will trigger recompilation
    do_hotswap = sys.argv[1] == "1"
    # ranks is a string like '13,7'
    ranks = sys.argv[2].split(",")
    ranks = int(ranks[0]), int(ranks[1])
    check_hotswap(do_hotswap=do_hotswap, ranks=ranks, alpha_scalings=(8, 16))
