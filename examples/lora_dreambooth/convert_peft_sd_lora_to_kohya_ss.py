import argparse
import os
from typing import Dict

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from transformers import CLIPTextModel

from peft import PeftModel, get_peft_model_state_dict


# Default kohya_ss LoRA replacement modules
# https://github.com/kohya-ss/sd-scripts/blob/c924c47f374ac1b6e33e71f82948eb1853e2243f/networks/lora.py#L664
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_ADAPTER_NAME = "default"


def get_module_kohya_state_dict(
    module: PeftModel, prefix: str, dtype: torch.dtype, adapter_name: str = LORA_ADAPTER_NAME
) -> Dict[str, torch.Tensor]:
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--sd_checkpoint_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument("--peft_lora_path", default=None, type=str, required=True, help="Path to peft trained LoRA")

    parser.add_argument(
        "--dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output safetensors file for use with webui.",
    )

    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    # Store kohya_ss state dict
    kohya_ss_state_dict = {}
    dtype = torch.float16 if args.half else torch.float32

    # Load Text Encoder LoRA model
    text_encoder_peft_lora_path = os.path.join(args.peft_lora_path, "text_encoder")
    if os.path.exists(text_encoder_peft_lora_path):
        text_encoder = CLIPTextModel.from_pretrained(
            args.sd_checkpoint, subfolder="text_encoder", revision=args.sd_checkpoint_revision
        )
        text_encoder = PeftModel.from_pretrained(
            text_encoder, text_encoder_peft_lora_path, adapter_name=LORA_ADAPTER_NAME
        )
        kohya_ss_state_dict.update(
            get_module_kohya_state_dict(text_encoder, LORA_PREFIX_TEXT_ENCODER, dtype, LORA_ADAPTER_NAME)
        )

    # Load UNet LoRA model
    unet_peft_lora_path = os.path.join(args.peft_lora_path, "unet")
    if os.path.exists(unet_peft_lora_path):
        unet = UNet2DConditionModel.from_pretrained(
            args.sd_checkpoint, subfolder="unet", revision=args.sd_checkpoint_revision
        )
        unet = PeftModel.from_pretrained(unet, unet_peft_lora_path, adapter_name=LORA_ADAPTER_NAME)
        kohya_ss_state_dict.update(get_module_kohya_state_dict(unet, LORA_PREFIX_UNET, dtype, LORA_ADAPTER_NAME))

    # Save state dict
    save_file(
        kohya_ss_state_dict,
        args.dump_path,
    )
