import argparse
import os
import re
from typing import Callable, List, Optional, Union

import safetensors
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


# Default kohya_ss LoRA replacement modules
# https://github.com/kohya-ss/sd-scripts/blob/c924c47f374ac1b6e33e71f82948eb1853e2243f/networks/lora.py#L661
UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"


def get_modules_names(
    root_module: nn.Module,
    target_replace_modules_linear: Optional[List[str]] = [],
    target_replace_modules_conv2d: Optional[List[str]] = [],
):
    # Combine replacement modules
    target_replace_modules = target_replace_modules_linear + target_replace_modules_conv2d

    # Store result
    modules_names = set()
    # https://github.com/kohya-ss/sd-scripts/blob/c924c47f374ac1b6e33e71f82948eb1853e2243f/networks/lora.py#L720
    for name, module in root_module.named_modules():
        if module.__class__.__name__ in target_replace_modules:
            if len(name) == 0:
                continue
            for child_name, child_module in module.named_modules():
                if len(child_name) == 0:
                    continue
                is_linear = child_module.__class__.__name__ == "Linear"
                is_conv2d = child_module.__class__.__name__ == "Conv2d"

                if (is_linear and module.__class__.__name__ in target_replace_modules_linear) or (
                    is_conv2d and module.__class__.__name__ in target_replace_modules_conv2d
                ):
                    modules_names.add(f"{name}.{child_name}")

    return sorted(modules_names)


def get_rank_alpha(
    layer_names: List[str],
    value_getter: Callable[[str], Union[int, float]],
    filter_string: str,
) -> Union[int, float]:
    values = [value_getter(p) for p in filter(lambda x: bool(re.search(filter_string, x)), layer_names)]
    value = values[0]
    assert all(v == value for v in values), f"All LoRA ranks and alphas must be same, found: {values}"
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_checkpoint", default=None, type=str, required=True, help="SD checkpoint to use")

    parser.add_argument(
        "--kohya_lora_path", default=None, type=str, required=True, help="Path to kohya_ss trained LoRA"
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    # Find text encoder modules to add LoRA to
    text_encoder = CLIPTextModel.from_pretrained(args.sd_checkpoint, subfolder="text_encoder")
    text_encoder_modules_names = get_modules_names(
        text_encoder, target_replace_modules_linear=TEXT_ENCODER_TARGET_REPLACE_MODULE
    )

    # Find unet2d modules to add LoRA to
    unet = UNet2DConditionModel.from_pretrained(args.sd_checkpoint, subfolder="unet")
    unet_modules_names = get_modules_names(
        unet,
        target_replace_modules_linear=UNET_TARGET_REPLACE_MODULE,
        target_replace_modules_conv2d=UNET_TARGET_REPLACE_MODULE,
    )

    # Open kohya_ss checkpoint
    with safetensors.safe_open(args.kohya_lora_path, framework="pt", device="cpu") as f:
        # Extract information about LoRA structure
        metadata = f.metadata()
        if (metadata is not None) and ("ss_network_dim" in metadata) and ("ss_network_alpha" in metadata):
            # LoRA rank and alpha are in safetensors metadata, just get it
            lora_r = lora_text_encoder_r = int(metadata["ss_network_dim"])
            lora_alpha = lora_text_encoder_alpha = float(metadata["ss_network_alpha"])
        else:
            # LoRA rank and alpha are not present, so infer them
            lora_r = get_rank_alpha(
                f.keys(), lambda n: f.get_tensor(n).size(0), f"^{LORA_PREFIX_UNET}\w+\.lora_down\.weight$"
            )
            lora_text_encoder_r = get_rank_alpha(
                f.keys(), lambda n: f.get_tensor(n).size(0), f"^{LORA_PREFIX_TEXT_ENCODER}\w+\.lora_down\.weight$"
            )
            lora_alpha = get_rank_alpha(f.keys(), lambda n: f.get_tensor(n).item(), f"^{LORA_PREFIX_UNET}\w+\.alpha$")
            lora_text_encoder_alpha = get_rank_alpha(
                f.keys(), lambda n: f.get_tensor(n).item(), f"^{LORA_PREFIX_TEXT_ENCODER}\w+\.alpha$"
            )

        # Create LoRA for text encoder
        text_encoder_config = LoraConfig(
            r=lora_text_encoder_r,
            lora_alpha=lora_text_encoder_alpha,
            target_modules=text_encoder_modules_names,
            lora_dropout=0.0,
            bias="none",
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_config)
        text_encoder_lora_state_dict = {x: None for x in get_peft_model_state_dict(text_encoder).keys()}

        # Load text encoder values from kohya_ss LoRA
        for peft_te_key in text_encoder_lora_state_dict.keys():
            kohya_ss_te_key = peft_te_key.replace("base_model.model", LORA_PREFIX_TEXT_ENCODER)
            kohya_ss_te_key = kohya_ss_te_key.replace("lora_A", "lora_down")
            kohya_ss_te_key = kohya_ss_te_key.replace("lora_B", "lora_up")
            kohya_ss_te_key = kohya_ss_te_key.replace(".", "_", kohya_ss_te_key.count(".") - 2)
            text_encoder_lora_state_dict[peft_te_key] = f.get_tensor(kohya_ss_te_key).to(text_encoder.dtype)

        # Load converted kohya_ss text encoder LoRA back to PEFT
        set_peft_model_state_dict(text_encoder, text_encoder_lora_state_dict)

        if args.half:
            text_encoder.to(torch.float16)

        # Save text encoder result
        text_encoder.save_pretrained(
            os.path.join(args.dump_path, "text_encoder"),
        )

        # Create LoRA for unet2d
        unet_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=unet_modules_names, lora_dropout=0.0, bias="none"
        )
        unet = get_peft_model(unet, unet_config)
        unet_lora_state_dict = {x: None for x in get_peft_model_state_dict(unet).keys()}

        # Load unet2d values from kohya_ss LoRA
        for peft_unet_key in unet_lora_state_dict.keys():
            kohya_ss_unet_key = peft_unet_key.replace("base_model.model", LORA_PREFIX_UNET)
            kohya_ss_unet_key = kohya_ss_unet_key.replace("lora_A", "lora_down")
            kohya_ss_unet_key = kohya_ss_unet_key.replace("lora_B", "lora_up")
            kohya_ss_unet_key = kohya_ss_unet_key.replace(".", "_", kohya_ss_unet_key.count(".") - 2)
            unet_lora_state_dict[peft_unet_key] = f.get_tensor(kohya_ss_unet_key).to(unet.dtype)

        # Load converted kohya_ss unet LoRA back to PEFT
        set_peft_model_state_dict(unet, unet_lora_state_dict)

        if args.half:
            unet.to(torch.float16)

        # Save text encoder result
        unet.save_pretrained(
            os.path.join(args.dump_path, "unet"),
        )
