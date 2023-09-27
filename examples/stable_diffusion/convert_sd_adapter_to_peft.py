import argparse
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import safetensors
import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

from peft import LoHaConfig, LoraConfig, PeftType, get_peft_model, set_peft_model_state_dict


# Default kohya_ss LoRA replacement modules
# https://github.com/kohya-ss/sd-scripts/blob/c924c47f374ac1b6e33e71f82948eb1853e2243f/networks/lora.py#L661
UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
PREFIX_UNET = "lora_unet"
PREFIX_TEXT_ENCODER = "lora_te"


@dataclass
class LoRAInfo:
    kohya_key: str
    peft_key: str
    alpha: Optional[float] = None
    rank: Optional[int] = None
    lora_A: Optional[torch.Tensor] = None
    lora_B: Optional[torch.Tensor] = None

    def peft_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.lora_A is None or self.lora_B is None:
            raise ValueError("At least one of lora_A or lora_B is None, they must both be provided")
        return {
            f"base_model.model{self.peft_key}.lora_A.weight": self.lora_A,
            f"base_model.model.{self.peft_key}.lora_B.weight": self.lora_A,
        }


@dataclass
class LoHaInfo:
    kohya_key: str
    peft_key: str
    alpha: Optional[float] = None
    rank: Optional[int] = None
    hada_w1_a: Optional[torch.Tensor] = None
    hada_w1_b: Optional[torch.Tensor] = None
    hada_w2_a: Optional[torch.Tensor] = None
    hada_w2_b: Optional[torch.Tensor] = None
    hada_t1: Optional[torch.Tensor] = None
    hada_t2: Optional[torch.Tensor] = None

    def peft_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.hada_w1_a is None or self.hada_w1_b is None or self.hada_w2_a is None or self.hada_w2_b is None:
            raise ValueError(
                "At least one of hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b is missing, they all must be provided"
            )
        state_dict = {
            f"base_model.model.{self.peft_key}.hada_w1_a": self.hada_w1_a,
            f"base_model.model.{self.peft_key}.hada_w1_b": self.hada_w1_b,
            f"base_model.model.{self.peft_key}.hada_w2_a": self.hada_w2_a,
            f"base_model.model.{self.peft_key}.hada_w2_b": self.hada_w2_b,
        }
        if not (
            (self.hada_t1 is None and self.hada_t2 is None) or (self.hada_t1 is not None and self.hada_t2 is not None)
        ):
            raise ValueError("hada_t1 and hada_t2 must be either both present or not present at the same time")
        if self.hada_t1 is not None and self.hada_t2 is not None:
            state_dict[f"{self.peft_key}.hada_t1"] = self.hada_t1
            state_dict[f"{self.peft_key}.hada_t2"] = self.hada_t2
        return state_dict


def construct_peft_loraconfig(info: Dict[str, LoRAInfo]) -> LoraConfig:
    """Constructs LoraConfig from data extracted from adapter checkpoint

    Args:
        info (Dict[str, LoRAInfo]): Information extracted from adapter checkpoint

    Returns:
        LoraConfig: config for constructing LoRA
    """

    # Unpack all ranks and alphas
    ranks = {key: val.rank for key, val in info.items()}
    alphas = {x[0]: x[1].alpha or x[1].rank for x in info.items()}

    # Determine which modules needs to be transformed
    target_modules = sorted(info.keys())

    # Determine most common rank and alpha
    r = Counter(ranks.values()).most_common(1)[0]
    lora_alpha = Counter(alphas.values()).most_common(1)[0]

    # Determine which modules have different rank and alpha
    rank_pattern = dict(sorted(filter(lambda x: x[1] != r, ranks.items()), key=lambda x: x[0]))
    alpha_pattern = dict(sorted(filter(lambda x: x[1] != lora_alpha, alphas.items()), key=lambda x: x[0]))

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=False,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    return config


def construct_peft_lohaconfig(info: Dict[str, LoHaInfo]) -> LoHaConfig:
    """Constructs LoHaConfig from data extracted from adapter checkpoint

    Args:
        info (Dict[str, LoHaInfo]): Information extracted from adapter checkpoint

    Returns:
        LoHaConfig: config for constructing LoHA
    """

    # Unpack all ranks and alphas
    ranks = {x[0]: x[1].rank for x in info.items()}
    alphas = {x[0]: x[1].alpha or x[1].rank for x in info.items()}

    # Determine which modules needs to be transformed
    target_modules = sorted(info.keys())

    # Determine most common rank and alpha
    r = Counter(ranks.values()).most_common(1)[0]
    alpha = Counter(alphas.values()).most_common(1)[0]

    # Determine which modules have different rank and alpha
    rank_pattern = dict(sorted(filter(lambda x: x[1] != r, ranks.items()), key=lambda x: x[0]))
    alpha_pattern = dict(sorted(filter(lambda x: x[1] != alpha, alphas.items()), key=lambda x: x[0]))

    # Determine whether any of modules have effective conv2d decomposition
    use_effective_conv2d = any(((val.hada_t1 is not None) or (val.hada_t2 is not None) for val in info.values()))

    config = LoHaConfig(
        r=r,
        alpha=alpha,
        target_modules=target_modules,
        rank_dropout=0.0,
        module_dropout=0.0,
        init_weights=False,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        use_effective_conv2d=use_effective_conv2d,
    )

    return config


def combine_peft_state_dict(info: Dict[str, Union[LoRAInfo, LoHaInfo]]) -> Dict[str, torch.Tensor]:
    result = {}
    for key_info in info.values():
        result.update(key_info.peft_state_dict())
    return result


def detect_adapter_type(keys: List[str]) -> PeftType:
    # Detect type of adapter by keys
    # Inspired by this:
    # https://github.com/bmaltais/kohya_ss/blob/ed4e3b0239a40506de9a17e550e6cf2d0b867a4f/tools/lycoris_utils.py#L312
    for key in keys:
        if "alpha" in key:
            continue
        elif any(x in key for x in ["lora_down", "lora_up"]):
            # LoRA
            return PeftType.LORA
        elif any(x in key for x in ["hada_w1", "hada_w2", "hada_t1", "hada_t2"]):
            # LoHa may have the following keys:
            # hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b, hada_t1, hada_t2
            return PeftType.LOHA
        elif any(x in key for x in ["lokr_w1", "lokr_w2", "lokr_t1", "lokr_t2"]):
            # LoKr may have the following keys:
            # lokr_w1, lokr_w2, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t1, lokr_t2
            raise ValueError("Currently LoKr adapters are not implemented")
        elif "diff" in key:
            raise ValueError("Currently full diff adapters are not implemented")
        else:
            raise ValueError("Unkown adapter type, probably not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_checkpoint", default=None, type=str, required=True, help="SD checkpoint to use")

    parser.add_argument(
        "--adapter_path",
        default=None,
        type=str,
        required=True,
        help="Path to downloaded adapter to convert",
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output peft adapter.")

    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    # Load all models that we need to add adapter to
    text_encoder = CLIPTextModel.from_pretrained(args.sd_checkpoint, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.sd_checkpoint, subfolder="unet")

    # Construct possible mapping from kohya keys to peft keys
    models_keys = {}
    for model, model_key, model_name in [
        (text_encoder, PREFIX_TEXT_ENCODER, "text_encoder"),
        (unet, PREFIX_UNET, "unet"),
    ]:
        models_keys.update(
            {
                f"{model_key}.{peft_key}".replace(".", "_"): peft_key
                for peft_key in (x[0] for x in model.named_modules())
            }
        )

    # Store conversion info (model_type -> peft_key -> LoRAInfo | LoHaInfo)
    adapter_info: Dict[str, Dict[str, Union[LoRAInfo, LoHaInfo]]] = {
        "text_encoder": {},
        "unet": {},
    }

    # Open adapter checkpoint
    with safetensors.safe_open(args.adapter_path, framework="pt", device="cpu") as f:
        # Extract information about adapter structure
        metadata = f.metadata()

        # Detect adapter type based on keys
        adapter_type = detect_adapter_type(f.keys())
        adapter_info_cls = {
            PeftType.LORA: LoRAInfo,
            PeftType.LOHA: LoHaInfo,
        }[adapter_type]

        # Iterate through available info and unpack all the values
        for key in f.keys():
            kohya_key, kohya_type = key.split(".")[:2]

            # Find which model this key belongs to
            if kohya_key.startswith(PREFIX_TEXT_ENCODER):
                model_type = "text_encoder"
            elif kohya_key.startswith(PREFIX_UNET):
                model_type = "unet"
            else:
                raise ValueError(f"Cannot determine model for key: {key}")

            # Find corresponding peft key
            if kohya_key not in models_keys:
                raise ValueError(f"Cannot find corresponding key for diffusers/transformers model: {kohya_key}")
            peft_key = models_keys[kohya_key]

            # Create a corresponding adapter info
            if peft_key not in adapter_info[model_type]:
                adapter_info[model_type][peft_key] = adapter_info_cls(kohya_key=kohya_key, peft_key=peft_key)

            tensor = f.get_tensor(key)
            if kohya_type == "alpha":
                adapter_info[model_type][peft_key].alpha = tensor.item()
            elif kohya_type == "lora_down":
                adapter_info[model_type][peft_key].lora_A = tensor
                adapter_info[model_type][peft_key].rank = tensor.shape[0]
            elif kohya_type == "lora_up":
                adapter_info[model_type][peft_key].lora_B = tensor
                adapter_info[model_type][peft_key].rank = tensor.shape[1]
            elif kohya_type == "hada_w1_a":
                adapter_info[model_type][peft_key].hada_w1_a = tensor
            elif kohya_type == "hada_w1_b":
                adapter_info[model_type][peft_key].hada_w1_b = tensor
                adapter_info[model_type][peft_key].rank = tensor.shape[0]
            elif kohya_type == "hada_w2_a":
                adapter_info[model_type][peft_key].hada_w2_a = tensor
            elif kohya_type == "hada_w2_b":
                adapter_info[model_type][peft_key].hada_w2_b = tensor
                adapter_info[model_type][peft_key].rank = tensor.shape[0]
            elif kohya_type == "hada_t1":
                adapter_info[model_type][peft_key].hada_t1 = tensor
                adapter_info[model_type][peft_key].rank = tensor.shape[0]
            elif kohya_type == "hada_t2":
                adapter_info[model_type][peft_key].hada_t2 = tensor
                adapter_info[model_type][peft_key].rank = tensor.shape[0]
            else:
                raise ValueError(f"Unknown weight name in key: {key} - {kohya_type}")

    # Get function which will create adapter config based on extracted info
    construct_config_fn = {
        PeftType.LORA: construct_peft_loraconfig,
        PeftType.LOHA: construct_peft_lohaconfig,
    }[adapter_type]

    # Process each model sequentially
    for model, model_name in [(text_encoder, "text_encoder"), (unet, "unet")]:
        config = construct_config_fn(adapter_info[model_name])
        model = get_peft_model(model, config)
        set_peft_model_state_dict(model, combine_peft_state_dict(adapter_info[model_name]))

        if args.half:
            model.to(torch.float16)

        # Save model to disk
        model.save_pretrained(os.path.join(args.dump_path, model_name))
