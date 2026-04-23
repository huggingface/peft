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

import argparse
import os

import numpy as np
import torch
from datautils import get_calib_data
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import get_peft_model
from peft.tuners.lora.config import CordaConfig, LoraConfig
from peft.tuners.lora.corda import preprocess_corda


@torch.no_grad()
def run_model(model, calib_loader):
    model.eval()
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)


def main(args):
    # Setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    elif torch.xpu.is_available():
        torch.xpu.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.float16, trust_remote_code=True
    )

    # Collect data
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.calib_loader_size, seed=args.seed)

    # Evaluate the original model
    print("\n---- model before svd ---\n")
    print(model)

    # Perform decomposition
    corda_config = CordaConfig(
        corda_method="ipm" if args.first_eigen else "kpm",
    )
    lora_config = LoraConfig(
        init_lora_weights="corda",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        r=args.r,
        lora_alpha=args.r,
        corda_config=corda_config,
    )
    preprocess_corda(
        model,
        lora_config,
        run_model=lambda: run_model(model, calib_loader),
    )
    model = get_peft_model(model, lora_config)

    # Evaluate again to check if the model is consistent
    # Using `model.model` here because `get_peft_model` wraps a layer to the model
    print("\n---- model after svd ---\n")
    print(model)

    # Save as hugging face model
    if args.save_model:
        assert args.save_path is not None
        save_path = args.save_path

        # Save CorDA modules
        model.peft_config["default"].init_lora_weights = True
        model.save_pretrained(os.path.join(save_path, "corda_init"))

        # Save residual model
        model = model.unload()
        model.save_pretrained(save_path)

        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        print(f"Done building CorDA huggingface model in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--calib_loader_size",
        type=int,
        default=256,
        help="number of samples used for covariance matrices",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=[
            "wikitext2",
            "c4",
            "ptb",
            "traivia_qa",
            "nqopen",
            "MetaMATH",
            "codefeedback",
            "WizLMinstruct",
            "alpaca",
        ],
        help="calibration dataset",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--first_eigen",
        action="store_true",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    main(args)
