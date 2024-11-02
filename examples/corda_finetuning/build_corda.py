import argparse
import os

import numpy as np
import torch
from cordalib.datautils import get_calib_data
from cordalib.evaluate_utils import evaluate_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft.mapping import get_peft_model
from peft.tuners.lora.config import CordaConfig, LoraConfig


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
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # Collect data
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.calib_loader_size, seed=args.seed)

    # Perform decomposition
    config = LoraConfig(
        init_lora_weights="corda",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        r=args.r,
        lora_alpha=args.alpha,
        corda_config=CordaConfig(
            run_model=lambda: run_model(model, calib_loader),
            sample_count=args.calib_loader_size,
            corda_method="ipm" if args.first_eigen else "kpm",
            corda_rank=args.r,
        ),
    )
    model = get_peft_model(model, config)

    # Evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "mmlu" if args.eval_mmlu else "",
        eval_ppl="wikitext2,ptb",
        limit=-1,
    )
    print(result)

    # Save as hugging face model
    if args.save_model:
        assert args.save_path is not None
        save_path = args.save_path

        # Save CorDA modules
        model.peft_config["default"].init_lora_weights = True
        model.save_pretrained(os.path.join(save_path, "corda_init"))

        # Save residual model
        model = model.unload()
        model.save_pretrained(model, save_path)

        # Save tokenizer
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
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
