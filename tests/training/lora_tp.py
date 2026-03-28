# Copyright 2026-present the HuggingFace Inc. team.
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

"""
Test that a LoRA model on a tensor-parallel base model can overfit a fixed batch.

Run with:
    torchrun --nproc_per_node=2 tests/training/lora_tp.py --model_id <model_id>
"""

import argparse
import logging
import sys
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.testing_utils import ColoredFormatter, Colors

from peft import LoraConfig, get_peft_model
from peft.import_utils import is_transformers_ge_v5_4_0


TINY_MODEL_ID = "peft-internal-testing/zephyr-smol_llama-100m-sft-full"
TARGET_MODULES = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj"]
TP_PLAN = {
    "model.embed_tokens": "embedding_rowwise",
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
}
STEPS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
LOSS_REDUCTION_THRESHOLD = 0.9
GRAD_NORM_REDUCTION_THRESHOLD = 0.9


def init_test_logger(rank):
    # Taken from transformers.testing_utils.init_test_logger but modified:
    #   1. To use the proper logger name for this test file
    #   2. To handle multiprocessing without duplicate logs
    logger = logging.getLogger("peft.training_test")
    level = logging.INFO if rank == 0 else 100  # Higher than CRITICAL to suppress logs from non-master processes
    logger.setLevel(level)

    # Only add handler if not already present (avoid duplicate handlers on repeated calls)
    if not logger.handlers:
        # Use stderr instead of stdout - pytest-xdist captures stdout which can cause deadlocks
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)

        # Use colored formatter if terminal supports it, plain otherwise
        if sys.stderr.isatty():
            formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False  # Don't propagate to root logger to avoid duplicate output
    return logger


def main(model_id: str, target_modules: list[str]):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    logger = init_test_logger(rank)
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, tp_plan=TP_PLAN)
    config = model.config

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = model.to(device)

    lora_config = LoraConfig(r=4, target_modules=target_modules)
    model = get_peft_model(model, lora_config)
    model.train()

    sample_input = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    batch = {k: v.repeat(BATCH_SIZE, 1).to(device) for k, v in sample_input.items()}
    batch["labels"] = batch["input_ids"].clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, betas=(0.9, 0.999))

    initial_loss = None
    final_loss = None
    initial_grad_norm = None
    final_grad_norm = None
    training_start = time.perf_counter()

    for step in range(1, STEPS + 1):
        step_start = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if initial_grad_norm is None:
            initial_grad_norm = grad_norm.item()
        final_grad_norm = grad_norm.item()

        optimizer.step()

        step_time = time.perf_counter() - step_start
        logger.info(
            f"{Colors.CYAN}step:{Colors.RESET} {step}  "
            f"{Colors.GREEN}loss:{Colors.RESET} {loss.item():7.4f}  "
            f"{Colors.YELLOW}grad_norm:{Colors.RESET} {grad_norm.item():6.4f}  "
            f"{Colors.DIM}step_time:{Colors.RESET} {step_time:.3f}s"
        )

    training_time = time.perf_counter() - training_start

    logger.info("-" * 70)
    logger.info(f"{Colors.BOLD}Training completed{Colors.RESET}")
    logger.info(f"Total training time: {training_time:.2f}s")
    logger.info(f"Total steps: {STEPS}")

    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    logger.info(f"{Colors.BOLD}Loss metrics:{Colors.RESET}")
    logger.info(f"  {Colors.CYAN}initial_loss:{Colors.RESET} {initial_loss:.4f}")
    logger.info(f"  {Colors.CYAN}final_loss:{Colors.RESET} {final_loss:.4f}")
    logger.info(f"  {Colors.CYAN}loss_reduction:{Colors.RESET} {loss_reduction:.1f}%")

    grad_norm_reduction = (initial_grad_norm - final_grad_norm) / initial_grad_norm * 100
    logger.info(f"{Colors.BOLD}Grad norm metrics:{Colors.RESET}")
    logger.info(f"  {Colors.CYAN}initial_grad_norm:{Colors.RESET} {initial_grad_norm:.4f}")
    logger.info(f"  {Colors.CYAN}final_grad_norm:{Colors.RESET} {final_grad_norm:.4f}")
    logger.info(f"  {Colors.CYAN}grad_norm_reduction:{Colors.RESET} {grad_norm_reduction:.1f}%")

    logger.info("-" * 70)
    logger.info(f"{Colors.BOLD}Testing generation{Colors.RESET}")

    model.eval()
    expected_tokens = batch["input_ids"][0].tolist()
    prompt_ids = torch.tensor([[expected_tokens[0]]], dtype=torch.long)
    prompt_ids = prompt_ids.to(device)
    num_tokens_to_generate = len(expected_tokens) - 1

    logger.info(f"Prompt: {tokenizer.decode([expected_tokens[0]])}")

    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=num_tokens_to_generate,
            do_sample=False,
            pad_token_id=config.pad_token_id if hasattr(config, "pad_token_id") else 0,
            eos_token_id=0,
            use_cache=False,
        )

    generated_tokens = generated_ids[0].tolist()
    generation_matches = generated_tokens == expected_tokens

    if generation_matches:
        logger.info(f"Expected:  {Colors.GREEN}{tokenizer.decode(expected_tokens)}{Colors.RESET}")
        logger.info(f"Generated: {Colors.GREEN}{tokenizer.decode(generated_tokens)}{Colors.RESET}")
        logger.info(f"{Colors.GREEN}✓ Generation matches training sequence!{Colors.RESET}")
    else:
        logger.info(f"Expected:  {Colors.GREEN}{tokenizer.decode(expected_tokens)}{Colors.RESET}")
        logger.info(f"Generated: {Colors.RED}{tokenizer.decode(generated_tokens)}{Colors.RESET}")
        matches = sum(1 for g, e in zip(generated_tokens, expected_tokens) if g == e)
        logger.info(
            f"{Colors.YELLOW}✗ Generation mismatch: {matches}/{len(expected_tokens)} tokens match{Colors.RESET}"
        )

    logger.info("-" * 70)
    logger.info(f"{Colors.BOLD}Running assertions{Colors.RESET}")

    loss_reduction_ratio = (initial_loss - final_loss) / initial_loss
    assert loss_reduction_ratio >= LOSS_REDUCTION_THRESHOLD, (
        f"Expected loss to decrease by at least {LOSS_REDUCTION_THRESHOLD * 100:.0f}%, got {loss_reduction:.1f}%"
    )
    logger.info(f"{Colors.GREEN}✓ Loss decreased by more than {LOSS_REDUCTION_THRESHOLD * 100:.0f}%{Colors.RESET}")

    grad_norm_reduction_ratio = (initial_grad_norm - final_grad_norm) / initial_grad_norm
    assert grad_norm_reduction_ratio >= GRAD_NORM_REDUCTION_THRESHOLD, (
        f"Expected grad_norm to decrease by at least {GRAD_NORM_REDUCTION_THRESHOLD * 100:.0f}%, "
        f"got {grad_norm_reduction:.1f}%"
    )
    logger.info(
        f"{Colors.GREEN}✓ Grad norm decreased by more than {GRAD_NORM_REDUCTION_THRESHOLD * 100:.0f}%{Colors.RESET}"
    )

    assert generation_matches, "Expected model to generate the training sequence after overfitting"
    logger.info(f"{Colors.GREEN}✓ Generated sequence matches training sequence{Colors.RESET}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=False, default=TINY_MODEL_ID)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        required=False,
        default=TARGET_MODULES,
        help="List of target modules for LoRA adaptation",
    )
    args = parser.parse_args()
    if not is_transformers_ge_v5_4_0:
        print("This test requires transformers v5.4.0 or higher")
    else:
        main(model_id=args.model_id, target_modules=args.target_modules)
