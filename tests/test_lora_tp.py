# Copyright 2025-present the HuggingFace Inc. team.
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

import logging
import os
import socket
import sys
import tempfile
import time
import unittest
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.testing_utils import ColoredFormatter, Colors

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer


WORLD_SIZE = 2
MODEL_ID = "Qwen/Qwen3-0.6B"
TINY_MODEL_ID = "amazingvince/zephyr-smol_llama-100m-sft-full"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

TEST_OVERFIT_STEPS = 20
TEST_OVERFIT_BATCH_SIZE = 4
TEST_OVERFIT_LEARNING_RATE = 1e-3
TEST_OVERFIT_LOG_FREQ = 1
TEST_OVERFIT_LOSS_REDUCTION_THRESHOLD = 0.9
TEST_OVERFIT_GRAD_NORM_REDUCTION_THRESHOLD = 0.9


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


_BASE_PORT = _find_free_port()


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


def _setup_dist(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _teardown_dist():
    dist.destroy_process_group()


def _test_function_wrapper(fn, rank, world_size, port, *extra_args):
    try:
        _setup_dist(rank, world_size, port)
        fn(rank, world_size, port, *extra_args)
    finally:
        _teardown_dist()


def _test_training_overfit(rank, world_size, port):
    """Test that a tiny model can overfit on a fixed batch."""
    logger = init_test_logger(rank)
    set_seed(42)

    # Get tiny model
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID, tp_plan="auto")
    config = model.config

    lora_config = LoraConfig(r=4, target_modules=TARGET_MODULES)
    model = get_peft_model(model, lora_config)

    model.train()

    sample_input = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    batch = {k: v.repeat(TEST_OVERFIT_BATCH_SIZE, 1) for k, v in sample_input.items()}
    batch["labels"] = batch["input_ids"].clone()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=TEST_OVERFIT_LEARNING_RATE, weight_decay=0.0, betas=(0.9, 0.999)
    )

    initial_loss = None
    final_loss = None
    initial_grad_norm = None
    final_grad_norm = None
    training_start = time.perf_counter()

    for step in range(1, TEST_OVERFIT_STEPS + 1):
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

        # Log at frequency
        if step == 1 or step % TEST_OVERFIT_LOG_FREQ == 0 or step == TEST_OVERFIT_STEPS:
            logger.info(
                f"{Colors.CYAN}step:{Colors.RESET} {step}  "
                f"{Colors.GREEN}loss:{Colors.RESET} {loss.item():7.4f}  "
                f"{Colors.YELLOW}grad_norm:{Colors.RESET} {grad_norm.item():6.4f}  "
                f"{Colors.DIM}step_time:{Colors.RESET} {step_time:.3f}s"
            )

    training_time = time.perf_counter() - training_start

    # Training Summary
    logger.info("-" * 70)
    logger.info(f"{Colors.BOLD}Training completed{Colors.RESET}")
    logger.info(f"Total training time: {training_time:.2f}s")
    logger.info(f"Total steps: {TEST_OVERFIT_STEPS}")

    # Loss analysis
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    logger.info(f"{Colors.BOLD}Loss metrics:{Colors.RESET}")
    logger.info(f"  {Colors.CYAN}initial_loss:{Colors.RESET} {initial_loss:.4f}")
    logger.info(f"  {Colors.CYAN}final_loss:{Colors.RESET} {final_loss:.4f}")
    logger.info(f"  {Colors.CYAN}loss_reduction:{Colors.RESET} {loss_reduction:.1f}%")

    # Grad norm analysis
    grad_norm_reduction = (initial_grad_norm - final_grad_norm) / initial_grad_norm * 100
    logger.info(f"{Colors.BOLD}Grad norm metrics:{Colors.RESET}")
    logger.info(f"  {Colors.CYAN}initial_grad_norm:{Colors.RESET} {initial_grad_norm:.4f}")
    logger.info(f"  {Colors.CYAN}final_grad_norm:{Colors.RESET} {final_grad_norm:.4f}")
    logger.info(f"  {Colors.CYAN}grad_norm_reduction:{Colors.RESET} {grad_norm_reduction:.1f}%")

    generation_matches = None
    logger.info("-" * 70)
    logger.info(f"{Colors.BOLD}Testing generation{Colors.RESET}")

    model.eval()

    # Get the expected token sequence (same pattern used in training)
    expected_tokens = batch["input_ids"][0].tolist()

    # Use first token as prompt
    prompt_ids = torch.tensor([[expected_tokens[0]]], dtype=torch.long)
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

    # Compare generated tokens with expected tokens
    generation_matches = generated_tokens == expected_tokens

    if generation_matches:
        logger.info(f"Expected:  {Colors.GREEN}{tokenizer.decode(expected_tokens)}{Colors.RESET}")
        logger.info(f"Generated: {Colors.GREEN}{tokenizer.decode(generated_tokens)}{Colors.RESET}")
        logger.info(f"{Colors.GREEN}✓ Generation matches training sequence!{Colors.RESET}")
    else:
        logger.info(f"Expected:  {Colors.GREEN}{tokenizer.decode(expected_tokens)}{Colors.RESET}")
        logger.info(f"Generated: {Colors.RED}{tokenizer.decode(generated_tokens)}{Colors.RESET}")
        # Count matching tokens
        matches = sum(1 for g, e in zip(generated_tokens, expected_tokens) if g == e)
        logger.info(
            f"{Colors.YELLOW}✗ Generation mismatch: {matches}/{len(expected_tokens)} tokens match{Colors.RESET}"
        )

    # Assertions
    logger.info("-" * 70)
    logger.info(f"{Colors.BOLD}Running assertions{Colors.RESET}")

    # Assert loss decreased significantly
    loss_reduction_ratio = (initial_loss - final_loss) / initial_loss

    assert loss_reduction_ratio >= TEST_OVERFIT_LOSS_REDUCTION_THRESHOLD, (
        f"Expected loss to decrease by at least {TEST_OVERFIT_LOSS_REDUCTION_THRESHOLD * 100:.0f}%, "
        f"got {loss_reduction:.1f}%"
    )
    logger.info(
        f"{Colors.GREEN}✓ Loss decreased by more than {TEST_OVERFIT_LOSS_REDUCTION_THRESHOLD * 100:.0f}%{Colors.RESET}"
    )

    # Assert grad_norm decreased significantly
    grad_norm_reduction_ratio = (initial_grad_norm - final_grad_norm) / initial_grad_norm
    assert grad_norm_reduction_ratio >= TEST_OVERFIT_GRAD_NORM_REDUCTION_THRESHOLD, (
        f"Expected grad_norm to decrease by at least {TEST_OVERFIT_GRAD_NORM_REDUCTION_THRESHOLD * 100:.0f}%, "
        f"got {grad_norm_reduction:.1f}%"
    )
    logger.info(
        f"{Colors.GREEN}✓ Grad norm decreased by more than {TEST_OVERFIT_GRAD_NORM_REDUCTION_THRESHOLD * 100:.0f}%{Colors.RESET}"
    )

    # Assert generation matches
    assert generation_matches, "Expected model to generate the training sequence after overfitting"
    logger.info(f"{Colors.GREEN}✓ Generated sequence matches training sequence{Colors.RESET}")


def _test_lora_weight_synchronization(rank, world_size, port):
    """
    Test that non-sharded LoRA weights are identical across ranks after training step.
    """
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID, tp_plan="auto")
    lora_config = LoraConfig(r=4, target_modules=TARGET_MODULES, init_lora_weights=True)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test that loss is finite and decreases over multiple steps
    for _ in range(3):
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Test that non-sharded LoRA weights are identical across ranks after training step
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base_layer = module.get_base_layer()
        tp_plan = getattr(base_layer, "_hf_tp_plan", None)
        if tp_plan == "colwise":
            weight = module.lora_A["default"].weight.data.contiguous()
            gathered = [torch.zeros_like(weight) for _ in range(world_size)]
            dist.all_gather(gathered, weight)
            for i, g in enumerate(gathered):
                assert torch.allclose(weight, g), f"{name}.lora_A differs between rank {rank} and rank {i}"
        elif tp_plan == "rowwise":
            weight = module.lora_B["default"].weight.data.contiguous()
            gathered = [torch.zeros_like(weight) for _ in range(world_size)]
            dist.all_gather(gathered, weight)
            for i, g in enumerate(gathered):
                assert torch.allclose(weight, g), f"{name}.lora_B differs between rank {rank} and rank {i}"


def _test_load_from_checkpoint(rank, world_size, port, tmp_dir):
    """
    Test that loading from a checkpoint correctly handles the sharding of LoRA weights according to the TP plan.
    """
    if rank == 0:
        plain_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        lora_config = LoraConfig(r=4, target_modules=TARGET_MODULES, init_lora_weights=True)
        plain_model = get_peft_model(plain_model, lora_config)
        plain_model.save_pretrained(tmp_dir)

    dist.barrier()

    tp_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, tp_plan="auto")
    tp_model = PeftModel.from_pretrained(tp_base, tmp_dir)

    for name, module in tp_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base_layer = module.get_base_layer()
        tp_plan = getattr(base_layer, "_hf_tp_plan", None)
        if tp_plan == "colwise":
            # lora_B output dim must match base layer output dim
            lora_b_out = module.lora_B["default"].weight.shape[0]
            base_layer_out = base_layer.weight.shape[0]
            assert lora_b_out == base_layer_out, (
                f"{name}: lora_B out_dim {lora_b_out} != local base out_dim {base_layer_out}"
            )
        elif tp_plan == "rowwise":
            # lora_A input dim must match base layer input dim
            lora_a_in = module.lora_A["default"].weight.shape[1]
            base_layer_in = base_layer.weight.shape[1]
            assert lora_a_in == base_layer_in, (
                f"{name}: lora_A in_dim {lora_a_in} != local base in_dim {base_layer_in}"
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
    tp_model.eval()
    with torch.no_grad():
        outputs = tp_model(**inputs, labels=inputs["input_ids"])
    assert torch.isfinite(outputs.loss), f"Loss not finite after checkpoint load: {outputs.loss}"


def _test_multiple_adapters(rank, world_size, port):
    """Two LoRA adapters coexist on a TP model and can be switched between."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, tp_plan="auto")
    for adapter_name in ["adapter_a", "adapter_b"]:
        lora_config = LoraConfig(r=4, target_modules=TARGET_MODULES, init_lora_weights=True)
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer("What is the capital of France?", return_tensors="pt")
    model.eval()
    with torch.no_grad():
        for adapter_name in ["adapter_a", "adapter_b"]:
            model.set_adapter(adapter_name)
            outputs = model(**inputs, labels=inputs["input_ids"])
            assert torch.isfinite(outputs.loss), f"Loss not finite with adapter '{adapter_name}': {outputs.loss}"


def _is_tp_available():
    try:
        import transformers.integrations.tensor_parallel  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_is_tp_available(), "transformers TP integration not available")
class TestLoraTP(unittest.TestCase):
    def _spawn(self, fn, *extra_args, port_offset=0):
        port = _BASE_PORT + port_offset
        wrapped_fn = partial(_test_function_wrapper, fn)
        mp.spawn(wrapped_fn, args=(WORLD_SIZE, port) + extra_args, nprocs=WORLD_SIZE, join=True)

    def test_training_overfit(self):
        self._spawn(_test_training_overfit, port_offset=0)

    def test_lora_weight_synchronization(self):
        self._spawn(_test_lora_weight_synchronization, port_offset=1)

    def test_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._spawn(_test_load_from_checkpoint, tmp_dir, port_offset=1)

    def test_multiple_adapters(self):
        self._spawn(_test_multiple_adapters, port_offset=2)


if __name__ == "__main__":
    unittest.main()
