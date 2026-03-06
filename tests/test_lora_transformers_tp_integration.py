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

import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer


WORLD_SIZE = 2
MODEL_ID = "Qwen/Qwen3-0.6B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


_BASE_PORT = _find_free_port()


def _setup_dist(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _teardown_dist():
    dist.destroy_process_group()


def _test_training(rank, world_size, port):
    """
    Test multiple aspects of training:
        1. TP LoRA model can do a training step,
        2. Loss is finite and decreases over multiple steps,
        3. Non-sharded LoRA weights are identical across ranks after training step
    """
    _setup_dist(rank, world_size, port)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, tp_plan="auto")
    lora_config = LoraConfig(r=4, target_modules=TARGET_MODULES, init_lora_weights=True)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test that loss is finite and decreases over multiple steps
    prev_loss = None
    for _ in range(3):
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if prev_loss is not None:
            assert loss.item() < prev_loss, "Loss did not decrease"
        prev_loss = loss.item()

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

    _teardown_dist()


def _test_load_from_checkpoint(rank, world_size, port, tmp_dir):
    """
    Test that loading from a checkpoint correctly handles the sharding of LoRA weights
    according to the TP plan.
    """
    _setup_dist(rank, world_size, port)

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

    _teardown_dist()


def _test_multiple_adapters(rank, world_size, port):
    """Two LoRA adapters coexist on a TP model and can be switched between."""
    _setup_dist(rank, world_size, port)

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

    _teardown_dist()


def _is_tp_available():
    try:
        import transformers.integrations.tensor_parallel  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_is_tp_available(), "transformers TP integration not available")
class TestLoraAndTransformersTPIntegration(unittest.TestCase):
    def _spawn(self, fn, *extra_args, port_offset=0):
        port = _BASE_PORT + port_offset
        mp.spawn(fn, args=(WORLD_SIZE, port) + extra_args, nprocs=WORLD_SIZE, join=True)

    def test_training(self):
        self._spawn(_test_training, port_offset=0)

    def test_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._spawn(_test_load_from_checkpoint, tmp_dir, port_offset=1)

    def test_multiple_adapters(self):
        self._spawn(_test_multiple_adapters, port_offset=2)


if __name__ == "__main__":
    unittest.main()
