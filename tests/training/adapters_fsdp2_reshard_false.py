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
Test that LoRA adapters stay trainable after a reference pass with `disable_adapter()` under FSDP2 with
`reshard_after_forward=False`, see #3800.

Run with:
    accelerate launch --config_file tests/training/fsdp2_config.yaml tests/training/adapters_fsdp2_reshard_false.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM

from peft import LoraConfig, get_peft_model


# q_proj and v_proj are in the FSDP modules of the decoder layers, lm_head is in the root FSDP module, which is the
# PeftModel
TARGET_MODULES = ["q_proj", "v_proj", "lm_head"]
STEPS = 4
LEARNING_RATE = 0.1
KL_COEF = 0.1


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def get_model(device, mesh=None):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(device)
    model = get_peft_model(model, LoraConfig(r=8, target_modules=TARGET_MODULES, init_lora_weights=False))
    if mesh is not None:
        for layer in model.base_model.model.model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
    return model


def get_lora_params(model):
    # before the first forward, these are the sharded parameters that FSDP2 keeps
    return [param for name, param in model.named_parameters() if "lora_" in name]


def get_full_tensor(tensor):
    if isinstance(tensor, DTensor):
        # clone, as full_tensor() can return a view of the local shard
        return tensor.full_tensor().clone()
    return tensor.detach().clone()


def train_step(model, optimizer, input_ids, ref_logits=None):
    optimizer.zero_grad()
    output = model(input_ids, labels=input_ids)
    loss = output.loss
    if ref_logits is not None:
        # the KL term to the logits of the reference pass makes the loss depend on that pass, so the comparison with
        # the unsharded model also checks that the adapters were disabled for it
        log_probs = F.log_softmax(output.logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        loss = loss + KL_COEF * F.kl_div(log_probs, ref_log_probs, log_target=True, reduction="batchmean")
    assert loss.requires_grad, "the adapters are frozen"
    loss.backward()
    optimizer.step()
    return loss.item()


def test_disable_adapter_keeps_adapters_trainable(device, mesh, input_ids):
    # a reference pass with the adapters disabled after a training step, then several training steps
    model = get_model(device, mesh)
    lora_params = get_lora_params(model)
    optimizer = torch.optim.SGD(lora_params, lr=LEARNING_RATE)
    train_step(model, optimizer, input_ids)
    with torch.no_grad(), model.disable_adapter():
        model(input_ids)
    num_trainable = sum(param.requires_grad for param in lora_params)
    assert num_trainable == len(lora_params), f"{num_trainable}/{len(lora_params)} LoRA parameters require grad"

    for step in range(STEPS):
        before = [get_full_tensor(param) for param in lora_params]
        train_step(model, optimizer, input_ids)
        after = [get_full_tensor(param) for param in lora_params]
        num_updated = sum(not torch.equal(x, y) for x, y in zip(before, after))
        assert num_updated == len(lora_params), f"step {step}: {num_updated}/{len(lora_params)} LoRA params updated"


def test_training_matches_unsharded_model(device, mesh, input_ids):
    # an RL-like loop: one reference pass with the adapters disabled, followed by two training steps with a KL term;
    # every rank uses the same batch, so the FSDP2 run has to match the unsharded model
    results = []
    for model_mesh in (mesh, None):
        model = get_model(device, model_mesh)
        lora_params = get_lora_params(model)
        optimizer = torch.optim.SGD(lora_params, lr=LEARNING_RATE)
        losses = []
        for _ in range(STEPS // 2):
            with torch.no_grad(), model.disable_adapter():
                ref_logits = model(input_ids).logits
            for _ in range(2):
                losses.append(train_step(model, optimizer, input_ids, ref_logits=ref_logits))
        results.append((losses, [get_full_tensor(param) for param in lora_params]))

    (fsdp_losses, fsdp_params), (losses, params) = results
    log(f"losses with FSDP2: {fsdp_losses}, without: {losses}")
    torch.testing.assert_close(fsdp_losses, losses, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(fsdp_params, params, rtol=1e-4, atol=1e-5)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.accelerator.is_available():
        device = torch.device(torch.accelerator.current_accelerator().type, local_rank)
        torch.accelerator.set_device_index(local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=dist.get_default_backend_for_device(device))
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    input_ids = torch.randint(0, 128, (4, 16), generator=torch.Generator().manual_seed(0)).to(device)

    test_disable_adapter_keeps_adapters_trainable(device, mesh, input_ids)
    test_training_matches_unsharded_model(device, mesh, input_ids)
    log("All checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
