# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import pytest
import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper


# Seeding for deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@pytest.fixture(scope="module")
def setup_dist():
    """Setup and teardown a single-GPU FSDP process group."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FSDP tests")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    if not torch.distributed.is_initialized():
        init_process_group(backend="nccl", rank=0, world_size=1)

    yield

    if torch.distributed.is_initialized():
        destroy_process_group()


@pytest.fixture(scope="module")
def tokenizer():
    """Reusable tokenizer for consistency across tests."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def ensure_lora_initialized(model):
    """Ensure LoRA adapter weights are non-zero for stable testing."""
    for _, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for adapter_name in module.lora_A:
                lora_A = module.lora_A[adapter_name]
                lora_B = module.lora_B[adapter_name]
                if getattr(lora_A, "weight", None) is not None and lora_A.weight.std() < 1e-6:
                    nn.init.kaiming_uniform_(lora_A.weight, a=1)
                if getattr(lora_B, "weight", None) is not None and lora_B.weight.std() < 1e-6:
                    nn.init.kaiming_uniform_(lora_B.weight, a=1)


class TestLoraFSDP:
    """Test LoRA adapter disabling behavior with and without FSDP."""

    def test_disable_adapter_changes_output_and_matches_base_model(self, tokenizer):
        """
        Baseline: verify that disable_adapter:
          * actually changes the output when adapters are enabled vs disabled,
          * and the disabled output approximates the base model output.
        """
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)
        ensure_lora_initialized(model)

        inputs = tokenizer("The future of AI is", return_tensors="pt")

        # Output with adapters enabled
        with torch.no_grad():
            output_with_adapters = model(**inputs).logits.clone()

        # Output with adapters disabled
        with model.disable_adapter():
            with torch.no_grad():
                output_without_adapters = model(**inputs).logits.clone()

        max_diff = torch.abs(output_with_adapters - output_without_adapters).max().item()
        assert max_diff > 1e-3, f"Adapters not affecting output. Max diff: {max_diff}"

        # Compare disabled output to base model (no PEFT)
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        with torch.no_grad():
            base_output = base_model(**inputs).logits

        base_diff = torch.abs(output_without_adapters - base_output).max().item()
        assert base_diff < 1e-5, f"Disabled adapters don't match base model. Diff: {base_diff}"

    def test_disable_adapter_with_fsdp(self, setup_dist, tokenizer):
        """
        Verify that disable_adapter works under FSDP:
          * does not raise the old requires_grad RuntimeError,
          * actually changes the output when adapters are toggled.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for this test")

        torch.cuda.manual_seed_all(42)

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)
        ensure_lora_initialized(model)

        # Wrap with FSDP
        model = FSDP(model, use_orig_params=True).cuda()
        inputs = {k: v.cuda() for k, v in tokenizer("The future of AI is", return_tensors="pt").items()}

        # Ensure disable_adapter context does not produce the old requires_grad error
        try:
            with model.disable_adapter():
                pass
        except RuntimeError as e:
            if "requires_grad" in str(e):
                pytest.fail(f"Got requires_grad RuntimeError that should be fixed: {e}")
            else:
                raise

        # Check that adapters actually change output
        with torch.no_grad():
            output_with_adapters = model(**inputs).logits.clone()

        with model.disable_adapter():
            with torch.no_grad():
                output_without_adapters = model(**inputs).logits.clone()

        max_diff = torch.abs(output_with_adapters - output_without_adapters).max().item()
        assert max_diff > 1e-3, f"FSDP: Adapters not affecting output when disabled. Max diff: {max_diff}."

    def test_modules_to_save_wrapper_handling(self, setup_dist, tokenizer):
        """
        Test that ModulesToSaveWrapper is handled correctly and disable_adapter works without error under FSDP.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for this test")

        torch.cuda.manual_seed_all(42)

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoraConfig(r=8, target_modules=["c_attn"], modules_to_save=["ln_f"])
        model = get_peft_model(model, config)
        ensure_lora_initialized(model)

        # Verify presence of both tuner layer and wrapper
        has_tuner = any(isinstance(m, BaseTunerLayer) for m in model.modules())
        has_wrapper = any(isinstance(m, ModulesToSaveWrapper) for m in model.modules())
        assert has_tuner, "Expected a BaseTunerLayer in the model"
        assert has_wrapper, "Expected a ModulesToSaveWrapper in the model"

        # Wrap with FSDP
        model = FSDP(model, use_orig_params=True).cuda()

        # Use tokenized input instead of raw tensor
        inputs = {k: v.cuda() for k, v in tokenizer("Test", return_tensors="pt").items()}

        with model.disable_adapter():
            _ = model(**inputs)
