# tests/test_arrow.py
# ---------------------------------------------------------------------
# Stand-alone tests for Arrow routing & GenKnowSub
# ---------------------------------------------------------------------

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
from peft.tuners.lora import ArrowConfig, create_arrow_model


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_model():
    """Tiny GPT-2 variant that ships with HF for unit tests."""
    return AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """
    Create a temp directory and chdir into it for the duration of the module. This lets us use *relative* adapter paths
    like 'ts0' so split_repo_and_subfolder() recognises them as LOCAL repo-roots.
    """
    wd = tmp_path_factory.mktemp("arrow_workdir")
    old_cwd = os.getcwd()
    os.chdir(wd)
    yield Path(wd)
    os.chdir(old_cwd)
    # (pytest will auto-delete wd)


def _create_and_save_adapter(out_dir: Path, rank: int = 4):
    """Helper: build a LoRA adapter around `model` and save into `out_dir`."""
    # fan_in_fan_out is set to True because of GPT2 model that we use to avoid warning
    cfg = LoraConfig(r=rank, target_modules=["c_attn"], fan_in_fan_out=True)
    model = AutoModelForCausalLM.from_pretrained(  # FRESH each call
        "hf-internal-testing/tiny-random-gpt2"
    )
    peft_model = get_peft_model(model, cfg)
    # Explicitly initialise both A and B. If using the lora_init_weight solely, the B is always initialised as 0
    for layer in peft_model.modules():
        if hasattr(layer, "lora_A"):
            for x in layer.lora_A.values():
                torch.nn.init.uniform_(x.weight, -0.1, 0.1)
            for x in layer.lora_B.values():
                torch.nn.init.uniform_(x.weight, -0.1, 0.1)
    peft_model.save_pretrained(out_dir)


@pytest.fixture(scope="module")
def ts_adapters(workdir: Path, base_model):
    """
    Build 3 task-specific adapters and return their *relative* paths ('ts0', 'ts1', 'ts2').
    """
    rel_paths = []
    for i in range(3):
        sub = workdir / f"ts{i}"
        _create_and_save_adapter(sub)
        rel_paths.append(f"ts{i}")
    return rel_paths


@pytest.fixture(scope="module")
def gen_adapter(workdir: Path, base_model):
    """Build 1 general-knowledge adapter and return its relative path list."""
    sub = workdir / "gen0"
    _create_and_save_adapter(sub)
    return ["gen0"]  # list because create_arrow_model expects list


# ─── Tests ────────────────────────────────────────────────────────────


class TestArrowRouting:
    def test_arrow_differs_with_extra_expert(self, ts_adapters):
        """
        Arrow with 2 experts vs Arrow with 3 experts must produce different logits.
        """
        # Arrow over first 2 experts
        cfg_small = ArrowConfig(arrow_top_k=2)
        m_small = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters[:2],
            arrow_config=cfg_small,
        ).eval()

        # Arrow over all 3 experts
        cfg_big = ArrowConfig(arrow_top_k=2)
        m_big = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            arrow_config=cfg_big,
        ).eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert not torch.allclose(m_small(x).logits, m_big(x).logits)

    def test_genknowsub_changes_output(self, ts_adapters, gen_adapter):
        """
        Arrow+GenKnowSub vs plain Arrow must change logits.
        """
        # Plain Arrow
        cfg_plain = ArrowConfig(arrow_top_k=2)
        m_plain = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            arrow_config=cfg_plain,
        ).eval()

        # Arrow + GenKnowSub
        cfg_gks = ArrowConfig(arrow_top_k=2, use_gks=True)
        m_gks = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_gks,
        ).eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert not torch.allclose(m_plain(x).logits, m_gks(x).logits)
