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
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageClassification

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
    cfg = LoraConfig(r=rank, target_modules=["c_attn"], fan_in_fan_out=True, init_lora_weights=False)
    model = AutoModelForCausalLM.from_pretrained(  # FRESH each call
        "hf-internal-testing/tiny-random-gpt2"
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.save_pretrained(out_dir)


@pytest.fixture(scope="module")
def ts_adapters(workdir: Path, base_model):
    """
    Build 3 task-specific adapters and return their absolute paths
    """
    abs_paths = []
    for i in range(3):
        sub = f"{workdir}/ts{i}"
        _create_and_save_adapter(sub)
        abs_paths.append(sub)
    return abs_paths


@pytest.fixture(scope="module")
def gen_adapter(workdir: Path, base_model):
    """Build 1 general-knowledge adapter and return its absolute path list."""
    sub = f"{workdir}/gen0"
    _create_and_save_adapter(sub)
    return [sub]  # list because create_arrow_model expects list


# ─── Tests ────────────────────────────────────────────────────────────


class TestArrowRouting:
    def test_arrow_differs_with_extra_expert(self, ts_adapters):
        """
        Arrow with 2 experts vs Arrow with 3 experts must produce different logits.
        """
        # Arrow over first 2 experts
        cfg_small = ArrowConfig(top_k=2)
        m_small = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters[:2],
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            arrow_config=cfg_small,
        ).eval()

        # Arrow over all 3 experts
        cfg_big = ArrowConfig(top_k=2)
        m_big = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            arrow_config=cfg_big,
        ).eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert not torch.allclose(m_small(x).logits, m_big(x).logits)

    def test_arrow_gks_with_load_adapter_later_with_forward(self, ts_adapters, gen_adapter):
        """
        Loading the last expert after creating the arrow model should produce the same result as loading all the
        experts at once in create_arrow_model(), when forward path is called before adding the new adapter.
        """
        # Arrow over all three experts
        cfg_big = ArrowConfig(top_k=2, use_gks=True)
        m_big = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            general_adapter_paths=gen_adapter,
            gen_repo_id="/".join(gen_adapter[0].split("/")[:2]),
            arrow_config=cfg_big,
        ).eval()

        # Arrow over all 2 experts + loading the third expert later
        cfg_small_later_big = ArrowConfig(top_k=2, use_gks=True)
        m_small_later_big = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters[:2],
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            general_adapter_paths=gen_adapter,
            gen_repo_id="/".join(gen_adapter[0].split("/")[:2]),
            arrow_config=cfg_small_later_big,
        )

        # Ensuring that the prototypes and gks are done one time by running a forward path
        x = torch.ones(1, 4, dtype=torch.long)
        m_small_later_big(x)

        # Now loading the third expert
        parts = ts_adapters[-1].split("/")
        repo_id = "/".join(parts[:2])
        sub_folder = "/".join(parts[2:])
        m_small_later_big.load_adapter(
            model_id=repo_id,
            adapter_name="new_added_ts_expert",
            subfolder=sub_folder,
        )
        m_small_later_big.eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert torch.allclose(m_big(x).logits, m_small_later_big(x).logits)

    def test_arrow_gks_with_load_adapter_later_without_forward(self, ts_adapters, gen_adapter):
        """
        Loading the last expert after creating the arrow model should produce the same result as loading all the
        experts at once in create_arrow_model()
        """
        # Arrow over all three experts
        cfg_big = ArrowConfig(top_k=2, use_gks=True)
        m_big = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            general_adapter_paths=gen_adapter,
            gen_repo_id="/".join(gen_adapter[0].split("/")[:2]),
            arrow_config=cfg_big,
        ).eval()

        # Arrow over all 2 experts + loading the third expert later
        cfg_small_later_big = ArrowConfig(top_k=2, use_gks=True)
        m_small_later_big = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters[:2],
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            general_adapter_paths=gen_adapter,
            gen_repo_id="/".join(gen_adapter[0].split("/")[:2]),
            arrow_config=cfg_small_later_big,
        )

        # Now loading the third expert
        parts = ts_adapters[-1].split("/")
        repo_id = "/".join(parts[:2])
        sub_folder = "/".join(parts[2:])
        m_small_later_big.load_adapter(
            model_id=repo_id,
            adapter_name="new_added_ts_expert",
            subfolder=sub_folder,
        )
        m_small_later_big.eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert torch.allclose(m_big(x).logits, m_small_later_big(x).logits)

    def test_genknowsub_changes_output(self, ts_adapters, gen_adapter):
        """
        Arrow+GenKnowSub vs plain Arrow must change logits.
        """
        # Plain Arrow
        cfg_plain = ArrowConfig(top_k=2)
        m_plain = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            arrow_config=cfg_plain,
        ).eval()

        # Arrow + GenKnowSub
        cfg_gks = ArrowConfig(top_k=2, use_gks=True)
        m_gks = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters,
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            general_adapter_paths=gen_adapter,
            gen_repo_id="/".join(gen_adapter[0].split("/")[:2]),
            arrow_config=cfg_gks,
        ).eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert not torch.allclose(m_plain(x).logits, m_gks(x).logits)

    def test_merging_adapters_raise_typeerror_in_arrow(self, ts_adapters):
        """
        Merging/unmerging is not allowed while an ArrowLinearLayer is loaded on the model and active.
        """
        # Arrow over first 2 experts
        cfg_small = ArrowConfig(top_k=2)
        m_small = create_arrow_model(
            base_model=AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2"),
            task_specific_adapter_paths=ts_adapters[:2],
            ts_repo_id="/".join(ts_adapters[0].split("/")[:2]),
            arrow_config=cfg_small,
        ).eval()

        with pytest.raises(RuntimeError, match=r"Cannot merge an active Arrow router adapter"):
            m_small.merge_and_unload()

    def test_conv2d_targets_raise_typeerror_in_arrow(self, workdir):
        """
        Adapters applied to Conv2d must be rejected by create_arrow_model() which enforces Linear/Linear4bit-only
        targets.
        """

        model_id = "hf-internal-testing/tiny-random-ResNetForImageClassification"

        # Find a Conv2d module name in the ResNet test model
        probe = AutoModelForImageClassification.from_pretrained(model_id)
        conv_names = [n.split(".")[-1] for n, m in probe.named_modules() if isinstance(m, torch.nn.Conv2d)]
        assert len(conv_names) > 0, "No Conv2d modules found in the ResNet test model."
        target_name = conv_names[0]

        # Build a LoRA adapter targeting a Conv2d
        cfg = LoraConfig(r=4, target_modules=[target_name], init_lora_weights=False)
        base = AutoModelForImageClassification.from_pretrained(model_id)
        peft_model = get_peft_model(base, cfg)

        conv_dir = workdir / "cv0"
        peft_model.save_pretrained(conv_dir)

        # Expect create_arrow_model to raise TypeError
        with pytest.raises(TypeError, match=r"LoRA adapters must only target Linear"):
            _ = create_arrow_model(
                base_model=AutoModelForImageClassification.from_pretrained(model_id),
                task_specific_adapter_paths=[str(conv_dir)],
                ts_repo_id=str(workdir),
                arrow_config=ArrowConfig(top_k=1),
            )

    def test_arrow_forward_float16_no_autocast_with_merging(self, ts_adapters):
        """
        Run Arrow in float16 with autocast disabled; forward should work, while merge/unmerge operations must raise for
        Arrow models.
        """
        import platform

        try:
            _ = torch.zeros(1, dtype=torch.float16)
        except Exception:
            pytest.skip(reason="Test requires float16 support")

        if platform.system() == "Darwin":
            pytest.skip(reason="MacOS does not support multiple ops in float16")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "hf-internal-testing/tiny-random-gpt2"

        # Create base in fp16 (no manual assignment to .dtype)
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

        cfg = ArrowConfig(top_k=2)
        ts_repo_id = "/".join(ts_adapters[0].split("/")[:2])

        # Build Arrow model and disable adapter dtype autocast
        model = create_arrow_model(
            base_model=base,
            task_specific_adapter_paths=ts_adapters,
            ts_repo_id=ts_repo_id,
            arrow_config=cfg,
            autocast_adapter_dtype=False,
            torch_dtype=torch.float16,
        ).eval()

        X = {
            "input_ids": torch.ones(1, 4, dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, 4, dtype=torch.long, device=device),
        }

        # Forward should work in fp16
        _ = model(**X)

        # Merge must fail on Arrow models
        with pytest.raises(RuntimeError, match=r"Cannot merge an active Arrow router adapter"):
            model.merge_adapter(safe_merge=False)

        with pytest.raises(RuntimeError, match=r"Cannot merge an active Arrow router adapter"):
            _ = model.merge_and_unload()
