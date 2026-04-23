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

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageClassification

from peft import LoraConfig, get_peft_model
from peft.tuners.lora import ArrowConfig, create_arrow_model
from peft.tuners.lora.arrow import _resolve_adapter_source
from tests.testing_utils import hub_online_once


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """
    Create a temp directory and chdir into it for the duration of the module.
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
    model_id = "peft-internal-testing/tiny-random-gpt2"
    with hub_online_once(model_id):
        model = AutoModelForCausalLM.from_pretrained(model_id)
    peft_model = get_peft_model(model, cfg)
    peft_model.save_pretrained(out_dir)


@pytest.fixture(scope="module")
def ts_adapters(workdir: Path):
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
def gen_adapter(workdir: Path):
    """Build 1 general-knowledge adapter and return its absolute path list."""
    sub = f"{workdir}/gen0"
    _create_and_save_adapter(sub)
    return [sub]  # list because create_arrow_model expects list


class TestArrowRouting:
    def test_incompatible_rank_raises(self, workdir: Path):
        """
        Adding adapters with different ranks must raise a ValueError.
        """
        # Create two adapters with different ranks targeting the same modules
        sub_r4 = workdir / "rank4"
        sub_r8 = workdir / "rank8"
        _create_and_save_adapter(sub_r4, rank=4)
        _create_and_save_adapter(sub_r8, rank=8)

        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base = AutoModelForCausalLM.from_pretrained(model_id)

        # Expect create_arrow_model to raise due to rank mismatch
        with pytest.raises(ValueError, match=r"rank mismatch"):
            _ = create_arrow_model(
                base_model=base,
                task_specific_adapter_paths=[str(sub_r4), str(sub_r8)],
                arrow_config=ArrowConfig(top_k=1),
            )

    def test_arrow_differs_with_extra_expert(self, ts_adapters):
        """
        Arrow with 2 experts vs Arrow with 3 experts must produce different logits.
        """
        # Arrow over first 2 experts
        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base_model_1 = AutoModelForCausalLM.from_pretrained(model_id)
            base_model_2 = copy.deepcopy(base_model_1)
        cfg_small = ArrowConfig(top_k=2)
        m_small = create_arrow_model(
            base_model=base_model_1,
            task_specific_adapter_paths=ts_adapters[:2],
            arrow_config=cfg_small,
        ).eval()

        # Arrow over all 3 experts
        cfg_big = ArrowConfig(top_k=2)
        m_big = create_arrow_model(
            base_model=base_model_2,
            task_specific_adapter_paths=ts_adapters,
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
        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base_model_1 = AutoModelForCausalLM.from_pretrained(model_id)
            base_model_2 = copy.deepcopy(base_model_1)
        cfg_big = ArrowConfig(top_k=2, use_gks=True, rng_seed=42)
        m_big = create_arrow_model(
            base_model=base_model_1,
            task_specific_adapter_paths=ts_adapters,
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_big,
        ).eval()

        # Arrow over all 2 experts + loading the third expert later
        cfg_small_later_big = ArrowConfig(top_k=2, use_gks=True, rng_seed=42)
        m_small_later_big = create_arrow_model(
            base_model=base_model_2,
            task_specific_adapter_paths=ts_adapters[:2],
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_small_later_big,
        )

        # Ensuring that the prototypes and gks are done one time by running a forward path
        x = torch.ones(1, 4, dtype=torch.long)
        m_small_later_big(x)

        # Now loading the third expert
        m_small_later_big.load_adapter(
            model_id=ts_adapters[-1],
            adapter_name="new_added_ts_expert",
        )
        # Activating the new adapter and run forward path on it
        m_small_later_big.set_adapter("new_added_ts_expert")
        x = torch.ones(3, 5, dtype=torch.long)
        m_small_later_big(x)

        # Now we switch back to the arrow_router
        m_small_later_big.set_adapter("arrow_router")
        m_small_later_big.eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert torch.allclose(m_big(x).logits, m_small_later_big(x).logits)

    def test_arrow_with_load_adapter_later_with_forward_activate_new(self, ts_adapters, gen_adapter):
        """
        Loading the last expert after creating the arrow model and activate it should produce different result compared
        to the case where arrow_router is activate, and the model's using arrow.
        """
        # Arrow over all three experts
        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base_model_1 = AutoModelForCausalLM.from_pretrained(model_id)
            base_model_2 = copy.deepcopy(base_model_1)
        cfg_big = ArrowConfig(top_k=2, use_gks=True, rng_seed=42)
        m_big = create_arrow_model(
            base_model=base_model_1,
            task_specific_adapter_paths=ts_adapters,
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_big,
        ).eval()

        # Arrow over all 2 experts + loading the third expert later
        cfg_small_later_big = ArrowConfig(top_k=2, use_gks=True, rng_seed=42)
        m_small_later_big = create_arrow_model(
            base_model=base_model_2,
            task_specific_adapter_paths=ts_adapters[:2],
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_small_later_big,
        )

        # Ensuring that the prototypes and gks are done one time by running a forward path
        x = torch.ones(1, 4, dtype=torch.long)
        m_small_later_big(x)

        # Now loading the third expert
        m_small_later_big.load_adapter(
            model_id=ts_adapters[-1],
            adapter_name="new_added_ts_expert",
        )
        # The new adapter is activated
        m_small_later_big.set_adapter("new_added_ts_expert")
        m_small_later_big.eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert not torch.allclose(m_big(x).logits, m_small_later_big(x).logits)

    def test_arrow_gks_with_load_adapter_later_without_forward(self, ts_adapters, gen_adapter):
        """
        Loading the last expert after creating the arrow model should produce the same result as loading all the
        experts at once in create_arrow_model()
        """
        # Arrow over all three experts
        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base_model_1 = AutoModelForCausalLM.from_pretrained(model_id)
            base_model_2 = copy.deepcopy(base_model_1)
        cfg_big = ArrowConfig(top_k=2, use_gks=True, rng_seed=42)
        m_big = create_arrow_model(
            base_model=base_model_1,
            task_specific_adapter_paths=ts_adapters,
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_big,
        ).eval()

        # Arrow over all 2 experts + loading the third expert later
        cfg_small_later_big = ArrowConfig(top_k=2, use_gks=True, rng_seed=42)
        m_small_later_big = create_arrow_model(
            base_model=base_model_2,
            task_specific_adapter_paths=ts_adapters[:2],
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_small_later_big,
        )

        # Now loading the third expert
        m_small_later_big.load_adapter(
            model_id=ts_adapters[-1],
            adapter_name="new_added_ts_expert",
        )
        m_small_later_big.eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert torch.allclose(m_big(x).logits, m_small_later_big(x).logits)

    def test_genknowsub_changes_output(self, ts_adapters, gen_adapter):
        """
        Arrow+GenKnowSub vs plain Arrow must change logits.
        """
        # Plain Arrow
        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base_model_1 = AutoModelForCausalLM.from_pretrained(model_id)
            base_model_2 = copy.deepcopy(base_model_1)
        cfg_plain = ArrowConfig(top_k=2)
        m_plain = create_arrow_model(
            base_model=base_model_1,
            task_specific_adapter_paths=ts_adapters,
            arrow_config=cfg_plain,
        ).eval()

        # Arrow + GenKnowSub
        cfg_gks = ArrowConfig(top_k=2, use_gks=True)
        m_gks = create_arrow_model(
            base_model=base_model_2,
            task_specific_adapter_paths=ts_adapters,
            general_adapter_paths=gen_adapter,
            arrow_config=cfg_gks,
        ).eval()

        x = torch.ones(1, 4, dtype=torch.long)
        assert not torch.allclose(m_plain(x).logits, m_gks(x).logits)

    def test_merging_adapters_raise_error_in_arrow(self, ts_adapters):
        """
        Merging/unmerging is not allowed while an ArrowLinearLayer is loaded on the model and active.
        """
        # Arrow over first 2 experts
        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
        cfg_small = ArrowConfig(top_k=2)
        m_small = create_arrow_model(
            base_model=base_model,
            task_specific_adapter_paths=ts_adapters[:2],
            arrow_config=cfg_small,
        ).eval()

        with pytest.raises(RuntimeError, match=r"Cannot merge an active Arrow router adapter"):
            m_small.merge_and_unload()

    def test_conv2d_targets_raise_typeerror_in_arrow(self, workdir):
        """
        Adapters applied to Conv2d must be rejected by create_arrow_model() which enforces Linear/Linear4bit-only
        targets.
        """

        model_id = "peft-internal-testing/tiny-random-ResNetForImageClassification"
        with hub_online_once(model_id):
            base = AutoModelForImageClassification.from_pretrained(model_id)

        # Build a LoRA adapter targeting a Conv2d
        cfg = LoraConfig(r=4, target_modules=["convolution"], init_lora_weights=False)
        peft_model = get_peft_model(copy.deepcopy(base), cfg)

        conv_dir = workdir / "cv0"
        peft_model.save_pretrained(conv_dir)

        # Expect create_arrow_model to raise TypeError
        with pytest.raises(TypeError, match=r"LoRA adapters must only target Linear"):
            _ = create_arrow_model(
                base_model=base,
                task_specific_adapter_paths=[str(conv_dir)],
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

        model_id = "peft-internal-testing/tiny-random-gpt2"

        # Create base in fp16 (no manual assignment to .dtype)
        with hub_online_once(model_id):
            base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)

        cfg = ArrowConfig(top_k=2)

        # Build Arrow model and disable adapter dtype autocast
        model = create_arrow_model(
            base_model=base,
            task_specific_adapter_paths=ts_adapters,
            arrow_config=cfg,
            autocast_adapter_dtype=False,
            dtype=torch.float16,
        ).eval()

        X = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

        # Forward should work in fp16
        _ = model(**X)

        # Merge must fail on Arrow models
        with pytest.raises(RuntimeError, match=r"Cannot merge an active Arrow router adapter"):
            model.merge_adapter(safe_merge=False)

        with pytest.raises(RuntimeError, match=r"Cannot merge an active Arrow router adapter"):
            _ = model.merge_and_unload()

    def test_prototypes_not_recomputed_on_repeated_forward(self, ts_adapters):
        """
        Repeated calls to forward should not recompute prototypes. We verify by spying on
        ArrowLoraLinearLayer.top_right_singular_vec_from_BA(), which is only called when prototypes are (re)built.
        """
        model_id = "peft-internal-testing/tiny-random-gpt2"
        with hub_online_once(model_id):
            base = AutoModelForCausalLM.from_pretrained(model_id)

        cfg = ArrowConfig(top_k=2)
        model = create_arrow_model(
            base_model=base,
            task_specific_adapter_paths=ts_adapters,
            arrow_config=cfg,
        ).eval()

        # Find one Arrow layer instance on the model
        arrow_layer = None
        for _, module in model.named_modules():
            if hasattr(module, "lora_arrow") and "arrow_router" in module.lora_arrow:
                arrow_layer = module.lora_arrow["arrow_router"]
                break
        assert arrow_layer is not None, "Arrow router layer not found on model"

        x = torch.ones(1, 4, dtype=torch.long)

        # Spy on the internal proto computation; should run once (E calls for E experts)
        with patch.object(
            arrow_layer,
            "top_right_singular_vec_from_BA",
            wraps=arrow_layer.top_right_singular_vec_from_BA,
        ) as spy:
            _ = model(x)
            first_calls = spy.call_count
            assert first_calls == len(arrow_layer.task_adapter_names)

            # Call forward again; prototypes should be cached, so no extra calls
            _ = model(x)
        assert spy.call_count == first_calls


def test_training_updates_when_task_adapter_active(ts_adapters):
    """
    Ensure a simple training step works: compute a dummy loss, backward, and take an optimizer step. Verify that
    task-adapter parameters update.
    """
    model_id = "peft-internal-testing/tiny-random-gpt2"
    with hub_online_once(model_id):
        base = AutoModelForCausalLM.from_pretrained(model_id)

    # Build Arrow model over two experts
    cfg = ArrowConfig(top_k=2)
    model = create_arrow_model(
        base_model=base,
        task_specific_adapter_paths=ts_adapters[:2],
        arrow_config=cfg,
    )
    model.train()

    # Switch to a specific task adapter for training (vanilla LoRA)
    model.set_adapter("task_0")

    # Choose a representative parameter to check updates (task_0 A weight)
    rep_name = None
    for n, _ in model.named_parameters():
        if ".lora_A.task_0.weight" in n:
            rep_name = n
            break
    assert rep_name is not None, "task_0 LoRA A weight not found"
    rep_param = dict(model.named_parameters())[rep_name]
    before = rep_param.detach().clone()

    # Optimizer over trainable params (task_0 now active and trainable)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    # Dummy batch
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    # Compute loss and update
    opt.zero_grad()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    assert hasattr(out, "loss") and out.loss is not None
    out.loss.backward()
    opt.step()

    after = rep_param.detach().clone()
    assert not torch.allclose(before, after), "Active task adapter parameters did not update after optimizer step"


@pytest.mark.parametrize(
    "case",
    [
        "local_root",
        "local_nested",
        "hub_repo",
        "hub_with_sub",
    ],
)
def test_resolve_adapter_source_variants(tmp_path: Path, case: str):
    """
    Ensure `_resolve_adapter_source` correctly handles:
      - Local dir (containing adapter_config.json)
      - Local nested subfolder
      - Hub repo id "user/repo"
      - Hub repo with subfolder "user/repo/sub/folder"
    """
    if case == "local_root":
        d = tmp_path / "adapter_local_root"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text("{}")
        model_id, sub = _resolve_adapter_source(str(d))
        assert model_id == str(d)
        assert sub is None

    elif case == "local_nested":
        d = tmp_path / "repo_like" / "sub" / "folder"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text("{}")
        model_id, sub = _resolve_adapter_source(str(d))
        assert model_id == str(d)
        assert sub is None

    elif case == "hub_repo":
        model_id, sub = _resolve_adapter_source("user/repo")
        assert model_id == "user/repo"
        assert sub is None

    elif case == "hub_with_sub":
        model_id, sub = _resolve_adapter_source("user/repo/sub/folder")
        assert model_id == "user/repo"
        assert sub == "sub/folder"

    else:
        raise AssertionError(f"unknown case: {case}")
