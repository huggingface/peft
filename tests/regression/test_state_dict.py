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

# State dict regression testing: check that adapter checkpoints can be saved and restored consistently across PEFT
# versions. In contrast to test_regression.py, which checks numerical outputs of specific methods, this suite covers
# (almost) all PEFT methods and focuses on the serialization contract: the set of keys in the saved state dict,
# restoration of the model from the checkpoint, independence of the adapter name, and roundtrip stability of
# save -> load -> save.
#
# Run this only if there is a change affecting the saving and loading logic that could invalidate existing
# checkpoints. There is no need to run it on a regular basis. Remember to add new PEFT methods to the test cases if they
# should be covered.
#
# To verify the current code against the stored artifacts, run:
#
# `pytest tests/regression/test_state_dict.py --regression`
#
# The artifacts are downloaded from the Hub on a per-test-case basis, so `-k` can be used to only download and check a
# subset, e.g. `-k lora`. The artifact of each case is a `save_pretrained` output plus a manifest with the expected
# state dict keys and metadata, and the model output on a fixed input.
#
# To create and upload new regression artifacts (this will overwrite the existing ones, so only do this when the
# current state dict format is considered correct), run:
#
# `HF_TOKEN=<token> REGRESSION_CREATION_MODE=True pytest tests/regression/test_state_dict.py --regression`
#
# This will fail if the git worktree is dirty, to ensure that possibly buggy states are not "blessed" as the
# reference; override with REGRESSION_FORCE_MODE=True if you know what you're doing. The commit that created the
# artifacts is recorded in their manifest.json.
#
# The token requires write access to the repo below. With `-k`, only the selected cases are created and uploaded,
# leaving the other artifacts on the Hub untouched. This is also the way to go when a change to PEFT intentionally
# alters the saved state dict of a method (e.g. a bug fix that removes keys that were never needed): once the new
# format is considered correct, re-create the artifacts of the affected cases.

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
import transformers
from huggingface_hub import snapshot_download, upload_folder
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

import peft
from peft import (
    AdaLoraConfig,
    AdamssConfig,
    AdaptionPromptConfig,
    BeftConfig,
    BOFTConfig,
    C3AConfig,
    CartridgeConfig,
    CPTConfig,
    DeftConfig,
    DeloraConfig,
    FourierFTConfig,
    FrodConfig,
    GloraConfig,
    GraloraConfig,
    HiraConfig,
    HRAConfig,
    IA3Config,
    LilyConfig,
    LNTuningConfig,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    MissConfig,
    MultitaskPromptTuningConfig,
    OFTConfig,
    OSFConfig,
    PeanutConfig,
    PeftModel,
    PolyConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PsoftConfig,
    PveraConfig,
    RandLoraConfig,
    RoadConfig,
    ShiraConfig,
    TinyLoraConfig,
    TrainableTokensConfig,
    UniLoraConfig,
    VBLoRAConfig,
    VeraConfig,
    WaveFTConfig,
    get_peft_model,
)


def strtobool(val):
    """Copied from distutils.util"""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")


# the repo has to be created manually once, it is not automatically created
HF_REPO = "peft-internal-testing/regression-tests-state-dict"
HF_TOKEN = os.environ.get("HF_TOKEN")
CREATION_MODE = strtobool(os.environ.get("REGRESSION_CREATION_MODE", "0"))
FORCE_MODE = strtobool(os.environ.get("REGRESSION_FORCE_MODE", "0"))
REGRESSION_DIR = tempfile.mkdtemp(prefix="peft_state_dict_regression_")


def check_clean_git_status(force):
    """Ensure that the worktree is not dirty, so that possibly buggy code states are not "blessed" as the reference.

    In contrast to test_regression.py, there is no check for a tagged release, as the artifacts are typically created
    from a clean main commit right before a refactoring, not from a release. The manifest records the exact commit.
    """
    try:
        subprocess.check_output(["git", "diff", "--quiet", "HEAD"])
    except subprocess.CalledProcessError as exc:
        if force:
            print("Overriding despite dirty git worktree", file=sys.stderr)
        else:
            raise RuntimeError("Git worktree is dirty") from exc


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


MANIFEST_NAME = "manifest.json"
OUTPUT_NAME = "output.safetensors"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"

MODEL_OPT = "peft-internal-testing/tiny-random-OPTForCausalLM"
MODEL_LLAMA = "trl-internal-testing/tiny-random-LlamaForCausalLM"
MODEL_T5 = "peft-internal-testing/tiny-random-T5ForConditionalGeneration-calibrated"

MODEL_CLASSES = {
    "AutoModelForCausalLM": AutoModelForCausalLM,
    "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
}

INPUTS_DECODER = {"input_ids": [[1, 2, 3], [6, 5, 4]], "attention_mask": [[1, 1, 1], [1, 1, 1]]}
INPUTS_SEQ2SEQ = {**INPUTS_DECODER, "decoder_input_ids": [[0, 1, 2], [2, 0, 1]]}


@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    if FORCE_MODE and not CREATION_MODE:
        raise RuntimeError("REGRESSION_FORCE_MODE can only be used together with REGRESSION_CREATION_MODE")

    if CREATION_MODE:
        check_clean_git_status(FORCE_MODE)
        if HF_TOKEN is None:
            raise RuntimeError("HF_TOKEN environment variable must be set in creation mode")

    yield

    # optionally upload the created regression artifacts at the end of the test session, then delete the local copies
    if CREATION_MODE and os.listdir(REGRESSION_DIR):
        upload_folder(repo_id=HF_REPO, folder_path=REGRESSION_DIR, token=HF_TOKEN)
    shutil.rmtree(REGRESSION_DIR)


@dataclass
class Case:
    name: str
    config_cls: type
    config_kwargs: dict
    model_id: str = MODEL_OPT
    model_cls: str = "AutoModelForCausalLM"
    inputs: dict = field(default_factory=lambda: INPUTS_DECODER)
    atol: float = 1e-6
    rtol: float = 1e-6
    # special creation flows that go beyond get_peft_model + save_pretrained
    variant: str | None = None  # None | "multi_adapter" | "pissa_conversion"
    notes: str = ""


CASES = [
    Case("adalora", AdaLoraConfig, {"task_type": "CAUSAL_LM", "total_step": 1}),
    Case(
        "adamss",
        AdamssConfig,
        {"target_modules": ["q_proj", "v_proj"], "r": 8, "num_subspaces": 4, "subspace_rank": 1, "use_asa": False},
    ),
    Case(
        "adaption_prompt",
        AdaptionPromptConfig,
        {"task_type": "CAUSAL_LM", "adapter_layers": 1, "adapter_len": 4},
        model_id=MODEL_LLAMA,
    ),
    Case("beft", BeftConfig, {"task_type": "CAUSAL_LM"}),
    Case("boft", BOFTConfig, {"task_type": "CAUSAL_LM"}),
    Case("c3a", C3AConfig, {"task_type": "CAUSAL_LM", "block_size": 1}),
    Case("cartridge", CartridgeConfig, {"task_type": "CAUSAL_LM", "num_virtual_tokens": 4, "num_frozen_tokens": 1}),
    Case(
        "cpt",
        CPTConfig,
        {
            "task_type": "CAUSAL_LM",
            "cpt_token_ids": [0, 1, 2, 3, 4, 5, 6, 7],
            "cpt_mask": [1, 1, 1, 1, 1, 1, 1, 1],
            "cpt_tokens_type_mask": [1, 2, 2, 2, 3, 3, 4, 4],
        },
    ),
    Case("deft", DeftConfig, {"task_type": "CAUSAL_LM"}),
    Case("delora", DeloraConfig, {"task_type": "CAUSAL_LM", "r": 2}),
    Case("fourierft", FourierFTConfig, {"task_type": "CAUSAL_LM", "n_frequency": 10}),
    Case("frod", FrodConfig, {"task_type": "CAUSAL_LM", "sparse_rate": 0.01}),
    Case("glora", GloraConfig, {"task_type": "CAUSAL_LM", "init_weights": True}),
    Case(
        "gralora",
        GraloraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "alpha": 16, "gralora_k": 2, "hybrid_r": 0},
    ),
    Case(
        "gralora_hybrid",
        GraloraConfig,
        {"task_type": "CAUSAL_LM", "r": 16, "alpha": 32, "gralora_k": 4, "hybrid_r": 4},
    ),
    Case("hira", HiraConfig, {"task_type": "CAUSAL_LM"}),
    Case("hra", HRAConfig, {"task_type": "CAUSAL_LM"}),
    Case("ia3", IA3Config, {"task_type": "CAUSAL_LM"}),
    Case("lily", LilyConfig, {"target_modules": ["q_proj", "v_proj"], "r": 8, "stride_A": 1, "num_B": 2}),
    Case(
        "ln_tuning",
        LNTuningConfig,
        {"task_type": "CAUSAL_LM", "target_modules": ["self_attn_layer_norm", "final_layer_norm"]},
    ),
    Case("loha", LoHaConfig, {"target_modules": ["q_proj", "v_proj"]}),
    Case("lokr", LoKrConfig, {"target_modules": ["q_proj", "v_proj"]}),
    Case("lora", LoraConfig, {"task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 32}),
    Case("lora_alora", LoraConfig, {"task_type": "CAUSAL_LM", "r": 8, "alora_invocation_tokens": [1]}),
    Case("lora_bias_all", LoraConfig, {"task_type": "CAUSAL_LM", "r": 8, "bias": "all"}),
    Case("lora_dora", LoraConfig, {"task_type": "CAUSAL_LM", "r": 8, "use_dora": True}),
    Case(
        "lora_modules_to_save",
        LoraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "modules_to_save": ["final_layer_norm"]},
    ),
    Case(
        "lora_modules_to_save_tied",
        LoraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "modules_to_save": ["lm_head"]},
        notes="lm_head is tied to embed_tokens in OPT, exercises the tied-weights handling of ModulesToSaveWrapper",
    ),
    Case("lora_multi_adapter", LoraConfig, {"task_type": "CAUSAL_LM", "r": 8}, variant="multi_adapter"),
    Case(
        "lora_pissa_conversion",
        LoraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "init_lora_weights": "pissa", "target_modules": ["q_proj", "v_proj"]},
        atol=1e-5,
        rtol=1e-4,
        variant="pissa_conversion",
        notes="saved via path_initial_model_for_weight_conversion, loadable as a plain LoRA on the unmutated base",
    ),
    Case(
        "lora_target_embedding",
        LoraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "target_modules": ["embed_tokens"]},
        notes="triggers the save_embedding_layers='auto' path, the base embedding weight is part of the checkpoint",
    ),
    Case(
        "lora_trainable_tokens",
        LoraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "trainable_token_indices": [0, 1, 3]},
    ),
    Case("miss", MissConfig, {"task_type": "CAUSAL_LM", "r": 2}),
    Case(
        "multitask_prompt_tuning",
        MultitaskPromptTuningConfig,
        {"task_type": "CAUSAL_LM", "num_virtual_tokens": 10, "num_tasks": 2},
        model_id=MODEL_LLAMA,
        inputs={**INPUTS_DECODER, "task_ids": [0, 1]},
    ),
    Case("oft", OFTConfig, {"task_type": "CAUSAL_LM"}),
    Case("osf", OSFConfig, {"task_type": "CAUSAL_LM"}),
    Case(
        "peanut",
        PeanutConfig,
        {"target_modules": ["q_proj", "v_proj"], "r": 4, "depth": 1, "act_fn": "relu", "init_weights": True},
    ),
    Case(
        "poly",
        PolyConfig,
        {"task_type": "SEQ_2_SEQ_LM", "r": 2, "n_tasks": 2, "n_skills": 2, "n_splits": 1},
        model_id=MODEL_T5,
        model_cls="AutoModelForSeq2SeqLM",
        inputs={**INPUTS_SEQ2SEQ, "task_ids": [0, 1]},
    ),
    Case("prefix_tuning", PrefixTuningConfig, {"task_type": "CAUSAL_LM", "num_virtual_tokens": 10}),
    Case(
        "prompt_encoder",
        PromptEncoderConfig,
        {"task_type": "CAUSAL_LM", "num_virtual_tokens": 10, "encoder_hidden_size": 32},
    ),
    Case("prompt_tuning", PromptTuningConfig, {"task_type": "CAUSAL_LM", "num_virtual_tokens": 10}),
    Case("psoft", PsoftConfig, {"task_type": "CAUSAL_LM", "r": 4, "psoft_alpha": 4}),
    Case("pvera", PveraConfig, {"task_type": "CAUSAL_LM", "r": 8}),
    Case("randlora", RandLoraConfig, {"target_modules": ["q_proj", "v_proj"], "r": 4}),
    Case("road", RoadConfig, {"task_type": "CAUSAL_LM", "variant": "road_1", "group_size": 2}),
    Case("shira", ShiraConfig, {"task_type": "CAUSAL_LM", "r": 1, "init_weights": False}),
    Case("tinylora", TinyLoraConfig, {"task_type": "CAUSAL_LM"}),
    Case(
        "tinylora_no_projection",
        TinyLoraConfig,
        {"task_type": "CAUSAL_LM", "save_projection": False},
        notes="projection is regenerated from the projection seed on load, only exact on the same system configuration",
    ),
    Case("trainable_tokens", TrainableTokensConfig, {"task_type": "CAUSAL_LM", "token_indices": [0, 1, 3]}),
    Case("unilora", UniLoraConfig, {"task_type": "CAUSAL_LM", "theta_d_length": 257}),
    Case(
        "vblora",
        VBLoRAConfig,
        {"task_type": "CAUSAL_LM", "vector_length": 1, "num_vectors": 2},
    ),
    Case(
        "vblora_topk",
        VBLoRAConfig,
        {"task_type": "CAUSAL_LM", "vector_length": 1, "num_vectors": 2, "save_only_topk_weights": True},
        atol=1e-5,
        rtol=1e-4,
        notes="topk saving is intentionally lossy, the logits are reconstructed from topk weights when loading",
    ),
    Case(
        "vera",
        VeraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "projection_prng_key": 0xFF, "d_initial": 0.1, "save_projection": True},
    ),
    Case(
        "vera_no_projection",
        VeraConfig,
        {"task_type": "CAUSAL_LM", "r": 8, "projection_prng_key": 0xFF, "d_initial": 0.1, "save_projection": False},
        notes="projection is regenerated from the PRNG key on load, only exact on the same system configuration",
    ),
    Case("waveft", WaveFTConfig, {"task_type": "CAUSAL_LM", "n_frequency": 8}),
    # Not covered:
    # - X-LoRA: its config references the sub-adapters by filesystem path, which makes the artifacts non-portable.
    # - LoftQ/EVA/LoRA-GA and other init methods that require calibration data or quantized weights.
    # - bias="lora_only"/"boft_only": these used to silently drop the trained biases (see PR #3457), so no valid
    #   artifact can be created before the fix; add cases once that is resolved.
]

CASE_IDS = [case.name for case in CASES]


def download_artifact(case_name):
    """Download the artifact of a single test case from the Hub, so that with -k, only the needed artifacts are
    loaded."""
    snapshot_path = snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{case_name}/*"])
    case_dir = Path(snapshot_path) / case_name
    if not (case_dir / MANIFEST_NAME).exists():
        pytest.fail(
            f"No regression artifact found for case '{case_name}' in {HF_REPO}. Create it first by running with "
            "REGRESSION_CREATION_MODE=1."
        )
    return case_dir


def build_base_model(model_cls_name, model_id):
    torch.manual_seed(0)
    model_cls = MODEL_CLASSES[model_cls_name]
    model = model_cls.from_pretrained(model_id)
    model.eval()
    return model


def fill_trainable_params(model):
    """Overwrite all trainable parameters with deterministic values (simulates training).

    The values are derived from the parameter name, so that they depend neither on the order of iteration nor on the
    global RNG state. Why this approach:

    - Initializing the adapter as a non-identity transform (à la set_init_weights_false): a dropped or unrestored
      checkpoint entry is only observable if the corresponding tensor differs from the value it has after a fresh load.
      This must also hold for tensors whose initial value would be identical after loading, e.g. the biases trained
      with bias="lora_only" or the module copies of modules_to_save, hence all trainable parameters need to be
      "trained". For the same reason, using a different RNG seed for creation vs. loading would not be reliable, as not
      all initial values are drawn from the RNG.
    - Filling all parameters with the same magic value: the tensors need to differ from each other, otherwise a tensor
      that is stored under the key of another, same-shaped tensor goes unnoticed (e.g. the keys of two layers being
      swapped, or the weights of a filtered-out adapter leaking into the saved one). Likewise, the elements within each
      tensor need to differ, otherwise layout mistakes (transposing, slicing, scattering) go unnoticed, as a constant
      tensor is invariant under these operations.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Even if RNG is not stable between torch releases, this is fine, as we only test that the weights can be
            # loaded correctly, it doesn't really matter under which RNG they were created.
            generator = torch.Generator().manual_seed(zlib.crc32(name.encode("utf-8")))
            values = torch.rand(param.shape, generator=generator, dtype=torch.float32) * 0.2 - 0.1
            param.copy_(values.to(dtype=param.dtype))


def get_output(model, inputs):
    model.eval()
    inputs = {key: torch.tensor(val, dtype=torch.long) for key, val in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    return output.logits.detach().to(torch.float32).cpu()


def create_artifact(case, case_dir, tmp_path):
    torch.manual_seed(0)
    base_model = build_base_model(case.model_cls, case.model_id)
    # record the base model output before creating the PEFT model, as some methods mutate the base weights
    base_inputs = {k: v for k, v in case.inputs.items() if k != "task_ids"}  # the base model accepts no task_ids
    base_logits = get_output(base_model, base_inputs)

    torch.manual_seed(0)
    config = case.config_cls(**case.config_kwargs)
    model = get_peft_model(base_model, config)

    save_kwargs = {}
    if case.variant == "pissa_conversion":
        # The PiSSA -> LoRA conversion requires the initial adapter as saved directly after initialization, before any
        # training. Its config needs init_lora_weights=True, otherwise loading it would mutate the base weights again.
        init_dir = os.path.join(tmp_path, "pissa_init")
        model.peft_config["default"].init_lora_weights = True
        model.save_pretrained(init_dir)
        model.peft_config["default"].init_lora_weights = case.config_kwargs["init_lora_weights"]
        save_kwargs["path_initial_model_for_weight_conversion"] = init_dir
    elif case.variant == "multi_adapter":
        # A second adapter is present in the model but not saved; its weights must not leak into the checkpoint.
        model.add_adapter("other", case.config_cls(**case.config_kwargs))
        save_kwargs["selected_adapters"] = ["default"]

    fill_trainable_params(model)
    logits = get_output(model, case.inputs)

    assert torch.isfinite(logits).all(), "the model output must be finite"
    # sanity check that the simulated training changed the model output, otherwise broken serialization would go
    # unnoticed, as the recorded output could be reproduced without restoring the adapter; a shape mismatch (prompt
    # learning methods insert virtual tokens) always implies a changed output
    if logits.shape == base_logits.shape:
        assert not torch.allclose(logits, base_logits), "the adapter must change the output of the base model"

    model.save_pretrained(str(case_dir), **save_kwargs)

    state_dict = safe_load_file(case_dir / ADAPTER_WEIGHTS_NAME)
    manifest = {
        "case_name": case.name,
        "peft_version": peft.__version__,
        "git_commit": get_git_commit(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "base_model_id": case.model_id,
        "model_cls": case.model_cls,
        "inputs": case.inputs,
        "atol": case.atol,
        "rtol": case.rtol,
        "state_dict_keys": sorted(state_dict.keys()),
        "notes": case.notes,
    }
    with open(case_dir / MANIFEST_NAME, "w") as f:
        json.dump(manifest, f, indent=2)
    safe_save_file({"logits": logits}, case_dir / OUTPUT_NAME)
    return manifest


def load_model_from_artifact(case_dir, manifest, adapter_name="default"):
    base_model = build_base_model(manifest["model_cls"], manifest["base_model_id"])
    torch.manual_seed(0)
    model = PeftModel.from_pretrained(base_model, str(case_dir), adapter_name=adapter_name)
    model.eval()
    return model


@pytest.mark.regression
class TestCreateArtifacts:
    @pytest.mark.skipif(not CREATION_MODE, reason="Set REGRESSION_CREATION_MODE=1 to create regression artifacts")
    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_create_artifact(self, case, tmp_path):
        case_dir = Path(REGRESSION_DIR) / case.name
        case_dir.mkdir(parents=True)

        try:
            manifest = create_artifact(case, case_dir, tmp_path)

            # sanity check: the artifact must be restorable by the version that created it
            model = load_model_from_artifact(case_dir, manifest)
            expected = safe_load_file(case_dir / OUTPUT_NAME)["logits"]
            logits = get_output(model, case.inputs)
            torch.testing.assert_close(logits, expected, atol=case.atol, rtol=case.rtol)
        except Exception:
            # don't leave partial artifacts behind, they would be uploaded at the end of the session
            shutil.rmtree(case_dir, ignore_errors=True)
            raise


@pytest.mark.regression
@pytest.mark.skipif(CREATION_MODE, reason="Skipping tests in CREATION_MODE")
class TestStateDictRegression:
    def load_manifest(self, case_dir):
        with open(case_dir / MANIFEST_NAME) as f:
            return json.load(f)

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_load_and_forward(self, case):
        # the checkpoint must load into the current version and produce the recorded output
        case_dir = download_artifact(case.name)
        manifest = self.load_manifest(case_dir)
        model = load_model_from_artifact(case_dir, manifest)
        logits = get_output(model, manifest["inputs"])
        expected = safe_load_file(case_dir / OUTPUT_NAME)["logits"]
        torch.testing.assert_close(logits, expected, atol=manifest["atol"], rtol=manifest["rtol"])

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_load_with_different_adapter_name(self, case):
        # the checkpoint format is independent of the adapter name, so loading under any name must work
        case_dir = download_artifact(case.name)
        manifest = self.load_manifest(case_dir)
        model = load_model_from_artifact(case_dir, manifest, adapter_name="other")
        logits = get_output(model, manifest["inputs"])
        expected = safe_load_file(case_dir / OUTPUT_NAME)["logits"]
        torch.testing.assert_close(logits, expected, atol=manifest["atol"], rtol=manifest["rtol"])

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_save_load_roundtrip(self, case, tmp_path):
        # saving the loaded model must reproduce the checkpoint: same keys and same tensor values
        case_dir = download_artifact(case.name)
        manifest = self.load_manifest(case_dir)
        model = load_model_from_artifact(case_dir, manifest)
        model.save_pretrained(str(tmp_path))

        old_state_dict = safe_load_file(case_dir / ADAPTER_WEIGHTS_NAME)
        new_state_dict = safe_load_file(tmp_path / ADAPTER_WEIGHTS_NAME)
        assert set(new_state_dict.keys()) == set(manifest["state_dict_keys"])

        for key in sorted(new_state_dict.keys()):
            torch.testing.assert_close(
                new_state_dict[key],
                old_state_dict[key],
                atol=manifest["atol"],
                rtol=manifest["rtol"],
                msg=lambda m, key=key: f"Mismatch in key {key}:\n{m}",
            )
