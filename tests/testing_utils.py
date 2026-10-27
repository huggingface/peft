# Copyright 2023-present the HuggingFace Inc. team.
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
import math
import os
from contextlib import contextmanager
from functools import lru_cache, wraps
from importlib import import_module
from importlib.util import find_spec
from unittest import mock

import numpy as np
import pytest
import torch
from accelerate.test_utils.testing import get_backend
from datasets import load_dataset
from torch import nn

from peft import (
    AdaLoraConfig,
    IA3Config,
    LNTuningConfig,
    LoraConfig,
    MissConfig,
    PromptLearningConfig,
    TinyLoraConfig,
    VBLoRAConfig,
)
from peft.import_utils import (
    is_aqlm_available,
    is_eetq_available,
    is_gptqmodel_available,
    is_hqq_available,
    is_optimum_available,
    is_torchao_available,
    is_transformers_ge_v5,
)


# Globally shared model cache used by `hub_online_once`.
_HUB_MODEL_ACCESSES = {}
# Some tests with multi GPU require specific device maps to ensure that the models are loaded in two devices
DEVICE_MAP_MAP: dict[str, dict[str, int]] = {
    "facebook/opt-6.7b": {
        "model.decoder.embed_tokens": 0,
        "model.decoder.embed_positions": 0,
        "model.decoder.final_layer_norm": 0,
        "model.decoder.layers.0": 0,
        "model.decoder.layers.1": 0,
        "model.decoder.layers.2": 0,
        "model.decoder.layers.3": 0,
        "model.decoder.layers.4": 0,
        "model.decoder.layers.5": 0,
        "model.decoder.layers.6": 0,
        "model.decoder.layers.7": 0,
        "model.decoder.layers.8": 0,
        "model.decoder.layers.9": 0,
        "model.decoder.layers.10": 0,
        "model.decoder.layers.11": 0,
        "model.decoder.layers.12": 0,
        "model.decoder.layers.13": 0,
        "model.decoder.layers.14": 0,
        "model.decoder.layers.15": 0,
        "model.decoder.layers.16": 1,
        "model.decoder.layers.17": 1,
        "model.decoder.layers.18": 1,
        "model.decoder.layers.19": 1,
        "model.decoder.layers.20": 1,
        "model.decoder.layers.21": 1,
        "model.decoder.layers.22": 1,
        "model.decoder.layers.23": 1,
        "model.decoder.layers.24": 1,
        "model.decoder.layers.25": 1,
        "model.decoder.layers.26": 1,
        "model.decoder.layers.27": 1,
        "model.decoder.layers.28": 1,
        "model.decoder.layers.29": 1,
        "model.decoder.layers.30": 1,
        "model.decoder.layers.31": 1,
        "lm_head": 0,  # tied with embed_tokens
    },
    "peft-internal-testing/opt-125m": {
        "model.decoder.embed_tokens": 0,
        "model.decoder.embed_positions": 0,
        "model.decoder.final_layer_norm": 1,
        "model.decoder.layers.0": 0,
        "model.decoder.layers.1": 0,
        "model.decoder.layers.2": 0,
        "model.decoder.layers.3": 0,
        "model.decoder.layers.4": 0,
        "model.decoder.layers.5": 0,
        "model.decoder.layers.6": 1,
        "model.decoder.layers.7": 1,
        "model.decoder.layers.8": 1,
        "model.decoder.layers.9": 1,
        "model.decoder.layers.10": 1,
        "model.decoder.layers.11": 1,
        "lm_head": 0,
    },
    "marcsun13/opt-350m-gptq-4bit": {
        "model.decoder.embed_tokens": 0,
        "model.decoder.embed_positions": 0,
        "model.decoder.layers.0": 0,
        "model.decoder.layers.1": 0,
        "model.decoder.layers.2": 0,
        "model.decoder.layers.3": 0,
        "model.decoder.layers.4": 0,
        "model.decoder.layers.5": 0,
        "model.decoder.layers.6": 1,
        "model.decoder.layers.7": 1,
        "model.decoder.layers.8": 1,
        "model.decoder.layers.9": 1,
        "model.decoder.layers.10": 1,
        "model.decoder.layers.11": 1,
        "model.decoder.final_layer_norm": 1,
        "lm_head": 0,  # tied with embed_tokens
    },
    "google/flan-t5-base": {
        "shared": 0,
        "encoder": 0,
        "decoder": 1,
        "final_layer_norm": 1,
        "decoder.embed_tokens": 0,  # tied with encoder.embed_tokens
        "lm_head": 0,  # tied with encoder.embed_tokens
    },
}


torch_device, device_count, memory_allocated_func = get_backend()


def require_non_cpu(test_case):
    """
    Decorator marking a test that requires a hardware accelerator backend. These tests are skipped when there are no
    hardware accelerator available.
    """
    return pytest.mark.skipif(torch_device == "cpu", reason="test requires a hardware accelerator")(test_case)


def require_non_xpu(test_case):
    """
    Decorator marking a test that should be skipped for XPU.
    """
    return pytest.mark.skipif(torch_device == "xpu", reason="test requires a non-XPU")(test_case)


def require_torch_gpu(test_case):
    """
    Decorator marking a test that requires a GPU. Will be skipped when no GPU is available.
    """
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU")(test_case)


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires multiple GPUs. Will be skipped when less than 2 GPUs are available.
    """
    multi_cuda_unavailable = not torch.cuda.is_available() or (device_count < 2)
    return pytest.mark.skipif(multi_cuda_unavailable, reason="test requires multiple GPUs")(test_case)


def require_torch_multi_accelerator(test_case):
    """
    Decorator marking a test that requires multiple hardware accelerators. These tests are skipped on a machine without
    multiple accelerators.
    """
    multi_device_unavailable = (torch_device == "cpu") or (device_count < 2)
    return pytest.mark.skipif(multi_device_unavailable, reason="test requires multiple hardware accelerators")(
        test_case
    )


def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires the bitsandbytes library. Will be skipped when the library is not installed.
    """
    try:
        import bitsandbytes  # noqa: F401

        test_case = pytest.mark.bitsandbytes(test_case)
    except ImportError:
        test_case = pytest.mark.skip(reason="test requires bitsandbytes")(test_case)
    return test_case


def require_gptqmodel(test_case):
    """
    Decorator marking a test that requires gptqmodel. These tests are skipped when gptqmodel isn't installed.
    """
    return pytest.mark.skipif(not is_gptqmodel_available(), reason="test requires gptqmodel")(test_case)


def require_aqlm(test_case):
    """
    Decorator marking a test that requires aqlm. These tests are skipped when aqlm isn't installed.
    """
    return pytest.mark.skipif(not is_aqlm_available(), reason="test requires aqlm")(test_case)


def require_hqq(test_case):
    """
    Decorator marking a test that requires aqlm. These tests are skipped when aqlm isn't installed.
    """
    return pytest.mark.skipif(not is_hqq_available(), reason="test requires hqq")(test_case)


def require_eetq(test_case):
    """
    Decorator marking a test that requires eetq. These tests are skipped when eetq isn't installed.
    """
    return pytest.mark.skipif(not is_eetq_available(), reason="test requires eetq")(test_case)


def require_optimum(test_case):
    """
    Decorator marking a test that requires optimum. These tests are skipped when optimum isn't installed.
    """
    return pytest.mark.skipif(not is_optimum_available(), reason="test requires optimum")(test_case)


def require_torchao(test_case):
    """
    Decorator marking a test that requires torchao. These tests are skipped when torchao isn't installed.
    """
    return pytest.mark.skipif(not is_torchao_available(), reason="test requires torchao")(test_case)


def require_deterministic_for_xpu(test_case):
    @wraps(test_case)
    def wrapper(*args, **kwargs):
        if torch_device == "xpu":
            original_state = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                return test_case(*args, **kwargs)
            finally:
                torch.use_deterministic_algorithms(original_state)
        else:
            return test_case(*args, **kwargs)

    return wrapper


@contextmanager
def temp_seed(seed: int):
    """Temporarily set the random seed. This works for python numpy, pytorch."""

    np_state = np.random.get_state()
    np.random.seed(seed)

    torch_state = torch.random.get_rng_state()
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch_cuda_states = torch.cuda.get_rng_state_all()
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        np.random.set_state(np_state)

        torch.random.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_states)


def get_state_dict(model, unwrap_compiled=True):
    """
    Get the state dict of a model. If the model is compiled, unwrap it first.
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)
    return model.state_dict()


@lru_cache
def load_dataset_english_quotes():
    # can't use pytest fixtures for now because of unittest style tests
    data = load_dataset("ybelkada/english_quotes_copy")
    return data


@lru_cache
def load_cat_image():
    # can't use pytest fixtures for now because of unittest style tests
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    return image


def set_init_weights_false(config_cls, kwargs):
    # helper function that sets the config kwargs such that the model is *not* initialized as an identity transform
    kwargs = kwargs.copy()

    if issubclass(config_cls, PromptLearningConfig):
        return kwargs
    if config_cls in (LNTuningConfig, VBLoRAConfig):
        return kwargs
    if (config_cls == MissConfig) and (kwargs.get("init_weights") == "bat"):
        # don't override 'bat' init or else it's not being properly tested
        return kwargs

    if config_cls in (LoraConfig, AdaLoraConfig):
        kwargs["init_lora_weights"] = False
    elif config_cls == IA3Config:
        kwargs["init_ia3_weights"] = False
    elif config_cls == TinyLoraConfig:
        kwargs["init_weights"] = "uniform"
    else:
        kwargs["init_weights"] = False
    return kwargs


@contextmanager
def hub_online_once(model_id: str):
    """Set env[HF_HUB_OFFLINE]=1 (and patch transformers/hugging_face_hub to think that it was always that way)
    for model ids that were already to avoid contacting the hub twice for the same model id in the context. The global
    variable `_HUB_MODEL_ACCESSES` tracks the number of hits per model id between `hub_online_once` calls.

    The reason for doing a context manager and not patching specific methods (e.g., `from_pretrained`) is that there
    are a lot of places (`PeftConfig.from_pretrained`, `get_peft_state_dict`, `load_adapter`, ...) that possibly
    communicate with the hub to download files / check versions / etc.

    Note that using this context manager can cause problems when used in code sections that access different resources.
    Example:

    ```
    def test_something(model_id, config_kwargs):
        with hub_online_once(model_id):
            model = ...from_pretrained(model_id)
            self.do_something_specific_with_model(model)
    ```
    It is assumed that `do_something_specific_with_model` is an absract method that is implement by several tests.
    Imagine the first test simply does `model.generate([1,2,3])`. The second call from another test suite however uses
    a tokenizer (`AutoTokenizer.from_pretrained(model_id)`) - this will fail since the first pass was online but didn't
    use the tokenizer and we're now in offline mode and cannot fetch the tokenizer. The recommended workaround is to
    extend the cache key (`model_id` passed to `hub_online_once` in this case) by something in case the tokenizer is
    used, so that these tests don't share a cache pool with the tests that don't use a tokenizer.

    It is best to avoid using this context manager in *yield* fixtures (normal fixtures are fine) as this is equivalent
    to wrapping the whole test in the context manager without explicitly writing it out, leading to unexpected
    `HF_HUB_OFFLINE` behavior in the test body.
    """
    override = {}

    try:
        if model_id in _HUB_MODEL_ACCESSES:
            override = {"HF_HUB_OFFLINE": "1"}
            _HUB_MODEL_ACCESSES[model_id] += 1
        elif model_id not in _HUB_MODEL_ACCESSES:
            _HUB_MODEL_ACCESSES[model_id] = 0
        is_offline = override.get("HF_HUB_OFFLINE", False) == "1"

        with (
            # strictly speaking it is not necessary to set the environment variable since most code that's out there
            # is evaluating it at import time and we'd have to reload the modules for it to take effect. It's
            # probably still a good idea to have it if there's some dynamic code that checks it.
            mock.patch.dict(os.environ, override),
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", is_offline),
        ):
            if is_transformers_ge_v5:
                with mock.patch("transformers.utils.hub.is_offline_mode", lambda: is_offline):
                    yield
            else:  # TODO remove if transformers <= 4 no longer supported
                with mock.patch("transformers.utils.hub._is_offline_mode", is_offline):
                    yield
    except Exception:
        # in case of an error we have to assume that we didn't access the model properly from the hub
        # for the first time, so the next call cannot be considered cached.
        if _HUB_MODEL_ACCESSES.get(model_id) == 0:
            del _HUB_MODEL_ACCESSES[model_id]
        raise


def require_transformers_model(kind, *, packed=False):
    """Gate only the requested architecture; 5.17 is the validated packed-layout test baseline."""
    if packed:
        pytest.importorskip(
            "transformers", minversion="5.17.0", reason="Packed MoE tests use the validated Transformers 5.17 baseline"
        )
    from transformers.models.auto.configuration_auto import model_type_to_module_name

    if find_spec(f"transformers.models.{model_type_to_module_name(kind)}") is None:
        pytest.skip(f"Transformers does not provide the {kind} architecture")


def create_tiny_causal_lm(kind):
    """Build a local random dense or MoE causal LM with eager native backends."""
    from transformers import AutoConfig, AutoModelForCausalLM

    require_transformers_model(kind, packed=kind in {"gpt_oss", "qwen3_moe", "mixtral", "olmoe", "gemma4_text"})
    config = AutoConfig.for_model(
        kind,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=3,
        num_local_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        enable_moe_block=True,
        top_k_experts=2,
        num_kv_shared_layers=0,
        global_head_dim=8,
        hidden_size_per_layer_input=0,
        max_position_embeddings=131072 if kind == "gpt_oss" else 64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return AutoModelForCausalLM.from_config(config)


@contextmanager
def deterministic_device(name):
    """Restore RNG and precision settings around CPU/CUDA numerical checks."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable; this test has not run on a GPU.")
    previous_threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(123)
            if devices:
                torch.cuda.manual_seed(123)
            yield device
            if devices:
                torch.cuda.synchronize(device)
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_num_threads(previous_threads)


class CudaTestFixtures:
    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA runtime unavailable; no GPU check ran.")

    @pytest.fixture
    def device(self):
        with deterministic_device("cuda") as current:
            yield current

    @pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def dtype(self, request, device):
        if request.param == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("This CUDA GPU does not support native BF16.")
        return request.param


def activate_branches(model):
    """Give Cortex's zero-initialized down projections a measurable effect."""
    from peft.tuners.cortex.layer import CortexLayer

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CortexLayer):
                for name in ("cortex_down", "cortex_shared_down"):
                    for projection in getattr(module, name).values():
                        projection.weight.normal_(std=0.1)


ARCHITECTURES = {
    "minimax": ("MiniMaxConfig", "MiniMaxSparseMoeBlock"),
    "minimax_m2": ("MiniMaxM2Config", "MiniMaxM2SparseMoeBlock"),
    "minimax_m3_vl": ("MiniMaxM3VLTextConfig", "MiniMaxM3VLSparseMoeBlock"),
    "deepseek_v2": ("DeepseekV2Config", "DeepseekV2Moe"),
    "deepseek_v3": ("DeepseekV3Config", "DeepseekV3MoE"),
    "deepseek_v32": ("DeepseekV32Config", "DeepseekV32MoE"),
    "deepseek_v4": ("DeepseekV4Config", "DeepseekV4SparseMoeBlock"),
    "kimi_linear": ("KimiLinearConfig", "KimiLinearMoE"),
    "glm4_moe": ("Glm4MoeConfig", "Glm4MoeMoE"),
    "glm4_moe_lite": ("Glm4MoeLiteConfig", "Glm4MoeLiteMoE"),
    "glm_moe_dsa": ("GlmMoeDsaConfig", "GlmMoeDsaMoE"),
    "glm5_next": ("Glm5NextTextConfig", "Glm5NextTextMoE"),
    "qwen3_next": ("Qwen3NextConfig", "Qwen3NextSparseMoeBlock"),
    "qwen3_5_moe": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeSparseMoeBlock"),
    "nemotron_h": ("NemotronHConfig", "NemotronHMoE"),
}


class BlockModel(nn.Module):
    """Exercise real native FFNs without instantiating unrelated attention stacks."""

    def __init__(self, kind, config=None):
        super().__init__()
        require_transformers_model(kind, packed=True)
        config_name, block_name = ARCHITECTURES[kind]
        package = f"transformers.models.{kind}"
        config_cls = getattr(import_module(package + f".configuration_{kind}"), config_name)
        self.config = config or config_cls(
            hidden_size=16,
            intermediate_size=24,
            moe_intermediate_size=12,
            dense_intermediate_size=20,
            shared_intermediate_size=18,
            shared_expert_intermediate_size=18,
            moe_shared_expert_intermediate_size=18,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            num_experts=4,
            num_local_experts=4,
            n_routed_experts=4,
            n_shared_experts=2,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            mlp_layer_types=["moe" if kind == "deepseek_v4" else "sparse"] * 2,
            moe_latent_size=8,
            max_position_embeddings=64,
            layer_types=["sliding_attention" if kind == "deepseek_v4" else "full_attention"] * 2,
        )
        self.config._experts_implementation = "eager"
        cls = getattr(import_module(package + f".modeling_{kind}"), block_name)
        self.mlp = cls(self.config, layer_idx=0) if kind == "deepseek_v4" else cls(self.config)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(std=0.15)

    def forward(self, x):
        result = self.mlp(x)
        return result[0] if isinstance(result, tuple) else result


class NativeFFNModel(nn.Module):
    """Keep the native FFN/router API without requiring the research checkout."""

    def __init__(self, mlp, config):
        super().__init__()
        self.mlp, self.config = mlp, config

    def forward(self, inputs):
        return self.mlp(inputs)


def configured_branch(kind, role, dtype, init="random"):
    """Construct one native FFN with a Cortex branch for initialization/formula tests."""
    from transformers import AutoConfig

    from peft import CortexConfig, get_peft_model

    require_transformers_model(kind, packed=kind != "qwen3")
    model_module = import_module(f"transformers.models.{kind}.modeling_{kind}")
    cls_name = {"qwen3": "Qwen3MLP", "gpt_oss": "GptOssMLP", "qwen3_moe": "Qwen3MoeSparseMoeBlock"}[kind]
    config = AutoConfig.for_model(
        kind,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=12,
        num_experts=3,
        num_local_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
    )
    config._experts_implementation = "eager"
    cls = getattr(model_module, cls_name)
    mlp = cls(config).to(dtype)
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.normal_(std=0.2)
    current = get_peft_model(
        NativeFFNModel(mlp, config),
        CortexConfig(
            k=3,
            moe_target=role if role != "dense" else "routed",
            adapter_bias=True,
            adapter_bias_init="zeros",
            init_weights=init,
        ),
        autocast_adapter_dtype=False,
    ).eval()
    layer = current.get_base_model().mlp
    if kind != "qwen3":
        layer = layer.experts
    return current, layer


def hidden_output(model, inputs):
    # Native OSS additionally returns router logits; this contract concerns
    # the FFN output and leaves those auxiliary outputs unchanged.
    result = model(inputs)
    return result[0] if isinstance(result, tuple) else result


def assert_cortex_biases_follow_native_experts(kind, mode, dtype, device):
    from peft import CortexConfig, get_peft_model
    from peft.tuners.cortex.layer import CortexLayer

    if (
        torch.device(device).type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        pytest.skip("Native BF16 required")
    base = create_tiny_causal_lm(kind).to(device=device, dtype=dtype).eval()
    inputs = torch.tensor([[1, 2, 3]], device=device)
    with torch.no_grad():
        expected = base(inputs).logits
    model = get_peft_model(base, CortexConfig(k=7, moe_target=mode), autocast_adapter_dtype=False).eval()
    for layer in model.modules():
        if not isinstance(layer, CortexLayer):
            continue
        for prefix in ("", "shared_") if mode == ["shared", "routed"] else ("",):
            for projection_kind in ("gate", "up", "down"):
                projection = getattr(layer, f"cortex_{prefix}{projection_kind}")["default"]
                assert (projection.bias is not None) == (kind == "gpt_oss")
                if projection_kind == "down":
                    assert torch.count_nonzero(projection.weight) == 0
                    if projection.bias is not None:
                        assert torch.count_nonzero(projection.bias) == 0
                elif projection.bias is not None:
                    assert 0 < projection.bias.abs().max() <= 1 / math.sqrt(16)
    torch.testing.assert_close(model(inputs).logits, expected, rtol=0, atol=0)
