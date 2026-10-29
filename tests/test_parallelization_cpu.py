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

"""CPU tests for PEFT's parallelization integrations.

Both FSDP and tensor parallelism need a process group and can therefore not be tested in a single process, but neither
of them needs a GPU: the `gloo` backend is enough and the models can be kept tiny. Keeping these tests on CPU means
they run in the regular test suite (`make test`), including the "tests on transformers main" workflow, whereas the
scenarios that genuinely require accelerators stay in `test_gpu_examples.py` and `test_common_gpu.py`.
"""

import os
import platform

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
from peft.import_utils import is_transformers_dtensor_tp, is_transformers_ge_v5_17_0
from peft.tuners.lora.layer import LoraLayer

from .testing_utils import hub_online_once


# These tests start a process group with more than one rank, which is only reliable on Linux. The FSDP tests in
# `test_custom_models.py` are restricted to Linux for the same reason.
pytestmark = [
    pytest.mark.skipif(platform.system() != "Linux", reason="requires a Linux multi-process setup"),
    pytest.mark.skipif(not dist.is_available(), reason="requires torch.distributed"),
    pytest.mark.skipif(
        not (is_transformers_dtensor_tp and is_transformers_ge_v5_17_0),
        reason="LoRA with tensor parallelism requires a recent transformers version (>= 5.17.0)",
    ),
]

WORLD_SIZE = 2
MODEL_ID = "trl-internal-testing/tiny-random-LlamaForCausalLM"
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


class TestLoraTensorParallel:
    """Tests for LoRA on a tensor-parallel model that don't need a GPU, see `test_gpu_examples.py` for the rest.

    The helpers that do the actual work run in the spawned processes, so they are static methods: `mp.spawn` passes
    them to the children by reference, and a bound method would drag the test instance along with it.
    """

    @pytest.fixture(autouse=True)
    def cpu_device_mesh(self, monkeypatch):
        """
        Build the TP device mesh on CPU, also on machines that have a GPU.

        `from_pretrained(distributed_config=...)` builds the mesh on the accelerator that PyTorch detects and moves the
        model there, but the process groups in this class use `gloo`, which cannot handle accelerator tensors. Hiding
        the accelerator makes transformers fall back to a CPU mesh, so these tests behave the same way everywhere. The
        variable is inherited by the spawned processes.
        """
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    @pytest.fixture
    def init_method(self, tmp_path):
        """Rendezvous file for the spawned processes. `tmp_path` is unique per test and removed afterwards."""
        return (tmp_path / "rendezvous").as_uri()

    @staticmethod
    def _run_in_processes(rank, fn, world_size, init_method, *args):
        """Create this rank's process group, run `fn` in it, and tear the group down again."""
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=world_size)
        try:
            fn(rank, world_size, *args)
        finally:
            dist.destroy_process_group()

    @staticmethod
    def _load_model_with_lora(target_modules, **lora_kwargs):
        """Load the tiny model sharded over the ranks with tensor parallelism and add LoRA adapters to it."""
        # imported here because PEFT supports transformers versions that don't have the DTensor TP API
        from transformers.distributed import DistributedConfig

        with hub_online_once(MODEL_ID):
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, distributed_config=DistributedConfig(tp_plan=TP_PLAN, tp_size=dist.get_world_size())
            )
        return get_peft_model(model, LoraConfig(r=4, target_modules=target_modules, **lora_kwargs))

    @staticmethod
    def _test_sharded_adapters_report_global_features(rank, world_size):
        """
        The adapters of a TP-sharded base layer have to report the global, unsharded dimensions of that layer.

        The adapters are initialized with the *local* shard dimensions of the base layer and only afterwards wrapped in
        a DTensor that spans the whole layer. The TP styles, however, reconstruct the global shape of the activations
        they wrap from the module attributes (`RowwiseParallel` reads `in_features`, `ColwiseParallel` reads
        `out_features`), so those attributes have to describe the global layer as well. If they don't, the activations
        are wrapped with too small a global shape, which makes DTensor disagree with the size of this rank's tensor and
        raises on the first forward pass.
        """
        from torch.distributed.tensor import DTensor

        # sharded colwise and rowwise, so both sharding styles are covered
        model = TestLoraTensorParallel._load_model_with_lora(target_modules=["q_proj", "o_proj"])

        checked = 0
        for name, module in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            base_layer = module.get_base_layer()
            tp_plan = getattr(base_layer, "_hf_tp_plan", None)
            if tp_plan not in ("colwise", "rowwise"):
                continue
            checked += 1

            if tp_plan == "colwise":
                # the output dim is sharded, so `lora_B` is sharded and `lora_A` stays replicated
                adapter, shard_dim, attr = module.lora_B["default"], 0, "out_features"
            else:
                # the input dim is sharded, so `lora_A` is sharded and `lora_B` stays replicated
                adapter, shard_dim, attr = module.lora_A["default"], 1, "in_features"

            # the base layer's weight is a DTensor, so its shape is the global (unsharded) one
            global_value = base_layer.weight.shape[shard_dim]
            assert isinstance(adapter.weight, DTensor), f"{name}: sharded adapter weight should be a DTensor"
            assert adapter.weight.shape[shard_dim] == global_value, f"{name}: unexpected global {attr}"
            assert adapter.weight.to_local().shape[shard_dim] == global_value // world_size, (
                f"{name}: unexpected shard size"
            )

            reported = getattr(adapter, attr)
            assert reported == global_value, (
                f"{name}: adapter reports {attr}={reported}, but the base layer is sharded with a global {attr} of "
                f"{global_value}"
            )

        assert checked > 0, "no LoRA adapter on a TP-sharded layer was found"

    @staticmethod
    def _test_forward_backward(rank, world_size):
        """A forward and backward pass through adapters that are sharded along either dimension must work."""
        from torch.distributed.tensor import DTensor

        # `embed_tokens` is included to also cover the embedding adapter, the other layers cover both sharding styles
        model = TestLoraTensorParallel._load_model_with_lora(
            target_modules=["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj"], init_lora_weights=False
        )
        model.train()

        input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
        loss = model(input_ids=input_ids, labels=input_ids).loss
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        loss.backward()

        checked = 0
        for name, module in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            if getattr(module.get_base_layer(), "_hf_tp_plan", None) is None:
                continue
            # `lora_A`/`lora_B` only exist for linear layers; the adapter of `embed_tokens` is stored in
            # `lora_embedding_A`/`lora_embedding_B` as plain parameters instead, its gradients are covered by
            # `TestLoraTensorParallel::test_lora_gradient_synchronization` in `test_gpu_examples.py`
            for adapter in (*module.lora_A.values(), *module.lora_B.values()):
                checked += 1
                grad = adapter.weight.grad
                assert grad is not None, f"{name}: adapter did not receive a gradient"
                grad = grad.to_local() if isinstance(grad, DTensor) else grad
                assert torch.count_nonzero(grad) > 0, f"{name}: adapter gradient is all zeros"

        assert checked > 0, "no LoRA adapter on a TP-sharded layer was found"

    def _spawn(self, fn, init_method, *args):
        """Run `fn` in `WORLD_SIZE` processes, each of which creates its own process group."""
        mp.spawn(self._run_in_processes, args=(fn, WORLD_SIZE, init_method) + args, nprocs=WORLD_SIZE, join=True)

    def test_sharded_adapters_report_global_features(self, init_method):
        self._spawn(self._test_sharded_adapters_report_global_features, init_method)

    def test_forward_backward(self, init_method):
        self._spawn(self._test_forward_backward, init_method)
