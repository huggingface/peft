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

import os
import platform

import packaging.version
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
from peft.import_utils import is_transformers_ge_v5_17_0
from peft.utils import infer_device
from peft.utils.integrations import TpMeshDimRegistry

from .testing_utils import hub_online_once


@pytest.mark.skipif(platform.system() != "Linux", reason="Run distributed tests only on Linux")
@pytest.mark.skipif(not dist.is_available(), reason="These tests require torch.distributed")
@pytest.mark.skipif(
    packaging.version.parse(torch.__version__) < packaging.version.parse("2.6"), reason="FSDP2 requires torch >= 2.6"
)
class TestShardedBaseLayerShapes:
    """
    Check the shapes of LoRA adapters on layers with DTensor weights.

        * FSDP2 is applied to the base model but not the LoRA adapters, at least yet. They need to be initialized with
          the full shape and will never be sharded.
        * TP is applied to both the base model and the LoRA adapters. There are two cases:
            1. There is a Transformers TP plan, in which case the adapters are initialized with the full shape and
            sharded afterwards.
            2. There is no Transformers TP plan, in which case the adapters are initialized with the sharded shape.
    """

    model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

    @pytest.fixture
    def cache_model(self, monkeypatch):
        with hub_online_once(self.model_id):
            AutoModelForCausalLM.from_pretrained(self.model_id)
        # the spawned processes don't share the state of hub_online_once, so they have to load the model from the cache
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    @staticmethod
    def _run_worker(rank, init_file, worker_fn, args):
        # Transformers reads the rank from the environment when it builds its TP device map
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
        try:
            worker_fn(*args)
        finally:
            dist.destroy_process_group()

    def _spawn(self, tmp_path, worker_fn, *args):
        mp.spawn(self._run_worker, args=(str(tmp_path / "dist_init"), worker_fn, args), nprocs=2, join=True)

    @staticmethod
    def fully_shard_lora_shapes(model_id, mesh_shape, mesh_dim_names):
        # FSDP2 is public from torch 2.6 onwards.
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=mesh_dim_names)

        reference = AutoModelForCausalLM.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        torch.manual_seed(0)
        model = get_peft_model(model, LoraConfig(target_modules="all-linear", init_lora_weights=False))

        torch.manual_seed(0)
        reference = get_peft_model(reference, LoraConfig(target_modules="all-linear", init_lora_weights=False))

        shapes = {name: param.shape for name, param in model.named_parameters() if "lora_" in name}
        expected_shapes = {name: param.shape for name, param in reference.named_parameters() if "lora_" in name}
        assert shapes == expected_shapes
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            torch.testing.assert_close(model(input_ids=input_ids).logits, reference(input_ids=input_ids).logits)

    @staticmethod
    def parallelize_module_lora_shapes(mesh_dim_name, register_mesh_dim, expected_features):
        if register_mesh_dim:
            # The spawned process only runs this check, so the change doesn't reach other tests
            TpMeshDimRegistry.register(mesh_dim_name)

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=(mesh_dim_name,))

        # The first layer (10 -> 20) is sharded on its output dim, the second (20 -> 2) on its input dim
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 2))
        model = parallelize_module(model, mesh, {"0": ColwiseParallel(), "1": RowwiseParallel()})
        model = get_peft_model(model, LoraConfig(target_modules=["0", "1"], r=4))
        assert model.base_model.model[0].lora_B["default"].weight.shape == (expected_features, 4)
        assert model.base_model.model[1].lora_A["default"].weight.shape == (4, expected_features)

    @staticmethod
    def tp_plan_lora_matches_unsharded_model(model_id, mesh_dim_names, init_lora_weights):
        from transformers.distributed import DistributedConfig

        def full(tensor):
            return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=mesh_dim_names)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, distributed_config=DistributedConfig(tp_size=2), device_mesh=mesh
        )
        reference = AutoModelForCausalLM.from_pretrained(model_id)
        # Compare initialization as well as loss and gradients: default initialization zeros B, so forward alone
        # would not catch incorrect initialization of A. init_lora_weights=False exercises both non-zero factors.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
        torch.manual_seed(0)
        model = get_peft_model(model, LoraConfig(target_modules=target_modules, init_lora_weights=init_lora_weights))
        torch.manual_seed(0)
        reference = get_peft_model(
            reference, LoraConfig(target_modules=target_modules, init_lora_weights=init_lora_weights)
        )

        lora_params = {name: param for name, param in model.named_parameters() if "lora_" in name}
        reference_params = {name: param for name, param in reference.named_parameters() if "lora_" in name}
        assert lora_params.keys() == reference_params.keys()
        for name, param in lora_params.items():
            torch.testing.assert_close(full(param.detach()), reference_params[name].detach())

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        loss = model(input_ids=input_ids, labels=input_ids).loss
        reference_loss = reference(input_ids=input_ids, labels=input_ids).loss
        torch.testing.assert_close(full(loss), reference_loss)
        loss.backward()
        reference_loss.backward()
        for name, param in lora_params.items():
            torch.testing.assert_close(full(param.grad), reference_params[name].grad)

    # an unnamed mesh is what fully_shard builds by default
    @pytest.mark.parametrize(
        "mesh_shape, mesh_dim_names",
        [((2,), None), ((2,), ("fsdp",)), ((2,), ("dp",)), ((1, 2), ("dp_replicate", "dp_shard"))],
        ids=["unnamed", "fsdp", "dp", "hsdp"],
    )
    @pytest.mark.usefixtures("cache_model")
    def test_fully_shard_lora_shapes(self, tmp_path, mesh_shape, mesh_dim_names):
        self._spawn(tmp_path, self.fully_shard_lora_shapes, self.model_id, mesh_shape, mesh_dim_names)

    # Registered TP dimensions use local shapes (10 features); other dimensions use full shapes (20).
    @pytest.mark.parametrize(
        "mesh_dim_name, register_mesh_dim, expected_features",
        [("tp", False, 10), ("model", False, 20), ("model", True, 10)],
    )
    def test_parallelize_module_lora_shapes(self, tmp_path, mesh_dim_name, register_mesh_dim, expected_features):
        self._spawn(tmp_path, self.parallelize_module_lora_shapes, mesh_dim_name, register_mesh_dim, expected_features)

    # an unnamed mesh is what Transformers v5.17 builds when no device_mesh is passed
    @pytest.mark.parametrize("mesh_dim_names", [None, ("tp",)])
    @pytest.mark.parametrize("init_lora_weights", [True, False], ids=["default-init", "nonzero-init"])
    @pytest.mark.skipif(not is_transformers_ge_v5_17_0, reason="LoRA with DTensor TP requires transformers >= 5.17")
    @pytest.mark.skipif(
        infer_device() != "cpu", reason="Transformers builds the TP mesh on the accelerator if there is one"
    )
    @pytest.mark.usefixtures("cache_model")
    def test_tp_plan_lora_matches_unsharded_model(self, tmp_path, mesh_dim_names, init_lora_weights):
        self._spawn(
            tmp_path, self.tp_plan_lora_matches_unsharded_model, self.model_id, mesh_dim_names, init_lora_weights
        )


def test_tp_mesh_dim_registry(monkeypatch):
    # Restore the registry after the test so custom names do not affect other tests.
    monkeypatch.setattr(TpMeshDimRegistry, "_names", TpMeshDimRegistry._names)
    assert TpMeshDimRegistry.is_registered("tp")
    assert not TpMeshDimRegistry.is_registered("model")
    assert not TpMeshDimRegistry.is_registered("fsdp")

    TpMeshDimRegistry.register("model")
    TpMeshDimRegistry.register("model")
    assert TpMeshDimRegistry.is_registered("model")
    assert TpMeshDimRegistry.is_registered("tp")
    assert not TpMeshDimRegistry.is_registered("fsdp")
