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
from peft.utils.constants import TP_MESH_DIM_NAMES

from .testing_utils import hub_online_once


@pytest.mark.skipif(platform.system() != "Linux", reason="Run distributed tests only on Linux")
@pytest.mark.skipif(not dist.is_available(), reason="These tests require torch.distributed")
@pytest.mark.skipif(
    packaging.version.parse(torch.__version__) < packaging.version.parse("2.6"), reason="FSDP2 requires torch >= 2.6"
)
class TestShardedBaseLayerShapes:
    """
    Check the shapes of LoRA factors on layers with DTensor weights, see #3803.

    FSDP2 shards the stored weight but the layer still computes the full projection, whatever the mesh dimensions are
    called. A tensor parallel layer without a Transformers TP plan computes on its local shard. The tests run two
    processes with gloo on CPU meshes.
    """

    model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

    @pytest.fixture
    def cache_model(self, monkeypatch):
        with hub_online_once(self.model_id):
            AutoModelForCausalLM.from_pretrained(self.model_id)
        # the spawned processes don't share the state of hub_online_once, so they have to load the model from the cache
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    @staticmethod
    def _run_worker(rank, init_file, check_fn, args):
        # Transformers reads the rank from the environment when it builds its TP device map
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
        try:
            check_fn(*args)
        finally:
            dist.destroy_process_group()

    def _spawn(self, tmp_path, check_fn, *args):
        mp.spawn(self._run_worker, args=(str(tmp_path / "dist_init"), check_fn, args), nprocs=2, join=True)

    @staticmethod
    def _check_fully_shard(model_id, mesh_shape, mesh_dim_names):
        # imported here, as FSDP2 is public from torch 2.6 on
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=mesh_dim_names)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        reference = AutoModelForCausalLM.from_pretrained(model_id)
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
    def _check_parallelize_module(mesh_dim_name, added_tp_mesh_dim_name, expected_features):
        if added_tp_mesh_dim_name is not None:
            # the spawned process only runs this check, so the change doesn't reach other tests
            TP_MESH_DIM_NAMES.add(added_tp_mesh_dim_name)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=(mesh_dim_name,))
        # the first layer (10 -> 20) is sharded on its output dim, the second (20 -> 2) on its input dim
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 2))
        model = parallelize_module(model, mesh, {"0": ColwiseParallel(), "1": RowwiseParallel()})
        model = get_peft_model(model, LoraConfig(target_modules=["0", "1"], r=4))
        assert model.base_model.model[0].lora_B["default"].weight.shape == (expected_features, 4)
        assert model.base_model.model[1].lora_A["default"].weight.shape == (4, expected_features)

    @staticmethod
    def _check_tp_plan(model_id, mesh_dim_names):
        # imported here, as older Transformers versions don't have it
        from transformers.distributed import DistributedConfig

        def full(tensor):
            return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=mesh_dim_names)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, distributed_config=DistributedConfig(tp_size=2), device_mesh=mesh
        )
        reference = AutoModelForCausalLM.from_pretrained(model_id)
        # init_lora_weights=False makes both LoRA factors non-zero. Every rank uses the same seed, so the gathered
        # adapter must equal the adapter of the unsharded model.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
        torch.manual_seed(0)
        model = get_peft_model(model, LoraConfig(target_modules=target_modules, init_lora_weights=False))
        torch.manual_seed(0)
        reference = get_peft_model(reference, LoraConfig(target_modules=target_modules, init_lora_weights=False))

        lora_params = {name: param for name, param in model.named_parameters() if "lora_" in name}
        reference_params = {name: param for name, param in reference.named_parameters() if "lora_" in name}
        torch.testing.assert_close(
            {name: full(param.detach()) for name, param in lora_params.items()},
            {name: param.detach() for name, param in reference_params.items()},
        )

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        loss = model(input_ids=input_ids, labels=input_ids).loss
        reference_loss = reference(input_ids=input_ids, labels=input_ids).loss
        torch.testing.assert_close(full(loss), reference_loss)
        loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(
            {name: full(param.grad) for name, param in lora_params.items()},
            {name: param.grad for name, param in reference_params.items()},
        )

    # an unnamed mesh is what fully_shard builds by default
    @pytest.mark.parametrize(
        "mesh_shape, mesh_dim_names",
        [((2,), None), ((2,), ("fsdp",)), ((2,), ("dp",)), ((1, 2), ("dp_replicate", "dp_shard"))],
        ids=["unnamed", "fsdp", "dp", "hsdp"],
    )
    @pytest.mark.usefixtures("cache_model")
    def test_fully_shard_lora_shapes(self, tmp_path, mesh_shape, mesh_dim_names):
        self._spawn(tmp_path, self._check_fully_shard, self.model_id, mesh_shape, mesh_dim_names)

    # Without a TP plan, the mesh dim names decide whether a layer is tensor parallel. The sharded side of the LoRA
    # factors has 10 features if it is and 20 otherwise.
    @pytest.mark.parametrize(
        "mesh_dim_name, added_tp_mesh_dim_name, expected_features",
        [("tp", None, 10), ("model", None, 20), ("model", "model", 10)],
    )
    def test_parallelize_module_lora_shapes(self, tmp_path, mesh_dim_name, added_tp_mesh_dim_name, expected_features):
        self._spawn(tmp_path, self._check_parallelize_module, mesh_dim_name, added_tp_mesh_dim_name, expected_features)

    # an unnamed mesh is what Transformers v5.17 builds when no device_mesh is passed
    @pytest.mark.parametrize("mesh_dim_names", [None, ("tp",)])
    @pytest.mark.skipif(not is_transformers_ge_v5_17_0, reason="LoRA with DTensor TP requires transformers >= 5.17")
    @pytest.mark.skipif(
        infer_device() != "cpu", reason="Transformers builds the TP mesh on the accelerator if there is one"
    )
    @pytest.mark.usefixtures("cache_model")
    def test_tp_plan_lora_matches_unsharded_model(self, tmp_path, mesh_dim_names):
        self._spawn(tmp_path, self._check_tp_plan, self.model_id, mesh_dim_names)
