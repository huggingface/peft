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

import copy
import os
import platform
from datetime import timedelta
from pathlib import Path

import packaging.version
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file as safe_load_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoModelForCausalLM

from peft import IA3Config, LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.import_utils import is_transformers_dtensor_tp, is_transformers_ge_v5_17_0
from peft.utils import infer_device
from peft.utils.integrations import TpMeshDimRegistry
from peft.utils.save_and_load import _gather_state_dict_for_save

from .test_custom_models import MLP, ModelMha
from .testing_utils import hub_online_once, set_init_weights_false


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
        # Older Transformers versions do not provide DistributedConfig.
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


def test_gather_plain_state_dict_preserves_tensors():
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().requires_grad_()
    extra_state = {"step": 1}
    result = _gather_state_dict_for_save({"weight": weight, "extra_state": extra_state})
    assert result["weight"] is weight
    assert result["weight"].requires_grad
    assert result["weight"].stride() == weight.stride()
    assert result["extra_state"] is extra_state


@pytest.mark.skipif(platform.system() != "Linux", reason="Run FSDP tests only on Linux")
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
class TestFsdp2StateDict:
    @staticmethod
    def _save_load(rank, world_size, root, device_type, config_cls, model_id="MLP"):
        from torch.distributed.fsdp import fully_shard

        torch.set_num_threads(1)
        device = torch.device(device_type, rank) if device_type != "cpu" else torch.device("cpu")
        if device_type != "cpu":
            torch.get_device_module(device_type).set_device(device)
        dist.init_process_group(
            dist.get_default_backend_for_device(device_type),
            init_method=Path(root, "rendezvous").as_uri(),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        try:
            torch.manual_seed(0)
            base = {"MLP": MLP, "MHA": ModelMha}[model_id]().to(device).eval()
            target, saved_module = ("lin0", "lin1") if model_id == "MLP" else ("mha", "lin0")
            kwargs = set_init_weights_false(
                config_cls, {"target_modules": [target], "modules_to_save": [saved_module]}
            )
            if config_cls == IA3Config:
                kwargs["feedforward_modules"] = []
            config = config_cls(**kwargs)
            model = get_peft_model(copy.deepcopy(base), config)
            model.add_adapter("selected", copy.deepcopy(config))
            model.set_adapter("selected")
            reference = copy.deepcopy(model).eval()
            mesh_shape = (world_size,) if world_size == 2 else (2, 2)
            mesh_names = ("shard",) if world_size == 2 else ("replicate", "shard")
            mesh = init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_names)
            fully_shard(model.base_model.model.get_submodule(target), mesh=mesh)
            fully_shard(model, mesh=mesh)
            assert not getattr(model, "_tp_size", None)

            inputs = torch.arange(90, device=device).view(9, 10).float()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3, foreach=False)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                actual_output = model(inputs)
                reference_output = reference(inputs)
                torch.testing.assert_close(actual_output, reference_output)
                actual_output.sum().backward()
                reference_output.sum().backward()
                reference_params = dict(reference.named_parameters())
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    assert param.grad is not None, name
                    gradient = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                    torch.testing.assert_close(gradient, reference_params[name].grad)
                optimizer.step()
                reference_optimizer.step()
            expected = {
                key: value.clone()
                for key, value in get_peft_model_state_dict(reference, adapter_name="selected").items()
            }

            state_dict = model.state_dict(keep_vars=True)
            gathered = []
            full_tensor = DTensor.full_tensor

            def spy_full_tensor(tensor, *args, **kwargs):
                gathered.append(tensor)
                return full_tensor(tensor, *args, **kwargs)

            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(DTensor, "full_tensor", spy_full_tensor)
                actual = get_peft_model_state_dict(model, state_dict=state_dict, adapter_name="selected")
            assert actual.keys() == expected.keys()
            unselected_ids = {id(value) for name, value in state_dict.items() if "selected" not in name.split(".")}
            assert len(gathered) == len(expected)
            assert not any(id(tensor) in unselected_ids for tensor in gathered)
            for key in expected:
                assert not isinstance(actual[key], DTensor)
                assert actual[key].device.type == "cpu"
                assert not actual[key].requires_grad
                assert actual[key].is_contiguous()
                torch.testing.assert_close(actual[key], expected[key].cpu())
            # Check serialization and restoration exactly against the gathered distributed weights.
            expected = {key: value.clone() for key, value in actual.items()}

            for safe_serialization in (True, False):
                destination = Path(root, "safe" if safe_serialization else "bin")
                model.save_pretrained(
                    destination,
                    selected_adapters=["selected"],
                    is_main_process=rank == 0,
                    safe_serialization=safe_serialization,
                )
                dist.barrier()
                adapter_path = destination / "selected"
                saved = (
                    safe_load_file(adapter_path / "adapter_model.safetensors")
                    if safe_serialization
                    else torch.load(adapter_path / "adapter_model.bin", weights_only=True, map_location="cpu")
                )
                assert saved.keys() == expected.keys()
                for key in expected:
                    assert not isinstance(saved[key], DTensor)
                    torch.testing.assert_close(saved[key], expected[key].cpu(), rtol=0, atol=0)
                reloaded = PeftModel.from_pretrained(copy.deepcopy(base), adapter_path).to(device).eval()
                inputs = torch.arange(90, device=device).view(9, 10).float()
                with torch.no_grad():
                    torch.testing.assert_close(reloaded(inputs), reference(inputs))
                    # Direct restoration into DTensors requires Transformers' DTensor loading support.
                    if not is_transformers_dtensor_tp:
                        continue
                    # Make restoring into distributed parameters observable, rather than reloading unchanged weights.
                    for name, param in model.named_parameters():
                        if "selected" in name.split("."):
                            param.to_local().zero_()
                set_peft_model_state_dict(model, saved, adapter_name="selected")
                restored = get_peft_model_state_dict(model, adapter_name="selected")
                for key in expected:
                    torch.testing.assert_close(restored[key], expected[key].cpu(), rtol=0, atol=0)
        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("world_size", [2, 4])
    # LoRA and IA3 cover additive low-rank and multiplicative parameters; the FSDP method matrix covers other configs.
    @pytest.mark.parametrize(
        "model_id, config_cls",
        [("MLP", LoraConfig), ("MLP", IA3Config), ("MHA", LoraConfig)],
        ids=["mlp-lora", "mlp-ia3", "mha-lora"],
    )
    def test_save_load_without_tp_metadata(self, world_size, model_id, config_cls, tmp_path):
        # Regression for #3802, including HSDP on a 2x2 CPU mesh and both checkpoint formats.
        mp.spawn(
            self._save_load,
            args=(world_size, str(tmp_path), "cpu", config_cls, model_id),
            nprocs=world_size,
            join=True,
        )
