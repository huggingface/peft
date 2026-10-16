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

import math
import os
import platform

import packaging.version
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.import_utils import is_transformers_ge_v5_17_0
from peft.utils import infer_device
from peft.utils.constants import TP_MESH_DIM_NAMES

from .testing_utils import hub_online_once, require_torch_multi_gpu


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


class _ExpertWeights(nn.Module):
    def __init__(self, is_transposed, num_experts=4):
        super().__init__()
        self.is_transposed = is_transposed
        shape = (num_experts, 6, 8) if is_transposed else (num_experts, 8, 6)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        weight = self.weight
        if not self.is_transposed:
            weight = weight.transpose(-2, -1)
        return torch.bmm(x, weight)


class _ExpertParameterModel(nn.Module):
    def __init__(self, is_transposed, num_experts=4):
        super().__init__()
        self.experts = _ExpertWeights(is_transposed, num_experts)

    def forward(self, x):
        return self.experts(x)


@pytest.mark.skipif(not dist.is_available(), reason="These tests require torch.distributed")
@pytest.mark.skipif(
    packaging.version.parse(torch.__version__) < packaging.version.parse("2.6"), reason="DTensor requires torch >= 2.6"
)
class TestExpertParameterParallelism:
    @staticmethod
    def _check(rank, init_uri, shard_dim, is_transposed, num_experts=4, mesh_shape=(2,), squeeze_mesh=False):
        dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=math.prod(mesh_shape))
        try:
            torch.set_num_threads(1)
            torch.manual_seed(42)
            base = _ExpertParameterModel(is_transposed, num_experts)
            weight = base.experts.weight.detach().clone()
            reference_base = _ExpertParameterModel(is_transposed, num_experts)
            reference_base.experts.weight = nn.Parameter(weight.clone())
            shard_dims = shard_dim if isinstance(shard_dim, tuple) else (shard_dim,)
            mesh_names = ("efsdp", "ep") if squeeze_mesh else tuple("ep" if dim == 0 else "tp" for dim in shard_dims)
            mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=mesh_names)
            placements = [Shard(dim) if dim is not None else Replicate() for dim in shard_dims]
            base.experts.weight = nn.Parameter(distribute_tensor(weight.clone(), mesh, placements))
            config = LoraConfig(r=2, lora_alpha=2, target_parameters=["experts.weight"], init_lora_weights=False)
            torch.manual_seed(7)
            model = get_peft_model(base, config)
            if squeeze_mesh:
                # The base is already unsharded during checkpoint recomputation, dropping its singleton FSDP axis.
                # Adapter parameters keep their original mesh; only their temporary factors should follow the base.
                wrapper = base.experts
                original = wrapper.get_param()
                wrapper.get_base_layer().weight = nn.Parameter(
                    DTensor.from_local(
                        original.to_local(), mesh["ep"], [Shard(0)], shape=original.shape, stride=original.stride()
                    ),
                    requires_grad=False,
                )
            torch.manual_seed(7)
            reference = get_peft_model(reference_base, LoraConfig(**config.to_dict()))
            x = torch.randn(num_experts, 3, 6)
            input_dim = 1 if is_transposed else 2
            input_placements = [
                Shard(0) if dim == 0 else Shard(2) if dim == input_dim else Replicate() for dim in shard_dims
            ]
            input_mesh = mesh["ep"] if squeeze_mesh else mesh
            distributed_x = distribute_tensor(
                x, input_mesh, input_placements[1:] if squeeze_mesh else input_placements
            )
            actual = model(distributed_x).full_tensor()
            expected = reference(x)
            torch.testing.assert_close(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            actual_params = {name: param for name, param in model.named_parameters() if "lora_" in name}
            expected_params = {name: param for name, param in reference.named_parameters() if "lora_" in name}
            for name, param in actual_params.items():
                assert isinstance(param, DTensor)
                assert param.grad is not None
                torch.testing.assert_close(param.full_tensor(), expected_params[name])
                torch.testing.assert_close(param.grad.full_tensor(), expected_params[name].grad)

            # Distributed checkpoints keep the ordinary unsharded two-dimensional LoRA format.
            state = get_peft_model_state_dict(model)
            expected_state = get_peft_model_state_dict(reference)
            assert state.keys() == expected_state.keys()
            for name, value in state.items():
                assert not isinstance(value, DTensor)
                torch.testing.assert_close(value, expected_state[name])
            state = {name: value.detach().clone() for name, value in state.items()}
            with torch.no_grad():
                for param in actual_params.values():
                    param.zero_()
            set_peft_model_state_dict(model, state)
            torch.testing.assert_close(model(distributed_x).full_tensor(), expected)
        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("shard_dim", [None, 0, 1, 2], ids=["replicated", "experts", "dim1", "dim2"])
    @pytest.mark.parametrize("is_transposed", [False, True])
    def test_expert_parameters_match_unsharded(self, tmp_path, shard_dim, is_transposed):
        mp.spawn(
            self._check,
            args=((tmp_path / "dist_init").as_uri(), shard_dim, is_transposed),
            nprocs=2,
            join=True,
        )

    @pytest.mark.parametrize("is_transposed", [False, True])
    @pytest.mark.parametrize(
        "shard_dim,num_experts,mesh_shape",
        [(0, 3, (2,)), ((0, 1), 4, (2, 2)), ((0, 2), 4, (2, 2))],
        ids=["uneven-experts", "ep-dim1", "ep-dim2"],
    )
    def test_expert_parameter_meshes(self, tmp_path, shard_dim, num_experts, mesh_shape, is_transposed):
        mp.spawn(
            self._check,
            args=((tmp_path / "dist_init").as_uri(), shard_dim, is_transposed, num_experts, mesh_shape),
            nprocs=math.prod(mesh_shape),
            join=True,
        )

    @pytest.mark.parametrize("is_transposed", [False, True])
    def test_expert_factors_follow_unsharded_base_mesh(self, tmp_path, is_transposed):
        mp.spawn(
            self._check,
            args=((tmp_path / "dist_init").as_uri(), (0, 0), is_transposed, 4, (1, 2), True),
            nprocs=2,
            join=True,
        )

    @staticmethod
    def _check_local_proxy_checkpoint(rank, init_uri):
        from torch.utils.checkpoint import checkpoint

        from peft.tuners.lora.layer import _LoraFactorsProxy

        dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=2)
        try:
            torch.set_num_threads(1)
            torch.manual_seed(7)
            mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("efsdp", "ep"))
            placements = [Shard(0), Shard(0)]
            weight = distribute_tensor(torch.randn(4, 6, 8), mesh, placements)
            lhs = nn.Parameter(distribute_tensor(torch.randn(4, 6, 2), mesh, placements))
            rhs = nn.Parameter(distribute_tensor(torch.randn(4, 2, 8), mesh, placements))
            x = torch.randn(2, 3, 6, requires_grad=True)
            expected = torch.bmm(x, weight.to_local() + 0.5 * torch.bmm(lhs.to_local(), rhs.to_local()))
            expected.square().sum().backward()
            expected_grads = [param.grad.full_tensor().clone() for param in (lhs, rhs)]
            lhs.grad = rhs.grad = x.grad = None
            calls = 0

            def forward(local_x):
                nonlocal calls
                # The original FSDP forward and checkpoint recomputation expose different logical meshes.
                tensors = (weight, lhs, rhs)
                if calls:
                    tensors = tuple(
                        DTensor.from_local(t.to_local(), mesh["ep"], [Shard(0)], shape=t.shape, stride=t.stride())
                        for t in tensors
                    )
                calls += 1
                base, left, right = tensors
                return torch.bmm(local_x, _LoraFactorsProxy(left, right, 0.5, local_tensor=True)(base))

            actual = checkpoint(forward, x, use_reentrant=False)
            torch.testing.assert_close(actual, expected)
            actual.square().sum().backward()
            assert calls == 2
            for param, expected_grad in zip((lhs, rhs), expected_grads):
                torch.testing.assert_close(param.grad.full_tensor(), expected_grad)
        finally:
            dist.destroy_process_group()

    def test_local_expert_proxy_checkpoint_mesh_transition(self, tmp_path):
        mp.spawn(self._check_local_proxy_checkpoint, args=((tmp_path / "dist_init").as_uri(),), nprocs=2, join=True)

    @staticmethod
    def _check_transformers(rank, init_uri, checkpoint, ep_size, device_type):
        from transformers.distributed import DistributedConfig

        # Transformers chooses the CUDA device from LOCAL_RANK, independently of torch.cuda.set_device.
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
        if device_type == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl" if device_type == "cuda" else "gloo", init_method=init_uri, rank=rank, world_size=2
        )
        try:
            torch.set_num_threads(1)
            device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
            model_kwargs = {"dtype": torch.float32, "attn_implementation": "eager", "experts_implementation": "eager"}
            base = AutoModelForCausalLM.from_pretrained(
                checkpoint, distributed_config=DistributedConfig(tp_size=2, ep_size=ep_size), **model_kwargs
            )
            config = LoraConfig(
                r=2,
                lora_alpha=2,
                target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                init_lora_weights=False,
            )
            torch.manual_seed(7)
            model = get_peft_model(base, config)
            reference_base = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs).to(device)
            torch.manual_seed(7)
            reference = get_peft_model(reference_base, LoraConfig(**config.to_dict()))
            ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
            actual = model(input_ids=ids, labels=ids)
            expected = reference(input_ids=ids, labels=ids)
            torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual.loss, expected.loss)
            actual.loss.backward()
            expected.loss.backward()
            reference_params = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                if "lora_" in name:
                    assert param.grad is not None
                    grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                    torch.testing.assert_close(grad, reference_params[name].grad, atol=2e-6, rtol=2e-5)
            state, expected_state = get_peft_model_state_dict(model), get_peft_model_state_dict(reference)
            assert state.keys() == expected_state.keys()
            for name, value in state.items():
                assert not isinstance(value, DTensor)
                # Transformers gathers distributed checkpoints onto CPU; compare portable values, not devices.
                torch.testing.assert_close(value.cpu(), expected_state[name].cpu())
            adapter_dir = os.path.join(checkpoint, "adapter")
            model.save_pretrained(adapter_dir, is_main_process=rank == 0)
            dist.barrier()
            for low_cpu_mem_usage in (False, True):
                restored_base = AutoModelForCausalLM.from_pretrained(
                    checkpoint, distributed_config=DistributedConfig(tp_size=2, ep_size=ep_size), **model_kwargs
                )
                restored = PeftModel.from_pretrained(
                    restored_base,
                    adapter_dir,
                    adapter_name="restored",
                    is_trainable=True,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                torch.testing.assert_close(restored(input_ids=ids).logits, expected.logits, atol=2e-6, rtol=2e-5)
            model.merge_adapter()
            torch.testing.assert_close(model(input_ids=ids).logits, expected.logits, atol=2e-6, rtol=2e-5)
            model.unmerge_adapter()
            torch.testing.assert_close(model(input_ids=ids).logits, expected.logits, atol=2e-6, rtol=2e-5)
        finally:
            dist.destroy_process_group()

    @staticmethod
    def _save_base(path):
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        torch.manual_seed(42)
        config = Qwen3MoeConfig(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=32,
        )
        Qwen3MoeForCausalLM(config).save_pretrained(path)

    @pytest.mark.skipif(
        packaging.version.parse(transformers.__version__) < packaging.version.parse("5.19"),
        reason="DistributedConfig expert parallelism requires transformers >= 5.19",
    )
    @pytest.mark.skipif(torch.cuda.is_available(), reason="The CPU loader test requires a CPU accelerator runtime")
    @pytest.mark.parametrize("ep_size", [1, 2], ids=["tp", "tp-ep"])
    def test_transformers_experts_cpu(self, tmp_path, ep_size):
        checkpoint = str(tmp_path / "base")
        self._save_base(checkpoint)
        mp.spawn(
            self._check_transformers, args=((tmp_path / "dist_init").as_uri(), checkpoint, ep_size, "cpu"), nprocs=2
        )

    @pytest.mark.skipif(
        packaging.version.parse(transformers.__version__) < packaging.version.parse("5.19"),
        reason="DistributedConfig expert parallelism requires transformers >= 5.19",
    )
    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    @pytest.mark.parametrize("ep_size", [1, 2], ids=["tp", "tp-ep"])
    def test_transformers_experts_cuda(self, tmp_path, ep_size):
        checkpoint = str(tmp_path / "base")
        self._save_base(checkpoint)
        mp.spawn(
            self._check_transformers, args=((tmp_path / "dist_init").as_uri(), checkpoint, ep_size, "cuda"), nprocs=2
        )
