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
import transformers
from safetensors.torch import load_file as safe_load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoModelForCausalLM, Qwen3Config

from peft import IA3Config, LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.import_utils import is_transformers_dtensor_tp
from peft.tuners.lora.layer import LoraLayer
from peft.utils import infer_device
from peft.utils.integrations import TpMeshDimRegistry
from peft.utils.save_and_load import _gather_state_dict_for_save

from .test_custom_models import MLP
from .testing_utils import set_init_weights_false


@pytest.mark.skipif(platform.system() != "Linux", reason="Run distributed tests only on Linux")
@pytest.mark.skipif(not dist.is_available(), reason="These tests require torch.distributed")
@pytest.mark.skipif(
    packaging.version.parse(torch.__version__) < packaging.version.parse("2.6"), reason="FSDP2 requires torch >= 2.6"
)
class TestShardedBaseLayerShapes:
    """
    Check the shapes of LoRA adapters on layers with DTensor weights.

        * LoRA adapters injected after FSDP2 sharding need full-sized initialization and remain unsharded.
        * Transformers TP initializes full-sized adapters and then shards them using the base layer's TP plan. The loss
          and gradient tests compare this path against an unsharded reference.
        * PyTorch's parallelize_module() shards the base weights without setting a Transformers TP plan. The shape
          tests check that adapters use local dimensions for registered TP mesh names and full dimensions otherwise.
          PEFT does not apply Transformers TP hooks to these adapters.

    Note: Don't overly parametrize the tests, as each individual test case incurs an overhead from creating and
    destroying the process group. Instead, pass a list of test cases to `_run_worker`, which are all run within the
    same process group. These test cases should be able to run without affecting each other, e.g. via side effects.
    """

    @pytest.fixture
    def model_config(self):
        # Independent head dimensions keep attention valid with hidden widths not divisible by the TP size.
        return Qwen3Config(
            vocab_size=21,
            hidden_size=17,
            intermediate_size=33,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )

    @pytest.fixture
    def model_checkpoint(self, tmp_path, model_config):
        checkpoint = tmp_path / "model"
        AutoModelForCausalLM.from_config(model_config).save_pretrained(checkpoint)
        return checkpoint

    @staticmethod
    def _run_worker(rank, init_file, worker_fn, cases):
        # Transformers reads the rank from the environment when it builds its TP device map
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
        try:
            # Note: Run cases in a loop instead of running separate parametrized tests to save the time for test setup
            # and teardown, which is signifcant. For this to work, the cases should be able to run without affecting one
            # another.
            for args in cases:
                # ensure that rng is reset for each test case
                with torch.random.fork_rng(devices=[]):
                    try:
                        worker_fn(*args)
                    except Exception as exc:
                        exc.add_note(f"{worker_fn.__name__} parameters: {args!r}")
                        raise
        finally:
            dist.destroy_process_group()

    def _spawn(self, tmp_path, worker_fn, cases):
        # Each test owns its workers and rendezvous file, including under xdist; only its parameter cases share them.
        mp.spawn(self._run_worker, args=(str(tmp_path / "dist_init"), worker_fn, cases), nprocs=2, join=True)

    @staticmethod
    def fsdp2_lora_shapes_injected_after_sharding(model_id, mesh_shape, mesh_dim_names):
        """
        Check that LoRA adapters injected after FSDP2 sharding have the full shapes of the base weights and that the
        forward pass matches a reference model with unsharded weights.
        """
        # FSDP2 is public from torch 2.6 onwards.
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=mesh_dim_names)

        reference = AutoModelForCausalLM.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        target_modules = [
            "embed_tokens",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        # Inject after sharding to check shape inference from FSDP-managed base weights.
        torch.manual_seed(0)
        model = get_peft_model(model, LoraConfig(target_modules=target_modules, init_lora_weights=False))

        # Check that the LoRA adapters are not DTensors.
        # The adapters are injected after fully_shard, so they are not sharded and remain regular tensors.
        assert all(not isinstance(param, DTensor) for name, param in model.named_parameters() if "lora_" in name)

        torch.manual_seed(0)
        reference = get_peft_model(reference, LoraConfig(target_modules=target_modules, init_lora_weights=False))

        # Check that global shapes match the reference model, which is not sharded.
        shapes = {name: param.shape for name, param in model.named_parameters() if "lora_" in name}
        expected_shapes = {name: param.shape for name, param in reference.named_parameters() if "lora_" in name}
        assert shapes == expected_shapes

        # Check that the forward pass matches the reference model.
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            torch.testing.assert_close(model(input_ids=input_ids).logits, reference(input_ids=input_ids).logits)

    @staticmethod
    def pytorch_tp_lora_adapter_shapes(model_config, mesh_dim_name, register_mesh_dim, embedding_style):
        """
        Check that LoRA adapters on layers with DTensor weights have the correct shapes when the base model is sharded
        with PyTorch's parallelize_module().
        """
        if register_mesh_dim:
            TpMeshDimRegistry.register(mesh_dim_name)

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=(mesh_dim_name,))

        model = AutoModelForCausalLM.from_config(model_config)
        model = parallelize_module(
            model,
            mesh,
            {
                "model.embed_tokens": embedding_style(),
                "model.layers.0.self_attn.q_proj": ColwiseParallel(),
                "model.layers.0.self_attn.o_proj": RowwiseParallel(),
            },
        )
        model = get_peft_model(model, LoraConfig(target_modules=["embed_tokens", "q_proj", "o_proj"], r=4))

        for adapter in model.modules():
            if not isinstance(adapter, LoraLayer):
                continue
            base_layer = adapter.get_base_layer()
            assert isinstance(base_layer.weight, DTensor)
            # Here we are not using Transformers TP API, so the base layer should not have a TP plan.
            assert getattr(base_layer, "_hf_tp_plan", None) is None

            # Registered TP mesh names use local dimensions; other names use full dimensions.
            expected_shape = (
                base_layer.weight.to_local().shape
                if TpMeshDimRegistry.is_registered(mesh_dim_name)
                else base_layer.weight.shape
            )
            if adapter.lora_embedding_A:
                in_features, out_features = expected_shape
                weight_A = adapter.lora_embedding_A["default"]
                weight_B = adapter.lora_embedding_B["default"]
            else:
                out_features, in_features = expected_shape
                weight_A = adapter.lora_A["default"].weight
                weight_B = adapter.lora_B["default"].weight
            assert (adapter.in_features, adapter.out_features) == (in_features, out_features)
            assert weight_A.shape == (4, in_features)
            assert weight_B.shape == (out_features, 4)

    @staticmethod
    def transformers_tp_lora_loss_and_gradients_match(model_id, mesh_dim_names, embedding_plan):
        """
        Check that LoRA adapters on a model sharded with Transformers TP have the correct shapes and that the loss and
        gradients match a reference model with unsharded weights.
        """
        from transformers.distributed import DistributedConfig

        def full(tensor):
            return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=mesh_dim_names)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            distributed_config=DistributedConfig(tp_size=2, tp_plan={"model.embed_tokens": embedding_plan}),
            device_mesh=mesh,
        )

        # Nonzero factors exercise both adapters in the forward and backward passes.
        target_modules = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
        torch.manual_seed(0)
        model = get_peft_model(model, LoraConfig(target_modules=target_modules, init_lora_weights=False))

        torch.manual_seed(0)
        reference = AutoModelForCausalLM.from_pretrained(model_id)
        reference = get_peft_model(reference, LoraConfig(target_modules=target_modules, init_lora_weights=False))

        expected_plans = {"colwise", "rowwise", embedding_plan}
        checked_plans = set()
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            base_layer = module.get_base_layer()
            tp_plan = getattr(base_layer, "_hf_tp_plan", None)
            if tp_plan not in expected_plans:
                continue

            rank = module.r["default"]

            # Expected: hooks will shard the adapter weights after initialization.
            #   * in_features, out_features are the full dimensions
            #   * adapter weights are sharded along the same dimension as the base layer's weight
            if module.lora_embedding_A:
                in_features, out_features = base_layer.weight.shape
                weight_A = module.lora_embedding_A["default"]
                weight_B = module.lora_embedding_B["default"]
            else:
                out_features, in_features = base_layer.weight.shape
                adapter_A = module.lora_A["default"]
                adapter_B = module.lora_B["default"]
                weight_A, weight_B = adapter_A.weight, adapter_B.weight

                # adapter_A and adapter_B are linear layers and should have the same in/out features metadata
                # as the base layer weight's global shape.
                assert (adapter_A.in_features, adapter_A.out_features) == (in_features, rank)
                assert (adapter_B.in_features, adapter_B.out_features) == (rank, out_features)

            # The LoRA layer should have the same in/out features metadata as the base layer.
            assert (module.in_features, module.out_features) == (in_features, out_features)

            assert weight_A.shape == (rank, in_features)
            assert weight_B.shape == (out_features, rank)
            if tp_plan in ("rowwise", "embedding_rowwise"):
                sharded_weight, replicated_weight, dim = weight_A, weight_B, 1
            else:
                sharded_weight, replicated_weight, dim = weight_B, weight_A, 0
            assert isinstance(sharded_weight, DTensor)
            assert not isinstance(replicated_weight, DTensor)

            # Check that the sharded weight as the correct local shape for the current rank.
            expected_local = torch.chunk(torch.empty(sharded_weight.shape), dist.get_world_size(), dim=dim)[
                dist.get_rank()
            ].shape
            assert sharded_weight.to_local().shape == expected_local
            checked_plans.add(tp_plan)

        assert checked_plans == expected_plans

        lora_params = {name: param for name, param in model.named_parameters() if "lora_" in name}
        reference_params = {name: param for name, param in reference.named_parameters() if "lora_" in name}
        assert lora_params.keys() == reference_params.keys()

        # Check that the LoRA weights match the reference model after initialization.
        for name, param in lora_params.items():
            torch.testing.assert_close(full(param.detach()), reference_params[name].detach())

        input_ids = torch.tensor([[0, 1, reference.config.vocab_size // 2, reference.config.vocab_size - 1]])
        output = model(input_ids=input_ids, labels=input_ids)
        reference_output = reference(input_ids=input_ids, labels=input_ids)

        # Check that logits and losses match from the same forward pass.
        torch.testing.assert_close(full(output.logits), reference_output.logits)
        torch.testing.assert_close(full(output.loss), reference_output.loss)

        # Check that the gradients match.
        output.loss.backward()
        reference_output.loss.backward()
        for name, param in lora_params.items():
            torch.testing.assert_close(full(param.grad), reference_params[name].grad)

    @staticmethod
    def transformers_tp_replicated_weights_match(model_id, embedding_plan, init_lora_weights):
        """
        Check that LoRA adapters on a model sharded with Transformers TP have the same replicated weights on all ranks.
        """
        from transformers.distributed import DistributedConfig

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            distributed_config=DistributedConfig(tp_size=2, tp_plan={"model.embed_tokens": embedding_plan}),
            device_mesh=mesh,
        )
        # Identical seeds could hide a missing broadcast of replicated adapter weights.
        torch.manual_seed(dist.get_rank())
        model = get_peft_model(
            model,
            LoraConfig(target_modules=["embed_tokens", "q_proj", "o_proj"], init_lora_weights=init_lora_weights),
        )
        replicated_weights = {
            name: param
            for name, param in model.named_parameters()
            if "lora_" in name and not isinstance(param, DTensor)
        }
        # One replicated factor for each of the embedding, colwise, and rowwise adapters.
        assert len(replicated_weights) == 3
        for name, param in replicated_weights.items():
            weight = param.detach().contiguous()
            gathered = [torch.empty_like(weight) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, weight)
            for rank, other in enumerate(gathered):
                torch.testing.assert_close(weight, other, rtol=0, atol=0, msg=f"{name} differs from rank {rank}")

    def test_fsdp2_lora_shapes_injected_after_sharding(self, tmp_path, model_checkpoint):
        cases = [
            (model_checkpoint, (2,), None),
            (model_checkpoint, (2,), ("fsdp",)),
            (model_checkpoint, (2,), ("dp",)),
            (model_checkpoint, (1, 2), ("dp_replicate", "dp_shard")),
        ]
        self._spawn(tmp_path, self.fsdp2_lora_shapes_injected_after_sharding, cases)

    def test_pytorch_tp_lora_adapter_shapes(self, tmp_path, model_config):
        # Run all unregistered-name checks before registration, which stays local to the spawned workers.
        cases = [
            (model_config, mesh_dim_name, register_mesh_dim, embedding_style)
            for mesh_dim_name, register_mesh_dim in [("tp", False), ("random_name", False), ("random_name", True)]
            for embedding_style in (RowwiseParallel, ColwiseParallel)
        ]
        self._spawn(tmp_path, self.pytorch_tp_lora_adapter_shapes, cases)

    @pytest.mark.skipif(
        packaging.version.parse(transformers.__version__) < packaging.version.parse("5.21.0"),
        reason="Embedding TP styles require transformers >= 5.21",
    )
    @pytest.mark.skipif(
        infer_device() != "cpu", reason="Transformers builds the TP mesh on the accelerator if there is one"
    )
    def test_transformers_tp_loss_and_gradients_match(self, tmp_path, model_checkpoint):
        cases = [
            (model_checkpoint, mesh_dim_names, embedding_plan)
            for mesh_dim_names in (None, ("tp",))
            for embedding_plan in ("embedding_rowwise", "embedding_colwise")
        ]
        self._spawn(tmp_path, self.transformers_tp_lora_loss_and_gradients_match, cases)

    @pytest.mark.skipif(
        packaging.version.parse(transformers.__version__) < packaging.version.parse("5.21.0"),
        reason="Embedding TP styles require transformers >= 5.21",
    )
    @pytest.mark.skipif(
        infer_device() != "cpu", reason="Transformers builds the TP mesh on the accelerator if there is one"
    )
    def test_transformers_tp_replicated_weights_match(self, tmp_path, model_checkpoint):
        cases = [
            (model_checkpoint, embedding_plan, init_lora_weights)
            for init_lora_weights in (True, False, "gaussian")
            for embedding_plan in ("embedding_rowwise", "embedding_colwise")
        ]
        self._spawn(tmp_path, self.transformers_tp_replicated_weights_match, cases)


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
    def _save_load(rank, world_size, root, device_type, config_cls):
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
            base = MLP().to(device).eval()
            kwargs = set_init_weights_false(config_cls, {"target_modules": ["lin0"], "modules_to_save": ["lin1"]})
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
            fully_shard(model.base_model.model.lin0, mesh=mesh)
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
    @pytest.mark.parametrize("config_cls", [LoraConfig, IA3Config])
    def test_save_load_without_tp_metadata(self, world_size, config_cls, tmp_path):
        # Regression for #3802, including HSDP on a 2x2 CPU mesh and both checkpoint formats.
        mp.spawn(
            self._save_load,
            args=(world_size, str(tmp_path), "cpu", config_cls),
            nprocs=world_size,
            join=True,
        )
