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
import transformers
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoModelForCausalLM, Qwen3Config

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from peft.utils import infer_device
from peft.utils.integrations import TpMeshDimRegistry


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
            # The spawned process only runs this check, so the change doesn't reach other tests
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

    @pytest.mark.parametrize(
        "mesh_shape, mesh_dim_names",
        [((2,), None), ((2,), ("fsdp",)), ((2,), ("dp",)), ((1, 2), ("dp_replicate", "dp_shard"))],
        ids=["unnamed", "fsdp", "dp", "hsdp"],
    )
    def test_fsdp2_lora_shapes_injected_after_sharding(self, tmp_path, model_checkpoint, mesh_shape, mesh_dim_names):
        self._spawn(
            tmp_path, self.fsdp2_lora_shapes_injected_after_sharding, model_checkpoint, mesh_shape, mesh_dim_names
        )

    @pytest.mark.parametrize(
        "mesh_dim_name, register_mesh_dim",
        [("tp", False), ("random_name", False), ("random_name", True)],
    )
    @pytest.mark.parametrize("embedding_style", [RowwiseParallel, ColwiseParallel], ids=["vocab", "hidden"])
    def test_pytorch_tp_lora_adapter_shapes(
        self, tmp_path, model_config, mesh_dim_name, register_mesh_dim, embedding_style
    ):
        self._spawn(
            tmp_path,
            self.pytorch_tp_lora_adapter_shapes,
            model_config,
            mesh_dim_name,
            register_mesh_dim,
            embedding_style,
        )

    @pytest.mark.parametrize("embedding_plan", ["embedding_rowwise", "embedding_colwise"])
    @pytest.mark.parametrize("mesh_dim_names", [None, ("tp",)])
    @pytest.mark.skipif(
        packaging.version.parse(transformers.__version__) < packaging.version.parse("5.21.0"),
        reason="Embedding TP styles require transformers >= 5.21",
    )
    @pytest.mark.skipif(
        infer_device() != "cpu", reason="Transformers builds the TP mesh on the accelerator if there is one"
    )
    def test_transformers_tp_loss_and_gradients_match(
        self, tmp_path, model_checkpoint, mesh_dim_names, embedding_plan
    ):
        self._spawn(
            tmp_path,
            self.transformers_tp_lora_loss_and_gradients_match,
            model_checkpoint,
            mesh_dim_names,
            embedding_plan,
        )

    @pytest.mark.parametrize("embedding_plan", ["embedding_rowwise", "embedding_colwise"])
    @pytest.mark.parametrize("init_lora_weights", [True, False, "gaussian"], ids=["default", "nonzero", "gaussian"])
    @pytest.mark.skipif(
        packaging.version.parse(transformers.__version__) < packaging.version.parse("5.21.0"),
        reason="Embedding TP styles require transformers >= 5.21",
    )
    @pytest.mark.skipif(
        infer_device() != "cpu", reason="Transformers builds the TP mesh on the accelerator if there is one"
    )
    def test_transformers_tp_replicated_weights_match(
        self, tmp_path, model_checkpoint, embedding_plan, init_lora_weights
    ):
        self._spawn(
            tmp_path,
            self.transformers_tp_replicated_weights_match,
            model_checkpoint,
            embedding_plan,
            init_lora_weights,
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
