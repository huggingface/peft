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
from copy import deepcopy

import pytest
import torch

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.config import BdLoraConfig
from peft.tuners.lora.variants import BlockDiagonalLinear


def _fill_deterministic_weight(weight: torch.Tensor) -> None:
    values = torch.arange(weight.numel(), dtype=weight.dtype, device=weight.device).reshape_as(weight)
    weight.data.copy_(values)


def _dense_weight(module: torch.nn.Module) -> torch.Tensor:
    if isinstance(module, BlockDiagonalLinear):
        return module.weight_as_blockdiagonal_matrix()
    return module.weight


def _copy_adapter_weights(dst_layer, src_layer) -> None:
    dst_layer.lora_A["default"].weight.data.copy_(src_layer.lora_A["default"].weight.data)
    dst_layer.lora_B["default"].weight.data.copy_(src_layer.lora_B["default"].weight.data)


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = torch.nn.Linear(10, 20)
        self.relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(20, 2)

    def forward(self, x):
        x = self.lin0(x)
        x = self.relu(x)
        return self.lin1(x)


class TestBdLora:
    @pytest.mark.parametrize(
        "nblocks,in_features,out_features,input_shape",
        [
            (2, 8, 8, (3, 8)),
            (2, 8, 8, (2, 5, 8)),
            (4, 12, 8, (3, 12)),
            (4, 12, 8, (2, 5, 12)),
        ],
    )
    def test_block_diagonal_linear_forward_matches_dense_equivalent(
        self, nblocks, in_features, out_features, input_shape
    ):
        torch.manual_seed(0)

        layer = BlockDiagonalLinear(in_features=in_features, out_features=out_features, nblocks=nblocks)
        _fill_deterministic_weight(layer.weight)

        x = torch.randn(*input_shape)
        # Compare the packed implementation against the explicit dense reconstruction
        dense_weight = layer.weight_as_blockdiagonal_matrix()

        expected = x @ dense_weight.T
        actual = layer(x)

        assert torch.allclose(
            actual,
            expected,
            atol=1e-5,
            rtol=1e-4,
        ), f"BlockDiagonalLinear forward mismatch for input_shape={input_shape}, nblocks={nblocks}"

    @pytest.mark.parametrize(
        "bdlora_config",
        [
            BdLoraConfig(target_modules_bd_a=["lin0"], nblocks=2, match_strict=True),
            BdLoraConfig(target_modules_bd_b=["lin0"], nblocks=2, match_strict=True),
        ],
    )
    def test_bdlora_merge_delta_matches_manual_dense_equivalent(self, bdlora_config):
        torch.manual_seed(0)

        model = TinyMLP()
        base_weight = model.lin0.weight.detach().clone()

        config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["lin0"],
            use_bdlora=bdlora_config,
        )
        peft_model = get_peft_model(deepcopy(model), config).eval()
        lora_layer = peft_model.base_model.model.lin0

        _fill_deterministic_weight(lora_layer.lora_A["default"].weight)
        _fill_deterministic_weight(lora_layer.lora_B["default"].weight)

        # Convert any block-diagonal factor back to its dense view before multiplying
        dense_a = _dense_weight(lora_layer.lora_A["default"])
        dense_b = _dense_weight(lora_layer.lora_B["default"])
        expected_delta = (dense_b @ dense_a) * lora_layer.scaling["default"]

        peft_model.merge_adapter()

        merged_weight = peft_model.base_model.model.lin0.base_layer.weight.data
        assert torch.allclose(
            merged_weight,
            base_weight + expected_delta,
            atol=1e-5,
            rtol=1e-4,
        ), f"Merged weight mismatch for BD-LoRA config={bdlora_config}"

    def test_bdlora_nblocks_one_matches_vanilla_lora(self):
        # With nblocks=1, there is no block split: out_features // 1 = out_features and r // 1 = r
        # So the BD-LoRA packing reduces to the same shapes as vanilla LoRA, and the outputs should match
        torch.manual_seed(0)

        base_model = TinyMLP()
        x = torch.randn(5, 10)

        lora_rank = 4
        lora_alpha = 8
        lora_dropout = 0.0
        target_modules = ["lin0"]

        bd_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_bdlora=BdLoraConfig(target_modules_bd_a=target_modules, nblocks=1, match_strict=True),
        )
        vanilla_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules
        )

        bd_model = get_peft_model(deepcopy(base_model), bd_config).eval()
        vanilla_model = get_peft_model(deepcopy(base_model), vanilla_config).eval()

        # Copy adapter tensors so the only remaining difference is the block packing path
        _copy_adapter_weights(vanilla_model.base_model.model.lin0, bd_model.base_model.model.lin0)

        bd_output = bd_model(x)
        vanilla_output = vanilla_model(x)

        assert torch.allclose(
            bd_output,
            vanilla_output,
            atol=1e-5,
            rtol=1e-4,
        ), "nblocks=1 BD-LoRA forward output should match vanilla LoRA"

        bd_model.merge_adapter()
        vanilla_model.merge_adapter()

        assert torch.allclose(
            bd_model.base_model.model.lin0.base_layer.weight.data,
            vanilla_model.base_model.model.lin0.base_layer.weight.data,
            atol=1e-5,
            rtol=1e-4,
        ), "nblocks=1 merged weights should match vanilla LoRA"

    @pytest.mark.parametrize(
        "bdlora_config,expected_a_shape,expected_b_shape,expected_adapter_params",
        [
            # A-block: only LoRA-A is block-diagonal. With in_features=10, nblocks=2, and r=4,
            # A stores (4, 10 // 2) = (4, 5) parameters, while B stays unchanged as dense at (20, 4)
            # Total trainable adapter params: 4 * 5 + 20 * 4 = 100
            (BdLoraConfig(target_modules_bd_a=["lin0"], nblocks=2, match_strict=True), (4, 5), (20, 4), 100),
            # B-block: only LoRA-B is block-diagonal. The packed parameter stores (out_features, r // nblocks),
            # so B keeps 20 rows but only 4 // 2 = 2 columns per block. That is 2 blocks of shape (10, 2),
            # for 2 * 10 * 2 = 40 B parameters. A stays unchanged as dense at (4, 10), so the total is 40 + 40 = 80
            (BdLoraConfig(target_modules_bd_b=["lin0"], nblocks=2, match_strict=True), (4, 10), (20, 2), 80),
        ],
    )
    def test_bdlora_packed_shapes_and_adapter_param_counts_vs_vanilla(
        self, bdlora_config, expected_a_shape, expected_b_shape, expected_adapter_params
    ):
        torch.manual_seed(0)

        base_model = TinyMLP()

        lora_rank = 4
        lora_alpha = 8
        lora_dropout = 0.0
        target_modules = ["lin0"]

        bd_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_bdlora=bdlora_config,
        )
        vanilla_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules
        )

        bd_model = get_peft_model(deepcopy(base_model), bd_config).eval()
        vanilla_model = get_peft_model(deepcopy(base_model), vanilla_config).eval()

        bd_layer = bd_model.base_model.model.lin0
        vanilla_layer = vanilla_model.base_model.model.lin0

        bd_a = bd_layer.lora_A["default"]
        bd_b = bd_layer.lora_B["default"]
        vanilla_a = vanilla_layer.lora_A["default"]
        vanilla_b = vanilla_layer.lora_B["default"]

        assert tuple(bd_a.weight.shape) == expected_a_shape
        assert tuple(bd_b.weight.shape) == expected_b_shape
        assert tuple(vanilla_a.weight.shape) == (4, 10)
        assert tuple(vanilla_b.weight.shape) == (20, 4)

        vanilla_adapter_params = sum(
            p.numel() for module in (vanilla_a, vanilla_b) for p in module.parameters() if p.requires_grad
        )
        bd_adapter_params = sum(p.numel() for module in (bd_a, bd_b) for p in module.parameters() if p.requires_grad)

        # For vanilla LoRA on lin0: A has shape (r, in)=(4,10) and B has shape (out, r)=(20,4),
        # so trainable adapter params are 4*10 + 20*4 = 120
        assert vanilla_adapter_params == 120

        assert bd_adapter_params == expected_adapter_params

        # BD-LoRA must reduce trainable adapter parameters vs vanilla LoRA
        assert bd_adapter_params < vanilla_adapter_params
