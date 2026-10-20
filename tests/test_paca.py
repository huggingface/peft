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

import pytest
import torch
from torch import nn
from torch.library import opcheck

from peft import PacaConfig, get_peft_model
from peft.tuners.paca.layer import Linear as PacaLinear
from peft.utils import infer_device


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(32, 8)

    def forward(self, x):
        return self.lin1(self.relu(self.lin0(x)))


def _paca_layers(model):
    return [module for module in model.modules() if isinstance(module, PacaLinear)]


class TestPaca:
    torch_device = infer_device()

    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        return MLP().to(self.torch_device)

    def test_gradcheck(self):
        if self.torch_device == "mps":
            pytest.skip("MPS does not support float64")
        torch.manual_seed(0)
        model = MLP().to(self.torch_device).double()
        config = PacaConfig(r=3, target_modules=["lin0"], init_weights=False, random_seed=0)
        model = get_peft_model(model, config)
        layer = model.base_model.model.lin0
        x = torch.randn(4, 16, device=self.torch_device, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(lambda inp: layer(inp), (x,))

    def test_base_weight_is_restored_after_training_step(self, mlp):
        base_weights = {name: module.weight.detach().clone() for name, module in mlp.named_children() if "lin" in name}
        x = torch.randn(8, 16, device=self.torch_device)
        with torch.no_grad():
            base_output = mlp(x)
        config = PacaConfig(r=4, target_modules=["lin0", "lin1"], random_seed=0)
        model = get_peft_model(mlp, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        for _ in range(3):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        for name, weight in base_weights.items():
            assert torch.equal(getattr(model.base_model.model, name).get_base_layer().weight, weight)
        # sanity check: the adapter was actually trained
        assert not torch.allclose(model(x), base_output)

    def test_only_selected_input_channels_are_saved_for_backward(self):
        # The activation memory saving of PaCA: instead of the full input, only the `r` input channels that correspond
        # to the trainable columns are kept for the backward pass.
        in_features, out_features, r = 64, 48, 4
        base_layer = nn.Linear(in_features, out_features).to(self.torch_device)
        base_layer.requires_grad_(False)
        config = PacaConfig(r=r, paca_alpha=r, init_weights=False)
        layer = PacaLinear(base_layer, "default", r, r, config=config)
        x = torch.randn(2, 7, in_features, device=self.torch_device, requires_grad=True)

        saved_shapes = []

        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            layer(x)

        assert (2, 7, r) in saved_shapes
        assert all(shape[-1] != in_features for shape in saved_shapes), saved_shapes

    def _add_overlapping_adapters(self, model):
        # Two adapters on lin0 whose selected columns partially overlap. The indices are set explicitly, as with the
        # random selection it depends on the seed whether there is an overlap at all. The scalings differ (2.0, 0.5).
        model = get_peft_model(
            model, PacaConfig(r=4, target_modules=["lin0"], init_weights=False), adapter_name="adapter0"
        )
        model.add_adapter("adapter1", PacaConfig(r=4, paca_alpha=2, target_modules=["lin0"], init_weights=False))
        layer = model.base_model.model.lin0
        device = layer.paca_delta["adapter0"].device
        layer.paca_indices["adapter0"] = torch.tensor([1, 3, 5, 7], device=device)
        layer.paca_indices["adapter1"] = torch.tensor([3, 5, 9, 11], device=device)
        model.base_model.set_adapter(["adapter0", "adapter1"])
        return model, layer

    @pytest.mark.parametrize("custom_op", [False, True])
    def test_overlapping_adapters_forward_and_gradients_match_reference(self, custom_op):
        if self.torch_device == "mps":
            pytest.skip("MPS does not support float64")
        torch.manual_seed(0)
        _, layer = self._add_overlapping_adapters(MLP().to(self.torch_device).double())
        base_weight = layer.get_base_layer().weight.detach().clone()
        x = torch.randn(3, 5, 16, device=self.torch_device, dtype=torch.double, requires_grad=True)
        grad_output = torch.randn(3, 5, 32, device=self.torch_device, dtype=torch.double)

        if custom_op:
            # the custom op that replaces _PacaLinearFunction under torch.compile, called directly
            names = ["adapter0", "adapter1"]
            out = torch.ops.peft.paca_linear_forward(
                x,
                layer.get_base_layer().weight,
                layer.get_base_layer().bias,
                [layer.paca_indices[name] for name in names],
                [float(layer.scaling[name]) for name in names],
                [layer.paca_delta[name] for name in names],
            )[0]
        else:
            out = layer(x)
        out.backward(grad_output)
        grads = [x.grad.clone(), layer.paca_delta["adapter0"].grad.clone(), layer.paca_delta["adapter1"].grad.clone()]
        assert torch.equal(layer.get_base_layer().weight, base_weight)

        # reference: both updates are added to the overlapping columns
        x.grad = None
        layer.paca_delta["adapter0"].grad = layer.paca_delta["adapter1"].grad = None
        weight = base_weight
        for name in ["adapter0", "adapter1"]:
            weight = weight.index_add(1, layer.paca_indices[name], layer.scaling[name] * layer.paca_delta[name])
        out_ref = nn.functional.linear(x, weight, layer.get_base_layer().bias)
        out_ref.backward(grad_output)

        torch.testing.assert_close(out, out_ref)
        for grad, grad_ref in zip(
            grads, [x.grad, layer.paca_delta["adapter0"].grad, layer.paca_delta["adapter1"].grad]
        ):
            torch.testing.assert_close(grad, grad_ref)

    def test_overlapping_adapters_merge_and_unmerge(self, mlp):
        model, layer = self._add_overlapping_adapters(mlp)
        model.eval()
        base_weight = layer.get_base_layer().weight.detach().clone()
        x = torch.randn(5, 16, device=self.torch_device)
        expected = model(x)

        model.base_model.merge_adapter(["adapter0", "adapter1"])
        torch.testing.assert_close(model(x), expected)
        model.base_model.unmerge_adapter()
        torch.testing.assert_close(layer.get_base_layer().weight, base_weight)
        torch.testing.assert_close(model(x), expected)

        unloaded = model.merge_and_unload(adapter_names=["adapter0", "adapter1"])
        assert not _paca_layers(unloaded)
        torch.testing.assert_close(unloaded(x), expected)

    def test_autocast_bf16(self, mlp):
        config = PacaConfig(r=4, target_modules=["lin0", "lin1"], init_weights=False)
        model = get_peft_model(mlp, config)
        x = torch.randn(5, 16, device=self.torch_device)
        with torch.autocast(torch.device(self.torch_device).type, dtype=torch.bfloat16):
            out = model(x)
        assert out.dtype == torch.bfloat16
        out.float().sum().backward()
        for layer in _paca_layers(model):
            grad = layer.paca_delta["default"].grad
            assert grad is not None
            assert grad.dtype == layer.paca_delta["default"].dtype == torch.float32

    def test_compile_ops_opcheck(self):
        torch.manual_seed(0)
        x = torch.randn(3, 5, 16, device=self.torch_device, requires_grad=True)
        weight = torch.randn(8, 16, device=self.torch_device)
        bias = torch.randn(8, device=self.torch_device)
        delta = torch.randn(8, 4, device=self.torch_device, requires_grad=True)
        idx = torch.tensor([1, 3, 7, 12], device=self.torch_device)
        # an input that does not require grad (e.g. the first adapted layer): the backward op skips the input gradient
        opcheck(torch.ops.peft.paca_linear_forward.default, (x.detach(), weight, bias, [idx], [2.0], [delta]))
        # several adapters whose selected columns overlap
        delta2 = torch.randn(8, 3, device=self.torch_device, requires_grad=True)
        idx2 = torch.tensor([3, 7, 9], device=self.torch_device)
        opcheck(
            torch.ops.peft.paca_linear_forward.default, (x, weight, bias, [idx, idx2], [2.0, 0.5], [delta, delta2])
        )

    def test_compile_ops_several_forwards_before_backward(self):
        # The custom op swaps the columns in and out of the weight, which is saved for the backward pass. This must not
        # bump the autograd version counter of the weight, otherwise a loss summed over two forward passes fails.
        torch.manual_seed(0)
        weight = torch.randn(8, 16, device=self.torch_device)
        delta = torch.randn(8, 4, device=self.torch_device, requires_grad=True)
        idx = torch.tensor([1, 3, 7, 12], device=self.torch_device)
        xs = [torch.randn(3, 16, device=self.torch_device, requires_grad=True) for _ in range(2)]

        outs = [torch.ops.peft.paca_linear_forward(x, weight, None, [idx], [2.0], [delta])[0] for x in xs]
        (outs[0].sum() + outs[1].sum()).backward()
        grad = delta.grad.clone()

        delta.grad = None
        for x in xs:
            torch.ops.peft.paca_linear_forward(x, weight, None, [idx], [2.0], [delta])[0].sum().backward()
        torch.testing.assert_close(grad, delta.grad)

    @pytest.mark.skipif(
        os.environ.get("PEFT_DEBUG_WITH_TORCH_COMPILE") != "1", reason="slow torch.compile tests are opt-in"
    )
    def test_torch_compile_matches_eager(self):
        torch.manual_seed(0)
        base = MLP().to(self.torch_device)
        config = PacaConfig(r=4, target_modules=["lin0", "lin1"], init_weights=False, random_seed=0)
        eager = get_peft_model(copy.deepcopy(base), config)
        compiled = get_peft_model(copy.deepcopy(base), config)
        compiled.load_state_dict(eager.state_dict())
        x = torch.randn(6, 16, device=self.torch_device, requires_grad=True)

        out_eager = eager(x)
        out_eager.sum().backward()
        torch._dynamo.reset()
        out_compiled = torch.compile(compiled)(x)
        out_compiled.sum().backward()

        torch.testing.assert_close(out_compiled, out_eager, atol=1e-5, rtol=1e-5)
        for (name, p_e), (_, p_c) in zip(eager.named_parameters(), compiled.named_parameters()):
            if p_e.requires_grad:
                torch.testing.assert_close(p_c.grad, p_e.grad, atol=1e-5, rtol=1e-5)
            else:
                # the base weights are left unchanged by the compiled forward and backward
                assert torch.equal(p_c, p_e), name
