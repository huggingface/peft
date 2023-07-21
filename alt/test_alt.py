import copy
from functools import partial

import pytest
import torch
from ia3 import IA3Config, LinearIA3Layer
from lora import LoraConfig, LinearLoraLayer
from torch import nn
from wrapper import AdapterWrapper


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(10, 300)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(300, 1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X)
        return X


class EmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 300)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(300, 1)

    def forward(self, X):
        X = self.emb(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X).squeeze(-1)
        return X


TEST_CASES = [
    (MLP, LoraConfig(target_modules="lin0")),
    (MLP, LoraConfig(target_modules="lin1")),
    (MLP, LoraConfig(target_modules=["lin0"])),
    (MLP, LoraConfig(target_modules=["lin0", "lin1"])),
    (MLP, LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])),

    (EmbModel, LoraConfig(target_modules="emb")),
    (EmbModel, LoraConfig(target_modules="lin1")),
    (EmbModel, LoraConfig(target_modules=["emb"])),
    (EmbModel, LoraConfig(target_modules=["emb", "lin1"])),
    (EmbModel, LoraConfig(target_modules=["emb"], modules_to_save=["lin1"])),

    (MLP, IA3Config(target_modules="lin0")),
    (MLP, IA3Config(target_modules="lin1")),
    (MLP, IA3Config(target_modules=["lin0"])),
    (MLP, IA3Config(target_modules=["lin0", "lin1"])),
    (MLP, IA3Config(target_modules=["lin0"], modules_to_save=["lin1"])),
]


def get_prefix_from_config(config):
    # used to identify if a parameter is an adapter parameter
    prefix = {
        LoraConfig: "lora",
        IA3Config: "ia3",
    }[type(config)]
    return prefix


class TestAltLora:
    device = "cpu"

    @pytest.fixture(autouse=True, params=["cpu", "cuda:0"])
    def set_device(self, request):
        if request.param == "cuda:0":
            if not torch.cuda.is_available():
                pytest.skip(reason="device is not CUDA-capble")
        self.device = request.param

    def get_data(self, model_cls):
        if model_cls == MLP:
            X = torch.rand(9, 10)
            y = torch.rand(9, 1)
        elif model_cls == EmbModel:
            X = torch.randint(0, 10, (9, 1))
            y = torch.rand(9, 1)
        X, y = X.to(self.device), y.to(self.device)
        return X, y

    def get_peft_model(self, model_cls, config, **kwargs):
        torch.manual_seed(0)
        model = model_cls()
        peft_model = AdapterWrapper.from_config(model, config, **kwargs)
        peft_model = peft_model.to(self.device).eval()
        return peft_model

    def fit(self, model, X, y, epochs=3):
        torch.manual_seed(0)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # train at least 3 steps for all parameters to be updated (probably this is required because of symmetry
        # breaking of some layers that are initialized with constants)
        for _ in range(3):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = nn.functional.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_applying_lora_does_not_change_output(self, model_cls, config):
        X, _ = self.get_data(model_cls)
        model = model_cls().to(self.device).eval()
        output_base = model(X)

        peft_model = AdapterWrapper.from_config(model, config)
        output_peft = peft_model(X)

        torch.testing.assert_close(output_base, output_peft)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_training_changes_output(self, model_cls, config):
        X, y = self.get_data(model_cls)
        peft_model = self.get_peft_model(model_cls, config)

        output_base = peft_model.model(X)
        output_before = peft_model(X)
        torch.testing.assert_close(output_base, output_before)

        self.fit(peft_model, X, y)
        output_after = peft_model(X)
        assert not torch.allclose(output_before, output_after)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_only_adapter_layers_are_updated(self, model_cls, config):
        X, y = self.get_data(model_cls)
        peft_model = self.get_peft_model(model_cls, config)
        params_before = {k: v.clone().detach() for k, v in peft_model.named_parameters()}
        self.fit(peft_model, X, y)
        params_after = dict(peft_model.named_parameters())

        assert params_before.keys() == params_after.keys()

        prefix = get_prefix_from_config(config)
        for key, param_before in params_before.items():
            param_after = params_after[key]
            is_adapter_param = prefix in key
            is_module_to_save = any(key.startswith(m2s + ".new_module") for m2s in config.modules_to_save)
            if is_adapter_param or is_module_to_save:
                assert not torch.allclose(param_before, param_after)
                assert param_after.requires_grad
            else:
                torch.testing.assert_close(param_before, param_after)
                assert not param_after.requires_grad

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    @pytest.mark.parametrize("merge", [True, False])
    def test_unload(self, model_cls, config, merge):
        X, y = self.get_data(model_cls)
        peft_model = self.get_peft_model(model_cls, config)
        output_before = peft_model(X)
        self.fit(peft_model, X, y)

        if merge:
            peft_model.merge_adapter()

        model = peft_model.unload()
        output_unloaded = model(X)
        torch.testing.assert_close(output_before, output_unloaded)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    @pytest.mark.parametrize("merge", [True, False])
    def test_merge_and_unload(self, model_cls, config, merge):
        X, y = self.get_data(model_cls)
        peft_model = self.get_peft_model(model_cls, config)

        self.fit(peft_model, X, y)
        output_after = peft_model(X)

        if merge:
            peft_model.merge_adapter()
        model = peft_model.merge_and_unload()
        output_unloaded = model(X)
        torch.testing.assert_close(output_after, output_unloaded)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_merge_unmerge(self, model_cls, config):
        X, y = self.get_data(model_cls)
        peft_model = self.get_peft_model(model_cls, config)

        self.fit(peft_model, X, y)
        output_after = peft_model(X)

        peft_model.merge_adapter()
        output_merged = peft_model(X)
        torch.testing.assert_close(output_after, output_merged)

        peft_model.unmerge_adapter()
        output_unmerged = peft_model(X)
        torch.testing.assert_close(output_after, output_unmerged)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_multiple_adapters_only_active_one_is_updated(self, model_cls, config):
        X, y = self.get_data(model_cls)

        config_A = config
        config_B = copy.copy(config)
        peft_model = self.get_peft_model(model_cls, config_B, adapter_name="adapter-B")
        # config B is the active adapter

        peft_model.add_adapter_from_config(config_A, "adapter-A")
        # config A is now the active adapter

        params_A_before = {
            k: v.clone().detach() for k, v in peft_model._adapter_registry["adapter-A"].named_adapter_parameters()
        }
        params_B_before = {
            k: v.clone().detach() for k, v in peft_model._adapter_registry["adapter-B"].named_adapter_parameters()
        }
        self.fit(peft_model, X, y, epochs=3)
        params_A_after = dict(peft_model._adapter_registry["adapter-A"].named_adapter_parameters())
        params_B_after = dict(peft_model._adapter_registry["adapter-B"].named_adapter_parameters())

        prefix = get_prefix_from_config(config)
        # adapter layers for A should have changed
        for key, param_before in params_A_before.items():
            param_after = params_A_after[key]
            is_adapter_param = prefix in key
            is_module_to_save = key.startswith("new_module.")
            if is_adapter_param or is_module_to_save:
                assert param_after.requires_grad
                assert not torch.allclose(param_before, param_after)
            else:
                assert not param_after.requires_grad
                torch.testing.assert_allclose(param_before, param_after)

        # all B layers should be the same
        for key, param_before in params_B_before.items():
            param_after = params_B_after[key]
            torch.testing.assert_close(param_before, param_after)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_multiple_adapters_output(self, model_cls, config):
        X, y = self.get_data(model_cls)

        config_A = config
        config_B = copy.copy(config)
        peft_model = self.get_peft_model(model_cls, config_B, adapter_name="adapter-B")
        # config B is the active adapter

        prefix = get_prefix_from_config(config)
        grad_dict = {name: param.requires_grad for name, param in peft_model.named_parameters()}
        grad_dict_expected = {name: (prefix in name) or ("new_module" in name) for name in grad_dict}
        assert grad_dict == grad_dict_expected

        peft_model.add_adapter_from_config(config_A, "adapter-A")
        # config_A is now the active adapter
        grad_dict = {name: param.requires_grad for name, param in peft_model.named_parameters()}
        assert grad_dict == grad_dict_expected

        output_A_1 = peft_model(X)

        peft_model.set_adapter("adapter-B")
        grad_dict = {name: param.requires_grad for name, param in peft_model.named_parameters()}
        assert grad_dict == grad_dict_expected
        self.fit(peft_model, X, y, epochs=10)  # different number of epochs to ensure different weights
        output_B_1 = peft_model(X)

        # first, ensure that the two adapters produce different outputs
        assert not torch.allclose(output_A_1, output_B_1)

        # going back to A should produce the same A output as before
        peft_model.set_adapter("adapter-A")
        grad_dict = {name: param.requires_grad for name, param in peft_model.named_parameters()}
        assert grad_dict == grad_dict_expected
        output_A_2 = peft_model(X)
        torch.testing.assert_close(output_A_1, output_A_2)

        # going back to B should produce the same B output as before
        peft_model.set_adapter("adapter-B")
        grad_dict = {name: param.requires_grad for name, param in peft_model.named_parameters()}
        assert grad_dict == grad_dict_expected
        output_B_2 = peft_model(X)
        torch.testing.assert_close(output_B_1, output_B_2)

        # delete the currently active adapter => adapter A should be active now
        peft_model.delete_adapter("adapter-B")
        output_A_3 = peft_model(X)
        torch.testing.assert_close(output_A_1, output_A_3)

        # deleting B again raises a KeyError
        with pytest.raises(KeyError):
            peft_model.delete_adapter("adapter-B")
        # trying to delete A, which is the last adapter, raises a ValueError
        with pytest.raises(ValueError):
            peft_model.delete_adapter("adapter-A")

    def test_mixing_adaptations_parameters_works(self):
        X, y = self.get_data(MLP)
        model = MLP().to(self.device).eval()
        output_base = model(X)

        peft_model = AdapterWrapper(model)
        peft_model.add_adapter()

        lin_lora_0 = partial(LinearLoraLayer, r=4)
        peft_model.add_adapter_layer("lin0", lin_lora_0)
        lin_lora_1 = partial(LinearLoraLayer, r=5)
        peft_model.add_adapter_layer("lin1", lin_lora_1)
        output_peft = peft_model(X)

        torch.testing.assert_close(output_base, output_peft)
        self.fit(peft_model, X, y)

        output_after = peft_model(X)
        assert not torch.allclose(output_peft, output_after)

        lora_layers = [layer for layer in peft_model.modules() if isinstance(layer, LinearLoraLayer)]
        assert len(lora_layers) == 2
        assert lora_layers[0].lora_A.out_features == 4
        assert lora_layers[0].lora_B.in_features == 4
        assert lora_layers[1].lora_A.out_features == 5
        assert lora_layers[1].lora_B.in_features == 5

        peft_model.delete_adapter_layer("lin0")
        output_deleted = peft_model(X)
        assert not torch.allclose(output_after, output_deleted)

        lora_layers = [layer for layer in peft_model.modules() if isinstance(layer, LinearLoraLayer)]
        assert len(lora_layers) == 1
        assert lora_layers[0].lora_A.out_features == 5
        assert lora_layers[0].lora_B.in_features == 5

    def test_mixing_adaptations_types_works(self):
        X, y = self.get_data(MLP)
        model = MLP().to(self.device).eval()
        output_base = model(X)

        peft_model = AdapterWrapper(model)
        peft_model.add_adapter()

        lin_lora = partial(LinearLoraLayer, r=4)
        peft_model.add_adapter_layer("lin0", lin_lora)
        lin_ia3 = LinearIA3Layer
        peft_model.add_adapter_layer("lin1", lin_ia3)
        output_peft = peft_model(X)

        torch.testing.assert_close(output_base, output_peft)
        self.fit(peft_model, X, y)

        output_after = peft_model(X)
        assert not torch.allclose(output_peft, output_after)

        lora_layers = [layer for layer in peft_model.modules() if isinstance(layer, LinearLoraLayer)]
        assert len(lora_layers) == 1
        assert lora_layers[0].lora_A.out_features == 4
        assert lora_layers[0].lora_B.in_features == 4

        ia3_layers = [layer for layer in peft_model.modules() if isinstance(layer, LinearIA3Layer)]
        assert len(ia3_layers) == 1
        assert ia3_layers[0].ia3_weight.shape == (1, 300)

    def test_nesting_adaptations_types_raises(self):
        X, y = self.get_data(MLP)
        model = MLP().to(self.device).eval()

        peft_model = AdapterWrapper(model)
        peft_model.add_adapter()

        # trying to apply IA³ to a Lora layer should raise an error
        lin_lora = partial(LinearLoraLayer, r=4)
        peft_model.add_adapter_layer("lin0", lin_lora)
        lin_ia3 = LinearIA3Layer

        with pytest.raises(ValueError):
            peft_model.add_adapter_layer("lin0", lin_ia3)

    @pytest.mark.skip(reason="TODO")
    def test_wrong_module_names_raises(self):
        pass
