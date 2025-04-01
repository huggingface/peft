# Copyright 2023-present the HuggingFace Inc. team.
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

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.other import ModulesToSaveWrapper


class ModelWithModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ModuleDict({"foo": nn.Linear(10, 10)})

    def forward(self):
        return self.module["foo"](torch.rand(1, 10))


class ModelWithModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ModuleList([nn.Linear(10, 10)])

    def forward(self):
        return self.module[0](torch.rand(1, 10))


class ModelWithParameterDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ParameterDict({"foo": nn.Parameter(torch.rand(10, 10))})

    def forward(self):
        return self.module["foo"]


class ModelWithParameterList(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ParameterList([nn.Parameter(torch.rand(10, 10))])

    def forward(self):
        return self.module[0]


@pytest.mark.parametrize(
    "cls", [ModelWithModuleDict, ModelWithModuleList, ModelWithParameterDict, ModelWithParameterList]
)
def test_modules_to_save_targets_module_dict_raises(cls):
    model = cls()
    peft_config = LoraConfig(
        target_modules=["other_layer"],
        modules_to_save=["module"],
    )
    model()  # sanity check that the model would normally work

    msg = "modules_to_save cannot be applied to modules of type"
    with pytest.raises(TypeError, match=msg):
        get_peft_model(model=model, peft_config=peft_config)


def test_modules_to_save_targets_tuner_layer_raises():
    # See e.g. issue 2027
    # Prevent users from (accidentally) targeting the same layer both with a tuner and modules_to_save. Normally, PEFT
    # will not target the same layer with both a tuner and ModulesToSaveWrapper. However, if modules_to_save is
    # automatically inferred, e.g. when using AutoModelForSequenceClassification, the ModulesToSaveWrapper is applied ex
    # post, which can lead to the double wrapping.
    model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Note: target_modules="all-linear" would also work and is closer to the original issue, but let's explicitly target
    # "score" here in case that "all-linear" will be fixed to no longer target the score layer.
    peft_config = LoraConfig(target_modules=["score"], task_type="SEQ_CLS")
    msg = "modules_to_save cannot be applied to modules of type"
    with pytest.raises(TypeError, match=msg):
        get_peft_model(model, peft_config)


def test_get_peft_model_revision_warning(tmp_path):
    base_model_id = "peft-internal-testing/tiny-random-BertModel"
    base_revision = "v2.0.0"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, revision=base_revision).eval()
    lora_config = LoraConfig(revision=base_revision)

    overwrite_revision = "main"
    overwrite_warning = f"peft config has already set base model revision to {base_revision}, overwriting with revision {overwrite_revision}"
    with pytest.warns(UserWarning, match=overwrite_warning):
        _ = get_peft_model(base_model, lora_config, revision=overwrite_revision)


def test_load_multiple_adapters_different_modules_to_save(tmp_path):
    # This tests the error described in #2422 where loading multiple adapters with different modules_to_save
    # attributes fails (due to a regression from #2376).

    model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")

    def peft_config(**kwargs):
        return LoraConfig(target_modules="all-linear", **kwargs)

    original_model = copy.deepcopy(model)

    peft_config_0 = peft_config(modules_to_save=["0.post_attention_layernorm"])
    peft_config_1 = peft_config(modules_to_save=["0.post_attention_layernorm"])
    peft_config_2 = peft_config(modules_to_save=["1.post_attention_layernorm"])

    # Save adapter 0, nothing fancy, should be equal to base model weighs
    peft_model = get_peft_model(copy.deepcopy(original_model), peft_config_0)
    peft_model.save_pretrained(tmp_path / "adapter_0")

    # Save adapter 1, modules to save weights are modified randomly, should be unique to adapter 1
    peft_model = get_peft_model(copy.deepcopy(original_model), peft_config_1)
    peft_model.model.model.layers[0].post_attention_layernorm.weight.data = torch.rand_like(
        peft_model.model.model.layers[0].post_attention_layernorm.weight.data
    )
    adapter_1_saved = peft_model.model.model.layers[0].post_attention_layernorm.weight.data.clone()
    peft_model.save_pretrained(tmp_path / "adapter_1")

    # Save adapter 2, modules to save weights are modified randomly, should be unique to adapter 2
    peft_model = get_peft_model(copy.deepcopy(original_model), peft_config_2)
    peft_model.model.model.layers[1].post_attention_layernorm.weight.data = torch.rand_like(
        peft_model.model.model.layers[1].post_attention_layernorm.weight.data
    )
    adapter_2_saved = peft_model.model.model.layers[1].post_attention_layernorm.weight.data.clone()
    peft_model.save_pretrained(tmp_path / "adapter_2")

    del peft_model

    combined_model = PeftModel.from_pretrained(original_model, tmp_path / "adapter_0", adapter_name="adapter_0")
    combined_model.load_adapter(tmp_path / "adapter_1", adapter_name="adapter_1")
    combined_model.load_adapter(tmp_path / "adapter_2", adapter_name="adapter_2")

    # For adapter 0 we expect every mentioned modules to save layer of this test to be equal to the original model
    # since we didn't modify it for adapter 0 and only adapter 0 is active.
    combined_model.set_adapter("adapter_0")
    assert torch.allclose(
        combined_model.model.model.layers[0].post_attention_layernorm.weight,
        original_model.model.layers[0].post_attention_layernorm.weight,
    )
    assert torch.allclose(
        combined_model.model.model.layers[1].post_attention_layernorm.weight,
        original_model.model.layers[1].post_attention_layernorm.weight,
    )

    # For adapter 1 we expect that the modified module to save 0.post_attention_layernorm is modified, the other
    # module to save layers mentioned above should be untouched.
    combined_model.set_adapter("adapter_1")
    assert torch.allclose(
        combined_model.model.model.layers[0].post_attention_layernorm.weight,
        adapter_1_saved,
    )
    assert torch.allclose(
        combined_model.model.model.layers[1].post_attention_layernorm.weight,
        original_model.model.layers[1].post_attention_layernorm.weight,
    )

    # For adapter 2 we expect its module to save layer (1.post_attention_layernorm) to be modified but the other
    # module to save weights should be kept original.
    combined_model.set_adapter("adapter_2")
    assert torch.allclose(
        combined_model.model.model.layers[0].post_attention_layernorm.weight,
        original_model.model.layers[0].post_attention_layernorm.weight,
    )
    assert torch.allclose(
        combined_model.model.model.layers[1].post_attention_layernorm.weight,
        adapter_2_saved,
    )


class TestModulesToSaveAttributeAccess:
    """Test attribute access on the ModulesToSaveWrapper class.

    When we have modules_to_save, the original module is wrapped. As long as only forward was called on this wrapped
    module, we were good. However, if, for instance, model parameters were directly accessed by another module, this
    would typically fail, as the wrapper does not have this attribute. We had special properties for weight and bias,
    but this is not enough. Therefore, attribute access is now transiently delegated to the active adapter (or original
    module, if the adapter is disabled).

    For one example, see #2099.

    """

    @pytest.fixture
    def mlp(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(1, 2)
                self.lin1 = nn.Linear(3, 4)

        return MLP()

    def test_transient_attribute_access_default_adapter(self, mlp):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = get_peft_model(mlp, config)
        assert model.lin1.weight is model.lin1.modules_to_save["default"].weight
        assert model.lin1.bias is model.lin1.modules_to_save["default"].bias

    def test_transient_attribute_access_non_default_adapter(self, mlp):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = get_peft_model(mlp, config)
        model.add_adapter("other", config)

        # at this point, default is still active
        assert model.lin1.weight is model.lin1.modules_to_save["default"].weight
        assert model.lin1.bias is model.lin1.modules_to_save["default"].bias
        assert model.lin1.weight is not model.lin1.modules_to_save["other"].weight
        assert model.lin1.bias is not model.lin1.modules_to_save["other"].bias

        model.set_adapter("other")
        assert model.lin1.weight is not model.lin1.modules_to_save["default"].weight
        assert model.lin1.bias is not model.lin1.modules_to_save["default"].bias
        assert model.lin1.weight is model.lin1.modules_to_save["other"].weight
        assert model.lin1.bias is model.lin1.modules_to_save["other"].bias

    def test_transient_attribute_access_disabled_adapter(self, mlp):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = get_peft_model(mlp, config)

        # at this point, default is still active
        assert model.lin1.weight is model.lin1.modules_to_save["default"].weight
        assert model.lin1.bias is model.lin1.modules_to_save["default"].bias
        assert model.lin1.weight is not model.lin1.original_module.weight
        assert model.lin1.bias is not model.lin1.original_module.bias

        with model.disable_adapter():
            assert model.lin1.weight is not model.lin1.modules_to_save["default"].weight
            assert model.lin1.bias is not model.lin1.modules_to_save["default"].bias
            assert model.lin1.weight is model.lin1.original_module.weight
            assert model.lin1.bias is model.lin1.original_module.bias

    def test_transient_attribute_access_uninitialized_adapter(self, mlp):
        # ensure that there is no weird infinite recursion when accessing a non-existing attribute on the class itself
        with pytest.raises(AttributeError, match="has no attribute 'original_module'"):
            ModulesToSaveWrapper.original_module

    def test_transient_attribute_access_attr_does_not_exist_on_modules_to_save(self, mlp):
        # ensure that there is no weird infinite recursion when accessing a non-existing attribute on the
        # ModelToSaveWrapper instance
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = get_peft_model(mlp, config)

        with pytest.raises(AttributeError, match="has no attribute 'foo'"):
            model.lin1.foo

    def test_transient_attribute_access_attr_does_not_exist_on_original_module(self, mlp):
        # ensure that there is no weird infinite recursion when accessing a non-existing attribute on the
        # original module of the ModelToSaveWrapper instance
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = get_peft_model(mlp, config)

        with pytest.raises(AttributeError, match="has no attribute 'foo'"):
            with model.disable_adapter():
                model.lin1.foo

    def test_transient_attribute_access_non_existing_adapter(self, mlp):
        # This should normally never happen, as the active adapter should always exist, but it's a failsafe
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = get_peft_model(mlp, config)
        model.base_model.model.lin1._active_adapter = "does-not-exist"
        with pytest.raises(AttributeError, match="has no attribute 'weight'"):
            model.lin1.weight


class TestModulesToSaveNameSubstringBug:
    """Test a bug that could occur with multiple modules to save where one adapter's name is a substring of another
    adapter's name.

    This bug was the result of an error in the logic of modifying the state_dict for modules_to_save in
    set_peft_model_state_dict. The error in the logic was that it was checked if an entry from modules_to_save (a set
    of strings) is a substring of a key of the state_dict. If it was, a new name was assigned to that key in the
    state_dict, which would allow to load the weight later.

    The issue that stems from the substring check occurs if there are multiple modules_to_save, and one of them has a
    name that is a substring of another. So e.g. if one is named "classifier" and the other is named "classifier2",
    there could be a false match.


    This bug was reported in #2289.

    """

    def get_model(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(5, 4)
                # important: "classifier" is a substring of "classifier2", "classifier3", "classifier4"
                self.classifier = nn.Linear(4, 2)
                self.classifier2 = nn.Linear(4, 2)
                self.classifier3 = nn.Linear(4, 2)
                self.classifier4 = nn.Linear(4, 2)

            def forward(self, x):
                x = self.lin(x)
                return self.classifier(x) + self.classifier2(x) + self.classifier3(x) + self.classifier4(x)

        torch.manual_seed(0)
        return MyModule()

    @pytest.fixture
    def path_merged_and_unmerged(self, tmp_path):
        # Create 2 checkpoints:
        # 1. merged: the model after calling merge_and_unload
        # 2. unmerged: the PEFT model saved without calling merge_and_unload
        path = tmp_path / "model.pt"

        lora_config = LoraConfig(
            target_modules=["lin"],
            # important: "classifier" is a substring of "classifier2", "classifier3", "classifier4"
            modules_to_save=["classifier", "classifier2", "classifier3", "classifier4"],
        )
        model = get_peft_model(self.get_model(), lora_config)
        # mock training
        for _ in range(5):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            output = model(torch.randn(10, 5))
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # save the peft model without merging
        path_unmerged = tmp_path / "unmerged"
        model.save_pretrained(path_unmerged)

        # merge the model and save state_dict
        path_merged = tmp_path / "merged"
        merged = model.merge_and_unload()
        state_dict = merged.state_dict()
        torch.save(state_dict, path_merged)

        return path_merged, path_unmerged

    def test_load_merged_and_unmerged_same_weights(self, path_merged_and_unmerged):
        # Note that this test is quasi flaky, it has a 1 in 4 chance of passing even without the bugfix. It passes when
        # "classifier" happens to be the last element of the set model.modules_to_save. The order of the set is random.
        # It is not possible just run this test multiple times to minimize the probability of this happening, because
        # within the same process, the hash order is consistent. With the bug fix, this doesn't matter, as the test will
        # always pass, but if there is a regression, there is a 1 in 4 chance of not catching it. Since the CI runs many
        # tests, it is overall very unlikely that none will catch it though. If you see this test failing in CI, thus be
        # aware that some of the passing tests may just pass owing to randomness.
        path_merged, path_unmerged = path_merged_and_unmerged

        # load the merged model directly
        state_dict = torch.load(path_merged, weights_only=True)
        model = self.get_model()
        model.load_state_dict(state_dict)
        sd_merged = model.state_dict()
        del model

        # load the unmerged model and merge it
        unmerged = PeftModel.from_pretrained(self.get_model(), path_unmerged)
        sd_unmerged = unmerged.merge_and_unload().state_dict()

        assert sd_merged.keys() == sd_unmerged.keys()
        for key in sd_merged.keys():
            param_merged = sd_merged[key]
            param_unmerged = sd_unmerged[key]
            assert torch.allclose(param_merged, param_unmerged)
