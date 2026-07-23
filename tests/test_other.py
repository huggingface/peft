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
import warnings
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
from accelerate import infer_auto_device_map
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    LlavaForConditionalGeneration,
)

from peft import LoraConfig, PeftModel, VeraConfig, get_peft_model
from peft.import_utils import is_transformers_ge_v5_1_0, is_transformers_ge_v5_6_0
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import (
    ModulesToSaveWrapper,
    _get_module_names_tied_with_embedding,
    _get_no_split_modules,
    detached_copy,
    prepare_model_for_kbit_training,
)

from .testing_utils import hub_online_once


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


class TestModulesToSaveKwargsOnlyForward:
    """Regression test for #3191: modules listed in `modules_to_save` whose parent calls them with keyword arguments
    only (e.g. Gemma's `vision_tower(pixel_values=...)`) used to crash with `TypeError: forward() missing 1 required
    positional argument: 'x'` because the wrapper required a positional first arg. The wrapper now forwards `*args,
    **kwargs` as-is.
    """

    @pytest.fixture
    def model(self):
        class KwargsOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 8)

            def forward(self, *, pixel_values):
                return self.lin(pixel_values)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Linear(8, 8)
                self.vision = KwargsOnly()

            def forward(self, x):
                return self.vision(pixel_values=self.trunk(x))

        return Outer()

    def test_kwargs_only_forward_active_adapter(self, model):
        config = LoraConfig(target_modules=["trunk"], modules_to_save=["vision"])
        peft_model = get_peft_model(model, config)
        # would previously raise TypeError about missing positional 'x'
        out = peft_model(torch.randn(2, 8))
        assert out.shape == (2, 8)

    def test_kwargs_only_forward_disabled_adapter(self, model):
        config = LoraConfig(target_modules=["trunk"], modules_to_save=["vision"])
        peft_model = get_peft_model(model, config)
        with peft_model.disable_adapter():
            out = peft_model(torch.randn(2, 8))
        assert out.shape == (2, 8)

    def test_kwargs_only_forward_multi_adapter(self, model):
        config = LoraConfig(target_modules=["trunk"], modules_to_save=["vision"])
        peft_model = get_peft_model(model, config)
        peft_model.add_adapter("other", config)
        peft_model.set_adapter("other")
        out = peft_model(torch.randn(2, 8))
        assert out.shape == (2, 8)


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


class TestTargetingAuxiliaryTrainingWrapper:
    """AuxiliaryTrainingWrapper such as ModulesToSaveWrapper and TrainableTokensWrapper are
    in general not to be targeted by PEFT methods such as adapters. For example, a ModulesToSaveWrapper's children
    modules should not be targeted by `LoraConfig(target_modules='all-linear')`, among other things.
    """

    @pytest.fixture
    def plain_model_cls(self):
        class PlainModel(nn.Module):
            def __init__(self, i, o):
                super().__init__()
                self.layer1 = nn.Linear(i, o)

            def forward(self, x):
                return self.layer1(x)

        return PlainModel

    @pytest.fixture
    def nested_model_cls(self, plain_model_cls):
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 5)
                self.layer3 = plain_model_cls(5, 10)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                return x

        return NestedModel

    def test_nested_ignores_modules_to_save(self, nested_model_cls, plain_model_cls):
        # Make sure that `target_modules` is not targeting the nested modules of a module marked as module to save.
        model = nested_model_cls()
        config = LoraConfig(
            target_modules=["layer1"],
            modules_to_save=["layer3"],
        )

        peft_model = get_peft_model(model, config)
        assert isinstance(peft_model.model.layer3.modules_to_save.default, plain_model_cls)

    def test_targeting_module_to_save_raises(self, nested_model_cls):
        model = nested_model_cls()
        config = LoraConfig(
            target_modules=["layer1"],
            modules_to_save=["layer1"],
        )
        msg = "No modules were targeted for adaptation. This might be caused by a combination"
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)

    def test_modules_to_save_targets_tuner_layer_raises(self):
        # See e.g. issue 2027 and 2477
        # Prevent users from (accidentally) targeting the same layer both with a tuner and modules_to_save. Normally, PEFT
        # will not target the same layer with both a tuner and ModulesToSaveWrapper. However, if modules_to_save is
        # automatically inferred, e.g. when using AutoModelForSequenceClassification, the ModulesToSaveWrapper is applied ex
        # post, which can lead to the double wrapping.
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

        # Note: target_modules="all-linear" would also work and is closer to the original issue, but let's explicitly target
        # "score" here in case that "all-linear" will be fixed to no longer target the score layer.
        peft_config = LoraConfig(target_modules=["score"], task_type="SEQ_CLS")

        # Since the `score` layer is in `model.modules_to_save` it should be ignored when targeted,
        # therefore the layer should not be adapted.
        msg = "No modules were targeted for adaptation. This might be caused by a combination"
        with pytest.raises(ValueError, match=msg) as e:
            get_peft_model(model, peft_config)

    def test_targeting_trainable_tokens_raises(self):
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

        peft_config = LoraConfig(target_modules=["embed_tokens"], task_type="SEQ_CLS", trainable_token_indices=[0, 1])

        # While this message might not be the most helpful message, at least it is not silently failing
        msg = "trainable_token_indices cannot be applied to modules of type <class 'peft.tuners.lora.layer.Embedding'>"
        with pytest.raises(TypeError, match=msg) as e:
            get_peft_model(model, peft_config)


class TestAdapterTargeting:
    """Make sure that already existing adapters cannot be targeted to avoid conflicts."""

    @pytest.fixture
    def base_model_cls(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(10, 20)
                self.l2 = torch.nn.Conv2d(1, 1, 2)

            def forward(self, x):
                return self.l2(self.l1(x))

        return M

    @pytest.mark.parametrize(
        "config_cls, config_kwargs",
        [
            (LoraConfig, {"target_modules": "l1.*"}),
            (LoraConfig, {"target_modules": "l2.*"}),
            (VeraConfig, {"target_modules": "l1.*"}),
            (VeraConfig, {"target_modules": "(l1|vera_A).*"}),  # also target the shared layer
        ],
    )
    def test_self_targeting_is_ignored(self, base_model_cls, config_cls, config_kwargs):
        base_model = base_model_cls()
        config1 = config_cls(**config_kwargs)
        config2 = config_cls(**config_kwargs)

        adapter1_name = "ADAPTER_1_512858"  # sufficiently unique names to make reliable testing easier
        adapter2_name = "ADAPTER_2_845781"

        peft_model = get_peft_model(base_model, config1, adapter_name=adapter1_name)
        state_dict_keys_1 = peft_model.state_dict().keys()

        peft_model.add_adapter(adapter2_name, config2)
        state_dict_keys_2 = peft_model.state_dict().keys()

        # Ideally there should be no new modules targeted beyond existing ModuleDicts. Therefore the keys
        # of the new state dict should only differ after the adapter name portion of the keys - not before.
        # Expected:
        # - a.b.<adapter_name_1>.xyz
        # - a.b.<adapter_name_2>.xyz
        # We're not expecting this to happen and test against it:
        # - a.b.<adapter_name_1>.xyz
        # - a.<adapter_name_2>.xyz
        def remove_adapter_portion(adapter_name, key):
            if key.endswith(f".{adapter_name}"):
                return key.removesuffix(f".{adapter_name}")
            return key.split(f".{adapter_name}.")[0]

        adapter_invariant_keys1 = {remove_adapter_portion(adapter1_name, key) for key in state_dict_keys_1}
        adapter_invariant_keys2 = {
            remove_adapter_portion(adapter2_name, remove_adapter_portion(adapter1_name, key))
            for key in state_dict_keys_2
        }

        assert adapter_invariant_keys1 == adapter_invariant_keys2


class TestGetNoSplitModules:
    # Ensure that children are considered when determining _no_split_modules
    # see https://github.com/huggingface/transformers/pull/38141

    def test_get_no_split_modules_simple(self):
        # choose a model where recursively visiting children is *not* required
        model_id = "peft-internal-testing/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        assert list(model._no_split_modules) == ["OPTDecoderLayer"]
        no_split_modules = _get_no_split_modules(model)
        assert no_split_modules == {"OPTDecoderLayer"}

    def test_get_no_split_modules_recursive(self):
        # choose a model where recursively visiting children is required
        model_id = "hf-internal-testing/tiny-random-LlavaForConditionalGeneration"
        model = LlavaForConditionalGeneration.from_pretrained(model_id)

        # model._no_split_modules is recursively generated as of transformers 5.1.0 so
        # depending on which transformers version we have in the test environment the
        # attribute will deliver either the same result as `_get_no_split_modules`
        # or an empty list.
        #
        # TODO remove this distinction once transformers <5.1.0 is not supported anymore
        if not is_transformers_ge_v5_1_0:
            # sanity check: just visiting the model itself is not enough:
            assert model._no_split_modules == []
            no_split_modules = _get_no_split_modules(model)
            assert no_split_modules == {"CLIPEncoderLayer", "LlamaDecoderLayer"}
        elif not is_transformers_ge_v5_6_0:
            # TODO remove this once transformers <5.6.0 is not supported anymore
            assert model._no_split_modules == {"CLIPEncoderLayer", "LlamaDecoderLayer"}
            no_split_modules = _get_no_split_modules(model)
            assert no_split_modules == {"CLIPEncoderLayer", "LlamaDecoderLayer"}
        else:
            # in transformers > 5.5.0, the structure of the model was changed, see
            # https://github.com/huggingface/transformers/pull/45361
            # https://github.com/huggingface/transformers/pull/45448
            assert model._no_split_modules == {
                "CLIPEncoderLayer",
                "CLIPTextEmbeddings",
                "CLIPVisionEmbeddings",
                "LlamaDecoderLayer",
            }
            no_split_modules = _get_no_split_modules(model)
            assert no_split_modules == {
                "CLIPEncoderLayer",
                "CLIPTextEmbeddings",
                "CLIPVisionEmbeddings",
                "LlamaDecoderLayer",
            }


class TestGetModuleNamesTiedWithEmbedding:
    # TODO remove mapping when transformers <5 is not supported anymore as it is the default
    # from there on. also remove the 'list' tied weights type
    model_tied_weights_mapping = {
        "peft-internal-testing/tiny-random-BertModel": {
            "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
            "cls.predictions.decoder.bias": "bert.embeddings.word_embeddings.bias",
        },
        "peft-internal-testing/opt-125m": {
            "lm_head.weight": "model.decoder.embed_tokens.weight",
        },
        "peft-internal-testing/tiny-random-t5": {
            "lm_head.weight": "shared.weight",
            "encoder.embed_tokens.weight": "shared.weight",
            "decoder.embed_tokens.weight": "shared.weight",
        },
    }

    model_ids = [
        "peft-internal-testing/opt-125m",
        "peft-internal-testing/tiny-random-BertModel",
        "peft-internal-testing/tiny-random-t5",
    ]

    @contextmanager
    def patch_model(self, model_id, tied_weights_type):
        with hub_online_once(model_id):
            if "t5" in model_id:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id)

            tied_weights_keys = list(self.model_tied_weights_mapping[model_id].keys())
            expected_module_names = sorted({k.rpartition(".")[0] for k in tied_weights_keys})

            if tied_weights_type == "list":
                # for transformers >=5 this tests compatibility with transformers <5
                with patch.object(model, "_tied_weights_keys", list(tied_weights_keys)):
                    yield model, expected_module_names

            elif tied_weights_type == "mapping":
                # for transformers <5 this tests compatibility with transformers >=5
                mapping = self.model_tied_weights_mapping[model_id]

                with patch.object(model, "_tied_weights_keys", mapping):
                    yield model, expected_module_names

            else:
                raise RuntimeError("Invalid fixture request")

    @pytest.mark.parametrize("tied_weights_type", ["list", "mapping"])
    @pytest.mark.parametrize("model_id", model_ids)
    def test_get_modules_tied_to_embedding(self, model_id, tied_weights_type):
        with self.patch_model(model_id, tied_weights_type) as (model, expected):
            if tied_weights_type == "mapping":
                assert isinstance(model._tied_weights_keys, dict)

            # transformers defines the bias as tied even if it doesn't exist, filter out in that case
            if not hasattr(model.get_input_embeddings(), "bias"):
                expected = list(filter(lambda k: "bias" not in k, expected))

            modules = _get_module_names_tied_with_embedding(model)

            assert expected == modules

    @pytest.mark.parametrize("tied_weights_type", ["list", "mapping"])
    @pytest.mark.parametrize("model_id", model_ids)
    def test_get_modules_tied_to_embedding_peft(self, model_id, tied_weights_type):
        with self.patch_model(model_id, tied_weights_type) as (model, expected):
            if tied_weights_type == "mapping":
                assert isinstance(model._tied_weights_keys, dict)

            # transformers defines the bias as tied even if it doesn't exist, filter out in that case
            if not hasattr(model.get_input_embeddings(), "bias"):
                expected = list(filter(lambda k: "bias" not in k, expected))

            peft_model = get_peft_model(model, LoraConfig())

            modules = peft_model._get_module_names_tied_with_embedding()

            assert expected == modules

    @pytest.mark.parametrize("tied_weights_type", ["list", "mapping"])
    @pytest.mark.parametrize("model_id", model_ids)
    def test_get_modules_tied_returns_empty_when_tying_disabled(self, model_id, tied_weights_type):
        # When tie_word_embeddings=False, no tied modules should be reported even if _tied_weights_keys exists
        # Linked to origin issue #2944
        with self.patch_model(model_id, tied_weights_type) as (model, _):
            # Model has _tied_weights_keys (architectural capability) but tying is disabled
            model.config.tie_word_embeddings = False

            modules = _get_module_names_tied_with_embedding(model)
            assert modules == []


class TestPrepareModelForKbitTraining:
    """CPU tests for prepare_model_for_kbit_training.

    GPU tests for this function (issue #3265 memory fix) live in test_common_gpu.py.
    """

    @pytest.fixture
    def fp16_model(self):
        model = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 32),
        ).to(torch.float16)
        model.is_loaded_in_8bit = True
        return model

    def test_fp32_cast(self, fp16_model):
        # all non-Params4bit fp16/bf16 params become fp32 after the call
        for param in fp16_model.parameters():
            assert param.dtype == torch.float16

        prepare_model_for_kbit_training(fp16_model, use_gradient_checkpointing=False)

        for param in fp16_model.parameters():
            if param.__class__.__name__ != "Params4bit":
                assert param.dtype == torch.float32

    def test_auto_clear_cache_default(self, fp16_model):
        # auto_clear_cache=True (default): empty_cache() is called after the fp32 casts
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
        ):
            prepare_model_for_kbit_training(fp16_model, use_gradient_checkpointing=False)
        mock_empty_cache.assert_called_once()

    def test_auto_clear_cache_disabled(self, fp16_model):
        # auto_clear_cache=False: empty_cache() is never called
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
        ):
            prepare_model_for_kbit_training(fp16_model, use_gradient_checkpointing=False, auto_clear_cache=False)
        mock_empty_cache.assert_not_called()


class TestDetachedCopy:
    """Tests for the detached_copy utility"""

    def get_mlp(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.lin1 = nn.Linear(20, 2)

            def forward(self, x):
                return self.lin1(self.relu(self.lin0(x)))

        torch.manual_seed(0)
        return MLP().eval()

    def test_get_peft_model_without_detached_copy_modifies_base_model(self):
        # sanity check: without a detached copy, the model is modified in-place
        model = self.get_mlp()
        peft_model = get_peft_model(model, LoraConfig(target_modules=["lin0"]))
        assert peft_model.base_model.model is model
        assert isinstance(model.lin0, BaseTunerLayer)

    def test_detached_copy_modules_are_new_but_params_are_shared(self):
        model = self.get_mlp()
        model_copy = detached_copy(model)

        assert model_copy is not model
        modules = dict(model.named_modules())
        modules_copy = dict(model_copy.named_modules())
        assert modules.keys() == modules_copy.keys()
        for name in modules:
            assert modules[name] is not modules_copy[name]

        params = dict(model.named_parameters())
        params_copy = dict(model_copy.named_parameters())
        assert params.keys() == params_copy.keys()
        for name in params:
            assert params[name] is params_copy[name]

        x = torch.randn(5, 10)
        assert torch.allclose(model(x), model_copy(x))

    def test_detached_copy_buffers_are_shared_by_default(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.BatchNorm1d(20))
        model_copy = detached_copy(model)
        buffers = dict(model.named_buffers())
        buffers_copy = dict(model_copy.named_buffers())
        assert buffers.keys() == buffers_copy.keys()
        for name in buffers:
            assert buffers[name] is buffers_copy[name]

    def test_detached_copy_share_buffers_false(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.BatchNorm1d(20))
        model_copy = detached_copy(model, share_buffers=False)

        # the buffers are independent copies with the same values
        buffers = dict(model.named_buffers())
        buffers_copy = dict(model_copy.named_buffers())
        assert buffers.keys() == buffers_copy.keys()
        for name in buffers:
            assert buffers[name] is not buffers_copy[name]
            assert torch.equal(buffers[name], buffers_copy[name])

        # the parameters are still shared
        for name, param in model.named_parameters():
            assert dict(model_copy.named_parameters())[name] is param

        # updating the running statistics of the copy during training does not affect the original model
        model_copy.train()
        model_copy(torch.randn(5, 10))
        assert not torch.allclose(model_copy[1].running_mean, model[1].running_mean)

    def test_detached_copy_preserves_weight_tying(self):
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 5)
                self.head = nn.Linear(5, 10, bias=False)
                self.head.weight = self.emb.weight

        model = TiedModel()
        model_copy = detached_copy(model)
        # tying is preserved within the copy and the tensors are shared with the original model
        assert model_copy.head.weight is model_copy.emb.weight
        assert model_copy.head.weight is model.emb.weight

    def test_get_peft_model_detached_leaves_base_model_unmodified(self):
        model = self.get_mlp()
        x = torch.randn(5, 10)
        output_before = model(x)

        peft_model = get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))

        assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
        assert any(isinstance(module, BaseTunerLayer) for module in peft_model.modules())
        assert torch.allclose(model(x), output_before)

    def test_get_peft_model_detached_base_weights_are_shared(self):
        model = self.get_mlp()
        peft_model = get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))
        assert peft_model.base_model.model.lin0.base_layer.weight is model.lin0.weight
        # non-targeted modules also share their weights
        assert peft_model.base_model.model.lin1.weight is model.lin1.weight

    def test_get_peft_model_detached_freezes_shared_base_weights(self):
        # Since requires_grad is an attribute of the shared tensors, freezing the base weights of the PEFT model also
        # affects the original model. This test is not really about enforcing this behavior but about checking the
        # status quo.
        model = self.get_mlp()
        get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))
        assert not model.lin0.weight.requires_grad
        assert not model.lin1.weight.requires_grad

    def test_two_detached_peft_models_on_same_base_model(self):
        model = self.get_mlp()
        x = torch.randn(5, 10)
        output_base = model(x)

        peft_model_1 = get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            peft_model_2 = get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))
        # the base model is unmodified, so there should be no warning about applying PEFT a second time
        assert not any("for a second time" in str(warning.message) for warning in recorded)

        # LoRA B is initialized to zero, so initially all outputs are identical
        assert torch.allclose(peft_model_1(x), output_base)
        assert torch.allclose(peft_model_2(x), output_base)

        # modifying the adapter of one PEFT model affects neither the other PEFT model nor the base model
        with torch.no_grad():
            peft_model_2.base_model.model.lin0.lora_B["default"].weight.fill_(0.5)
        assert not torch.allclose(peft_model_2(x), output_base)
        assert torch.allclose(peft_model_1(x), output_base)
        assert torch.allclose(model(x), output_base)

    def test_get_peft_model_detached_modules_to_save_are_not_shared(self):
        model = self.get_mlp()
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(detached_copy(model), config)

        wrapper = peft_model.base_model.model.lin1
        assert wrapper.original_module.weight is model.lin1.weight
        # the trainable copy is independent of the base model
        assert wrapper.modules_to_save["default"].weight is not model.lin1.weight
        with torch.no_grad():
            wrapper.modules_to_save["default"].weight.zero_()
        assert not (model.lin1.weight == 0).all()

    def test_get_peft_model_detached_save_load_roundtrip(self, tmp_path):
        model = self.get_mlp()
        x = torch.randn(5, 10)

        peft_model = get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))
        with torch.no_grad():
            peft_model.base_model.model.lin0.lora_B["default"].weight.fill_(0.5)
        output = peft_model(x)
        peft_model.save_pretrained(tmp_path)

        # since the base model was left unmodified, it can be used directly to load the adapter
        loaded = PeftModel.from_pretrained(model, tmp_path)
        assert torch.allclose(loaded(x), output)

    def test_from_pretrained_detached_leaves_base_model_unmodified(self, tmp_path):
        model = self.get_mlp()
        x = torch.randn(5, 10)
        output_base = model(x)

        peft_model = get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))
        with torch.no_grad():
            peft_model.base_model.model.lin0.lora_B["default"].weight.fill_(0.5)
        output_peft = peft_model(x)
        peft_model.save_pretrained(tmp_path)

        loaded = PeftModel.from_pretrained(detached_copy(model), tmp_path)
        assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
        assert torch.allclose(model(x), output_base)
        assert torch.allclose(loaded(x), output_peft)
        # the base weights are shared with the original model
        assert loaded.base_model.model.lin0.base_layer.weight is model.lin0.weight

    def test_get_peft_model_without_detached_copy_changes_base_model_behavior(self):
        model = self.get_mlp()
        config = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        x = torch.randn(5, 10)
        output_before = model(x)
        get_peft_model(model, config)
        assert not torch.allclose(model(x), output_before)

    def test_get_peft_model_with_detached_copy_keeps_base_model_behavior(self):
        model = self.get_mlp()
        config = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        x = torch.randn(5, 10)
        output_before = model(x)
        peft_model = get_peft_model(detached_copy(model), config)
        assert torch.allclose(model(x), output_before)
        # sanity check: the PEFT model itself does produce different outputs
        assert not torch.allclose(peft_model(x), output_before)

    def test_from_pretrained_with_and_without_detached_copy(self, tmp_path):
        model = self.get_mlp()
        config = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        x = torch.randn(5, 10)
        output_before = model(x)
        get_peft_model(detached_copy(model), config).save_pretrained(tmp_path)

        # loading the adapter onto a detached copy does not affect the base model
        peft_model = PeftModel.from_pretrained(detached_copy(model), tmp_path)
        assert torch.allclose(model(x), output_before)
        assert not torch.allclose(peft_model(x), output_before)

        # loading it onto the base model directly changes its behavior
        PeftModel.from_pretrained(model, tmp_path)
        assert not torch.allclose(model(x), output_before)

    def test_cast_dtype_propagates_to_original_model(self):
        # Casting dtype or device works by assigning to param.data, i.e. it mutates the shared tensors. Therefore,
        # casting the detached PEFT model propagates to the original model.
        model = self.get_mlp()
        peft_model = get_peft_model(detached_copy(model), LoraConfig(target_modules=["lin0"]))
        peft_model.to(torch.float16)
        assert model.lin0.weight.dtype == torch.float16
        assert model.lin1.weight.dtype == torch.float16

    def test_two_compiled_detached_models_on_same_base_model(self):
        torch.manual_seed(0)
        model = self.get_mlp().eval()
        x = torch.randn(5, 10)
        output_base = model(x)

        config_kwargs = {"target_modules": ["lin0"], "init_lora_weights": False}
        peft_model_1 = get_peft_model(detached_copy(model), LoraConfig(**config_kwargs))
        peft_model_2 = get_peft_model(detached_copy(model), LoraConfig(**config_kwargs))

        compiled_1 = torch.compile(peft_model_1)
        compiled_2 = torch.compile(peft_model_2)
        output_1 = compiled_1(x)
        output_2 = compiled_2(x)

        # compiled outputs correspond to the eager outputs of the respective model
        assert torch.allclose(output_1, peft_model_1(x), atol=1e-6, rtol=1e-5)
        assert torch.allclose(output_2, peft_model_2(x), atol=1e-6, rtol=1e-5)
        # the two models have different (randomly initialized) adapters, so their outputs differ
        assert not torch.allclose(output_1, output_2)
        # the base model is unaffected
        assert torch.allclose(model(x), output_base)

    def test_quant_state_attributes_are_shared(self):
        # bitsandbytes stores the quantization state as an attribute of the parameter and additionally as a module
        # attribute referencing the same object; ensure that neither is duplicated by detached_copy
        class FakeQuantState:
            def __init__(self):
                self.absmax = torch.randn(10)

        model = self.get_mlp()
        quant_state = FakeQuantState()
        model.lin0.weight.quant_state = quant_state
        model.lin0.quant_state = quant_state
        model.lin1.weight.SCB = torch.randn(2)

        model_copy = detached_copy(model)
        assert model_copy.lin0.weight.quant_state is quant_state
        assert model_copy.lin0.quant_state is quant_state
        assert model_copy.lin1.weight.SCB is model.lin1.weight.SCB

    def test_from_pretrained_detached_with_cpu_and_disk_offload(self, tmp_path):
        # mirrors test_offload_load from test_gpu_examples.py: load a LoRA adapter onto a detached copy of a model
        # with CPU- and disk-offloaded modules; this exercises copying a model whose modules carry accelerate hooks
        # and whose disk-offloaded parameters are on the meta device
        torch.manual_seed(0)
        model_id = "gpt2"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            input_ids = torch.arange(10).view(1, -1)

            config = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False, target_modules=["c_attn"])
            peft_model = get_peft_model(detached_copy(model), config)
            expected = peft_model(input_ids).logits
            peft_model.save_pretrained(tmp_path / "adapter")
            del peft_model

            memory_limits = {"cpu": "0.4GIB"}  # no "disk" for PeftModel.from_pretrained() compatibility
            device_map = infer_auto_device_map(model, max_memory=memory_limits)
            assert "disk" in device_map.values()
            offloaded_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=device_map, offload_folder=str(tmp_path / "offload")
            )
            loaded = PeftModel.from_pretrained(
                detached_copy(offloaded_model),
                tmp_path / "adapter",
                max_memory=memory_limits,
                offload_folder=str(tmp_path / "offload"),
            ).eval()

            logits = loaded(input_ids).logits
            assert torch.allclose(logits, expected, atol=1e-5, rtol=1e-5)
            # the offloaded base model was left unmodified and remains functional
            assert not any(isinstance(module, BaseTunerLayer) for module in offloaded_model.modules())
            logits_base = offloaded_model(input_ids).logits
            assert torch.isfinite(logits_base).all()
