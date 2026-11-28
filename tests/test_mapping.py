# Copyright 2025-present the HuggingFace Inc. team.
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
import pickle
import warnings

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PeftWarning, TrainableTokensConfig, get_peft_model
from peft.utils.runtime_checks import MAX_CHECKED_BATCHES, RuntimeCheck, ZeroEmbeddingCheck

from .testing_utils import hub_online_once


class TestGetPeftModel:
    RELOAD_WARNING_EXPECTED_MATCH = r"You are trying to modify a model .*"

    @pytest.fixture
    def lora_config_0(self):
        return LoraConfig(target_modules="0")

    @pytest.fixture
    def base_model(self):
        return torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Linear(2, 10))

    def test_get_peft_model_warns_when_reloading_model(self, lora_config_0, base_model):
        get_peft_model(base_model, lora_config_0)

        with pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH):
            get_peft_model(base_model, lora_config_0)

    def test_get_peft_model_proposed_fix_in_warning_helps(self, lora_config_0, base_model, recwarn):
        peft_model = get_peft_model(base_model, lora_config_0)
        peft_model.unload()
        get_peft_model(base_model, lora_config_0)

        warning_checker = pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH)

        for warning in recwarn:
            if warning_checker.matches(warning):
                pytest.fail("Warning raised even though model was unloaded.")

    def test_get_peft_model_repeated_invocation(self, lora_config_0, base_model):
        peft_model = get_peft_model(base_model, lora_config_0)

        # use direct-addressing of the other layer to accommodate for the nested model
        lora_config_1 = LoraConfig(target_modules="base_model.model.1")

        with pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH):
            get_peft_model(peft_model, lora_config_1)


class TestZeroEmbeddingCheck:
    # Runtime check attached by get_peft_model that reports frozen, all-zero input embedding rows of tokens that occur
    # in the training data, see #4130. The battery test in testing_common.py covers all PEFT methods, these tests cover
    # the details of the check.
    MODEL_ID = "peft-internal-testing/tiny-random-LlamaForCausalLM"
    ZERO_IDS = [5, 6]
    PAD_ID = 31999  # the padding_idx of the embedding in this model
    MATCH = "input embedding rows are all zero and frozen"

    @pytest.fixture
    def base_model(self):
        with hub_online_once(self.MODEL_ID):
            model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID)
        with torch.no_grad():
            # the tokens 5 and 6 get the all-zero rows of never-trained special tokens; the padding row is zero too,
            # as it would be in a real checkpoint, but must not be reported
            model.get_input_embeddings().weight[[*self.ZERO_IDS, self.PAD_ID]] = 0
        return model

    @pytest.fixture
    def input_ids(self):
        # contains the zero rows 5 and 6 as well as the pad token
        return torch.tensor([[1, 5, 6, 7, self.PAD_ID, self.PAD_ID]])

    def get_hooks(self, model):
        embedding = model.get_input_embeddings()
        return [hook for hook in embedding._forward_hooks.values() if isinstance(hook, ZeroEmbeddingCheck)]

    def test_off_by_default(self, base_model, input_ids, recwarn):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"])).train()
        assert not self.get_hooks(model)
        model(input_ids=input_ids)
        assert not any(self.MATCH in str(w.message) for w in recwarn)

    def test_is_a_runtime_check(self):
        assert issubclass(ZeroEmbeddingCheck, RuntimeCheck)

    def test_warns(self, base_model, input_ids):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="warn").train()
        with pytest.warns(PeftWarning, match=self.MATCH) as record:
            model(input_ids=input_ids)
        # the pad token also has a zero row but is not reported
        assert "token ids [5, 6]" in str(record[0].message)
        assert "trainable_token_indices" in str(record[0].message)
        # the hook removes itself after the first report
        assert not self.get_hooks(model)

    def test_raise(self, base_model, input_ids):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        with pytest.raises(RuntimeError, match=self.MATCH):
            model(input_ids=input_ids)
        # the hook also removes itself when it raises
        assert not self.get_hooks(model)

    def test_disabled(self, base_model, input_ids, recwarn):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks=False).train()
        assert not self.get_hooks(model)
        model(input_ids=input_ids)
        assert not any(self.MATCH in str(w.message) for w in recwarn)

    def test_invalid_value_raises(self, base_model):
        with pytest.raises(ValueError, match="runtime_checks must be one of"):
            get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="error")

    def test_inactive_in_eval_mode_and_without_grad(self, base_model, input_ids):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise")
        model.eval()
        model(input_ids=input_ids)
        model.train()
        with torch.no_grad():
            model(input_ids=input_ids)
        assert self.get_hooks(model)

    def test_removed_after_max_checked_batches(self, base_model):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        clean_input_ids = torch.tensor([[1, 2, 3]])
        for _ in range(MAX_CHECKED_BATCHES - 1):
            model(input_ids=clean_input_ids)
        assert self.get_hooks(model)
        model(input_ids=clean_input_ids)
        assert not self.get_hooks(model)

    def test_reports_keyword_call(self, base_model, input_ids):
        # Transformers models call the embedding with positional ids, but custom code may use `embedding(input=ids)`.
        # A forward hook only sees keyword arguments when registered with `with_kwargs=True`; without it, such calls
        # would never be checked.
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        with pytest.raises(RuntimeError, match=self.MATCH):
            model.get_input_embeddings()(input=input_ids)

    def test_removed_after_max_batches_with_unsupported_calls(self, base_model):
        # Calls the check cannot inspect still count, so that the hook does not stay attached forever. Calling the
        # hook directly with no input ids and no tensor output simulates such a call: it returns early without
        # checking anything, but still counts as a training batch.
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        embedding = model.get_input_embeddings()
        [hook] = self.get_hooks(model)
        # after one call fewer than the limit, the hook is still attached; the next call removes it
        for _ in range(MAX_CHECKED_BATCHES - 1):
            hook(embedding, (), {}, None)
        assert self.get_hooks(model)
        with pytest.raises(RuntimeError, match="could not inspect the input embedding"):
            hook(embedding, (), {}, None)
        assert not self.get_hooks(model)

    @pytest.mark.parametrize("trainable", [False, True])
    def test_last_batch_is_handled_after_unsupported_calls(self, base_model, trainable):
        # A valid final batch, including a trainable output, prevents an unsupported-signature report.
        config = LoraConfig(target_modules=["q_proj"], modules_to_save=["embed_tokens"] if trainable else None)
        model = get_peft_model(base_model, config, runtime_checks="raise").train()
        embedding = model.get_input_embeddings()
        [hook] = self.get_hooks(model)
        for _ in range(MAX_CHECKED_BATCHES - 1):
            hook(embedding, (), {}, None)
        model(input_ids=torch.tensor([[1, 2, 3]]))
        assert not self.get_hooks(model)

    def test_disabled_adapters_on_bare_embedding(self, base_model):
        # Disabling q_proj adapters must also pause the check on the plain embedding, without using its budget.
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        ids = torch.tensor([[1, 5, 2]])
        [hook] = self.get_hooks(model)
        with model.disable_adapter():
            for _ in range(MAX_CHECKED_BATCHES):
                model(ids)
        assert self.get_hooks(model) == [hook]
        assert hook.num_checked_batches == 0
        with pytest.raises(RuntimeError, match=r"token ids \[5\]"):
            model(ids)

    def test_modules_to_save_does_not_duplicate_check(self, base_model):
        # modules_to_save copies the embedding hooks; only one check may report on the wrapper.
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="warn").train()
        model.add_adapter("b", LoraConfig(target_modules=["q_proj"], modules_to_save=["embed_tokens"]))
        model.set_adapter("default")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model(torch.tensor([[1, 5, 2]]))
        assert len([w for w in record if issubclass(w.category, PeftWarning)]) == 1
        assert not self.get_hooks(model)

    def test_trainable_adapter_does_not_remove_check(self, base_model):
        # A trainable embedding batch counts, but a later frozen adapter must still be checked.
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        model.add_adapter("b", LoraConfig(target_modules=["q_proj"], modules_to_save=["embed_tokens"]))
        model.set_adapter("b")
        model(torch.tensor([[1, 5, 2]]))
        [hook] = self.get_hooks(model)
        assert hook.num_checked_batches == 1
        model.set_adapter("default")
        with pytest.raises(RuntimeError, match=r"token ids \[5\]"):
            model(torch.tensor([[1, 5, 2]]))

    def test_ignores_pad_token_of_text_config(self, base_model, input_ids):
        # Multimodal models store the token ids of the language model in config.text_config. The tokens 5 and 6 have
        # zero rows: 5 is set as the pad token in text_config only and 6 as the image token in the main config. If
        # text_config were not read, token 5 would be reported and the forward call would raise.
        text_config = copy.deepcopy(base_model.config)
        text_config.pad_token_id = 5
        base_model.config.text_config = text_config
        base_model.config.image_token_id = 6
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        model(input_ids=input_ids)

    def test_near_zero_rows_are_reported(self, base_model, input_ids):
        # Some checkpoints store untrained rows as tiny values instead of exact zeros.
        with torch.no_grad():
            base_model.get_input_embeddings().weight[self.ZERO_IDS] = 1e-37
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        with pytest.raises(RuntimeError, match=self.MATCH):
            model(input_ids=input_ids)

    def test_ignores_multimodal_placeholder_tokens(self, base_model, input_ids):
        # placeholder embeddings are replaced by encoder features, so a zero row is expected
        base_model.config.image_token_id = 5
        base_model.config.video_token_id = 6
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        model(input_ids=input_ids)

    def test_reports_with_activation_checkpointing(self, base_model, input_ids):
        # enable_input_require_grads sets requires_grad on the frozen embedding output, which must not count as
        # trainable
        base_model.enable_input_require_grads()
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        with pytest.raises(RuntimeError, match=self.MATCH):
            model(input_ids=input_ids)

    @pytest.mark.parametrize(
        "config",
        [
            LoraConfig(target_modules=["q_proj"], trainable_token_indices=[5]),
            TrainableTokensConfig(token_indices=[5]),
        ],
    )
    def test_reports_tokens_not_covered_by_trainable_tokens(self, base_model, input_ids, config):
        model = get_peft_model(base_model, config, runtime_checks="raise").train()
        with pytest.raises(RuntimeError, match=r"token ids \[6\]"):
            model(input_ids=input_ids)

    @pytest.mark.parametrize(
        "config_kwargs",
        [
            {"trainable_token_indices": [5, 6]},
            {"modules_to_save": ["embed_tokens"]},
            {"target_modules": ["q_proj", "embed_tokens"]},
        ],
    )
    def test_trainable_rows_are_not_reported(self, base_model, input_ids, config_kwargs):
        config = LoraConfig(**{"target_modules": ["q_proj"], **config_kwargs})
        model = get_peft_model(base_model, config, runtime_checks="raise").train()
        model(input_ids=input_ids)

    def test_trainable_tokens_on_a_trainable_embedding_are_not_reported(self, base_model, input_ids):
        # with the base embedding trained as well, every row is trainable, not only the token indices
        config = LoraConfig(target_modules=["q_proj"], trainable_token_indices=[5])
        model = get_peft_model(base_model, config, runtime_checks="raise").train()
        for param in model.get_input_embeddings().parameters():
            param.requires_grad_(True)
        model(input_ids=input_ids)
        assert len(self.get_hooks(model)) == 1

    def test_adapter_added_later_that_wraps_the_embedding(self, base_model, input_ids):
        # the check moves to the LoRA embedding layer instead of observing the frozen base layer inside it
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="raise").train()
        model.add_adapter("other", LoraConfig(target_modules=["q_proj", "embed_tokens"]))
        model.set_adapter("other")
        assert len(self.get_hooks(model)) == 1
        model(input_ids=input_ids)

    def test_inactive_with_disabled_adapters(self, base_model, input_ids):
        # e.g. a reference forward pass during preference optimization, the rows are trainable once the adapters are
        # enabled again
        config = LoraConfig(target_modules=["q_proj"], trainable_token_indices=[5, 6])
        model = get_peft_model(base_model, config, runtime_checks="raise").train()
        with model.disable_adapter():
            model(input_ids=input_ids)
        model(input_ids=input_ids)

    @pytest.mark.parametrize("method", ["unload", "merge_and_unload"])
    def test_unload_removes_hook(self, base_model, method):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="warn")
        assert self.get_hooks(model)
        base_model = getattr(model, method)()
        assert not self.get_hooks(base_model)

    def test_model_is_picklable(self, base_model):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), runtime_checks="warn")
        loaded = pickle.loads(pickle.dumps(model))
        assert len(self.get_hooks(loaded)) == 1
