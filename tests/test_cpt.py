# Copyright 2024-present the HuggingFace Inc. team.
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
import math
from typing import Any, Union

import pytest
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import CPTConfig, TaskType, get_peft_model


TEMPLATE = {"input": "input: {}", "intra_separator": " ", "output": "output: {}", "inter_separator": "\n"}

MODEL_NAME = "peft-internal-testing/tiny-random-OPTForCausalLM"
MAX_INPUT_LENGTH = 1024


@pytest.fixture(scope="module")
def global_tokenizer():
    """Load the tokenizer fixture for the model."""

    return AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")


@pytest.fixture(scope="module")
def config_text():
    """Load the SST2 dataset and prepare it for testing."""
    config = CPTConfig(
        cpt_token_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # Example token IDs for testing
        cpt_mask=[1, 1, 1, 1, 1, 1, 1, 1],
        cpt_tokens_type_mask=[1, 2, 2, 2, 3, 3, 3, 4],
        opt_weighted_loss_type="decay",
        opt_loss_decay_factor=0.95,
        opt_projection_epsilon=0.2,
        opt_projection_format_epsilon=0.1,
        tokenizer_name_or_path=MODEL_NAME,
        task_type=TaskType.CAUSAL_LM,
    )
    return config


@pytest.fixture(scope="module")
def config_random():
    """Load the SST2 dataset and prepare it for testing."""
    config = CPTConfig(
        opt_weighted_loss_type="decay",
        opt_loss_decay_factor=0.95,
        opt_projection_epsilon=0.2,
        opt_projection_format_epsilon=0.1,
        tokenizer_name_or_path=MODEL_NAME,
        task_type=TaskType.CAUSAL_LM,
    )
    return config


@pytest.fixture(scope="module")
def sst_data():
    """Load the SST2 dataset and prepare it for testing."""
    data = load_dataset("nyu-mll/glue", "sst2")

    def add_string_labels(example):
        if example["label"] == 0:
            example["label_text"] = "negative"
        elif example["label"] == 1:
            example["label_text"] = "positive"
        return example

    train_dataset = data["train"].select(range(4)).map(add_string_labels)
    test_dataset = data["validation"].select(range(10)).map(add_string_labels)

    return {"train": train_dataset, "test": test_dataset}


@pytest.fixture(scope="module")
def collator(global_tokenizer):
    class CPTDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
        def __init__(self, tokenizer, training=True, mlm=False):
            super().__init__(tokenizer, mlm=mlm)
            self.training = training
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # mk check why needed

        def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
            # Handle dict or lists with proper padding and conversion to tensor.
            list_sample_mask = []
            for i in range(len(examples)):
                if "sample_mask" in examples[i].keys():
                    list_sample_mask.append(examples[i].pop("sample_mask"))

            max_len = max(len(ex["input_ids"]) for ex in examples)

            def pad_sequence(sequence, max_len, pad_value=0):
                return sequence + [pad_value] * (max_len - len(sequence))

            input_ids = torch.tensor([pad_sequence(ex["input_ids"], max_len) for ex in examples])
            attention_mask = torch.tensor([pad_sequence(ex["attention_mask"], max_len) for ex in examples])
            input_type_mask = torch.tensor([pad_sequence(ex["input_type_mask"], max_len) for ex in examples])

            batch = {"input_ids": input_ids, "attention_mask": attention_mask, "input_type_mask": input_type_mask}

            tensor_sample_mask = batch["input_ids"].clone().long()
            tensor_sample_mask[:, :] = 0
            for i in range(len(list_sample_mask)):
                tensor_sample_mask[i, : len(list_sample_mask[i])] = list_sample_mask[i]

            batch["labels"] = batch["input_ids"].clone()
            if not self.training:
                batch["sample_mask"] = tensor_sample_mask

            return batch

    collator = CPTDataCollatorForLanguageModeling(global_tokenizer, training=True, mlm=False)
    return collator


def dataset(data, tokenizer):
    class CPTDataset(Dataset):
        def __init__(self, samples, tokenizer, template, max_length=MAX_INPUT_LENGTH):
            self.template = template
            self.tokenizer = tokenizer
            self.max_length = max_length

            self.attention_mask = []
            self.input_ids = []
            self.input_type_mask = []
            self.inter_separator_ids = self._get_input_ids(template["inter_separator"])

            for sample_i in tqdm(samples):
                input_text, label = sample_i["sentence"], sample_i["label_text"]
                input_ids, attention_mask, input_type_mask = self.preprocess_sentence(input_text, label)

                self.input_ids.append(input_ids)
                self.attention_mask.append(attention_mask)
                self.input_type_mask.append(input_type_mask)

        def _get_input_ids(self, text):
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]

        def preprocess_sentence(self, input_text, label):
            input_template_part_1_text, input_template_part_2_text = self.template["input"].split("{}")
            input_template_tokenized_part1 = self._get_input_ids(input_template_part_1_text)
            input_tokenized = self._get_input_ids(input_text)
            input_template_tokenized_part2 = self._get_input_ids(input_template_part_2_text)

            sep_tokenized = self._get_input_ids(self.template["intra_separator"])

            label_template_part_1, label_template_part_2 = self.template["output"].split("{}")
            label_template_part1_tokenized = self._get_input_ids(label_template_part_1)
            label_tokenized = self._get_input_ids(label)
            label_template_part2_tokenized = self._get_input_ids(label_template_part_2)

            eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            input_ids = (
                input_template_tokenized_part1
                + input_tokenized
                + input_template_tokenized_part2
                + sep_tokenized
                + label_template_part1_tokenized
                + label_tokenized
                + label_template_part2_tokenized
                + eos
            )

            # determine label tokens, to calculate loss only over them when labels_loss == True
            attention_mask = [1] * len(input_ids)
            input_type_mask = (
                [1] * len(input_template_tokenized_part1)
                + [2] * len(input_tokenized)
                + [1] * len(input_template_tokenized_part2)
                + [0] * len(sep_tokenized)
                + [3] * len(label_template_part1_tokenized)
                + [4] * len(label_tokenized)
                + [3] * len(label_template_part2_tokenized)
                + [0] * len(eos)
            )

            assert len(input_type_mask) == len(input_ids) == len(attention_mask)

            return input_ids, attention_mask, input_type_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "input_type_mask": self.input_type_mask[idx],
            }

    dataset = CPTDataset(data, tokenizer, TEMPLATE)

    return dataset


def test_model_initialization_text(global_tokenizer, config_text):
    """Test model loading and PEFT model initialization."""
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = get_peft_model(base_model, config_text)
    assert model is not None, "PEFT model initialization failed"


def test_model_initialization_random(global_tokenizer, config_random):
    """Test model loading and PEFT model initialization."""
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = get_peft_model(base_model, config_random)
    assert model is not None, "PEFT model initialization failed"


def test_model_initialization_wrong_task_type_raises():
    msg = "CPTConfig only supports task_type = CAUSAL_LM."
    with pytest.raises(ValueError, match=msg):
        CPTConfig(task_type=TaskType.SEQ_CLS)

    msg = "CPTConfig only supports task_type = CAUSAL_LM."
    with pytest.raises(ValueError, match=msg):
        CPTConfig()


def test_model_training_random(sst_data, global_tokenizer, collator, config_random):
    """Perform a short training run to verify the model and data integration."""

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, config_random)
    emb = model.prompt_encoder.default.embedding.weight.data.clone().detach()
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        remove_unused_columns=False,
        save_strategy="no",
        logging_steps=1,
    )

    train_dataset = dataset(sst_data["train"], global_tokenizer)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator)

    trainer.train()
    # Verify that the embedding tensor remains unchanged (frozen)
    assert torch.all(model.prompt_encoder.default.embedding.weight.data.clone().detach().cpu() == emb.cpu())

    delta_emb = model.prompt_encoder.default.get_projection().clone().detach()
    norm_delta = delta_emb.norm(dim=1).cpu()
    epsilon = model.prompt_encoder.default.get_epsilon().cpu()
    # Verify that the change in tokens is constrained to epsilon
    assert torch.all(norm_delta <= epsilon)


def test_model_batch_training_text(sst_data, global_tokenizer, collator, config_text):
    """Perform a short training run to verify the model and data integration."""

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, config_text)
    emb = model.prompt_encoder.default.embedding.weight.data.clone().detach()

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        remove_unused_columns=False,
        save_strategy="no",
        logging_steps=1,
    )

    train_dataset = dataset(sst_data["train"], global_tokenizer)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator)

    trainer.train()
    # Verify that the embedding tensor remains unchanged (frozen)
    assert torch.all(model.prompt_encoder.default.embedding.weight.data.clone().detach().cpu() == emb.cpu())

    cpt_tokens_type_mask = torch.Tensor(config_text.cpt_tokens_type_mask).long()
    non_label_idx = (cpt_tokens_type_mask == 1) | (cpt_tokens_type_mask == 2) | (cpt_tokens_type_mask == 3)

    delta_emb = model.prompt_encoder.default.get_projection().clone().detach()
    norm_delta = delta_emb.norm(dim=1).cpu()
    epsilon = model.prompt_encoder.default.get_epsilon().cpu()
    # Verify that the change in tokens is constrained to epsilon
    assert torch.all(norm_delta <= epsilon)
    # Ensure that label tokens remain unchanged
    assert torch.all((norm_delta == 0) == (~non_label_idx))


@pytest.mark.parametrize("adapter_name", ["default", "experiment"])
@pytest.mark.parametrize("loss_type", ["none", "decay"])
def test_cpt_forward_loss_matches_reference(adapter_name, loss_type):
    """
    Ensure that the loss calculated by CPT returns the expected value, as none of the general tests would cover
    accidental changes in the loss calculation.

    In particular, take care of this: the virtual tokens are prefixed to the labels (using the configured
    cpt_token_ids), then every position whose token type is not a positive multiple of 4 is ignored, and the loss is
    the shifted cross entropy over the remaining positions, optionally weighted by label-group decay.
    """
    config = CPTConfig(
        cpt_token_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        cpt_mask=[1, 1, 1, 1, 1, 1, 1, 1],
        cpt_tokens_type_mask=[1, 2, 2, 2, 3, 3, 4, 4],
        opt_weighted_loss_type=loss_type,
        opt_loss_decay_factor=0.25,
        tokenizer_name_or_path=MODEL_NAME,
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, config, adapter_name=adapter_name)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, base_model.config.vocab_size, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    input_type_mask = torch.full_like(input_ids, 4)

    with torch.no_grad():
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, input_type_mask=input_type_mask
        )
        # when input_type_mask is not passed, it defaults to all 4s, i.e. all input tokens are treated as labels
        output_default_mask = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # without labels, no loss is computed
        output_no_labels = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output_no_labels.loss is None
    assert torch.isfinite(output.loss)
    assert torch.allclose(output.loss, output_default_mask.loss, atol=1e-6, rtol=1e-6)

    # reference computation
    prefix_labels = torch.tensor(config.cpt_token_ids).view(1, -1).repeat(2, 1)
    cpt_labels = torch.cat([prefix_labels, labels], dim=1)
    prefix_type_mask = torch.tensor(config.cpt_tokens_type_mask).view(1, -1).repeat(2, 1)
    # the input token types are shifted to avoid clashing with the prefix token types
    shifted_input_type_mask = input_type_mask + prefix_type_mask.max()
    cpt_type_mask = torch.cat([prefix_type_mask, shifted_input_type_mask], dim=1)
    keep = (cpt_type_mask > 0) & (cpt_type_mask % 4 == 0)
    cpt_labels[~keep] = -100

    vocab_size = output.logits.shape[-1]
    # Keep one loss per shifted label so decay weights can be applied before averaging. Restore the label tensor's
    # shape after flattening for cross_entropy, so the token-type mask selects the corresponding positions.
    token_losses = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, vocab_size).float(),
        cpt_labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(cpt_labels[:, 1:])
    weights = torch.ones_like(token_losses)
    if loss_type == "decay":
        # Prefix labels (type 4) precede the input label group (type 8), so only the prefix loss is decayed.
        weights[cpt_type_mask[:, 1:] == 4] = config.opt_loss_decay_factor
    expected_loss = (token_losses * weights)[keep[:, 1:]].mean()
    assert torch.allclose(output.loss, expected_loss, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "model_class, model_config",
    [
        pytest.param(
            GPT2LMHeadModel,
            GPT2Config(
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=16,
                n_positions=32,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            ),
            id="gpt2",
        ),
        pytest.param(
            LlamaForCausalLM,
            LlamaConfig(
                vocab_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                hidden_size=16,
                intermediate_size=32,
            ),
            id="llama",
        ),
    ],
)
@pytest.mark.parametrize("prefix_types", [None, [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize("explicit_input_mask", [False, True])
@pytest.mark.parametrize("weighted_loss", ["none", "decay"])
def test_cpt_partial_context_keeps_input_labels(
    model_class, model_config, prefix_types, explicit_input_mask, weighted_loss
):
    # CPT uses four token types per demonstration; only positive multiples of four contribute to the loss.
    # Input types must be shifted past the context types without changing their roles. With an incomplete context,
    # an offset not divisible by four masks input answers (#3899), yielding NaN when no context answer remains.
    # Fixed logits let us write the expected answer losses explicitly for incomplete and complete contexts.
    torch.manual_seed(0)
    config_kwargs = (
        {}
        if prefix_types is None
        else {"cpt_token_ids": list(range(len(prefix_types))), "cpt_tokens_type_mask": prefix_types}
    )
    config = CPTConfig(
        task_type=TaskType.CAUSAL_LM,
        opt_weighted_loss_type=weighted_loss,
        opt_loss_decay_factor=0.95,
        **config_kwargs,
    )
    model = get_peft_model(model_class(model_config), config).eval()
    input_ids = torch.tensor([[5, 6, 7, 8], [8, 7, 6, 5]])
    input_types = torch.tensor([[1, 2, 4, 4], [0, 4, 2, 4]]) if explicit_input_mask else torch.full_like(input_ids, 4)
    kwargs = {"input_type_mask": input_types.clone()} if explicit_input_mask else {}

    # Logit j is always j, so an answer token t has loss log(sum(exp(j), j=0..31)) - t.
    with model.base_model.get_output_embeddings().register_forward_hook(
        lambda module, inputs, output: torch.arange(32, device=output.device, dtype=output.dtype).expand_as(output)
    ):
        output = model(input_ids=input_ids, labels=input_ids.clone(), **kwargs)

    log_normalizer = math.log(sum(math.exp(j) for j in range(32)))
    if explicit_input_mask:
        # The two samples select answers [7, 8] and [7, 5].
        loss_sum = 4 * log_normalizer - (7 + 8 + 7 + 5)
        label_count = 4
    else:
        # Without a mask, all four input tokens in each sample contribute.
        loss_sum = 8 * log_normalizer - 2 * (5 + 6 + 7 + 8)
        label_count = 8
    if prefix_types in ([1, 2, 3, 4], [1, 2, 3, 4, 5]):
        # Each sample also has one context answer, token 3, weighted by 0.95 only in decay mode.
        context_weight = 0.95 if weighted_loss == "decay" else 1
        loss_sum += 2 * context_weight * (log_normalizer - 3)
        label_count += 2

    torch.testing.assert_close(output.loss, output.loss.new_tensor(loss_sum / label_count))


@pytest.mark.parametrize("decay_factor", [0.25, 0.5])
def test_cpt_forward_loss_uses_switched_adapter_config(decay_factor):
    # Each CPT adapter has its own prompt encoder and loss configuration. Compare switching from the unweighted
    # "default" adapter to the exponential-decay "decay" adapter with a reference model using only decay loss.
    # Switching back must restore the default loss; matching logits isolates the loss configuration regression (#3879).
    unweighted_config = CPTConfig(
        cpt_token_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        cpt_mask=[1] * 8,
        cpt_tokens_type_mask=[1, 2, 2, 2, 3, 3, 4, 4],
        opt_weighted_loss_type="none",
        tokenizer_name_or_path=MODEL_NAME,
        task_type=TaskType.CAUSAL_LM,
    )
    decay_config = copy.deepcopy(unweighted_config)
    decay_config.opt_weighted_loss_type = "decay"
    decay_config.opt_loss_decay_factor = decay_factor
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    reference_model = get_peft_model(copy.deepcopy(base_model), copy.deepcopy(decay_config)).eval()
    model_under_test = get_peft_model(base_model, unweighted_config)
    model_under_test.add_adapter("decay", decay_config)
    model_under_test.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, base_model.config.vocab_size, (2, 6))
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids}

    with torch.no_grad():
        default_output = model_under_test(**inputs)
        model_under_test.set_adapter("decay")
        switched_output = model_under_test(**inputs)
        decay_reference_output = reference_model(**inputs)
        model_under_test.set_adapter("default")
        restored_output = model_under_test(**inputs)

    torch.testing.assert_close(switched_output.logits, decay_reference_output.logits)
    torch.testing.assert_close(switched_output.loss, decay_reference_output.loss)
    assert not torch.allclose(default_output.loss, decay_reference_output.loss)
    torch.testing.assert_close(restored_output.loss, default_output.loss)

    model_under_test.set_adapter("decay")
    model_under_test(**inputs).loss.backward()
    gradient = model_under_test.prompt_encoder["decay"].delta_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_cpt_forward_does_not_mutate_input_type_mask():
    """A forward pass must leave the caller-provided input_type_mask unchanged.

    The CPT forward used to shift the mask in place, so every call rewrote the tensor held by the caller (e.g. a batch
    stored in a dataset and reused across epochs) and the drift compounded on each forward.
    """
    config = CPTConfig(
        cpt_token_ids=[0, 1, 2, 3],
        cpt_mask=[1, 1, 1, 1],
        cpt_tokens_type_mask=[1, 2, 3, 4],
        opt_weighted_loss_type="none",
        tokenizer_name_or_path=MODEL_NAME,
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, config)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, base_model.config.vocab_size, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    input_type_mask = torch.full_like(input_ids, 4)
    input_type_mask_before = input_type_mask.clone()

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone(),
            input_type_mask=input_type_mask,
        )
        # a second forward catches the drift compounding across calls
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone(),
            input_type_mask=input_type_mask,
        )

    assert torch.equal(input_type_mask, input_type_mask_before)
