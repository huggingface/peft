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

import torch
from transformers import AutoTokenizer


class TokenizerMetaMath:
    PROMPT_NO_INPUT = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response: "
    )
    PROMPT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Input:\n{input}\n\n### Response: "
    )

    def format_prompt(self, query):
        query = query.split("\n", 1)
        if len(query) == 1 or query[1].strip("\n") == "":
            return self.PROMPT_NO_INPUT.format(query=query[0])
        else:
            return self.PROMPT.format(query=query[0], input=query[1])

    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, examples):
        prompts = [self.format_prompt(text) for text in examples["query"]]
        completions = examples["response"]
        return self._tokenize_fn(prompts, completions)

    def _tokenize_fn(self, prompts, completions):
        prompt_tokens = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        input_tokens = self.tokenizer([x + y for x, y in zip(prompts, completions)], add_special_tokens=False)[
            "input_ids"
        ]
        input_tokens = [[self.tokenizer.bos_token_id] + x + [self.tokenizer.eos_token_id] for x in input_tokens]
        prompt_length = [len(x) + 1 for x in prompt_tokens]  # +1 for the bos token
        input_length = [len(x) for x in input_tokens]
        return {"input_ids": input_tokens, "prompt_length": prompt_length, "input_length": input_length}


class DataCollator:
    def __init__(self, eos_token_id, max_length=None):
        self.eos_token_id = eos_token_id
        self.max_length = max_length

    def __call__(self, batch):
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        input_lengths = torch.stack(batch["input_length"])
        prompt_lengths = torch.stack(batch["prompt_length"])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self.eos_token_id
        )
        col_indices = torch.arange(input_ids.size(1)).unsqueeze(0)
        attention_mask = col_indices < input_lengths.unsqueeze(1)
        label_mask = torch.logical_or(col_indices < prompt_lengths.unsqueeze(1), ~attention_mask)
        labels = input_ids.masked_fill(label_mask, -100)
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            labels = labels[:, : self.max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
