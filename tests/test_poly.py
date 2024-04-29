#!/usr/bin/env python3

# coding=utf-8
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

import os
import tempfile
import unittest

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from peft import PeftModel, PolyConfig, TaskType, get_peft_model


class TestPoly(unittest.TestCase):
    def test_poly(self):
        torch.manual_seed(0)
        model_name_or_path = "google/flan-t5-small"

        atol, rtol = 1e-6, 1e-6
        r = 8  # rank of lora in poly
        n_tasks = 3  # number of tasks
        n_skills = 2  # number of skills (loras)
        n_splits = 4  # number of heads
        lr = 1e-2
        num_epochs = 10

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        peft_config = PolyConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            poly_type="poly",
            r=r,
            n_tasks=n_tasks,
            n_skills=n_skills,
            n_splits=n_splits,
        )

        model = get_peft_model(base_model, peft_config)

        # generate some dummy data
        text = os.__doc__.splitlines()
        assert len(text) > 10
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs["task_ids"] = torch.arange(len(text)) % n_tasks
        inputs["labels"] = tokenizer((["A", "B"] * 100)[: len(text)], return_tensors="pt")["input_ids"]

        # simple training loop
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        for _ in range(num_epochs):
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # loss improved by at least 50%
        assert losses[-1] < (0.5 * losses[0])

        # check that saving and loading works
        torch.manual_seed(0)
        model.eval()
        logits_before = model(**inputs).logits
        tokens_before = model.generate(**inputs)

        with model.disable_adapter():
            logits_disabled = model(**inputs).logits
            tokens_disabled = model.generate(**inputs)

        assert not torch.allclose(logits_before, logits_disabled, atol=atol, rtol=rtol)
        assert not torch.allclose(tokens_before, tokens_disabled, atol=atol, rtol=rtol)

        # saving and loading
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            loaded = PeftModel.from_pretrained(base_model, tmp_dir)

        torch.manual_seed(0)
        output_after = loaded(**inputs).logits
        tokens_after = loaded.generate(**inputs)
        assert torch.allclose(logits_before, output_after, atol=atol, rtol=rtol)
        assert torch.allclose(tokens_before, tokens_after, atol=atol, rtol=rtol)
