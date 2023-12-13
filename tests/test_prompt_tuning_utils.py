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
import unittest

from parameterized import parameterized
import torch
from transformers import AutoModelForCausalLM

from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from peft.utils import separate_pad_tokens, add_pad_tokens
from peft.tuners.tuners_utils import check_target_module_exists, inspect_matched_modules


SEPARATE_PAD_TOKENS_TEST_CASES = (
    (
        torch.tensor([[10, 2, 3, 3]]), None, torch.tensor([[0, 1, 1, 1]]), torch.tensor([[-100, 2, 3, 3]]),
        (
            [torch.tensor([2, 3, 3])], None, [torch.tensor([1, 1, 1])], [torch.tensor([2, 3, 3])], 
            ([torch.tensor([10])], None, [torch.tensor([0])], [torch.tensor([-100])])
        )
    ),
    (
        torch.tensor([[2, 3, 3], [1, 10, 12]]), None, torch.tensor([[1, 1, 1], [1, 1, 1]]), torch.tensor([[2, 3, 3], [1, 10, 12]]), 
        (
            [torch.tensor([2, 3, 3]), torch.tensor([1, 10, 12])], None, [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1])], [torch.tensor([2, 3, 3]), torch.tensor([1, 10, 12])],
            ([torch.tensor([]), torch.tensor([])], None, [torch.tensor([]), torch.tensor([])], [torch.tensor([]), torch.tensor([])])
        )
    ),
    (
        None, torch.tensor([[[10], [2], [3], [3]]]), torch.tensor([[0, 1, 1, 1]]), torch.tensor([[-100, 2, 3, 3]]),
        (
            None, [torch.tensor([[2], [3], [3]])], [torch.tensor([1, 1, 1])], [torch.tensor([2, 3, 3])], 
            (None, [torch.tensor([[10]])], [torch.tensor([0])], [torch.tensor([-100])])
        )
    ),
    (
        torch.tensor([[10, 2, 3, 3]]), None, None, torch.tensor([[-100, 2, 3, 3]]),
        (
            torch.tensor([[10, 2, 3, 3]]), None, None, torch.tensor([[-100, 2, 3, 3]]), 
            None
        )
    ),
)

class PromptTuningUtilsTester(unittest.TestCase):
    r"""
    Test if the helper functions used for prompt tuning work as expected.
    """
    @parameterized.expand(SEPARATE_PAD_TOKENS_TEST_CASES)
    def test_separate_pad_tokens(self, input_ids, input_embeds, attn_mask, labels, expected_res):

        expected_inp_ids, expected_embeds, expected_attn_mask, expected_labels, expected_pad_els = expected_res
        actual_inp_ids, actual_embeds, actual_attn_mask, actual_labels, actual_pad_els = separate_pad_tokens(
            input_ids=input_ids, inputs_embeds=input_embeds, attention_mask=attn_mask, labels=labels
        )

        self.assert_true_for_tensor_list(actual_inp_ids, expected_inp_ids)
        self.assert_true_for_tensor_list(actual_embeds, expected_embeds)
        self.assert_true_for_tensor_list(actual_attn_mask, expected_attn_mask)
        self.assert_true_for_tensor_list(actual_labels, expected_labels)

        if expected_pad_els is None or actual_pad_els is None:
            self.assertEqual(actual_pad_els, expected_pad_els)
        else:
            self.assertTrue(len(expected_pad_els) == len(actual_pad_els))
            for i in range(len(actual_pad_els)):
                self.assert_true_for_tensor_list(expected_pad_els[i], actual_pad_els[i])
    
    def assert_true_for_tensor_list(self, list1, list2):
        if list1 is None or list2 is None:
            self.assertEqual(list1, list2)
        else:
            [self.assertTrue(torch.equal(tensor1, tensor2)) for tensor1, tensor2 in zip(list1, list2)]
        
        

        