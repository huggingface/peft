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
"""
Test for tie_word_embeddings handling in merge_and_unload()

Resolves: https://github.com/huggingface/peft/issues/2777
"""

import gc
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model


class TestTieWordEmbeddingsMerge(unittest.TestCase):
    """Test that merge_and_unload correctly handles tie_word_embeddings."""

    model_id = "HuggingFaceTB/SmolLM-135M"

    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_merge_unties_embeddings_when_both_targeted(self):
        """
        Test that when both embed_tokens and lm_head have adapters via modules_to_save,
        merge_and_unload() automatically unties the weights and updates config.

        Resolves: https://github.com/huggingface/peft/issues/2777
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )

        # Skip if model doesn't have tied embeddings
        if not getattr(model.config, "tie_word_embeddings", False):
            self.skipTest("Model does not have tie_word_embeddings=True")

        # Verify initial state
        self.assertTrue(model.config.tie_word_embeddings)

        # Configure LoRA to target both embed_tokens and lm_head via modules_to_save
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        )

        peft_model = get_peft_model(model, lora_config)

        # Merge and unload
        merged_model = peft_model.merge_and_unload()

        # After merge: tie_word_embeddings should be False
        self.assertFalse(
            merged_model.config.tie_word_embeddings,
            "config.tie_word_embeddings should be False after merging with both embeddings targeted",
        )

        # Verify weights are actually untied (different memory addresses)
        embed_weight = merged_model.model.embed_tokens.weight
        lm_head_weight = merged_model.lm_head.weight
        self.assertNotEqual(
            embed_weight.data_ptr(),
            lm_head_weight.data_ptr(),
            "embed_tokens and lm_head weights should be untied (different memory)",
        )

    def test_merge_preserves_tie_when_embeddings_not_targeted(self):
        """
        Test that when only LoRA target_modules (not modules_to_save) are used,
        tie_word_embeddings is preserved if embed_tokens/lm_head are not targeted.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )

        if not getattr(model.config, "tie_word_embeddings", False):
            self.skipTest("Model does not have tie_word_embeddings=True")

        original_tie = model.config.tie_word_embeddings

        # Configure LoRA without targeting embeddings
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

        peft_model = get_peft_model(model, lora_config)
        merged_model = peft_model.merge_and_unload()

        # tie_word_embeddings should be unchanged
        self.assertEqual(
            merged_model.config.tie_word_embeddings,
            original_tie,
            "tie_word_embeddings should be unchanged when embeddings are not targeted",
        )

    def test_merged_model_produces_valid_output(self):
        """
        Test that merged model produces coherent output (not garbage).
        This is a sanity check to ensure the merge doesn't break inference.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )

        if not getattr(model.config, "tie_word_embeddings", False):
            self.skipTest("Model does not have tie_word_embeddings=True")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        )

        peft_model = get_peft_model(model, lora_config)
        merged_model = peft_model.merge_and_unload()

        # Generate some text
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("Hello, how are", return_tensors="pt")
        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check that output is not garbage (contains some readable characters)
        self.assertTrue(
            any(c.isalpha() for c in decoded),
            f"Generated output appears to be garbage: {decoded}",
        )


if __name__ == "__main__":
    unittest.main()
