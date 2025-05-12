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

"""
Data handling utilities for PEFT benchmarking.
"""

import json
import os
import textwrap
from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer


# Path to the default prompts file
DEFAULT_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts.json")


def load_test_prompts(config: dict) -> dict[str, list[str]]:
    """
    Load prompts from JSON file.

    Args:
        config: Configuration containing prompts file path

    Returns:
        dictionary with prompts by category
    """
    # Use the specified prompts file or fall back to default
    prompts_file = config.get("prompts_file", DEFAULT_PROMPTS_PATH)

    with open(prompts_file) as f:
        prompts = json.load(f)

    # Apply textwrap.dedent to remove leading spaces from multiline prompts
    for category, prompt_list in prompts.items():
        prompts[category] = [textwrap.dedent(prompt) for prompt in prompt_list]

    return prompts


def truncate_prompt_for_model(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    reserve_output_tokens: int = 50,
) -> str:
    """
    Truncate a prompt to fit within the model's context window.

    Args:
        prompt: Input prompt
        tokenizer: Model tokenizer
        max_length: Maximum sequence length (if None, uses model's max_length)
        reserve_output_tokens: Number of tokens to reserve for response

    Returns:
        Truncated prompt
    """
    if max_length is None:
        if hasattr(tokenizer, "model_max_length"):
            max_length = tokenizer.model_max_length
        else:
            max_length = 2048  # Default fallback

    max_prompt_length = max_length - reserve_output_tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]

    if len(input_ids) <= max_prompt_length:
        return prompt

    truncated_ids = input_ids[:max_prompt_length]
    truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    return truncated_prompt


def prepare_benchmark_prompts(
    config: dict,
    tokenizer: PreTrainedTokenizer,
    max_input_length: Optional[int] = None,
    num_samples: int = 5,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Prepare prompts for benchmarking, ensuring appropriate length and variety.

    Args:
        config: Benchmark configuration
        tokenizer: Model tokenizer
        max_input_length: Maximum input length (overrides model default if provided)
        num_samples: Number of prompts to use from each category
        seed: Random seed (kept for backwards compatibility)

    Returns:
        Dictionary with processed prompts by category
    """
    # Load prompts
    all_prompts = load_test_prompts(config)

    # Process each category
    processed_prompts = {}
    for category, prompts in all_prompts.items():
        # Take the first num_samples prompts
        selected_prompts = prompts[:num_samples] if len(prompts) > num_samples else prompts

        # Truncate prompts if needed
        truncated_prompts = [
            truncate_prompt_for_model(
                prompt,
                tokenizer,
                max_length=max_input_length,
                reserve_output_tokens=config.get("reserve_output_tokens", 50),
            )
            for prompt in selected_prompts
        ]

        processed_prompts[category] = truncated_prompts

    return processed_prompts
