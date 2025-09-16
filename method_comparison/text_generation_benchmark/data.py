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
from typing import Optional

from transformers import PreTrainedTokenizer
from utils import BenchmarkConfig


DEFAULT_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "configs", "prompts.json")


def load_test_prompts(config: dict) -> dict[str, list[str]]:
    """
    Load prompts from JSON file.

    Args:
        config: Configuration containing prompts file path

    Returns:
        dictionary with prompts by category
    """
    prompts_file = getattr(config, "prompts_file", DEFAULT_PROMPTS_PATH)

    with open(prompts_file) as f:
        prompts = json.load(f)

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
            max_length = 2048

    max_prompt_length = max_length - reserve_output_tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]

    if len(input_ids) <= max_prompt_length:
        return prompt

    truncated_ids = input_ids[:max_prompt_length]
    truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    return truncated_prompt


def prepare_benchmark_prompts(
    config: BenchmarkConfig,
    tokenizer: PreTrainedTokenizer,
    max_input_length: Optional[int] = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Prepare prompts for benchmarking, ensuring appropriate length and variety.
    Always returns all prompt categories for consistent benchmarking.

    Args:
        config: Benchmark configuration
        tokenizer: Model tokenizer
        max_input_length: Maximum input length (overrides model default if provided)
        seed: Random seed (kept for backwards compatibility)

    Returns:
        Dictionary with processed prompts by category (all categories included)
    """
    all_prompts = load_test_prompts(config)

    processed_prompts = {}
    for category, prompts in all_prompts.items():
        truncated_prompts = [
            truncate_prompt_for_model(
                prompt,
                tokenizer,
                max_length=max_input_length,
                reserve_output_tokens=getattr(config, "reserve_output_tokens", 50),
            )
            for prompt in prompts
        ]

        processed_prompts[category] = truncated_prompts

    return processed_prompts
