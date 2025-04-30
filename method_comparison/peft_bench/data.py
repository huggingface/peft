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
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer


# Default test prompts collection with varying lengths and domains
DEFAULT_TEST_PROMPTS = {
    "short": [
        "Explain what PEFT means.",
        "Summarize the benefits of fine-tuning.",
        "What is parameter-efficient fine-tuning?",
        "Compare LoRA and full fine-tuning.",
        "List three NLP tasks.",
    ],
    "medium": [
        "Write a function in Python to calculate the Fibonacci sequence up to n terms. Include comments explaining your implementation.",
        "Explain the concept of parameter-efficient fine-tuning methods in natural language processing. What are the advantages compared to full fine-tuning?",
        "Summarize the key differences between LoRA, Prefix Tuning, and Prompt Tuning approaches. Which situations might each be most suitable for?",
        "Describe the architecture of a transformer model and explain how attention mechanisms work. Why are they effective for sequence data?",
        "Write a detailed analysis of how fine-tuning affects model performance across different domains. Include considerations about catastrophic forgetting.",
    ],
    "long": [
        """Analyze the evolution of parameter-efficient fine-tuning methods from 2020 to present. 
        Include a detailed comparison of at least five different approaches, their theoretical foundations, 
        computational requirements, and empirical performance on standard benchmarks. 
        Discuss the trade-offs between parameter count, training time, inference latency, and downstream performance.
        Finally, suggest potential future directions for research in this area and explain why they might be promising.
        """,
        """Write a comprehensive tutorial explaining how to implement LoRA (Low-Rank Adaptation) for fine-tuning 
        a large language model. Start with the mathematical foundations, then provide step-by-step Python code 
        using the PEFT library. Include examples for both training and inference, error handling, and best practices 
        for hyperparameter selection. The tutorial should be accessible to ML practitioners who understand basic 
        concepts but haven't implemented PEFT methods before. Add detailed comments throughout the code.
        """,
        """Design a benchmark suite for comparing different parameter-efficient fine-tuning methods across various tasks.
        Specify the metrics to be collected (e.g., parameter count, training time, inference latency, memory usage,
        accuracy/perplexity on downstream tasks), the baseline models to be used, and the evaluation methodology.
        Include both natural language understanding and generation tasks. Propose a visualization approach for presenting
        the multi-dimensional results in an intuitive way. Discuss potential limitations and biases in your benchmark design.
        """,
    ],
    "code": [
        "Write a Python function to implement a binary search algorithm.",
        "Create a class in JavaScript for managing a simple to-do list application.",
        "Implement a sorting algorithm of your choice in C++ and explain its time complexity.",
    ],
    "math": [
        "Find the derivative of f(x) = x^3 * ln(x^2 + 1) with respect to x.",
        "Solve the system of equations: 3x + 2y - z = 5, 2x - y + 2z = 7, x + y + z = 3.",
        "Prove that the sum of the first n odd natural numbers equals n².",
    ],
    "reasoning": [
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
    ],
}


def load_test_prompts(config: Dict) -> Dict[str, List[str]]:
    """
    Load test prompts based on configuration.

    Args:
        config: Dictionary containing prompt configuration
               Can specify 'prompts_file' or 'custom_prompts'

    Returns:
        Dictionary of prompt categories and their prompts
    """
    # Start with default prompts
    prompts = DEFAULT_TEST_PROMPTS.copy()

    # If a prompts file is specified, load it
    if "prompts_file" in config:
        file_path = config["prompts_file"]
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                file_prompts = json.load(f)
            # Update or add to default prompts
            for category, prompt_list in file_prompts.items():
                prompts[category] = prompt_list

    # If custom prompts are specified directly in config
    if "custom_prompts" in config:
        for category, prompt_list in config["custom_prompts"].items():
            prompts[category] = prompt_list

    # If specific categories are requested, filter to just those
    if "prompt_categories" in config:
        categories = config["prompt_categories"]
        prompts = {k: v for k, v in prompts.items() if k in categories}

    return prompts


def get_prompts_by_length(prompts: Dict[str, List[str]], length: str = "all") -> List[str]:
    """
    Get prompts filtered by length category.

    Args:
        prompts: Dictionary of prompts by category
        length: One of 'short', 'medium', 'long', or 'all'

    Returns:
        List of prompts matching the requested length
    """
    if length == "all":
        # Flatten all prompts into a single list
        return [prompt for category in prompts.values() for prompt in category]

    if length in prompts:
        return prompts[length]

    # If length category not found, return a mix of available prompts
    result = []
    for category, prompt_list in prompts.items():
        if prompt_list:
            result.append(prompt_list[0])  # Take first prompt from each category

    return result


def truncate_prompt_for_model(
    prompt: str, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None, reserve_output_tokens: int = 50
) -> str:
    """
    Truncate a prompt to fit within a model's context window.

    Args:
        prompt: The input prompt to truncate if needed
        tokenizer: Model tokenizer
        max_length: Maximum context length (if None, uses tokenizer's model_max_length)
        reserve_output_tokens: Number of tokens to reserve for model output

    Returns:
        Truncated prompt that fits within model constraints
    """
    if max_length is None:
        max_length = tokenizer.model_max_length

    # Reserve tokens for output
    available_length = max(1, max_length - reserve_output_tokens)

    # Tokenize prompt to check length
    tokens = tokenizer.encode(prompt, truncation=False, add_special_tokens=True)

    if len(tokens) <= available_length:
        return prompt  # No truncation needed

    # Truncate token sequence
    truncated_tokens = tokens[:available_length]

    # Decode back to text
    truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    # Add indicator that truncation happened
    truncated_prompt += " [truncated...]"

    return truncated_prompt


def prepare_benchmark_prompts(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    max_input_length: Optional[int] = None,
    num_samples: int = 5,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Prepare prompts for benchmarking, ensuring appropriate length and variety.

    Args:
        config: Benchmark configuration
        tokenizer: Model tokenizer
        max_input_length: Maximum input length (overrides model default if provided)
        num_samples: Number of prompts to sample from each category
        seed: Random seed for reproducible sampling

    Returns:
        Dictionary with processed prompts by category
    """
    np.random.seed(seed)

    # Load prompts from configuration
    all_prompts = load_test_prompts(config)

    # Process each category
    processed_prompts = {}
    for category, prompts in all_prompts.items():
        # Sample if we have more prompts than needed
        if len(prompts) > num_samples:
            selected_prompts = np.random.choice(prompts, size=num_samples, replace=False).tolist()
        else:
            selected_prompts = prompts

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
