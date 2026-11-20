"""Activated LoRA test benchmark.

This includes speed benchmarks.

Regression testing can be implemented by providing expected outputs.
The benchmark script will write the generated outputs to a file.

"""

import argparse
import json

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_base_model(input_sequence, tokenizer, base_model, **kwargs):
    inputs = tokenizer(input_sequence, return_tensors="pt").to(base_model.device)
    a = perf_counter()
    output = base_model.generate(**inputs, **kwargs)
    b = perf_counter()
    return output, b - a


def benchmark_peft_model(input_sequence, tokenizer, peft_model, **kwargs):
    inputs = tokenizer(input_sequence, return_tensors="pt").to(peft_model.device)
    a = perf_counter()
    output = peft_model.generate(**inputs, **kwargs)
    b = perf_counter()
    return output, b - a


def fixed_sequence(tokenizer, n, invocation_pos=None, invocation_tokens=None):
    """Build a string sequence of length n with invocation_tokens starting at position invocation_pos.
    We use a string since batching of strings is easier and the device matching is handled that way too.
    """
    assert invocation_pos is None or invocation_pos >= 0

    sequence = "nonsense " * n
    sequence = tokenizer.encode(sequence)
    sequence = sequence[:n]

    if invocation_pos is not None:
        if invocation_pos + len(invocation_tokens) > n:
            raise ValueError(f"{invocation_pos=} (+{len(invocation_tokens)}) exceeds sequence length {n}")
        sequence[invocation_pos : invocation_pos + len(invocation_tokens)] = invocation_tokens

    # make sure that we don't lose the invocation tokens in the process
    assert tokenizer.encode(tokenizer.decode(sequence)) == sequence
    assert len(sequence) == n
    return tokenizer.decode(sequence)


@dataclass
class BenchmarkOutputs:
    output_base_clear: list[int]
    output_base_token: list[int]
    output_base_batched_smooth: list[list[int]]
    output_base_batched_ragged: list[list[int]]

    output_peft_clear: list[int]
    output_peft_token: list[int]
    output_peft_batched_smooth: list[list[int]]
    output_peft_batched_ragged: list[list[int]]


def main(args):
    generation_kwargs = {
        "max_new_tokens": 200,
        "min_new_tokens": 2,
        "do_sample": False,
    }

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Build sequences and batches for inference.
    #
    # We will test smooth (invocation token index is the same for the batch) and ragged
    # (not the same index across batch) batches since we assume that the two have different
    # performance and validity characteristics.
    invocation_tokens = tokenizer.encode("foobar")

    input_sequence_token = fixed_sequence(tokenizer, n=100, invocation_pos=50, invocation_tokens=invocation_tokens)
    input_sequence_clear = fixed_sequence(tokenizer, n=100, invocation_pos=None)
    input_sequence_smooth_batch = [
        fixed_sequence(tokenizer, n=100, invocation_pos=50, invocation_tokens=invocation_tokens),
        fixed_sequence(tokenizer, n=100, invocation_pos=50, invocation_tokens=invocation_tokens),
        fixed_sequence(tokenizer, n=100, invocation_pos=50, invocation_tokens=invocation_tokens),
        fixed_sequence(tokenizer, n=100, invocation_pos=50, invocation_tokens=invocation_tokens),
    ]
    input_sequence_ragged_batch = [
        fixed_sequence(tokenizer, n=100, invocation_pos=None),
        fixed_sequence(tokenizer, n=100, invocation_pos=25, invocation_tokens=invocation_tokens),
        fixed_sequence(tokenizer, n=100, invocation_pos=50, invocation_tokens=invocation_tokens),
        fixed_sequence(tokenizer, n=100, invocation_pos=75, invocation_tokens=invocation_tokens),
    ]

    # Benchmarking
    #
    print("\nBase model timings (overall baseline):")
    out_base_clear, t_base_clear = benchmark_base_model(input_sequence_token, tokenizer, base_model, **generation_kwargs)
    print(f"{t_base_clear=}")

    out_base_token, t_base_token = benchmark_base_model(input_sequence_token, tokenizer, base_model, **generation_kwargs)
    print(f"{t_base_token=}")

    out_base_batched_smooth, t_base_batched_smooth = benchmark_base_model(
        input_sequence_smooth_batch, tokenizer, base_model, **generation_kwargs
    )
    print(f"{t_base_batched_smooth=}")

    out_base_batched_ragged, t_base_batched_ragged = benchmark_base_model(
        input_sequence_ragged_batch, tokenizer, base_model, **generation_kwargs
    )
    print(f"{t_base_batched_ragged=}")


    print("\nLoRA timings (PEFT baseline):")
    torch.manual_seed(42)
    alora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=False,  # make the output (seeded) random
    )
    peft_model = get_peft_model(deepcopy(base_model), alora_config).to(args.device)

    _, t_peft_base_clear = benchmark_peft_model(input_sequence_clear, tokenizer, peft_model, **generation_kwargs)
    print(f"{t_peft_base_clear=}")

    _, t_peft_base_token = benchmark_peft_model(input_sequence_token, tokenizer, peft_model, **generation_kwargs)
    print(f"{t_peft_base_token=}")

    _, t_peft_base_batched_smooth = benchmark_peft_model(
        input_sequence_smooth_batch, tokenizer, peft_model, **generation_kwargs
    )
    print(f"{t_peft_base_batched_smooth=}")

    _, t_peft_base_batched_ragged = benchmark_peft_model(
        input_sequence_ragged_batch, tokenizer, peft_model, **generation_kwargs
    )
    print(f"{t_peft_base_batched_ragged=}")


    print("\naLoRA timings:")
    torch.manual_seed(42)
    alora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # silence aLoRA task-type warning
        alora_invocation_tokens=invocation_tokens,
        init_lora_weights=False,  # make the output (seeded) random
    )
    peft_model = get_peft_model(deepcopy(base_model), alora_config).to(args.device)

    out_peft_clear, t_peft_clear = benchmark_peft_model(input_sequence_clear, tokenizer, peft_model, **generation_kwargs)
    print(f"{t_peft_clear=}")

    out_peft_token, t_peft_token = benchmark_peft_model(input_sequence_token, tokenizer, peft_model, **generation_kwargs)
    print(f"{t_peft_token=}")

    out_peft_batched_smooth, t_peft_batched_smooth = benchmark_peft_model(
        input_sequence_smooth_batch, tokenizer, peft_model, **generation_kwargs
    )
    print(f"{t_peft_batched_smooth=}")

    out_peft_batched_ragged, t_peft_batched_ragged = benchmark_peft_model(
        input_sequence_ragged_batch, tokenizer, peft_model, **generation_kwargs
    )
    print(f"{t_peft_batched_ragged=}")


    # Comparison with reference outputs
    #
    # Correctness verification with outputs from another run.
    if args.reference_output_file:
        with open(args.reference_output_file) as f:
            data = json.load(f)
            reference_outputs = BenchmarkOutputs(**data)

        assert out_base_clear.cpu().tolist() == reference_outputs.output_base_clear
        assert out_base_token.cpu().tolist() == reference_outputs.output_base_token
        assert out_base_batched_smooth.cpu().tolist() == reference_outputs.output_base_batched_smooth
        assert out_base_batched_ragged.cpu().tolist() == reference_outputs.output_base_batched_ragged
        assert out_peft_clear.cpu().tolist() == reference_outputs.output_peft_clear
        assert out_peft_token.cpu().tolist() == reference_outputs.output_peft_token
        assert out_peft_batched_smooth.cpu().tolist() == reference_outputs.output_peft_batched_smooth
        assert out_peft_batched_ragged.cpu().tolist() == reference_outputs.output_peft_batched_ragged

    # Creating reference artifact
    #
    # If instructed, create a reference output file that allows us to verify
    # correctness with other implementations at a later point.
    if args.output_file:
        outputs = BenchmarkOutputs(
            output_base_clear=out_base_clear.cpu().tolist(),
            output_base_token=out_base_token.cpu().tolist(),
            output_base_batched_smooth=out_base_batched_smooth.cpu().tolist(),
            output_base_batched_ragged=out_base_batched_ragged.cpu().tolist(),
            output_peft_clear=out_peft_clear.cpu().tolist(),
            output_peft_token=out_peft_token.cpu().tolist(),
            output_peft_batched_smooth=out_peft_batched_smooth.cpu().tolist(),
            output_peft_batched_ragged=out_peft_batched_ragged.cpu().tolist(),
        )

        with open(args.output_file, "w") as f:
            json.dump(asdict(outputs), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--reference_output_file", type=str, default=None)

    args = parser.parse_args()
    main(args)
