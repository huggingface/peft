# Copyright 2026-present the HuggingFace Inc. team.
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
Convert a Mixture-of-Experts (MoE) model into a dense model with PEFT's MoE-to-dense method.

The MoE model is used as the teacher to distill a dense student that is initialized from the most important experts of
each MoE layer. Distillation uses the MetaMathQA dataset and evaluation is done on GSM8K, mirroring the MetaMathQA
benchmark in `method_comparison/MetaMathQA`. Note that the paper this method is based on distills on billions of tokens
of general pretraining data; this example instead distills on a narrow domain (math), for which a much smaller token
budget is needed to obtain a useful dense model.

Example:

    python distill_moe_to_dense.py --model_id Qwen/Qwen3-30B-A3B --output_dir ./qwen3-30b-a3b-math-dense
"""

import argparse
import gc
import math
import os
import textwrap
import time
from contextlib import nullcontext

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, set_seed
from utils import BucketIterator, evaluate, get_accuracy, get_tokenizer, get_train_valid_test_datasets

from peft import MoeToDenseConfig, get_peft_model
from peft.utils import infer_device


# Prompt format of the MetaMathQA benchmark. A worked example is prepended to every prompt (one-shot) so that the models
# produce the answer format that the GSM8K evaluation expects ("The answer is: <number>"). Without it, the accuracy of
# most base/instruct models is 0 regardless of their math ability, simply because their answers cannot be parsed
FEW_SHOT_EXAMPLE = (
    "Question: A bakery sells muffins for $3 each and cookies for $1 each. Tom buys 4 muffins and 6 cookies. "
    "How much does he pay in total? Think step by step.\n"
    "Answer: The muffins cost 4 * 3 = 12 dollars. The cookies cost 6 * 1 = 6 dollars. In total, Tom pays "
    "12 + 6 = 18 dollars.\nThe answer is: 18\n\n"
)
QUERY_TEMPLATE = "Question: {query} Think step by step.\nAnswer:"
# the example adds ~100 tokens to every sequence, hence the larger limits compared to the MetaMathQA benchmark
GENERATION_KWARGS = {"max_length": 1100, "max_new_tokens": 300}
# models that were not fine-tuned on the format tend to continue with made-up questions after their answer
STOP_STRING = "\nQuestion:"
BUCKET_FACTOR = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-30B-A3B", help="The MoE model to convert.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the dense model.")
    parser.add_argument(
        "--adapter_output_dir",
        type=str,
        default=None,
        help="Optionally, where to save the PEFT adapter before compressing the model.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--dense_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="The dtype of the trainable dense FFNs. float32 with autocast is more precise but needs more memory.",
    )
    parser.add_argument("--attn_implementation", type=str, default=None, help="Passed to from_pretrained.")
    parser.add_argument("--device_map", type=str, default=None, help="E.g. 'auto' to spread the model over GPUs.")
    parser.add_argument("--use_gc", action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--num_experts_to_keep", type=int, default=None, help="Defaults to the router's top-k.")
    parser.add_argument(
        "--modules_to_save",
        type=str,
        nargs="*",
        default=None,
        help="Additional modules to fully fine-tune, e.g. q_proj k_proj v_proj o_proj input_layernorm "
        "post_attention_layernorm. The teacher keeps using the original modules.",
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--batch_size_eval", type=int, default=50, help="batch size for evaluation")
    parser.add_argument(
        "--calibration_steps",
        type=int,
        default=100,
        help="Number of training batches used to collect the routing statistics for scoring the experts.",
    )
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum number of steps to train.")
    parser.add_argument("--eval_steps", type=int, default=250, help="Number of evaluation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature of the KL loss.")
    parser.add_argument(
        "--eval_test",
        action="store_true",
        help="Evaluate the final model on the GSM8K test set (slow).",
    )
    parser.add_argument(
        "--skip_benchmark",
        action="store_true",
        help="Skip the comparison of inference speed and memory between the MoE model and the dense model.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def benchmark_model(model, tokenizer, prompts: list[list[int]], device: str, num_new_tokens: int = 64) -> dict:
    """Measure parameter count, weight memory, prefill throughput, decode latency, and peak memory of a model."""
    accelerator = getattr(torch, device, None) if device != "cpu" else None

    def sync():
        if accelerator is not None:
            accelerator.synchronize()

    def timed(fn, repeats: int = 3) -> float:
        fn()  # warmup
        sync()
        tic = time.perf_counter()
        for _ in range(repeats):
            fn()
        sync()
        return (time.perf_counter() - tic) / repeats

    results = {
        "parameters (B)": sum(p.numel() for p in model.parameters()) / 1e9,
        "weight memory (GB)": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30,
    }
    with torch.inference_mode():
        # prefill: forward pass over a batch of prompts
        batch = tokenizer.pad({"input_ids": prompts[:8]}, return_tensors="pt", padding_side="left").to(model.device)
        duration = timed(lambda: model(**batch, use_cache=False))
        results["prefill (tokens/s)"] = batch["attention_mask"].sum().item() / duration
        # decode: greedy generation of a fixed number of tokens, at batch size 1 and 16
        for batch_size in (1, 16):
            batch = tokenizer.pad({"input_ids": prompts[:batch_size]}, return_tensors="pt", padding_side="left").to(
                model.device
            )

            def generate(batch=batch):
                return model.generate(
                    **batch,
                    max_new_tokens=num_new_tokens,
                    min_new_tokens=num_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            if accelerator is not None:
                accelerator.reset_peak_memory_stats()
            duration = timed(generate, repeats=1)
            results[f"decode bs={batch_size} (ms/step)"] = 1000 * duration / num_new_tokens
            if accelerator is not None:
                results[f"peak memory bs={batch_size} (GB)"] = accelerator.max_memory_allocated() / 2**30
    return results


def run_benchmark(args, tokenizer, prompts: list[list[int]], device: str, dtype: torch.dtype) -> dict[str, dict]:
    """
    Load the original MoE model and the exported dense model one after the other and measure them under the same
    conditions. Returns the measurements per model.
    """
    results = {}
    for name, path in (("MoE (original)", args.model_id), ("dense (distilled)", args.output_dir)):
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map=args.device_map or device, attn_implementation=args.attn_implementation
        ).eval()
        results[name] = benchmark_model(model, tokenizer, prompts, device)
        del model
        gc.collect()
        if device != "cpu":
            getattr(torch, device).empty_cache()
    return results


def format_benchmark_table(results: dict[str, dict]) -> str:
    """Format the benchmark measurements of `run_benchmark` as a comparison table."""
    names = list(results)
    width = max(len(metric) for metric in results[names[0]])
    lines = [f"{'':{width}}  " + "  ".join(f"{name:>18}" for name in names)]
    for metric in results[names[0]]:
        values = "  ".join(f"{results[name][metric]:>18.2f}" for name in names)
        lines.append(f"{metric:{width}}  {values}")
    return "\n".join(lines)


def evaluate_accuracy(model, tokenizer, ds, batch_size: int) -> tuple[float, list[str]]:
    """Generate answers for the prompts of the dataset and return the accuracy and the predictions."""
    model.eval()
    predictions, responses = evaluate(
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        batch_size=batch_size,
        generate_kwargs={**GENERATION_KWARGS, "stop_strings": [STOP_STRING]},
    )
    predictions = [prediction.split(STOP_STRING, 1)[0].strip() for prediction in predictions]
    accuracy = get_accuracy(predictions=predictions, responses=responses)
    return accuracy, predictions


@torch.no_grad()
def evaluate_kl(model, tokenizer, ds, batch_size: int) -> tuple[float, int]:
    """
    Mean KL divergence between teacher and student (in nats per token) on the reference answers of the dataset.

    Returns the KL divergence and the number of answer tokens it was computed on.
    """
    model.eval()
    total, count = 0.0, 0
    for j in range(0, len(ds), batch_size):
        sliced = ds[j : j + batch_size]
        answer_ids = tokenizer([" " + response for response in sliced["response"]], add_special_tokens=False)
        input_ids = [p + a for p, a in zip(sliced["input_ids"], answer_ids["input_ids"])]
        batch = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt").to(model.device)
        # only the answer tokens count, not the prompt and not the padding
        labels = batch["input_ids"].clone()
        for i, (prompt_ids, ids) in enumerate(zip(sliced["input_ids"], input_ids)):
            labels[i, : len(prompt_ids)] = -100
            labels[i, len(ids) :] = -100
        num_tokens = (labels[:, 1:] != -100).sum().item()
        kl = model.get_distillation_loss(**batch, labels=labels, use_cache=False)
        total += kl.item() * num_tokens
        count += num_tokens
    return total / max(count, 1), count


def main():
    args = parse_args()
    set_seed(args.seed)
    device = infer_device()
    dtype = getattr(torch, args.dtype)
    dense_dtype = getattr(torch, args.dense_dtype)

    tokenizer = get_tokenizer(model_id=args.model_id, max_seq_length=args.max_seq_length)
    query_template = FEW_SHOT_EXAMPLE + QUERY_TEMPLATE
    # the constant few-shot prefix is excluded from the distillation loss
    prefix_length = len(tokenizer(FEW_SHOT_EXAMPLE)["input_ids"])
    ds_train, ds_valid, ds_test = get_train_valid_test_datasets(
        tokenizer=tokenizer, query_template=query_template, print_fn=print
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map=args.device_map or device,
        attn_implementation=args.attn_implementation,
    )
    if args.use_gc:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = MoeToDenseConfig(
        num_experts_to_keep=args.num_experts_to_keep,
        modules_to_save=args.modules_to_save or None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def get_batches():
        iterator = BucketIterator(
            ds_train, batch_size=args.batch_size, bucket_factor=BUCKET_FACTOR, delete_cols=["response"]
        )
        while True:
            for batch in iterator:
                tokens_per_sample = [len(input_ids) for input_ids in batch["input_ids"]]
                batch = tokenizer.pad(batch, return_tensors="pt").to(model.device)
                # The labels only serve to mask tokens from the distillation loss: the few-shot prefix and the padding
                # tokens, except for the first one, which is the EOS token (as in the MetaMathQA benchmark, the pad
                # token is the EOS token and the model should learn to predict it).
                labels = batch["input_ids"].clone()
                labels[:, :prefix_length] = -100
                for i, num_tokens in enumerate(tokens_per_sample):
                    labels[i, num_tokens + 1 :] = -100
                batch["labels"] = labels
                yield batch

    batches = get_batches()
    # the autocast context is only needed if the dense FFNs have a higher precision than the rest of the model
    if dense_dtype != dtype and dtype in (torch.bfloat16, torch.float16):
        autocast_ctx = torch.autocast(device_type=device, dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    # Reporting wrappers around the (pure) evaluation functions: they run the evaluations under the same autocast
    # context as the training (the trainable modules may be in a different dtype than the rest of the model) and print
    # the results.
    def report_accuracy(ds, desc: str, num_examples: int = 1) -> float:
        tic = time.perf_counter()
        with autocast_ctx:
            accuracy, predictions = evaluate_accuracy(model, tokenizer, ds, args.batch_size_eval)
        print(f"{desc}: accuracy {100 * accuracy:.1f}% ({len(ds)} samples, {time.perf_counter() - tic:.0f}s)")
        for prediction in predictions[:num_examples]:
            print(textwrap.indent(textwrap.shorten(f"example prediction: {prediction}", width=400), "    "))
        print()
        return accuracy

    def report_kl(desc: str) -> float:
        with autocast_ctx:
            kl, num_answer_tokens = evaluate_kl(model, tokenizer, ds_valid, args.batch_size_eval)
        print(
            f"{desc}: KL(teacher || student) = {kl:.4f} nats/token ({num_answer_tokens} held-out GSM8K answer tokens)"
        )
        return kl

    # 0. the teacher is the model with disabled adapters, i.e. the original MoE model
    with model.disable_adapter():
        report_accuracy(ds_valid, "Teacher (MoE)")

    # 1. collect routing statistics on training data and build the dense FFNs from the most important experts
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(args.calibration_steps), desc="Calibration"):
            batch = next(batches)
            batch.pop("labels")  # no need to compute the loss
            model(**batch, use_cache=False)
    model.update_and_allocate()
    if dense_dtype != dtype:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(dense_dtype)
    report_kl("Student before distillation")
    report_accuracy(ds_valid, "Student before distillation")

    # 2. distill the MoE teacher into the dense student
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_ratio * args.max_steps)

    def lr_lambda(step: int) -> float:
        # linear warmup, then cosine decay from lr to min_lr_ratio * lr (roughly the paper's schedule, Appendix J)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    losses = []
    num_tokens = 0
    tic = time.perf_counter()
    pbar = tqdm(range(1, args.max_steps + 1), desc="Distillation")
    for step in pbar:
        loss_step = 0.0
        for _ in range(args.grad_accumulation_steps):
            batch = next(batches)
            num_tokens += (batch["labels"] != -100).sum().item()
            with autocast_ctx:
                loss = model.get_distillation_loss(**batch, temperature=args.temperature, use_cache=False)
            loss = loss / args.grad_accumulation_steps
            loss.backward()
            loss_step += loss.item()
        if args.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_norm_clip)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        losses.append(loss_step)
        if not math.isfinite(loss_step):
            raise RuntimeError(f"Non-finite loss at step {step}")
        pbar.set_postfix({"kl": f"{loss_step:.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

        if step % args.eval_steps == 0:
            elapsed = time.perf_counter() - tic
            avg_loss = sum(losses[-args.eval_steps :]) / args.eval_steps
            # the same metric as the eval line below, but on the training batches (and averaged over the last
            # eval_steps optimizer steps, i.e. while the weights were still changing)
            print(
                f"\nStep {step}: KL(teacher || student) = {avg_loss:.4f} nats/token "
                f"(MetaMathQA training batches), {num_tokens / elapsed:.0f} tokens/s"
            )
            report_kl(f"Student at step {step}")
            report_accuracy(ds_valid, f"Student at step {step}")
            model.train()
            tic = time.perf_counter()
            num_tokens = 0

    # 3. evaluate, export the dense model, and save it
    model.eval()
    if args.adapter_output_dir:
        model.save_pretrained(args.adapter_output_dir)
        print(f"Saved the adapter to {args.adapter_output_dir}")
    report_kl("Student after distillation")
    if args.eval_test:
        print("Evaluating on test set, this can take some time...")
        with model.disable_adapter():
            report_accuracy(ds_test, "Teacher (MoE) on test set")
        report_accuracy(ds_test, "Student on test set")

    dense_model = model.compress_and_unload()
    if dense_dtype != dtype:
        dense_model.to(dtype)
    num_params = sum(param.numel() for param in dense_model.parameters())
    print(f"Dense model has {num_params / 1e9:.2f}B parameters")
    os.makedirs(args.output_dir, exist_ok=True)
    dense_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved the dense model to {args.output_dir}")

    # 4. compare inference speed and memory of the original MoE model and the dense model; both are loaded from scratch
    # after freeing the training model, so that they are measured under the same conditions
    if not args.skip_benchmark:
        prompts = ds_valid["input_ids"][:16]
        del model, dense_model, optimizer, lr_scheduler, trainable_params
        gc.collect()
        if device != "cpu":
            getattr(torch, device).empty_cache()
        results = run_benchmark(args, tokenizer, prompts, device, dtype)
        print("\n" + format_benchmark_table(results))


if __name__ == "__main__":
    main()
