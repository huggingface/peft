"""
Offline generation of self-generated training data for MTP self-distillation (Medusa-2/EAGLE-style).

The frozen base model (no adapter) greedily continues finewiki prefixes. Training the MTP head on
the base model's own continuations aligns the draft distribution with the verification
distribution, which is what speculative acceptance measures — corpus targets instead grade the
head on text the base model would not have produced.

Each output row is one sample of `--target_len + --roundtrip_margin` tokens (BOS + prefix +
greedy continuation). train.py consumes the `text` column through tokenize_wiki unchanged: it
chunks at `--seq_len` and keeps only full chunks, so each row yields exactly one training chunk.
Rows that stop early on EOS are skipped (a short sample cannot form a full chunk) and counted.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="prefix source: hub id or local path (same value train.py --dataset would take)",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="optional data_files passthrough for load_dataset",
    )
    parser.add_argument("--num_samples", type=int, default=50_000)
    parser.add_argument(
        "--target_len",
        type=int,
        default=384,
        help="tokens per sample; must match train.py --seq_len so tokenize_wiki keeps the chunk",
    )
    parser.add_argument(
        "--roundtrip_margin",
        type=int,
        default=8,
        help="extra generated tokens to absorb decode->re-tokenize jitter in tokenize_wiki",
    )
    parser.add_argument("--prefix_min", type=int, default=32)
    parser.add_argument("--prefix_max", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=2000)
    parser.add_argument("--local_dir", type=str, default="data/selfgen")
    parser.add_argument(
        "--hub_id",
        type=str,
        default=None,
        help="push the finished dataset to this hub repo",
    )
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--validation_rows", type=int, default=100)
    return parser.parse_args()


def load_manifest(local_dir: Path) -> dict:
    manifest_path = local_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"docs_consumed": 0, "samples_written": 0, "eos_skipped": 0, "empty_docs": 0}


def save_manifest(local_dir: Path, manifest: dict):
    (local_dir / "manifest.json").write_text(json.dumps(manifest))


def main():
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(local_dir)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(
        args.device
    )
    model.eval()
    model.config.pad_token_id = eos_id
    torch.manual_seed(args.seed)

    if args.dataset is None:
        raise SystemExit(
            "--dataset is required (e.g. the finewiki hub id your train.py run used, or a local file)"
        )
    ds = load_dataset(
        args.dataset, split="train", streaming=True, data_files=args.data_files
    )
    if manifest["docs_consumed"]:
        print(f"Resuming: skipping {manifest['docs_consumed']} already-consumed docs")
        ds = ds.skip(manifest["docs_consumed"])
    stream = iter(ds)

    def next_doc():
        nonlocal manifest
        try:
            doc = next(stream)
            manifest["docs_consumed"] += 1
            return doc["text"]
        except StopIteration:
            return None

    rows = []
    samples_done = manifest["samples_written"]
    validation: list[tuple[list[int], str]] = []
    validation_stride = max(1, args.num_samples // max(args.validation_rows, 1))
    total_len = args.target_len + args.roundtrip_margin
    t0 = time.perf_counter()
    gen_tokens = 0
    batch_idx = 0

    while samples_done < args.num_samples:
        # one prefix length per batch: prompts differ only when a doc is shorter than it
        rng = np.random.default_rng(args.seed + batch_idx)
        prefix_len = int(rng.integers(args.prefix_min, args.prefix_max + 1))

        prompts = []
        while len(prompts) < args.batch_size:
            text = next_doc()
            if text is None:
                break
            # NOTE that we don't use apply_chat_template here even when the model does have a
            # chat template defined. This is not instruction fine-tuning and we assume that
            # teaching the MTP tokens to mimic model outputs is enough to transfer to
            # instruction fine-tuned model outputs as well.
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                manifest["empty_docs"] += 1
                continue
            if len(ids) < args.prefix_min:
                # too little context to be a meaningful prefix
                manifest["short_docs"] = manifest.get("short_docs", 0) + 1
                continue
            prompts.append(([bos_id] + ids[:prefix_len], len(ids[:prefix_len])))
        if not prompts:
            print("Input stream exhausted before reaching --num_samples")
            break

        # left-pad ragged prompts (short docs) to the batch max length
        prompt_lens = [len(p) for p, _ in prompts]
        max_prompt_len = max(prompt_lens)
        pad_column = [
            torch.full((max_prompt_len - length,), eos_id, dtype=torch.long)
            for length in prompt_lens
        ]
        prompt_ids = [p for p, _ in prompts]
        input_ids = torch.stack(
            [
                torch.cat([pad, torch.tensor(p, dtype=torch.long)])
                for pad, p in zip(pad_column, prompt_ids)
            ]
        ).to(args.device)
        attention_mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.zeros(len(pad), dtype=torch.long),
                        torch.ones(len(p), dtype=torch.long),
                    ]
                )
                for pad, p in zip(pad_column, prompt_ids)
            ]
        ).to(args.device)

        # short docs keep their smaller real prefix but get a larger continuation budget, so
        # every kept sample reaches exactly total_len + 1 tokens (BOS + total_len text tokens)
        budgets = [total_len + 1 - length for length in prompt_lens]
        max_new_tokens = max(budgets)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=eos_id,
            )
        gen_tokens += output.shape[0] * max_new_tokens

        for row, (_, p_len), length in zip(output.tolist(), prompts, prompt_lens):
            budget = total_len + 1 - length
            # early EOS within this row's own budget: sample is shorter than target_len,
            # tokenize_wiki would drop it (later EOS is truncated away and harmless)
            if eos_id in row[max_prompt_len : max_prompt_len + budget]:
                manifest["eos_skipped"] += 1
                continue
            # keep [BOS] + real prefix + continuation truncated to this row's own budget
            kept = row[max_prompt_len - length : max_prompt_len - length + total_len + 1]
            text = tokenizer.decode(kept[1:], skip_special_tokens=True)
            rows.append(
                {
                    "text": text,
                    "input_ids": kept,
                    "prefix_len": p_len,
                    "source": "finewiki_selfgen",
                }
            )
            if (
                samples_done % validation_stride == 0
                and len(validation) < args.validation_rows
            ):
                validation.append((kept[1:], text))
            samples_done += 1
        batch_idx += 1

        if len(rows) >= args.shard_size:
            shard = local_dir / f"shard-{manifest['samples_written']:06d}.parquet"
            Dataset.from_list(rows).to_parquet(str(shard))
            manifest["samples_written"] = samples_done
            save_manifest(local_dir, manifest)
            print(f"Wrote {shard.name} ({samples_done}/{args.num_samples} samples)")
            rows = []

        if batch_idx % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[{samples_done}/{args.num_samples}] eos_skipped={manifest['eos_skipped']} "
                f"gen={gen_tokens / max(elapsed, 1e-9):.0f} tok/s"
            )

    if rows:
        shard = local_dir / f"shard-{manifest['samples_written']:06d}.parquet"
        Dataset.from_list(rows).to_parquet(str(shard))
    manifest["samples_written"] = samples_done
    save_manifest(local_dir, manifest)

    # roundtrip validation: quantify the decode->re-tokenize fidelity of the text format
    if validation:
        exact = 0
        token_matches = 0
        token_total = 0
        for stored_ids, text in validation:
            re_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            exact += int(re_ids == stored_ids)
            token_matches += sum(a == b for a, b in zip(re_ids, stored_ids))
            token_total += max(len(re_ids), len(stored_ids))
        print(
            f"Roundtrip check on {len(validation)} rows: {exact}/{len(validation)} rows identical, "
            f"{token_matches}/{token_total} tokens identical ({token_matches / max(token_total, 1) * 100:.2f}%)"
        )

    print(
        f"Done: {samples_done} samples, {manifest['eos_skipped']} early-EOS skips, "
        f"{manifest.get('short_docs', 0)} short docs, {manifest['empty_docs']} empty docs, "
        f"{gen_tokens} tokens generated"
    )

    if args.hub_id:
        shards = sorted(str(p) for p in local_dir.glob("shard-*.parquet"))
        dataset = load_dataset("parquet", data_files=shards, split="train")
        dataset.push_to_hub(args.hub_id, private=args.private)
        print(f"Pushed {len(dataset)} rows to {args.hub_id}")


if __name__ == "__main__":
    main()
