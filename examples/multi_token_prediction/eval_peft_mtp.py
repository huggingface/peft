"""
Prototype: integrate ALoRA-MTP (rewrite.py / inference_mtp.py) into transformers
`model.generate()` using the existing `_assisted_decoding` loop + Cache.

Design:
- ONE PeftModel (base + aLoRA adapters) + a tiny Sampler head. No second model.
- Drafts come from the *same* forward pass that verifies: mask tokens are appended to
  the candidate, aLoRA turns ON only on those trailing mask positions, the base (aLoRA-off)
  logits verify the previous draft, and the mask-position hidden states feed the Sampler
  to produce the next draft.
- The KV cache is the standard DynamicCache. The mask positions' (aLoRA-on) K/V are exactly
  the trailing tokens that `_assisted_decoding` crops every step, so the persistent cache
  holds only aLoRA-off (base) K/V -> correct, no recompute, no doubling.

Memory: 1 model + 1 cache + tiny sampler. No `detached_copy` / second assistant needed.
"""

import argparse
import json
import os
import time
import warnings

import packaging.version
import safetensors
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel, PeftModelForCausalLM
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.variants import calculate_alora_offsets


if packaging.version.parse(transformers.__version__) <= packaging.version.parse("5.15.0"):
    raise RuntimeError("needs transformers > 5.15")

torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# Sampler (mirrors rewrite.SamplerModule, simplified for the per-step draft)
# --------------------------------------------------------------------------------------
class SamplerModule(torch.nn.Module):
    def __init__(self, unembedding, hidden_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden_size),
        )
        self.unembedding = unembedding
        torch.nn.init.zeros_(self.mlp[-3].weight)
        torch.nn.init.zeros_(self.mlp[-3].bias)

    def forward(self, hidden_states, prev_token_embs):
        combined = torch.cat([prev_token_embs, hidden_states], dim=-1)
        transformed = self.mlp(combined) + hidden_states
        return torch.nn.functional.linear(transformed, self.unembedding.weight)


def make_draft(model_outputs, embedding, sampler_module, K, use_sampler, prev_token):
    """Produce K draft tokens from the mask-position hidden states of the last forward.

    Masks are always the last K tokens of the computed window, so negative indexing is
    robust across prefill (full hidden) and decode (kept-logits) views.

    `prev_token` must be the last *accepted* token (input_ids[:, -1:]): the mask hidden
    at position p is trained to predict token p+1 given emb(token at p), so the sampler
    chain must start from the token sitting right before the first predicted position.
    (The old code used logits[:, 0].argmax, which in decode cycles is the *first* window
    position -- the verification of draft_1 -- not the last verified token.)
    """
    last_hidden = model_outputs.hidden_states[-1]  # [B, T, H]
    logits = model_outputs.logits  # [B, T', V] (T' = logits_to_keep window)
    if os.environ.get("MTP_DEBUG"):
        print(
            f"  [MTP] make_draft: logits.shape={logits.shape} hidden.shape={last_hidden.shape} "
            f"logits_to_keep_arg={getattr(model_outputs, 'logits_to_keep', '?')}"
        )
        print(f"  [MTP] make_draft: logits[:,-K:].argmax={logits[:, -K:, :].argmax(dim=-1)[0].tolist()}")
    mask_hidden = last_hidden[:, -K:, :]  # [B, K, H]
    if use_sampler:
        draft = []
        for i in range(K):
            emb = embedding(prev_token)  # [B, 1, H]
            sl = sampler_module(mask_hidden[:, i : i + 1, :], emb)  # [B, 1, V]
            prev_token = sl.argmax(dim=-1)  # [B, 1]
            draft.append(prev_token)
        return torch.cat(draft, dim=-1)  # [B, K]
    return logits[:, -K:, :].argmax(dim=-1)  # [B, K]


# --------------------------------------------------------------------------------------
# Candidate generator: plugs into transformers `_assisted_decoding`
# --------------------------------------------------------------------------------------
class AloraMTPCandidateGenerator:
    requires_model_outputs = True
    model_kwargs_overrides = {"output_hidden_states": True}

    def __init__(
        self,
        main_model,
        generation_config,
        model_kwargs,
        logits_processor=None,
        sampler_module=None,
        use_sampler=True,
        mask_token_ids=None,
    ):
        self.main_model = main_model
        self.device = main_model.device
        self.K = len(mask_token_ids)
        self.use_sampler = use_sampler
        self.sampler_module = sampler_module
        self.mask_ids = torch.tensor(mask_token_ids, dtype=torch.long, device=self.device)
        self.n_matches_history = []  # n_matches per spec cycle (acceptance stats)

    def get_candidates(
        self, input_ids, model_kwargs=None, model_outputs=None, is_first_iteration=False, n_last_matches=0, **kw
    ):
        masks = self.mask_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
        if is_first_iteration:
            # No draft yet: just append masks so the first (prefill) forward produces
            # mask hidden states. Verification will reject the masks (n_matches=0) and
            # keep only the NTP bonus token -- exactly inference_mtp.generate_linear iter 1.
            self._last_had_drafts = False
            return torch.cat([input_ids, masks], dim=-1), None
        if getattr(self, "_last_had_drafts", False) and n_last_matches < self.K:
            # Partial acceptance: the last forward's mask hidden states attended to
            # REJECTED draft tokens, so drafts made from them would be garbage. Emit a
            # masks-only candidate instead: the forward re-computes fresh mask hiddens on
            # top of verified tokens only (== inference_mtp.generate_linear's
            # `speculated = []` reset). Verification accepts just the NTP token here.
            self._last_had_drafts = False
            self._last_draft = None
            return torch.cat([input_ids, masks], dim=-1), None
        # Draft from the previous forward's mask hidden states, then re-append masks.
        # The previous candidate either was masks-only (fresh hiddens) or fully accepted
        # (hiddens attended to correct drafts only), so its mask hiddens are valid.
        draft = make_draft(
            model_outputs,
            self.main_model.get_input_embeddings(),
            self.sampler_module,
            self.K,
            self.use_sampler,
            prev_token=input_ids[:, -1:],
        )  # [B, K]
        self._last_draft = draft[0]
        self._last_had_drafts = True
        cand = torch.cat([draft, masks], dim=-1)  # [B, 2K]

        # Crop the draft after the first EOS token to avoid accidentally accepting the
        # tokens after the EOS. If the EOS token is wrongly predicted we'll waste one step but
        # that's better than generating potential garbage. We can crop the candidates (and therefore
        # the mask tokens) because either EOS is correct and we stop here or EOS is incorrect, in
        # which case _last_had_drafts == True and, by definition, n_last_matches < K since at least
        # the EOS did not match which will trigger the case above where we return input_ids + masks.
        eos_token_id = self.main_model.generation_config.eos_token_id
        if eos_token_id is not None:
            if draft.shape[0] > 1:
                warnings.warn("EOS handling is not supported for ALoRAMTP, there may be garbage output.")
            else:
                eos_positions = torch.isin(draft[0], torch.tensor(eos_token_id).to(draft)).nonzero()
                if eos_positions.numel() > 0:
                    num_drafted = eos_positions[0].item() + 1
                    cand = cand[0:1, :num_drafted]

        return torch.cat([input_ids, cand], dim=-1), None

    def update_candidate_strategy(self, input_ids=None, scores=None, num_matches=0, **kwargs):
        self.n_matches_history.append(int(num_matches))
        if os.environ.get("MTP_DEBUG"):
            draft = self._last_draft.tolist() if getattr(self, "_last_draft", None) is not None else "?"
            sel = (
                scores[0, : num_matches + 2].argmax(dim=-1).tolist()
                if scores is not None and num_matches >= 0
                else "?"
            )
            print(
                f"  [MTP DEBUG] n_matches={num_matches} draft={draft} sel_prefix={sel} "
                f"input_ids_len={input_ids.shape[1] if input_ids is not None else '?'}"
            )


# --------------------------------------------------------------------------------------
# PeftModel generate override: recompute aLoRA offsets every forward.
# PEFT's default `PeftModelForCausalLM.generate` freezes `alora_offsets` from the prompt
# for the whole generation (it computes them once, then strips them before base.generate).
# That is correct when invocation tokens live in the prompt, but for MTP the masks move
# every step, so offsets must be recomputed from the *current* (sliced) input_ids each
# forward. We install our own per-LoraLayer pre-hook that does exactly that.
# --------------------------------------------------------------------------------------
class AloraMTPModel(PeftModelForCausalLM):
    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        if getattr(peft_config, "alora_invocation_tokens", None) is None:
            return super().generate(*args, **kwargs)

        # Track the current input_ids of each forward (top-level). `_assisted_decoding`
        # calls `LlamaForCausalLM.forward` directly (generate runs on the underlying
        # model, not the LoraModel wrapper), so the hook must live on that model.
        gen_model = self.base_model.model if hasattr(self.base_model, "model") else self.base_model
        rec = {"input_ids": None, "_is_mtp": getattr(self, "_mtp_dbg_flag", False)}

        def top_pre_hook(module, args_, kwargs_):
            inp = kwargs_.get("input_ids")
            if inp is None and len(args_) > 0 and isinstance(args_[0], torch.Tensor):
                inp = args_[0]
            rec["input_ids"] = inp
            if os.environ.get("MTP_DEBUG") and rec.get("_is_mtp"):
                cnt = getattr(module, "_top_dbg", 0)
                if cnt < 3:
                    object.__setattr__(module, "_top_dbg", cnt + 1)
                    pos_ids = kwargs_.get("position_ids")
                    pos_vals = pos_ids[0, :5].tolist() if pos_ids is not None else None
                    cache = kwargs_.get("past_key_values")
                    cache_len = cache.get_seq_length() if cache is not None else "no cache"
                    print(
                        f"  [MTP] top_pre_hook: input_ids={inp.shape if inp is not None else None} "
                        f"pos_start={pos_vals} cache_len={cache_len}"
                    )

        # ... and inject freshly-recomputed alora_offsets into every LoRA layer, recording them.
        self._alora_offsets_seen = []

        def layer_pre_hook(module, args_, kwargs_):
            inp = rec["input_ids"]
            off = calculate_alora_offsets(self.peft_config, self.active_adapter, inp) if inp is not None else None
            kwargs_["alora_offsets"] = off
            self._alora_offsets_seen.append(off)
            if os.environ.get("MTP_DEBUG") and rec.get("_is_mtp") and getattr(module, "_lp_dbg", 0) < 5:
                object.__setattr__(module, "_lp_dbg", getattr(module, "_lp_dbg", 0) + 1)
                print(f"  [MTP] layer_pre_hook: input_ids={inp.shape if inp is not None else None} offsets={off}")

        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config

        handles = [gen_model.register_forward_pre_hook(top_pre_hook, with_kwargs=True)]
        for m in self.base_model.modules():
            if isinstance(m, LoraLayer):
                handles.append(m.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))
        try:
            kwargs.pop("alora_offsets", None)  # never use frozen offsets
            return self.base_model.generate(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation


def wire_candidate_generator(model, sampler, mask_token_ids, use_sampler):
    """Monkeypatch `_get_candidate_generator` to return ours when use_mtp is set.

    `_assisted_decoding` runs on the underlying HF model (PeftModel.base_model.model),
    so we must patch *that* instance, not the PeftModel wrapper.
    """
    gen = model
    base = model.base_model.model  # the LlamaForCausalLM that generate() actually runs on

    def patched(
        self,
        generation_config,
        input_ids,
        inputs_tensor,
        logits_processor,
        model_kwargs,
        assistant_model=None,
        target_tokenizer=None,
        assistant_tokenizer=None,
    ):
        cg = AloraMTPCandidateGenerator(
            main_model=gen,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            logits_processor=logits_processor,
            sampler_module=sampler,
            use_sampler=use_sampler,
            mask_token_ids=mask_token_ids,
        )
        gen._last_mtp_generator = cg
        return cg

    base._get_candidate_generator = patched.__get__(base, type(base))


def load_model(model_path: str, dtype=torch.bfloat16):
    """Load base model and apply trained LoRA adapter with mask tokens."""
    # Load tokenizer (has the mask tokens we added during training)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Read adapter config to get base model name
    with open(f"{model_path}/adapter_config.json") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    # Load base model (original Llama without LoRA)
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, device_map="auto")

    # Resize embeddings to match tokenizer (added mask tokens during training)
    base_model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA adapter from training
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    model.__class__ = AloraMTPModel
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Get mask token IDs
    mask_token_ids = model.peft_config["default"].alora_invocation_tokens

    print("Model loaded successfully")
    print(f"Mask token IDs: {mask_token_ids}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer, mask_token_ids, len(mask_token_ids)


def load_sampler(model_path, model, hidden_size, device):
    """Load the trained sampler head from sampler_model.safetensors."""
    sampler = SamplerModule(model.get_output_embeddings(), hidden_size)
    sd = safetensors.torch.load_file(f"{model_path}/sampler_model.safetensors")
    # Strip 'sampler.' prefix (keys are 'sampler.mlp.0.weight' etc.)
    sd = {k.removeprefix("sampler."): v for k, v in sd.items()}
    sampler.load_state_dict(sd, strict=False)
    sampler = sampler.to(device).to(model.dtype)
    sampler.eval()
    return sampler


def greedy_ref(model, input_ids, min_new_tokens, max_new_tokens, device):
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids.to(device),
        do_sample=False,
        use_cache=True,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
    )
    return out, time.perf_counter() - t0


def mtp_gen(model, input_ids, min_new_tokens, max_new_tokens, device):
    model.generation_config.use_mtp = True
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids.to(device),
        do_sample=False,
        use_cache=True,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
    )
    model.generation_config.use_mtp = False
    return out, time.perf_counter() - t0


def build_prompt_ids(row, tokenizer, prompt_len=None):
    """Tokenize the first user turn of a chat sample into prompt ids.

    Uses the tokenizer's chat template when one is defined, otherwise the raw user
    content (the MTP head was trained on raw text). Truncates to `prompt_len` ids when
    given, otherwise keeps the whole utterance. Returns None for rows without a
    usable user turn.
    """
    user = next(
        (m.get("content") for m in row["messages"] if m.get("role") == "user"), None
    )
    if user is None or not user.strip():
        return None
    if getattr(tokenizer, "chat_template", None):
        # return_dict=False: transformers >= 5 defaults to return_dict=True, which wraps
        # the ids in a BatchEncoding instead of returning a plain list of token ids.
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if isinstance(ids, dict):  # older transformers without the return_dict kwarg
            ids = ids["input_ids"]
    else:
        ids = tokenizer(user, add_special_tokens=True)["input_ids"]
    if prompt_len is not None:
        ids = ids[:prompt_len]
    return ids or None


def load_eval_rows(
    dataset_name, tokenizer, prompt_len=None, num_samples=None, per_source=None
):
    """Load the eval dataset and build prompts; returns (rows, skipped_per_source)."""
    ds = load_dataset(dataset_name, split="train")
    rows = []
    skipped = {}
    for row in ds:
        ids = build_prompt_ids(row, tokenizer, prompt_len)
        if ids is None:
            skipped[row["source"]] = skipped.get(row["source"], 0) + 1
            continue
        rows.append({"source": row["source"], "ids": ids})

    if per_source is not None:
        kept, counts = [], {}
        for r in rows:
            if counts.get(r["source"], 0) < per_source:
                kept.append(r)
                counts[r["source"]] = counts.get(r["source"], 0) + 1
        rows = kept
    if num_samples is not None:
        rows = rows[:num_samples]
    return rows, skipped


def format_summary_row(name, a):
    acc_pct = a["n_acc"] / max(a["n_tok"], 1) * 100
    tps_ref = a["n_tok"] / a["t_ref"] if a["t_ref"] > 0 else float("nan")
    tps_mtp = a["n_tok"] / a["t_mtp"] if a["t_mtp"] > 0 else float("nan")
    speedup = a["t_ref"] / a["t_mtp"] if a["t_mtp"] > 0 else float("nan")
    return (
        f"{name:<42} {a['n']:>4} {a['n_acc']:>5}/{a['n_tok']:<5} {acc_pct:>5.1f} "
        f"{tps_ref:>10.2f} {tps_mtp:>10.2f} {speedup:>7.2f}x {a['lossless']:>4}/{a['n']:<4}"
    )


def print_summary(records, K, args, skipped):
    mode = "sampler" if args.use_sampler else "no-sampler"
    keys = ("n", "n_tok", "n_acc", "fwd", "t_ref", "t_mtp", "lossless")

    order = []
    agg = {}
    for r in records:
        s = r["source"]
        if s not in agg:
            agg[s] = dict.fromkeys(keys, 0)
            order.append(s)
        a = agg[s]
        a["n"] += 1
        a["n_tok"] += r["n_tok"]
        a["n_acc"] += r["n_acc"]
        a["fwd"] += r["fwd"]
        a["t_ref"] += r["t_ref"]
        a["t_mtp"] += r["t_mtp"]
        a["lossless"] += int(r["lossless"])

    header = (
        f"{'source':<42.42} {'n':>4} {'accepted':>11} {'acc%':>6} {'tok/s ref':>10} "
        f"{'tok/s mtp':>10} {'speedup':>8} {'lossless':>9}"
    )
    lines = [
        "",
        f"=== Summary ({len(records)} samples, K={K}, {mode}, {args.dtype}, "
        f"max_new_tokens={args.max_new_tokens}, prompt_len={args.prompt_len}) ===",
        header,
        "-" * len(header),
    ]
    for s in order:
        lines.append(format_summary_row(s.removeprefix("ai2-adapt-dev/"), agg[s]))

    total = {k: sum(agg[s][k] for s in order) for k in keys}
    lines.append("-" * len(header))
    lines.append(format_summary_row("TOTAL", total))
    print("\n".join(lines))

    n_skipped = sum(skipped.values())
    if n_skipped:
        print(
            f"\n  Skipped {n_skipped} samples with empty/missing user turn: {skipped}"
        )
    if args.dtype == "bfloat16" and total["lossless"] < total["n"]:
        print(
            "  NOTE: lossless=False is expected with bf16 (batched vs single-token numerical diff)"
        )
        print("        float32 is lossless — confirmed by test_precision.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="mtp_a3_seq384")
    ap.add_argument("--dataset", default="hubnemo/tulu3-sft-mini")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument(
        "--prompt_len",
        type=int,
        default=None,
        help="truncate the prompt to N tokens (default: use the whole user utterance)",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="evaluate only the first N samples overall",
    )
    ap.add_argument(
        "--per_source",
        type=int,
        default=None,
        help="evaluate only the first N samples per source",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--use_sampler",
        action="store_true",
        help="use trained sampler head instead of base unembedding",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--verbose", action="store_true", help="print divergence details")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model, tokenizer, mask_ids, K = load_model(args.model_path, dtype)
    model = model.to(args.device)

    # make sure that we generate at least K tokens in every generationi so that every sample can
    # be used to determine how well the MTP portion models the base model - even when the base
    # model would generate EOS immediately. While this is certainly possible, it is unlikely to
    # be the case for instruction-trained models and also not the focus of this benchmark.
    min_new_tokens = K

    sampler = None
    if args.use_sampler:
        hidden_size = model.config.hidden_size
        sampler = load_sampler(args.model_path, model, hidden_size, args.device)
        print(f"Loaded sampler head (hidden_size={hidden_size})")
    wire_candidate_generator(
        model, sampler=sampler, mask_token_ids=mask_ids, use_sampler=args.use_sampler
    )

    rows, skipped = load_eval_rows(
        args.dataset, tokenizer, args.prompt_len, args.num_samples, args.per_source
    )
    print(f"Evaluating {len(rows)} samples from {args.dataset}")

    records = []
    for i, row in enumerate(rows):
        ids = torch.tensor([row["ids"]], dtype=torch.long)
        ref, t_ref = greedy_ref(model, ids.clone(), min_new_tokens, args.max_new_tokens, args.device)
        out, t_mtp = mtp_gen(model, ids.clone(), min_new_tokens, args.max_new_tokens, args.device)

        ref_new = ref[0, len(row["ids"]) :].tolist()
        out_new = out[0, len(row["ids"]) :].tolist()
        match = ref_new == out_new
        if args.verbose and not match:
            for j, (r, o) in enumerate(zip(ref_new, out_new)):
                if r != o:
                    print(
                        f"  DIVERGE at pos {j}: ref={tokenizer.decode([r])!r} mtp={tokenizer.decode([o])!r}"
                    )
                    break

        cg = getattr(model, "_last_mtp_generator", None)
        nm = cg.n_matches_history if cg else []
        n_acc = sum(nm)
        n_tok = len(out_new)
        records.append(
            {
                "source": row["source"],
                "n_tok": n_tok,
                "n_acc": n_acc,
                "fwd": len(nm),
                "t_ref": t_ref,
                "t_mtp": t_mtp,
                "lossless": match,
            }
        )

        src = row["source"].removeprefix("ai2-adapt-dev/")
        print(
            f"[{i + 1}/{len(rows)}] {src}: accepted {n_acc}/{n_tok} "
            f"({n_acc / max(n_tok, 1) * 100:4.0f}%) | fwd {len(nm)} | "
            f"t_ref {t_ref:6.1f}s t_mtp {t_mtp:6.1f}s | {t_ref / max(t_mtp, 1e-9):5.2f}x"
        )

    print_summary(records, K, args, skipped)


if __name__ == "__main__":
    main()
