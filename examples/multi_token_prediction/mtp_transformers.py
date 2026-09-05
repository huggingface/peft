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

import packaging.version
import safetensors
import torch
import transformers
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
                    pos_end = pos_ids[0, -5:].tolist() if pos_ids is not None else None
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


def greedy_ref(model, input_ids, max_new_tokens, device):
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids.to(device),
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
    )
    return out, time.perf_counter() - t0


def mtp_gen(model, input_ids, max_new_tokens, device):
    model.generation_config.use_mtp = True
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids.to(device),
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
    )
    model.generation_config.use_mtp = False
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="mtp_a3_seq384")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--samples", type=str, default="infer.txt")
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use_sampler", action="store_true", help="use trained sampler head instead of base unembedding")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model, tokenizer, mask_ids, K = load_model(args.model_path, dtype)
    model = model.to(args.device)

    sampler = None
    if args.use_sampler:
        hidden_size = model.config.hidden_size
        sampler = load_sampler(args.model_path, model, hidden_size, args.device)
        print(f"Loaded sampler head (hidden_size={hidden_size})")
    wire_candidate_generator(model, sampler=sampler, mask_token_ids=mask_ids, use_sampler=args.use_sampler)

    with open(args.samples) as f:
        samples = f.read().split("\n\n")

    total_accepted = 0
    total_tokens = 0
    total_fwd = 0
    all_match = True

    for i, text in enumerate(samples[: args.num_samples]):
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = ids[:, : args.prompt_len]
        print(f"\n--- sample {i} (prompt {ids.shape[1]} tok) ---")

        ref, t_ref = greedy_ref(model, ids.clone(), args.max_new_tokens, args.device)
        out, t_mtp = mtp_gen(model, ids.clone(), args.max_new_tokens, args.device)

        ref_new = ref[0, ids.shape[1] :].tolist()
        out_new = out[0, ids.shape[1] :].tolist()
        match = ref_new == out_new
        all_match &= match
        if not match:
            for j, (r, o) in enumerate(zip(ref_new, out_new)):
                if r != o:
                    print(f"  DIVERGE at pos {j}: ref={tokenizer.decode([r])!r} mtp={tokenizer.decode([o])!r}")
                    break

        cg = getattr(model, "_last_mtp_generator", None)
        nm = cg.n_matches_history if cg else []
        n_acc = sum(nm)
        n_tok = len(out_new)
        total_accepted += n_acc
        total_tokens += n_tok
        total_fwd += len(nm)

        print(f"  ref : {tokenizer.decode(ref_new, skip_special_tokens=True)[:120]}")
        print(f"  mtp : {tokenizer.decode(out_new, skip_special_tokens=True)[:120]}")
        print(
            f"  lossless: {match} | accepted {n_acc}/{n_tok} ({n_acc / max(n_tok, 1) * 100:.0f}%) | "
            f"fwd: {len(nm)} | n_matches: {nm} | t_ref {t_ref:.1f}s t_mtp {t_mtp:.1f}s"
        )

    mode = "sampler" if args.use_sampler else "no-sampler"
    print(f"\n=== Summary ({args.num_samples} samples, K={K}, {mode}, {args.dtype}) ===")
    print(
        f"  Overall acceptance: {total_accepted}/{total_tokens} ({total_accepted / max(total_tokens, 1) * 100:.0f}%)"
    )
    print(f"  All lossless: {all_match}")
    if not all_match and dtype == torch.bfloat16:
        print("  NOTE: lossless=False is expected with bf16 (batched vs single-token numerical diff)")
        print("        float32 is lossless — confirmed by test_precision.py")
    print("  Target: ~30% acceptance on wikipedia")


if __name__ == "__main__":
    main()
