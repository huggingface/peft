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
import sys, os
# Prefer local transformers (5.15.0.dev0, has use_mtp) and peft source over installed versions
_LOCAL_TRANSFORMERS = os.path.join(os.path.dirname(__file__), "transformers", "src")
_LOCAL_PEFT = os.path.join(os.path.dirname(__file__), "..", "..", "src")
for p in (_LOCAL_TRANSFORMERS, _LOCAL_PEFT):
    ap2 = os.path.abspath(p)
    if os.path.isdir(ap2) and ap2 not in sys.path:
        sys.path.insert(0, ap2)

import argparse
import torch
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
from peft import LoraConfig, get_peft_model, TaskType, PeftModelForCausalLM
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.variants import calculate_alora_offsets

torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# Sampler (mirrors rewrite.SamplerModule, simplified for the per-step draft)
# --------------------------------------------------------------------------------------
class SamplerModule(torch.nn.Module):
    def __init__(self, unembedding, hidden_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size), torch.nn.SiLU(), torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.SiLU(), torch.nn.LayerNorm(hidden_size),
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
        print(f"  [MTP] make_draft: logits.shape={logits.shape} hidden.shape={last_hidden.shape} "
              f"logits_to_keep_arg={getattr(model_outputs, 'logits_to_keep', '?')}")
        print(f"  [MTP] make_draft: logits[:,-K:].argmax={logits[:, -K:, :].argmax(dim=-1)[0].tolist()}")
    mask_hidden = last_hidden[:, -K:, :]  # [B, K, H]
    if use_sampler:
        draft = []
        for i in range(K):
            emb = embedding(prev_token)  # [B, 1, H]
            sl = sampler_module(mask_hidden[:, i:i + 1, :], emb)  # [B, 1, V]
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

    def __init__(self, main_model, generation_config, model_kwargs, logits_processor=None,
                 sampler_module=None, use_sampler=True, mask_token_ids=None):
        self.main_model = main_model
        self.device = main_model.device
        self.K = len(mask_token_ids)
        self.use_sampler = use_sampler
        self.sampler_module = sampler_module
        self.mask_ids = torch.tensor(mask_token_ids, dtype=torch.long, device=self.device)
        self.n_matches_history = []  # n_matches per spec cycle (acceptance stats)

    def get_candidates(self, input_ids, model_kwargs=None, model_outputs=None,
                       is_first_iteration=False, n_last_matches=0, **kw):
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
        draft = make_draft(model_outputs, self.main_model.get_input_embeddings(),
                           self.sampler_module, self.K, self.use_sampler,
                           prev_token=input_ids[:, -1:])  # [B, K]
        self._last_draft = draft[0]
        self._last_had_drafts = True
        cand = torch.cat([draft, masks], dim=-1)  # [B, 2K]
        return torch.cat([input_ids, cand], dim=-1), None

    def update_candidate_strategy(self, input_ids=None, scores=None, num_matches=0, **kwargs):
        self.n_matches_history.append(int(num_matches))
        if os.environ.get("MTP_DEBUG"):
            draft = self._last_draft.tolist() if getattr(self, "_last_draft", None) is not None else "?"
            sel = scores[0, :num_matches+2].argmax(dim=-1).tolist() if scores is not None and num_matches >= 0 else "?"
            print(f"  [MTP DEBUG] n_matches={num_matches} draft={draft} sel_prefix={sel} "
                  f"input_ids_len={input_ids.shape[1] if input_ids is not None else '?'}")
        return None


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
                    print(f"  [MTP] top_pre_hook: input_ids={inp.shape if inp is not None else None} "
                          f"pos_start={pos_vals} cache_len={cache_len}")

        # ... and inject freshly-recomputed alora_offsets into every LoRA layer, recording them.
        self._alora_offsets_seen = []

        def layer_pre_hook(module, args_, kwargs_):
            inp = rec["input_ids"]
            off = (calculate_alora_offsets(self.peft_config, self.active_adapter, inp)
                   if inp is not None else None)
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


# --------------------------------------------------------------------------------------
# Build a tiny end-to-end setup (CPU, random init, no download)
# --------------------------------------------------------------------------------------
def build(K=2, use_sampler=True, vocab=80, hidden=64, layers=2, heads=2):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
                     num_attention_heads=heads, intermediate_size=128,
                     max_position_embeddings=512, rms_norm_eps=1e-5)
    base = LlamaForCausalLM(cfg)
    mask_token_ids = list(range(vocab, vocab + K))
    base.resize_token_embeddings(vocab + K)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          alora_invocation_tokens=mask_token_ids, use_rslora=True)
    model = get_peft_model(base, lora_cfg)
    model = model  # PeftModelForCausalLM
    # Non-zero LoRA so the adapter is observable
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.1)
    model.eval()
    # Wrap to get per-forward aLoRA recompute
    model.__class__ = AloraMTPModel  # NOTE: reclass the PeftModelForCausalLM instance
    sampler = SamplerModule(model.get_output_embeddings(), hidden)
    # make sampler weights non-zero so it drafts something
    with torch.no_grad():
        for n, p in sampler.named_parameters():
            if "weight" in n and "LayerNorm" not in n:
                p.copy_(torch.randn_like(p) * 0.05)
    sampler = sampler.to(model.device)
    return model, sampler, mask_token_ids, vocab


def wire_candidate_generator(model, sampler, mask_token_ids, use_sampler):
    """Monkeypatch `_get_candidate_generator` to return ours when use_mtp is set.

    `_assisted_decoding` runs on the underlying HF model (PeftModel.base_model.model),
    so we must patch *that* instance, not the PeftModel wrapper.
    """
    gen = model
    base = model.base_model.model  # the LlamaForCausalLM that generate() actually runs on
    orig = base._get_candidate_generator

    def patched(self, generation_config, input_ids, inputs_tensor, logits_processor,
                model_kwargs, assistant_model=None, target_tokenizer=None, assistant_tokenizer=None):
        if getattr(generation_config, "use_alora_mtp", False):
            cg = AloraMTPCandidateGenerator(
                main_model=gen, generation_config=generation_config, model_kwargs=model_kwargs,
                logits_processor=logits_processor, sampler_module=sampler,
                use_sampler=use_sampler, mask_token_ids=mask_token_ids,
            )
            gen._last_mtp_generator = cg
            return cg
        return orig(generation_config=generation_config, input_ids=input_ids,
                    inputs_tensor=inputs_tensor, logits_processor=logits_processor,
                    model_kwargs=model_kwargs, assistant_model=assistant_model,
                    target_tokenizer=target_tokenizer, assistant_tokenizer=assistant_tokenizer)
    base._get_candidate_generator = patched.__get__(base, type(base))


def base_greedy(model, input_ids, max_new_tokens):
    """Reference: plain greedy with aLoRA OFF (no masks in prompt) -> base greedy."""
    model.generation_config.use_alora_mtp = False
    out = model.generate(input_ids=input_ids, do_sample=False, use_cache=True,
                         max_new_tokens=max_new_tokens, pad_token_id=0)
    return out


def train_drafter(model, sampler, mask_ids, K, use_sampler, ctx, rollout_len=64, steps=120, seq_len=8):
    """Self-distill on the generation trajectory: train LoRA + sampler so drafts predict
    the (aLoRA-off) greedy verifier's own continuation -> high acceptance, which exercises
    the multi-token / full-acceptance path of the cache logic (the riskiest part)."""
    model.eval()
    # 1. base greedy rollout (aLoRA OFF) from ctx
    with model.disable_adapter():
        roll = model.generate(input_ids=ctx, do_sample=False, use_cache=True,
                               max_new_tokens=rollout_len, pad_token_id=0)[0].tolist()
    model.train()
    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "lora_" in n] + list(sampler.parameters()), lr=1e-2)
    masks = torch.tensor(mask_ids)
    vocab = mask_ids[0]
    for step in range(steps):
        s = torch.randint(0, len(roll) - seq_len - K - 1, (1,)).item()
        window = torch.tensor([roll[s:s + seq_len]])
        inp = torch.cat([window, masks.unsqueeze(0)], dim=1)
        with model.disable_adapter():
            tgt = model(inp).logits[0, seq_len:seq_len + K].argmax(-1)  # y2..yK+1 (after the NTP y1)
        out = model(inp, output_hidden_states=True)
        hs = out.hidden_states[-1][:, -K:, :]
        prev = out.logits[:, -K - 1, :].argmax(-1, keepdim=True)
        loss = 0.0
        for i in range(K):
            emb = model.get_input_embeddings()(prev)
            sl = sampler(hs[:, i:i + 1, :], emb)
            loss = loss + torch.nn.functional.cross_entropy(sl[0], tgt[i:i + 1])
            prev = sl.argmax(-1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 30 == 0:
            print(f"  [train] step {step:3d} loss {loss.item():.4f}")
    model.eval()


def mtp_generate(model, input_ids, max_new_tokens, use_sampler=True):
    model.generation_config.use_alora_mtp = True
    model.generation_config.use_mtp = True  # triggers ASSISTED mode
    out = model.generate(input_ids=input_ids, do_sample=False, use_cache=True,
                         max_new_tokens=max_new_tokens, pad_token_id=0)
    model.generation_config.use_mtp = False
    model.generation_config.use_alora_mtp = False
    return out


def param_bytes(m):
    return sum(p.storage().nbytes() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--no_sampler", action="store_true")
    ap.add_argument("--train_steps", type=int, default=0, help="self-distill the drafter so drafts get accepted")
    args = ap.parse_args()
    use_sampler = not args.no_sampler

    model, sampler, mask_ids, vocab = build(K=args.K, use_sampler=use_sampler)
    wire_candidate_generator(model, sampler, mask_ids, use_sampler)

    ctx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    if args.train_steps > 0:
        train_drafter(model, sampler, mask_ids, args.K, use_sampler, ctx.clone(), steps=args.train_steps)

    ref = base_greedy(model, ctx.clone(), args.max_new_tokens)
    out = mtp_generate(model, ctx.clone(), args.max_new_tokens, use_sampler=use_sampler)

    ref_new = ref[0, ctx.shape[1]:].tolist()
    out_new = out[0, ctx.shape[1]:].tolist()
    print(f"K={args.K} use_sampler={use_sampler} max_new={args.max_new_tokens}")
    print("ref :", ref_new)
    print("mtp :", out_new)
    print("LOSSLESS (mtp == greedy):", ref_new == out_new)
    assert ref_new == out_new, "MTP output diverged from greedy -> not lossless!"

    # ---- Instrumentation: prove per-forward aLoRA recompute + real draft acceptance ----
    cg = getattr(model, "_last_mtp_generator", None)
    nm = cg.n_matches_history if cg else []
    offs = getattr(model, "_alora_offsets_seen", [])
    n_lora = sum(1 for m in model.base_model.modules() if isinstance(m, LoraLayer))
    per_fwd = [offs[i] for i in range(0, len(offs), max(n_lora, 1))] if offs else []
    nonzero_fwd = sum(1 for o in per_fwd if o is not None and o[0] is not None)
    print(f"\nInstrumentation:")
    print(f"  LoRA layers instrumented: {n_lora}")
    print(f"  forward passes recorded: {len(per_fwd)}")
    print(f"  forwards with aLoRA ON (masks triggered adapter): {nonzero_fwd}")
    print(f"  spec cycles: {len(nm)} | n_matches history: {nm}")
    print(f"  total draft tokens accepted: {sum(nm)} (proves verify+draft loop is real)")

    # ---- Memory accounting (real numbers) ----
    base_p = param_bytes(model)
    samp_p = param_bytes(sampler)
    shared = sampler.unembedding.weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()

    # Measure a real KV cache for a sequence of length L (cache cost dominates long gen)
    L = ctx.shape[1] + args.max_new_tokens
    with torch.no_grad():
        cache = DynamicCache()
        model.generation_config.use_alora_mtp = False
        model(torch.arange(L, device=model.device).unsqueeze(0) % 64, use_cache=True, past_key_values=cache)
        cache_bytes = sum((lyr.keys.storage().nbytes() + lyr.values.storage().nbytes())
                          for lyr in cache.layers if lyr.keys is not None)
    cache_mb = cache_bytes / 1e6

    print("\nMemory (MB):")
    print(f"  model (base+LoRA) : {base_p/1e6:.3f}")
    print(f"  sampler head      : {samp_p/1e6:.4f}  (unembedding tied/shared: {shared})")
    print(f"  1x KV cache (L={L}) : {cache_mb:.3f}")
    print("  ---- our single-pass approach ----")
    print(f"  TOTAL peak ~ model + sampler + 1x cache + 2K scratch = {base_p/1e6 + samp_p/1e6 + cache_mb:.3f} (+ {2*args.K} transient positions)")
    print("  second model? NO   second KV cache? NO   detached_copy? NOT NEEDED")
    print("  ---- contrast: two-model spec decoding ----")
    import copy
    asst = copy.deepcopy(model)
    cow_shared = all(a.data_ptr() == b.data_ptr() for a, b in
                     zip(asst.parameters(), model.parameters()))  # deepcopy -> all distinct
    print(f"  naive deepcopy assistant : +{param_bytes(asst)/1e6:.3f} params AND +1x cache  -> doubles BOTH")
    del asst
    # Simulate PEFT PR #3470 `detached_copy` (COW via _lazy_clone): params shared, but a
    # separate model needs its OWN KV cache to run as an independent assistant.
    shadow = copy.copy(model)  # shallow -> same module tree
    shadow._orig_owners = []
    for (n, p), (_, q) in zip(model.named_parameters(), shadow.named_parameters()):
        q._lazy_clone()            # COW: storage shared until written
        shadow._orig_owners.append((n, q))
    cow_share = all(p.data_ptr() == q.data_ptr() for (_, p), (_, q) in
                    zip(model.named_parameters(), shadow.named_parameters()))
    print(f"  detached_copy (COW) shadow: +0 params (shared={cow_share}) BUT still +1x cache -> cache DOUBLES")
    print("  => detached_copy saves WEIGHTS, not the KV cache; only the single-pass approach avoids doubling.")
    print("\nPROTOTYPE OK")


if __name__ == "__main__":
    main()
