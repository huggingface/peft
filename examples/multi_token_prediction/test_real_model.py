"""Test the ALoRA-MTP generate integration on the pretrained mtp_a3_seq384 model.

Reuses the prototype's AloraMTPModel / AloraMTPCandidateGenerator / wire_candidate_generator.
Loads the real PeftModel, runs greedy reference vs MTP-generate, measures draft acceptance.

Supports both no-sampler (base unembedding at mask positions) and sampler (trained MLP head).
"""

import argparse
import json
import time
import torch
import safetensors.torch
import packaging.version

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from mtp_transformers import (
    AloraMTPModel, AloraMTPCandidateGenerator, SamplerModule, wire_candidate_generator,
)


if packaging.version.parse(transformers.__version__) < packaging.version.parse("5.16.0"):
    raise RuntimeError("needs transformers >= 5.16")


def load_model(model_path, base_model_override=None, device="cpu", dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open(f"{model_path}/adapter_config.json") as f:
        adapter_config = json.load(f)
    base_name = base_model_override or adapter_config["base_model_name_or_path"]
    mask_ids = adapter_config["alora_invocation_tokens"]
    K = len(mask_ids)

    print(f"Loading base model: {base_name} ({dtype}, {device})")
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
    base.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings to {len(tokenizer)} (K={K} mask tokens)")

    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    model.__class__ = AloraMTPModel
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, mask_ids, K


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
    model.generation_config.use_alora_mtp = False
    t0 = time.perf_counter()
    out = model.generate(input_ids=input_ids.to(device), do_sample=False, use_cache=True,
                         max_new_tokens=max_new_tokens, pad_token_id=model.generation_config.pad_token_id)
    return out, time.perf_counter() - t0


def mtp_gen(model, input_ids, max_new_tokens, device):
    model.generation_config.use_alora_mtp = True
    model.generation_config.use_mtp = True
    t0 = time.perf_counter()
    out = model.generate(input_ids=input_ids.to(device), do_sample=False, use_cache=True,
                         max_new_tokens=max_new_tokens, pad_token_id=model.generation_config.pad_token_id)
    model.generation_config.use_mtp = False
    model.generation_config.use_alora_mtp = False
    return out, time.perf_counter() - t0


WIKI_SAMPLES = [
    "- July 20 – Johannes Bohn, German physician (d. 1718)",
    "Note: As the New Zealand Bowling Association at this time consists largely of South Island clubs, the first truly \"national\" championships are not deemed to have begun until 1914.",
    "Robert Boulter is an English film , television and theatre actor . He had a guest starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre .",
    "In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke .",
    "The history of artificial intelligence began in antiquity , with myths and stories of artificial beings endowed with intelligence . The field of modern AI research was founded at a conference at Dartmouth College in 1956 , where the term was coined by John McCarthy .",
    "Artificial intelligence ( AI ) is the intelligence of machines or software , as opposed to the intelligence of human beings or animals . It is a field of study in computer science that develops and studies intelligent machines . Such machines may be called AIs .",
    "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data . It is seen as a part of artificial intelligence .",
    "Deep learning is a subset of machine learning that focuses on algorithms inspired by the structure and function of the human brain called artificial neural networks . The term deep refers to the number of hidden layers in the neural network .",
    "Natural language processing ( NLP ) is an interdisciplinary subfield of computer science and linguistics . It is primarily concerned with giving computers the ability to support and manipulate human language .",
    "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world . Using digital images from cameras and videos and deep learning models , machines can accurately identify and classify objects .",
    "Reinforcement learning is a machine learning approach where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards . It is one of three basic machine learning paradigms .",
    "A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain . It is a type of machine learning process called deep learning that uses interconnected nodes or neurons .",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="mtp_a3_seq384")
    ap.add_argument("--base_model", default=None, help="override base model path/id")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use_sampler", action="store_true", help="use trained sampler head instead of base unembedding")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model, tokenizer, mask_ids, K = load_model(args.model_path, args.base_model, args.device, dtype)
    model = model.to(args.device)

    sampler = None
    if args.use_sampler:
        hidden_size = model.config.hidden_size
        sampler = load_sampler(args.model_path, model, hidden_size, args.device)
        print(f"Loaded sampler head (hidden_size={hidden_size})")
    wire_candidate_generator(model, sampler=sampler, mask_token_ids=mask_ids, use_sampler=args.use_sampler)

    total_accepted = 0
    total_tokens = 0
    total_fwd = 0
    all_match = True

    for i, text in enumerate(WIKI_SAMPLES[:args.num_samples]):
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = ids[:, :args.prompt_len]
        print(f"\n--- sample {i} (prompt {ids.shape[1]} tok) ---")

        ref, t_ref = greedy_ref(model, ids.clone(), args.max_new_tokens, args.device)
        out, t_mtp = mtp_gen(model, ids.clone(), args.max_new_tokens, args.device)

        ref_new = ref[0, ids.shape[1]:].tolist()
        out_new = out[0, ids.shape[1]:].tolist()
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
        print(f"  lossless: {match} | accepted {n_acc}/{n_tok} ({n_acc/max(n_tok,1)*100:.0f}%) | "
              f"fwd: {len(nm)} | n_matches: {nm} | t_ref {t_ref:.1f}s t_mtp {t_mtp:.1f}s")

    mode = "sampler" if args.use_sampler else "no-sampler"
    print(f"\n=== Summary ({args.num_samples} samples, K={K}, {mode}, {args.dtype}) ===")
    print(f"  Overall acceptance: {total_accepted}/{total_tokens} ({total_accepted/max(total_tokens,1)*100:.0f}%)")
    print(f"  All lossless: {all_match}")
    if not all_match and dtype == torch.bfloat16:
        print(f"  NOTE: lossless=False is expected with bf16 (batched vs single-token numerical diff)")
        print(f"        float32 is lossless — confirmed by test_precision.py")
    print(f"  Target: ~30% acceptance on wikipedia")


if __name__ == "__main__":
    main()
