"""
Inference script for Multi-Token Prediction with Linear Speculative Decoding.

Compares autoregressive baseline vs MTP speculative generation.
"""

import argparse
import json
import time
from copy import deepcopy

import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rewrite import Sampler



def load_model_and_tokenizer(model_path: str):
    """Load base model and apply trained LoRA adapter with mask tokens."""
    # Load tokenizer (has the mask tokens we added during training)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Read adapter config to get base model name
    with open(f"{model_path}/adapter_config.json", "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    # Load base model (original Llama without LoRA)
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16, device_map="auto")

    # Resize embeddings to match tokenizer (added mask tokens during training)
    base_model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA adapter from training
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Get mask token IDs
    K = len(model.peft_config['default'].alora_invocation_tokens)
    mask_token_ids = []
    mask_tokens = []
    for i in range(1, K + 1):
        mask_token = f"<mask_{i}>"
        token_id = tokenizer.convert_tokens_to_ids(mask_token)
        mask_token_ids.append(token_id)
        mask_tokens.append(mask_token)

    print("Model loaded successfully")
    print(f"Mask token IDs: {mask_token_ids}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer, mask_token_ids, mask_tokens


def load_sampler(model_path: str, model, hidden_size: int, num_mtp: int):
    """Load the trained sampler head."""
    sampler = Sampler(
        embedding=model.get_input_embeddings(),
        unembedding=model.get_output_embeddings(),
        hidden_size=hidden_size,
        num_mtp=num_mtp,
        skip_last=False,
    ).to(model.device)
    sampler_path = f"{model_path}/sampler_model.safetensors"

    # Load state dict, filtering out unembedding (it's shared, not part of SamplerHead)
    state_dict = safetensors.torch.load_file(sampler_path)

    for key in list(state_dict.keys()):
        if key.startswith("sampler.unembedding.") or key.startswith("embedding"):
            del state_dict[key]

    result = sampler.load_state_dict(state_dict, strict=False)

    # TODO can probably be removed with newer model artifacts
    for key in result.missing_keys:
        if not key.startswith("embedding.") and not key.startswith("sampler.unembedding."):
            raise ValueError(f"Oh no, {key} is missing :(")

    sampler.eval()

    return sampler


def generate_normal(model, tokenizer, samples, max_new_tokens=20, disable_eos=False):
    outputs = []
    benchmark = [{'time': None} for _ in range(len(samples))]

    for i_sample, sample in enumerate(samples):
        tic = time.perf_counter()
        output = []
        input_tokens = tokenizer(sample)['input_ids']

        while len(output) < max_new_tokens:
            inputs = {'input_ids': torch.tensor([input_tokens + output]).to(model.device)}
            model_outputs = model(**inputs)
            if disable_eos:
                model_outputs.logits[0, -1, tokenizer.eos_token_id] = -torch.inf
            output.append(model_outputs.logits[0, -1].argmax(dim=-1))
        toc = time.perf_counter()
        outputs.append(input_tokens + output)
        benchmark[i_sample]['time'] = toc - tic
    return outputs, benchmark


def generate_linear(model, tokenizer, sampler, mask_token_ids, mask_tokens, samples, max_new_tokens=20,
                    disable_eos=False):
    num_mtp = len(mask_token_ids)
    outputs = []
    bos_token = [tokenizer.bos_token_id]
    benchmark = [{'time': None, 'num_accepted': None, 'total_tokens': None} for _ in range(len(samples))]

    for i_sample, sample in enumerate(samples):
        tic = time.perf_counter()
        num_accepted_tokens = []

        # i_0:    i    can   have      m1       m2
        # o_0:  can   have     my     cat   stolen
        #         ^
        #           unverified_idx
        #
        # s_0:    i    can   have      my
        #
        # i_1:    i    can   have     my     cat   stolen        m1    m2
        # o_1:  can   have     my    cat  stolen     from       you     .
        #                                      ^ if i_1[unverified_idx] == o_1[unverified_idx - 1] we know that
        #                                        o_1[unverified_idx] is good! (for all k in [0..num_mtp])
        #                                      ^
        #                                        unverified_idx
        # s_1:    i    can   have     my      cat   stolen   from
        #                                           ^^^^^^^^^^^^^  both added if above holds, otherwise only 'cat'
        #
        # i_2:    i    can   have     my      cat   stolen    from    you   .    m1   m2
        # o_2:  can   have     my    cat   stolen     from     you      .   I  hate  you
        #                                                               ^
        #                                                                unverified_idx


        input_tokens = tokenizer(sample, add_special_tokens=False)['input_ids']
        sequence = bos_token + input_tokens
        speculated = []
        init_len = len(sequence)

        while len(sequence) - init_len < max_new_tokens:
            inputs = {'input_ids': torch.tensor([sequence + speculated + mask_token_ids]).to(model.device)}
            print("---------")
            print("INPUT:", tokenizer.decode(inputs['input_ids'][0]))
            print("INPUT_TOKENS:", inputs['input_ids'][0, init_len - 2*num_mtp:])

            model_outputs = model(**inputs, output_hidden_states=True)
            if disable_eos:
                model_outputs.logits[0, -2 * num_mtp + 1:, tokenizer.eos_token_id] = -torch.inf
            model_tokens = model_outputs.logits.argmax(dim=-1)
            print("OUTPUT_TOKENS:", model_tokens[0, init_len - 2*num_mtp:])

            if not speculated:
                # there are no tokens to verify yet, we just accept the next token prediction
                # and move on to the next iteration where we can verify the speculative tokens that we just added
                # via the mask tokens.
                # we mark the input of the model, except the MTP tokens, as verified.
                token_ntp = model_tokens[0, -(num_mtp+1)]  # [1]
                sequence.append(token_ntp.item())
                speculated = model_tokens[0, -num_mtp:].tolist()

                print("FIRST SEQUENCE, JUST ADDING")

                if sequence[-1] == tokenizer.eos_token_id:
                    break
                continue

            print("VERIFICATION POSSIBLE")

            # t1 t2 t3 u1 u2 m1 m2   .shape[-1] = 7   num_mtp=2  => unverified_idx = 7 - 4 = 3
            #  0  1  2  3  4  5  6
            #           ^
            unverified_idx = inputs['input_ids'].shape[-1] - 2 * num_mtp

            # handle the next token prediction token which is always safe as it comes from
            # the base model and not from the speculative gated LoRA portion.
            token_ntp = model_tokens[0, unverified_idx - 1]  # [1]
            sequence.append(token_ntp.item())
            if sequence[-1] == tokenizer.eos_token_id:
                break

            print(f"VERIFYING previous MTP tokens ({unverified_idx=}, {inputs['input_ids'].shape=}):", inputs['input_ids'][0, unverified_idx:unverified_idx+num_mtp])
            tokens_accepted = 0

            # the model output contains speculative tokens that we need to verify.
            # we have verified the outputs up to unverified_idx, so we know these are good.
            for i_k in range(num_mtp):
                token_ref = model_tokens[0, unverified_idx + i_k - 1]  # [1]
                token_mtp = model_tokens[0, unverified_idx + i_k]  # [1]
                input_ref = inputs['input_ids'][0, unverified_idx + i_k]  # [1]

                print(f"INDUCTION, {token_ref=}, {token_mtp=}, {input_ref=}")

                if token_ref == input_ref:
                    print(f"ACCEPT MTP TOKEN {i_k+1}/{num_mtp} ({token_mtp})")
                    sequence.append(token_mtp.item())
                    tokens_accepted += 1
                    if sequence[-1] == tokenizer.eos_token_id:
                        break
                else:
                    # verification failed, we must stop early and hope that the next run
                    # will be better.
                    break
            num_accepted_tokens.append(tokens_accepted)
            if sequence[-1] == tokenizer.eos_token_id:
                break

            # ok now we know that all the speculated tokens from the previous run were good so
            # we can add the newly speculated tokens to be verified in the next cycle.
            #
            # if we only have partial application of speculated tokens, we cannot use the newly
            # generated token of this run since they are based on false info. we need to generate
            # new ones the next cycle.
            if tokens_accepted == num_mtp:
                speculated = model_tokens[0, -num_mtp:].tolist()
            else:
                speculated = []

        toc = time.perf_counter()
        benchmark[i_sample]['time'] = toc - tic
        benchmark[i_sample]['num_accepted_tokens'] = sum(num_accepted_tokens)
        benchmark[i_sample]['total_tokens'] = len(sequence) - init_len
        outputs.append(sequence[:init_len + max_new_tokens])  # prevent exceeding max new tokens
    return outputs, benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="mtp_model")
    parser.add_argument('--text_file', type=str, default="train.txt")
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--chunk_length', type=int, default=120)
    args = parser.parse_args()

    model, tokenizer, mask_token_ids, mask_tokens = load_model_and_tokenizer(args.model_path)
    num_mtp = len(mask_token_ids)
    sampler = load_sampler(args.model_path, model, model.config.hidden_size, num_mtp)

    model = model.to(args.device)
    sampler = sampler.to(args.device)

    with open(args.text_file, "r") as f:
        text = f.read()

    chunks = text.split("\n\n")
    samples = [chunk.strip()[:args.chunk_length] for chunk in chunks if len(chunk.strip()) > 100][:args.num_samples]

    print(f"Loaded {len(samples)} samples from {args.text_file}")

    print("Normal auto-regressive decoding (no cache)")
    outputs_normal, benchmark_normal = generate_normal(model, tokenizer, samples, disable_eos=True)
    for sample in tokenizer.batch_decode(outputs_normal):
        print(sample)

    print("Linear decoding")
    outputs_linear, benchmark_linear = generate_linear(model, tokenizer, sampler, mask_token_ids, mask_tokens, samples,
                                                       disable_eos=True)
    for sample, benchmark_row in zip(tokenizer.batch_decode(outputs_linear), benchmark_linear):
        print('output:\n------')
        print(sample)
        print('---------------')
        print(benchmark_row)
        print('---------------')

    print()

    print("Making sure that the methods generated equal amounts of tokens.")
    for o_normal, o_linear in zip(outputs_normal, outputs_linear):
        print(f"{len(o_normal)=} vs. {len(o_linear)=}")
        assert len(o_normal) == len(o_linear)

    print()

    mtp_token_rates = []
    for b_normal, b_linear in zip(benchmark_normal, benchmark_linear):
        print(f"{b_normal['time']=} vs. {b_linear['time']=}")
        print(f"accepted tokens (linear): {b_linear['num_accepted_tokens']}")
        print(f"total tokens (linear): {b_linear['total_tokens']}")
        mtp_token_rates.append(b_linear['num_accepted_tokens'] / b_linear['total_tokens'])

    overall_mtp_rate = sum(mtp_token_rates) / len(mtp_token_rates)
    print(f"Overall MTP token rate: {overall_mtp_rate}")


if __name__ == "__main__":
    main()


if False:

    K = None  # determined on model load
    MAX_TOKENS = 100  # Tokens to generate per sample
    NUM_SAMPLES = 5  # Number of benchmark samples
    TEMPERATURE = 1.0

    MODEL_PATH = "./mtp_model"
    DATA_PATH = "./train.txt"




    def autoregressive_generate(
        model, tokenizer, prompt_ids: torch.Tensor, max_tokens: int, temperature: float = 1.0
    ) -> Tuple[List[int], float]:
        """Baseline: Generate tokens one at a time autoregressively."""
        device = model.device
        generated = prompt_ids.clone()

        start_time = time.time()

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = model(generated)
                logits = outputs.logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        elapsed = time.time() - start_time
        new_tokens = generated[0, len(prompt_ids[0]) :].tolist()

        return new_tokens, elapsed


    def speculative_generate(
        model,
        tokenizer,
        sampler_head,
        mask_token_ids: List[int],
        prompt_ids: torch.Tensor,
        max_tokens: int,
        k: int = 4,
        temperature: float = 1.0,
        use_sampler: bool = True,
    ) -> Tuple[List[int], float, dict]:
        """Speculative: Generate k tokens at once, verify, accept/resample.

        Per paper §2.4 the NTP token y_{n+1} comes from the base unembedding at z_n
        and is auto-verified; the sampler then produces y_{n+2}..y_{n+k+1} conditioned
        on emb(y_{n+1}) at step 0. Verification feeds [context, y_{n+1}, drafts] with
        no masks (ALoRA off → base only) and accepts drafts greedily, falling back to
        the base argmax on rejection. `temperature` is retained for API compatibility
        but unused while the draft is greedy (argmax), matching training.

        When `use_sampler=False` the draft is taken from the base unembedding at each
        mask position directly (the paper's no-sampler ablation, Fig 7/Fig 8 left),
        bypassing the sampler MLP. Comparing the two modes isolates whether the
        sampler or the mask hidden state is the bottleneck for draft acceptance.
        """
        device = model.device
        generated = prompt_ids.clone()
        total_tokens = 0
        total_accepted = 0
        total_draft_calls = 0
        get_emb = model.get_input_embeddings()

        start_time = time.time()

        with torch.no_grad():
            while total_tokens < max_tokens:
                current_ids = generated
                n = current_ids.shape[1]

                # Draft phase: forward [context, masks] → base logits + mask hiddens
                mask_ids = torch.tensor([mask_token_ids], device=device)
                draft_input = torch.cat([current_ids, mask_ids], dim=1)
                outputs = model(draft_input, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                # NTP token (auto-verified): base unembedding at the last context
                # position. ALoRA is off on context positions, so this is exactly the
                # standard autoregressive prediction.
                ntp_token = outputs.logits[:, n - 1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
                generated = torch.cat([generated, ntp_token], dim=1)
                total_tokens += 1
                if total_tokens >= max_tokens:
                    break

                # Sampler: y_{n+2}..y_{n+k+1} conditioned on emb(y_{n+1}) at step 0 and
                # on the sampler's own previous argmax afterwards (matches the
                # training-time conditioning in train_mtp.train_step). With
                # use_sampler=False the draft comes from the base unembedding at the
                # i-th mask position (outputs.logits[:, n+i, :]) instead.
                draft_tokens = []
                prev_emb = get_emb(ntp_token)  # [1, 1, H]
                hidden_for_sampler = hidden_states[:, n:, :]  # mask positions z_{n+1}..z_{n+k}
                for i in range(k):
                    if use_sampler:
                        h = hidden_for_sampler[:, i : i + 1, :]
                        sampler_logits = sampler_head(h, prev_emb).squeeze(1)  # [1, V]
                        draft_token = sampler_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
                    else:
                        draft_token = outputs.logits[:, n + i, :].argmax(dim=-1, keepdim=True)  # [1, 1]
                    draft_tokens.append(draft_token.item())
                    prev_emb = get_emb(draft_token)

                total_draft_calls += 1

                # Verification: [context, ntp_token, drafts] — no masks, ALoRA off, so
                # the base model produces the reference greedy distribution.
                draft_tensor = torch.tensor([draft_tokens], device=device)
                verify_input = torch.cat([current_ids, ntp_token, draft_tensor], dim=1)
                verify_logits = model(verify_input).logits

                # Sequential acceptance: verify_logits[:, n+j, :] predicts position
                # n+j+1 = draft_tokens[j], so pos = n + j (the old pos-1 was the bug).
                for j in range(k):
                    if total_tokens >= max_tokens:
                        break
                    pos = n + j
                    if pos >= verify_logits.shape[1]:
                        break
                    base_token = verify_logits[:, pos, :].argmax(dim=-1).item()
                    if base_token == draft_tokens[j]:
                        generated = torch.cat([generated, torch.tensor([[draft_tokens[j]]], device=device)], dim=1)
                        total_tokens += 1
                        total_accepted += 1
                    else:
                        generated = torch.cat([generated, torch.tensor([[base_token]], device=device)], dim=1)
                        total_tokens += 1
                        break

        elapsed = time.time() - start_time
        new_tokens = generated[0, len(prompt_ids[0]) :].tolist()

        stats = {
            "total_accepted": total_accepted,
            "total_draft_calls": total_draft_calls,
            "acceptance_rate": total_accepted / (total_draft_calls * k) if total_draft_calls > 0 else 0,
            "avg_accepted_per_call": total_accepted / total_draft_calls if total_draft_calls > 0 else 0,
        }

        return new_tokens, elapsed, stats


    def benchmark(model, tokenizer, sampler_head, mask_token_ids, text_samples: List[str]):
        """Run benchmark comparison on multiple samples."""
        results = []

        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: {len(text_samples)} samples, {MAX_TOKENS} tokens each")
        print(f"{'=' * 60}\n")

        for i, sample in enumerate(text_samples):
            prompt_text = sample[:50]
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

            print(f"Sample {i + 1}: '{prompt_text[:40]}...'")

            ar_tokens, ar_time = autoregressive_generate(model, tokenizer, prompt_ids, MAX_TOKENS)
            ar_speed = len(ar_tokens) / ar_time

            spec_tokens, spec_time, stats = speculative_generate(
                model, tokenizer, sampler_head, mask_token_ids, prompt_ids, MAX_TOKENS, K, TEMPERATURE
            )
            spec_speed = len(spec_tokens) / spec_time

            base_tokens, base_time, base_stats = speculative_generate(
                model, tokenizer, sampler_head, mask_token_ids, prompt_ids, MAX_TOKENS, K, TEMPERATURE,
                use_sampler=False,
            )
            base_speed = len(base_tokens) / base_time

            speedup = spec_speed / ar_speed if ar_time > 0 else 0

            result = {
                "sample": i + 1,
                "prompt": prompt_text[:40],
                "ar_tokens": len(ar_tokens),
                "ar_time": ar_time,
                "ar_speed": ar_speed,
                "spec_tokens": len(spec_tokens),
                "spec_time": spec_time,
                "spec_speed": spec_speed,
                "speedup": speedup,
                "acceptance_rate": stats["acceptance_rate"],
                "avg_accepted": stats["avg_accepted_per_call"],
                "base_acceptance_rate": base_stats["acceptance_rate"],
                "base_avg_accepted": base_stats["avg_accepted_per_call"],
            }
            results.append(result)

            print(f"  AR:      {len(ar_tokens)} tokens in {ar_time:.2f}s ({ar_speed:.1f} tok/s)")
            print(f"  Spec:    {len(spec_tokens)} tokens in {spec_time:.2f}s ({spec_speed:.1f} tok/s) | draft-match: {stats['acceptance_rate']:.2%}")
            print(f"  Base:    {len(base_tokens)} tokens in {base_time:.2f}s ({base_speed:.1f} tok/s) | draft-match: {base_stats['acceptance_rate']:.2%}")
            print(f"  Speedup (spec): {speedup:.2f}x")
            print()

        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        avg_acceptance = sum(r["acceptance_rate"] for r in results) / len(results)
        avg_base_acceptance = sum(r["base_acceptance_rate"] for r in results) / len(results)

        print(f"{'=' * 60}")
        print(f"SUMMARY")
        print(f"{'=' * 60}")
        print(f"Average Speedup (spec): {avg_speedup:.2f}x")
        print(f"Average Draft-Match (sampler): {avg_acceptance:.2%}")
        print(f"Average Draft-Match (base/no-sampler): {avg_base_acceptance:.2%}")
        print(f"{'=' * 60}\n")

        return results


    def __main():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('model_path', type=str, default=MODEL_PATH)
        args = parser.parse_args()

        model, tokenizer, mask_token_ids = load_model_and_tokenizer(args.model_path)

        hidden_size = model.config.hidden_size
        vocab_size = len(tokenizer)
        sampler_head = load_sampler_head(MODEL_PATH, hidden_size, vocab_size, model.device, model)

        with open(DATA_PATH, "r") as f:
            text = f.read()

        chunks = text.split("\n\n")
        samples = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100][:NUM_SAMPLES]

        print(f"Loaded {len(samples)} samples from {DATA_PATH}")

        results = benchmark(model, tokenizer, sampler_head, mask_token_ids, samples)

        with open("./benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to ./benchmark_results.json")


    if __name__ == "__main__":
        main()
