# MoE-to-dense: pruning and distilling a Mixture-of-Experts model into a dense model

This example converts a Mixture-of-Experts (MoE) model into a dense model using PEFT's `MoeToDenseConfig`, following [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://arxiv.org/abs/2605.28207) (Kim et al., 2026). The idea:

1. **Score and select experts.** During forward passes on calibration data, routing statistics are collected for each MoE layer. Experts are scored by their *conditional probability* (the average routing probability over the tokens for which they were selected) and the top-k experts are kept (k = the router's top-k, so the dense model has as many parameters as the MoE model has *active* parameters).
2. **Concatenate.** The kept experts are concatenated into a single dense FFN per layer, with the down projections scaled by 1/k. This preserves the experts' intermediate activations exactly; only the token-dependent routing weights are replaced by a constant.
3. **Distill.** The dense student is trained to match the logits of the MoE teacher (forward KL divergence). Teacher and student share all parameters except for the FFNs, so the teacher forward pass is simply the forward pass with disabled adapters.

Finally, `compress_and_unload()` replaces the MoE layers by the dense FFNs and adjusts the model config, so that the result is a regular transformers model that can be saved and loaded without PEFT. For Qwen3-MoE, the result is a true dense model (`mlp_only_layers` covers all layers); for architectures without a dense MLP class (e.g. GPT-OSS, Gemma 4), the MoE layers are kept with a single expert and a router that always selects it, which adds a little routing overhead at inference time (use `torch.compile` to remove it).

Unlike other PEFT methods, the trainable "adapter" is large: it is the full set of dense FFNs, i.e. the active parameters of the MoE model (~1.8B parameters for Qwen3-30B-A3B). The compression comes from dropping the inactive experts (~27B parameters for Qwen3-30B-A3B).

## The minimal workflow

```python
import torch
from transformers import AutoModelForCausalLM
from peft import MoeToDenseConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, device_map="cuda")
model = get_peft_model(model, MoeToDenseConfig())

# 1. collect routing statistics, then build the dense FFNs from the most important experts
with torch.no_grad():
    for batch in calibration_batches:
        model(**batch)
model.update_and_allocate()

# 2. distill the MoE teacher (= adapters disabled) into the dense student
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
for batch in train_batches:
    loss = model.get_distillation_loss(**batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 3. export a standalone dense model
dense_model = model.compress_and_unload()
dense_model.save_pretrained("qwen3-30b-a3b-dense")
```

The adapter (i.e. the dense FFNs) can be saved and loaded like any other PEFT adapter via `save_pretrained` and `PeftModel.from_pretrained`; a loaded adapter does not need calibration.

## Running the example

The script distills on [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) and evaluates the GSM8K accuracy of the teacher, the student before distillation, and the student during distillation, reusing the data pipeline of the [MetaMathQA benchmark](../../method_comparison/MetaMathQA):

```sh
python distill_moe_to_dense.py --model_id ibm-granite/granite-3.1-1b-a400m-instruct --output_dir ./granite-1b-10k --max_steps 10000 --calibration_steps 50 --eval_steps 500 --batch_size_eval 25 --eval_test --num_experts_to_keep 4
```

A worked example is prepended to every prompt (one-shot) so that teacher and student produce the answer format that the GSM8K evaluation can parse (`The answer is: <number>`); the prefix is excluded from the distillation loss. Without it, the accuracy of most models is 0 regardless of their math ability (pass `--no_few_shot` to disable it). Run once without `--skip_initial_eval` to see the accuracy of the teacher, which is the ceiling for the student.

Two metrics are reported: the GSM8K accuracy (generation, slow) and the KL divergence between teacher and student on the reference answers of the validation set (no generation, fast). The KL divergence is the more informative metric for distillation, as the accuracy of a small model is dominated by the prompt format and its ability to follow it (e.g. instruct models tend to do poorly with completion-style prompts; prefer base models for this example). As a rough guide: a KL below ~0.05 nats/token means the student is practically indistinguishable from the teacher, 0.1-0.3 is a noticeable but useful approximation, and values above 0.5 mean the student still behaves like a different model.

Some care is needed when comparing the reported numbers, as the same metrics are computed on data with different characteristics:

- Training KL vs. held-out KL: Both lines report the same per-token forward KL divergence, but the training line is computed on the MetaMathQA training batches (and averaged over the trailing `eval_steps` optimizer steps, so it lags slightly behind the current weights), while the held-out line is computed on the GSM8K reference answers only. A persistent gap between the two is expected, it mostly reflects the style difference between the two text distributions.
- Validation accuracy: The validation set is small (50 samples), thus validation accuracy is not representative of test accuracy. The test accuracy (1319 unseen samples) is the reliable number and can be evaluated with `--eval_test`.

When training finishes, you should see something like this:

```
Student after distillation: KL(teacher || student) = 0.3686 nats/token (6575 held-out GSM8K answer tokens)
Teacher (MoE) on test set: accuracy 28.7% (1319 samples, 558s)
Student on test set: accuracy 10.0% (1319 samples, 284s)
Dense model has 0.28B parameters

                            MoE (original)   dense (distilled)
parameters (B)                        1.33                0.28
weight memory (GB)                    2.49                0.52
prefill (tokens/s)                45955.71            76902.32
decode bs=1 (ms/step)                10.93               10.57
peak memory bs=1 (GB)                 2.53                0.55
decode bs=16 (ms/step)               29.12               14.46
peak memory bs=16 (GB)                3.08                0.93
```

As you can see, the student did not recover the accuracy of the teacher, which would most likely require much longer training  (the paper trains on the order of billions of tokens). But the distilled model requires 2 GB less memory to load and inference is slightly faster. For `num_experts_to_keep=8`, which corresponds to the number of active experts in the original model, we get:

```
Student after distillation: KL(teacher || student) 0.3496 nats/token on 6575 held-out answer tokens
Student on test set: accuracy 10.4% (1319 samples, 295s)
Dense model has 0.43B parameters

                            MoE (original)   dense (distilled)
parameters (B)                        1.33                0.43
weight memory (GB)                    2.49                0.80
prefill (tokens/s)                45907.53            64086.83
decode bs=1 (ms/step)                11.05               10.53
peak memory bs=1 (GB)                 2.54                0.85
decode bs=16 (ms/step)               29.11               25.47
peak memory bs=16 (GB)                3.09                1.40
```

### Useful options 

- `--max_steps`, `--batch_size`, `--grad_accumulation_steps`: the distillation budget. Distilling a narrow domain like math needs far fewer tokens than the billions of tokens of general data used in the paper, but expect to need at least a few thousand steps for good results.
- `--calibration_steps`: the number of training batches used for the routing statistics (a few hundred are plenty).
- `--num_experts_to_keep`: keep more (or fewer) experts than the router's top-k, resulting in a bigger (smaller) dense FFN.
- `--modules_to_save q_proj k_proj v_proj o_proj input_layernorm post_attention_layernorm`: also fully fine-tune the attention projections and norms (the paper trains the whole student). Trainable copies are created, the teacher keeps using the originals. The non-expert parameters are a small fraction of an MoE model, so this costs little memory and lets the rest of the network adapt to the replaced FFNs.
- `--dense_dtype`: the dense FFNs are trained in float32 (with autocast to the model dtype) by default; `bfloat16` uses less memory.
- `--use_gc`: gradient checkpointing.
- `--device_map auto`: spread the model over multiple GPUs.
- `--adapter_output_dir`: also save the PEFT adapter before compressing the model.
- `--eval_test`: evaluate on the GSM8K test set at the end.
- `--skip_benchmark`: skip the comparison at the end, which loads the original MoE model and the saved dense model one after the other and reports their parameter count, weight memory, prefill throughput, decode latency (batch size 1 and 16), and peak memory. Note that the per-token compute of the dense model equals the *active* compute of the MoE model, so the gains are in memory and in decoding throughput at larger batch sizes (where the MoE model has to read most of its experts per step); at batch size 1, the two are expected to be similar.

Run with `--help` for all.

### Memory requirements

The whole MoE teacher must stay in memory during distillation. For Qwen3-30B-A3B in bfloat16 that is ~61 GB for the teacher, plus the dense FFNs with their gradients and AdamW states (~29 GB in float32, ~15 GB in bfloat16), plus activations. This requires a GPU with more than 80 GB of memory or multiple GPUs (`--device_map auto`). For a quick functional test, use a small MoE model, e.g. `ibm-granite/granite-3.1-1b-a400m-instruct` (~10 GB in total).

### Notes

- Quantized experts are not supported; load the model with dequantized (floating point) experts.
- Hub kernels that replace the forward of the whole MoE block (e.g. `use_kernels=True` for GPT-OSS) bypass the dense FFNs; keep them disabled. The `experts_implementation` setting (`eager`, `grouped_mm`, ...) is supported.
- Only one adapter can be active at a time, and the adapter cannot be merged (use `compress_and_unload()` instead).
