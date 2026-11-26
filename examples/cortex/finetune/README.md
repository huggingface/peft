# Fine-tuning with Cortex

`finetune.py` trains a Cortex adapter on a dense or MoE causal LM with the Transformers `Trainer` and saves the adapter and the tokenizer.

## Training data

Supply a local JSONL file in which every line has either a `text` field or a `messages` field:

```json
{"text": "A temporal split places earlier observations in training and later observations in testing."}
{"messages": [{"role": "user", "content": "What is a temporal split?"}, {"role": "assistant", "content": "Training timestamps precede test timestamps."}]}
```

Conversations are rendered with the chat template of the model. All tokens are used as training targets.

## Training

```bash
python finetune.py \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --train-file train.jsonl \
  --output-dir outputs/qwen-cortex \
  --k 64 --init-weights sample
```

The same script works for the supported MoE models, e.g. Qwen3-MoE, GPT-OSS, or DeepSeek-V3. For GPT-OSS, use a dequantized BF16 checkpoint. `--moe-target` selects where the channels are added: `shared` (default) extends the shared experts, or adds channels common to all experts to MoE models without shared experts, `routed` adds channels to each expert, and `both` does both. Note that `routed` multiplies the number of added parameters by the number of experts; see the [Cortex documentation](https://huggingface.co/docs/peft/main/en/package_reference/cortex#moe-models).

`--cortex-bias` selects whether the added channels have biases: with `auto` (default), they have biases if the FFNs of the model have biases, e.g. for GPT-OSS, and with `none`, they have no biases. For models whose FFNs have no biases, like Qwen3, both options are the same; see the [Cortex documentation](https://huggingface.co/docs/peft/main/en/package_reference/cortex#biases).

To train on several GPUs, launch the script with `torchrun`, e.g. `torchrun --nproc_per_node 2 finetune.py` followed by the arguments above. The adapter and the tokenizer are saved once, by the main process.

## Loading the adapter

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "outputs/qwen-cortex")
```

To get a model without adapter layers, merge the adapter. The merged FFNs are wider than the original ones, and their widths are updated in the model config, so the merged model can be saved and loaded like any other Transformers model:

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("outputs/qwen-cortex-merged")

merged_model = AutoModelForCausalLM.from_pretrained("outputs/qwen-cortex-merged")
```

If Cortex cannot infer or verify a config for the merged model, `merge_and_unload()` warns and leaves the config unchanged. In that case, and for loading merged MoE models, see the [Cortex documentation](https://huggingface.co/docs/peft/main/en/package_reference/cortex#merging-and-saving).
