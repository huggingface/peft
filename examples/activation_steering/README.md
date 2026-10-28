# Activation steering with Qwen3.5-4B

This example applies configurable neuron interventions to [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The default is the original two-neuron recipe: zero-based layer 13, neuron 7956 at dose -1, and layer 14, neuron 8949 at dose +1.

Sources:

- [Instigating tweet](https://x.com/_lyraaaa_/status/2113833330429083662)
- [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): original steering implementation and calibration text.
- [Independent reproduction](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering): implementation, comparisons, and reported limitations.

Install PEFT from this checkout and a Transformers version supporting Qwen3.5. Run these commands from the PEFT repository root.

## Presets

Each preset has an `adapter_config.json` under `presets/`. These contain configuration only; calibrate a preset to create its adapter checkpoint before generation. The additional recipes encode the supplied suggestions from [this tweet](https://x.com/_lyraaaa_/status/2113887372610077136).

| Preset                                                       | Interventions (zero-based layer and neuron, signed dose)                       |
|--------------------------------------------------------------|--------------------------------------------------------------------------------|
| [original](presets/original/adapter_config.json)             | L13 N7956 -1; L14 N8949 +1                                                     |
| [sarcastiqwen](presets/sarcastiqwen/adapter_config.json)     | L15 N3920 +1                                                                   |
| [sycophancy-l12](presets/sycophancy-l12/adapter_config.json) | L15 N3920 +1; L12 N2920 -1                                                     |
| [sycophancy-l16](presets/sycophancy-l16/adapter_config.json) | L15 N3920 +1; L16 N8168 -1                                                     |
| [condolences](presets/condolences/adapter_config.json)       | L15 N3920 +1; L10 N6295 +1                                                     |
| [chill-friend](presets/chill-friend/adapter_config.json)     | L15 N3920 +1; L11 N6261 +0.4; L17 N7870 -0.4                                   |
| [mixed](presets/mixed/adapter_config.json)                   | L18 N5796 +0.6; L15 N3920 +0.6; L12 N2920 -0.5; L10 N6295 +0.4; L16 N8168 -0.3 |

Select a preset with `--preset` before the subcommand:

```bash
python examples/activation_steering/qwen3_5.py --preset chill-friend calibrate
python examples/activation_steering/qwen3_5.py --preset chill-friend generate --prompt "Tell me a short story."
```

Add another preset by placing its `adapter_config.json` in a new subdirectory under `presets/`; the CLI discovers it automatically.

## Calibrate and save

```bash
python examples/activation_steering/qwen3_5.py calibrate
```

Calibration runs an unsteered forward pass and measures the median outgoing residual L2 norm at each targeted block, excluding the first token. It uses the original author's fallback reference text, rather than precomputed corpus statistics. Override the text with:

```bash
python examples/activation_steering/qwen3_5.py calibrate --reference-text "Your calibration text goes here."
```

No actual training is needed. The configured doses and calibrated scales are saved to `qwen3.5-4b-steering` for `original`, or `qwen3.5-4b-steering-PRESET` for other presets. Use `--adapter-path PATH` before either subcommand to select another directory. Generation loads the saved checkpoint's configuration and scales.

## Generate

```bash
python examples/activation_steering/qwen3_5.py generate --prompt "Tell me a short story."
```

The example reloads the pinned base model and uses Transformers' `model.load_adapter()` to restore the intervention. No recalibration is needed.

Generation defaults follow [Qwen's recommendations for non-thinking general tasks](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices):

| Setting                | Default                                            |
|------------------------|----------------------------------------------------|
| Sampling               | Enabled (`--no-do-sample` selects greedy decoding) |
| `--temperature`        | 0.7                                                |
| `--top-p`              | 0.8                                                |
| `--top-k`              | 20                                                 |
| `min_p`                | 0.0                                                |
| `--repetition-penalty` | 1.0 (disabled)                                     |

Use `--seed` to control sampling and `--disable-adapter` to compare the same prompt without steering:

```bash
python examples/activation_steering/qwen3_5.py generate --prompt "Tell me a short story." --seed 42
python examples/activation_steering/qwen3_5.py generate --prompt "Tell me a short story." --seed 42 --disable-adapter
```

These sampling defaults differ from the reproduction's greedy decoding. To use greedy decoding:

```bash
python examples/activation_steering/qwen3_5.py generate --no-do-sample
```

## Create and share your own intervention

A calibrated intervention is a regular PEFT adapter checkpoint. Recipients need this PEFT checkout, or a release containing `ActivationSteeringConfig`, and a Transformers version supporting Qwen3.5.

### 1. Define the intervention

Create a configuration in Python:

```python
from peft import ActivationSteeringConfig, TaskType

config = ActivationSteeringConfig(
    base_model_name_or_path="Qwen/Qwen3.5-4B",
    task_type=TaskType.CAUSAL_LM,
    steering_pattern={
        "layers.15": {
            "source_module": "mlp.down_proj",
            "indices": [3920],
            "init_val": 0.5,
            "statistic": "median_l2",
        },
    },
)
```

You can change the block names, zero-based neuron indices, and signed doses to define your intervention. `source_module` is relative to the targeted block; for these Qwen MLP neurons, it is `mlp.down_proj`. A scalar `init_val` broadcasts over the indices, or provide a list with one dose per index. Only `median_l2` is currently supported for `statistic`.

### 2. Create and calibrate the PEFT model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import get_peft_model
from peft.tuners.activation_steering import calibrate_activation_steering

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = get_peft_model(base_model, config).eval()

calibration_text = (
    "The river runs past the village and through the fields. "
    "At the market, people trade fruit, bread, and stories from their travels. "
    "A programmer writes a function to sort a list of numbers."
)
inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
calibrate_activation_steering(model, {**inputs, "use_cache": False})
```

Replace the sample text with representative calibration data. It runs unsteered forwards and stores fixed residual scales; no actual training is needed.

### 3. Push the calibrated model directly to the Hub

After authenticating with `hf auth login`, call `push_to_hub()` on the PEFT model and tokenizer:

```python
repo_id = "your-username/qwen3.5-4b-my-intervention"
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

This creates the repository if needed and uploads `adapter_config.json`, `adapter_model.safetensors`, and tokenizer files. The adapter checkpoint stores both the doses and calibrated scales; the base model weights are not uploaded. Sharing only the configuration omits the calibration. Include a model card to describe the intervention.

### 4. Load the shared intervention

Others can pass the Hub repository ID to this example, which loads through Transformers' PEFT integration:

```bash
python examples/activation_steering/qwen3_5.py --adapter-path your-username/qwen3.5-4b-my-intervention generate --prompt "Tell me a short story." --seed 42
```

They can also load it directly with `PeftModel` and generate:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.utils import logging

from peft import PeftConfig, PeftModel

logging.set_verbosity_error()
set_seed(42)
adapter_id = "your-username/qwen3.5-4b-my-intervention"
config = PeftConfig.from_pretrained(adapter_id)
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id).eval()

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Tell me a short story."}],
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(model.device)
prompt_length = inputs.input_ids.shape[1]

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_end|>"]),
        pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True))
```

Loading restores the saved doses and scales, so recipients do not need to calibrate again.
