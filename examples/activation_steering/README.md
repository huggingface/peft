# Activation steering with Qwen3.5-4B

This example applies configurable neuron interventions to [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The default is the original two-neuron recipe: zero-based layer 13, neuron 7956 at dose -1, and layer 14, neuron 8949 at dose +1.

Sources:

- [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): original steering implementation and calibration text.
- [Independent reproduction](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering): implementation, comparisons, and reported limitations.

Install PEFT from this checkout and a Transformers version supporting Qwen3.5. Run these commands from the PEFT repository root.

## Presets

Each preset has an `adapter_config.json` under `presets/`. These contain configuration only; calibrate a preset to create its adapter checkpoint before generation. The additional recipes encode the supplied suggestions; their behavioral labels have not been validated here.

| Preset                                                       | Interventions (zero-based layer and neuron, signed dose)                       |
|--------------------------------------------------------------|--------------------------------------------------------------------------------|
| [original](presets/original/adapter_config.json)             | L13 N7956 -1; L14 N8949 +1                                                     |
| [sarcastiqwen](presets/sarcastiqwen/adapter_config.json)     | L15 N3920 +1                                                                   |
| [sycophancy-l12](presets/sycophancy-l12/adapter_config.json) | L15 N3920 +1; L12 N2920 -1                                                     |
| [sycophancy-l16](presets/sycophancy-l16/adapter_config.json) | L15 N3920 +1; L16 N8168 -1                                                     |
| [condolences](presets/condolences/adapter_config.json)       | L15 N3920 +1; L10 N6295 +1                                                     |
| [chill-friend](presets/chill-friend/adapter_config.json)     | L15 N3920 +1; L11 N6261 +0.4; L17 N7870 -0.4                                   |
| [mixed](presets/mixed/adapter_config.json)                   | L18 N5796 +0.6; L15 N3920 +0.6; L12 N2920 -0.5; L10 N6295 +0.4; L16 N8168 -0.3 |

Unspecified dose magnitudes are interpreted as 1. The "add" suggestions are interpreted as additions to `sarcastiqwen`; the two suggested sycophancy neurons are separate alternatives. The final five-neuron recipe is `mixed`. Select a preset with `--preset` before the subcommand:

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

No optimizer step is needed. The configured doses and calibrated scales are saved to `qwen3.5-4b-steering` for `original`, or `qwen3.5-4b-steering-PRESET` for other presets. Use `--adapter-path PATH` before either subcommand to select another directory. Generation loads the saved checkpoint's configuration and scales.

## Generate

```bash
python examples/activation_steering/qwen3_5.py generate --prompt "Tell me a short story."
```

The example reloads the pinned base model and uses Transformers' `model.load_adapter()` to restore the intervention. No recalibration is needed. Generation uses greedy decoding and disables thinking, matching the reproduction's decoding setup.

The example explicitly stops on both `<|endoftext|>` (248044) and `<|im_end|>` (248046). The pinned model's [config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json) only specifies the former, while its [tokenizer](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer_config.json) uses the latter to end chat turns. Include both EOS IDs when using another chat client; otherwise generation can continue into invented user and assistant turns. `skip_special_tokens=True` hides the delimiters but does not stop generation.

Repetition within an answer can still occur. Each full-dose intervention adds a vector whose L2 norm matches its block's calibrated median residual norm, so these are large perturbations. The independent reproduction reports reduced coherence. Compare the same prompt with adapters disabled (`model.disable_adapters()`, then `model.enable_adapters()` to restore). Sampling or smaller doses may help, but change the experiment; adjusting temperature alone has no effect with `do_sample=False`.
