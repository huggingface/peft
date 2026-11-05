# Cortex examples

Cortex adds trainable nonlinear FFN channels to a frozen model. The
[API reference](../../docs/source/package_reference/cortex.md) describes supported
layouts, initialization, adapter operations, and structural merging. Training
data, loss computation, and task evaluation belong to the examples below.

## Choose a workflow

| Example | Input and objective | Output |
| --- | --- | --- |
| [finetune/](finetune/README.md) | Local `messages` or `text` JSONL; ordinary cross-entropy with Transformers `Trainer` | Adapter and tokenizer; optional native model export |
| [consolidate/](consolidate/README.md) | Source documentation; generated questions and cached teacher targets; KL with optional hidden-state MSE | Prepared cache, adapter, tokenizer, experiment record, and evaluation reports |

Fine-tuning needs no teacher or cache. To use consolidation's generated QA with
it, follow the [messages conversion](finetune/README.md#reusing-consolidation-questions).
The consolidation evaluator requires that workflow's `run.json` and prepared
metadata; it does not directly accept the ordinary fine-tuning output.

The consolidation example performs one context-to-parameters update. Continual
replay, target regeneration, and the policy for retaining past tasks belong to
the broader [Cortex research workflow](https://github.com/zjunlp/cortex).
For supervised training configurations and benchmark commands, see the
[MetaMathQA benchmark](../../method_comparison/MetaMathQA/experiments/cortex/README.md).
A [custom FFN classification example](finetune/README.md#custom-ffn-classification)
also demonstrates the API outside a causal language model.

## Check a Transformers layout

The probe is an example script and is not installed by `pip install peft`.
Clone the [PEFT repository](https://github.com/huggingface/peft) and check out
the release tag or commit matching your PEFT installation. For development
versions, use the same source checkout as the installed package.

Run the probe from the repository root with a local model config directory or
a Hub model ID. To explore a model's layouts at two different channel widths:

```sh
python examples/cortex/validate_layout.py --model-name path/to/config-directory --k 2 4
```

Dense models have one layout per width. When selecting complete MoE blocks,
the probe checks shared, routed, and both branches together. To select just
the combined case, add `--moe-target shared routed`.

Explicitly selecting a shared FFN or experts module fixes its role to shared
or routed, respectively, regardless of `moe_target` in the JSON or CLI.
Selecting both leaves checks the combination once per width; dense targets
selected alongside them are retained. If a selection mixes a complete MoE
block and fixed leaves from another block, only the complete block varies
with `moe_target`.
Excluded branches, including those reserved by `modules_to_save`, are not
required by the probe.

The probe reports which expert projections ran
before routing weights were applied. When several branch kinds are present,
it disables each kind in turn to check that it affects the logits. It also
checks that native merge and reload preserve the full adapter output.

To check custom field names before creating an adapter, supply a new Cortex
configuration with `--cortex-config`. For example, a `cortex.json` describing an
FFN whose projections are named `w1`, `w3`, and `w2` can contain:

```json
{
  "ffn_specs": {
    "MyFfn": {
      "gate_proj": "w1",
      "up_proj": "w3",
      "down_proj": "w2",
      "act_fn": "activation",
      "width_field": "ffn_channels"
    }
  }
}
```

Use the actual class and field names from the model. Add `moe_specs` to describe
nonstandard expert or shared-FFN names; see the
[structure specifications](../../docs/source/package_reference/cortex.md#custom-structure-specifications).

```sh
python examples/cortex/validate_layout.py --model-name path/to/config-directory --cortex-config cortex.json --k 2 4
```

This accepts a JSON file or a directory containing `adapter_config.json`, without
resolved `target_roles`. No adapter weights or prior injection are needed.
Command-line probe options override the JSON. If `k` or `moe_target` is present
in the JSON, it selects the width or MoE branches to test; otherwise, the probe
uses widths 2 and 4 and discovers dense or MoE modes using the supplied specs.
As in training, JSON `moe_target` has no effect on a dense-only selection.
A file produced by `CortexConfig.save_pretrained` includes default values, so
it also selects those values. Omit `moe_target` from an exploratory JSON to
check all three MoE modes, or use `--moe-target` to select one.

If you already have an adapter, check its **saved configuration** instead:

```sh
python examples/cortex/validate_layout.py --model-name path/to/config-directory --adapter-config path/to/adapter
```

This uses the recorded physical targets and branch roles, including the
distinction between a public and a routed branch on the same expert module.
`--adapter-config` and `--cortex-config` are mutually exclusive. Saved-config
replay rejects probe overrides, preserving the recorded settings. For either
configuration source, `--model-name` takes precedence over its
`base_model_name_or_path`.

All probe modes build a reduced random model and use probe weights; they do not
load pretrained model weights or measure task quality. The script uses native
Transformers model loaders with `trust_remote_code=False`; a spec does not
import arbitrary custom model classes.
The probe keeps the layer schedule, expert count, and routing top-k. If the reduced
model still exceeds the CPU allocation limit, pass a smaller local config that
retains the target layers and routing behavior you want to check.

## Environment and execution

Install this PEFT checkout with `python -m pip install -e .` from the repository
root. The examples also use PyTorch, Transformers, Accelerate, Datasets, and
Safetensors. Change into `examples/cortex/finetune` or
`examples/cortex/consolidate` before running that directory's README commands.
Use new output directories for preparation, training, and native export.

The examples were checked with Transformers 5.17.0 and Accelerate 1.15.0. CPU
checks used Python 3.13, PyTorch 2.10.0, Datasets 5.0.1, and Safetensors 0.8.0.
The consolidation GPU recipes use PyTorch 2.9.0 / CUDA 12.9 with DeepSpeed 0.18.9
for ZeRO-0. Their full SEIS task-quality runs have not been validated. These
versions describe the example validation environments, not a global PEFT minimum.

Both training scripts use one device. Fine-tuning uses ordinary Trainer;
consolidation additionally supports single-process DeepSpeed ZeRO-0.
Only explicit code scoring requires `seismic-linter==0.2.0`, installed in the
execution environment you provide. Follow the scoring section before executing
generated programs.

The training CLI defaults are `--k 64 --init-weights sample`, one epoch, and
batch size one. The library's `CortexConfig` defaults are `k=16` and random
initialization. Recipes specify their own settings; model names do not select
hyperparameters. Both training scripts use `--warmup-ratio 0.1` by default.
Set `--warmup-steps` instead for a fixed number of optimizer steps, or
`--warmup-steps 0` to disable warmup. Run each script with `--help` for its complete options.

Both examples use `chat_utils.py` to preserve native Qwen3/GPT-OSS reasoning
formats. GPT-OSS rendering fixes the template's automatic current date to
`xxxx-xx-xx` through `strftime_now`; user-supplied dates and saved tokenizer files
are unchanged. Reuse the helper when reproducing that formatting after reload.

## Citation

If you use Cortex in your research, please cite
[Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399).

```bibtex
@misc{zhang2026cortex,
  title = {Cortex: Consolidating Context into Parametric Memory},
  author = {Jintian Zhang and Feng He and Xinnian Liang and Ziwen Xu and Haoming Xu and Haojing Huang and Shuangzhi Wu and Ningyu Zhang},
  year = {2026},
  eprint = {2610.20399},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2610.20399}
}
```
