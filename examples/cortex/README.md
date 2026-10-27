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
hyperparameters. Run each script with `--help` for its complete options.

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
