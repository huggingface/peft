# Cortex examples

Cortex adds trainable channels to the FFNs of a frozen model, see the [Cortex documentation](https://huggingface.co/docs/peft/main/en/package_reference/cortex).

- [finetune/](finetune/README.md): train a Cortex adapter on a dense or MoE causal LM with the Transformers `Trainer`.

The workflow for consolidating documents into an adapter, as described in the paper, is available in the [Cortex research repository](https://github.com/zjunlp/cortex).

## Citation

If you use Cortex in your research, please cite [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399).

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
