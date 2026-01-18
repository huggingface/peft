# AdaDoRA (AdaLoRA + DoRA) Example

This example demonstrates how to use **AdaDoRA** - a combination of [AdaLoRA's](https://arxiv.org/abs/2303.10512) adaptive rank allocation with [DoRA's](https://arxiv.org/abs/2402.09353) weight-decomposed low-rank adaptation.

## Usage

```bash
python run_glue.py \
    --model_name_or_path roberta-base \
    --task_name sst2 \ 
    --peft_type ADALORA \ 
    --use_dora \ 
    --per_device_train_batch_size 16 \ 
    --learning_rate 1e-4 \ 
    --num_train_epochs 3 \ 
    --output_dir ./output
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--peft_type ADALORA` | Use AdaLoRA as the base PEFT method |
| `--use_dora` | Enable DoRA's magnitude-direction decomposition |
| `--init_r` | Initial rank for each layer (default: 12) |
| `--target_r` | Target rank after pruning (default: 8) |

## What is AdaDoRA?

AdaDoRA combines:
- **AdaLoRA**: Adaptive rank allocation based on importance scores, allowing different layers to have different ranks
- **DoRA**: Weight Decomposition into magnitude and direction components for more stable training

This combination provides both parameter efficiency through adaptive pruning and training stability through weight decomposition.

## Citation

If you use AdaDoRA, please cite both papers:

```bibtex
@inproceedings{zhang2023adalora,
  title={AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning},
  author={Zhang, Qingru and Chen, Minshuo and Bukharin, Alexander and He, Pengcheng and Cheng, Yu and Chen, Weizhu and Zhao, Tuo},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://arxiv.org/abs/2303.10512}
}

@article{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  journal={arXiv preprint arXiv:2402.09353},
  year={2024},
  url={https://arxiv.org/abs/2402.09353}
}
```
