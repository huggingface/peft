<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models

[DEFT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/93a34a7138bdad95e874018d5f491cc6-Abstract-Conference.html)
(Decompositional Efficient Fine-Tuning) is a parameter-efficient fine-tuning method for text-to-image models. It
decomposes the update of a frozen weight matrix `W` into two trainable components: a projection that removes a low-rank
subspace from `W`, and a low-rank update that injects new content into that subspace. This formulation is designed to
balance aligning with a target distribution, learning new concepts from a few images (personalization), and preserving
the pretrained model's instruction-following ability and editability.

Concretely, DEFT combines two trainable low-rank components: (1) a projection onto the complement of a low-rank
subspace spanned by a low-rank matrix, and (2) a low-rank update. The first low-rank matrix defines the subspace, while
the second enables flexible parameter adaptation within that subspace.

When to use DEFT: it is best suited to adapting a model to new data or concepts while **retaining and even improving the
base model's instruction-following ability** and keeping **forgetting of its previous capabilities to a minimum**.

Per target layer, DEFT learns a projection direction `P` (shape `out_features x r`) and an injection matrix `R` (shape
`r x in_features`). The effective weight is the residual projection

```
W' = (I - P_proj) @ W + Q_P @ R
```

The projector `P_proj` is derived from `P` according to `decomposition_method`:

- `"relu"` (default): `Q_P = P`, `P_proj = P @ relu(P).T` — a non-orthogonal projection.
- `"qr"`: `Q_P = qr(P)`, `P_proj = Q_P @ Q_P.T` — an orthogonal projection.

The `(I - P_proj) @ W` term removes a sub-space of the pretrained weight while `Q_P @ R` injects new content into it.
By default (`init_weights=True`) `R` is initialized so that the update is an exact identity at initialization
(`W' == W`), so training starts from the pretrained weights and learns the injection. The update is equivalent to a
low-rank additive delta `Q_P @ (R - right.T @ W)`, which is computed without ever forming the `out x out`
projection matrix and can be merged into the base weights for inference-free deployment.

Setting `para=True` selects the
[PaRa](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f09e8dd9274cb7c2dd0dc65ffc6f427a-Abstract-Conference.html)
(Parameter Rank Reduction) variant: a removal-only update `W' = (I - P_proj) @ W` that keeps just the subspace-removal
term and drops the injection. Only the projection `P` is trained (no injection matrix `R`), so the adapter is not an
identity at initialization. PaRa was introduced for personalizing text-to-image diffusion models and is available here
as a special case of DEFT.

DEFT is currently implemented for `torch.nn.Linear` and `Conv1D` (e.g. gpt-2, via `fan_in_fan_out`) layers. The original implementation and the experiments from the
paper (Dreambooth, Dreambench Plus, InsDet, VisualCloze, on Stable Diffusion and a unified model) are available at
[github.com/MAXNORM8650/DEFT](https://github.com/MAXNORM8650/DEFT).

If you use DEFT in your work, please cite the paper:

```bibtex
@article{kumar2026deft,
  title={DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models},
  author={Kumar, Komal and Anwer, Rao and Shahbaz Khan, Fahad and Khan, Salman and Laptev, Ivan and Cholakkal, Hisham},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={102009--102035},
  year={2026}
}
```

If you use the PaRa variant (`para=True`), please also cite:

```bibtex
@inproceedings{chen2025personalizing,
  title={Para: Personalizing text-to-image diffusion via parameter rank reduction},
  author={Chen, Shangyu and Pan, Zizheng and Cai, Jianfei and Phung, Dinh},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## DeftConfig

[[autodoc]] tuners.deft.config.DeftConfig

## DeftModel

[[autodoc]] tuners.deft.model.DeftModel
