<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Expert Weighted Low-Rank Adaptation

[EWoRA](https://aclanthology.org/2025.findings-ijcnlp.108/) (Expert Weighted Low-Rank Adaptation) is a LoRA variant designed for finetuning on heterogeneous data. Instead of a single low-rank adapter, EWoRA uses several independent low-rank "expert" adapters (`num_experts`) and learns a lightweight routing matrix that dynamically weights the experts for each input. This lets a single adapter capture the diverse expertise needed across a heterogeneous corpus while keeping the same low-rank parameter budget as LoRA. EWoRA was introduced in [EWoRA: Expert Weighted Low-Rank Adaptation for Heterogeneous Data](https://aclanthology.org/2025.findings-ijcnlp.108/) (AACL-IJCNLP 2025, Findings).

EWoRA currently has the following constraints:

- Only `nn.Linear` and `transformers.pytorch_utils.Conv1D` layers are supported.
- Because the experts are weighted dynamically at the forward pass, EWoRA cannot be merged into the base model: `merge()` and `merge_and_unload()` raise `NotImplementedError`. Use `unload()` to remove the adapters and recover the base model.

The abstract from the paper is:

> Low-Rank Adaptation (LoRA) has emerged as a widely adopted parameter-efficient fine-tuning (PEFT) approach for language models. By restricting weight updates to a low-rank subspace, LoRA achieves cost-effective finetuning of large, generalist models to more specialized target domains. While LoRA achieves impressive results for a variety of individual downstream tasks, it struggles to capture the diverse expertise needed when presented with a more heterogeneous finetuning corpus. To address this, we propose Expert Weighted Low-Rank Adaptation (EWoRA), a novel LoRA variant that partitions a rank-r adapter into n independent adapters of rank r/n. A lightweight "routing" matrix (W<sub>r</sub> ∈ R<sup>r×n</sup>) aggregates the outputs of these adapters by learning specialized weights for each context. Experiments show EWoRA improves performance over LoRA when finetuning on heterogeneous data while generally matching or exceeding LoRA performance on individual finetuning tasks under the same low-rank parameter budget.

## Citation

```bibtex
@inproceedings{kohli-etal-2025-ewora,
    title = "{EW}o{RA}: Expert Weighted Low-Rank Adaptation for Heterogeneous Data",
    author = "Kohli, Harsh  and
      Feng, Helian  and
      Minorics, Lenon  and
      Vasani, Bhoomit  and
      He, Xin  and
      Kebarighotbi, Ali",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-ijcnlp.108/",
    doi = "10.18653/v1/2025.findings-ijcnlp.108",
    pages = "1729--1737",
    ISBN = "979-8-89176-303-6"
}
```

# API

## EworaConfig

[[autodoc]] tuners.ewora.config.EworaConfig

## EworaModel

[[autodoc]] tuners.ewora.model.EworaModel
