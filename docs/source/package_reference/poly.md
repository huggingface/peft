<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Polytropon

[Polytropon](https://hf.co/papers/2202.13914) is a multitask model with a number of different LoRA adapters in it's "inventory". The model learns the correct combination of adapters from the inventory with a routing function to choose the best subset of modules for a specific task. PEFT also supports [Multi-Head Adapter Routing (MHR)](https://hf.co/papers/2211.03831) for Polytropon which builds on and improves the routing function by combining the adapter heads more granularly. The adapter heads are separated into disjoint blocks and a different routing function is learned for each one, allowing for more expressivity.

<hfoptions id="paper">
<hfoption id="Combining Modular Skills in Multitask Learning">

The abstract from the paper is:

*A modular design encourages neural models to disentangle and recombine different facets of knowledge to generalise more systematically to new tasks. In this work, we assume that each task is associated with a subset of latent discrete skills from a (potentially small) inventory. In turn, skills correspond to parameter-efficient (sparse / low-rank) model parameterisations. By jointly learning these and a task-skill allocation matrix, the network for each task is instantiated as the average of the parameters of active skills. To favour non-trivial soft partitions of skills across tasks, we experiment with a series of inductive biases, such as an Indian Buffet Process prior and a two-speed learning rate. We evaluate our latent-skill model on two main settings: 1) multitask reinforcement learning for grounded instruction following on 8 levels of the BabyAI platform; and 2) few-shot adaptation of pre-trained text-to-text generative models on CrossFit, a benchmark comprising 160 NLP tasks. We find that the modular design of a network significantly increases sample efficiency in reinforcement learning and few-shot generalisation in supervised learning, compared to baselines with fully shared, task-specific, or conditionally generated parameters where knowledge is entangled across tasks. In addition, we show how discrete skills help interpretability, as they yield an explicit hierarchy of tasks.*

</hfoption>
<hfoption id="Multi-Head Adapter Routing for Cross-Task Generalization">

The abstract from the paper is:

*Parameter-efficient fine-tuning (PEFT) for cross-task generalization consists in pre-training adapters on a multi-task training set before few-shot adaptation to test tasks. Polytropon [Ponti et al., 2023] (Poly) jointly learns an inventory of adapters and a routing function that selects a (variable-size) subset of adapters for each task during both pre-training and few-shot adaptation. In this paper, we investigate the role that adapter routing plays in its success and design new variants based on our findings. First, we build on the intuition that finer-grained routing provides more expressivity. Hence, we propose MHR (Multi-Head Routing), which combines subsets of adapter parameters and outperforms Poly under a comparable parameter budget; by only fine-tuning the routing function and not the adapters (MHR-z), we achieve competitive performance with extreme parameter efficiency. Second, we find that Poly/MHR performance is a result of better multi-task optimization, rather than modular inductive biases that facilitate adapter recombination and local adaptation, as previously hypothesized. In fact, we find that MHR exhibits higher gradient alignment between tasks than any other method. Since this implies that routing is only crucial during multi-task pre-training, we propose MHR-mu, which discards routing and fine-tunes the average of the pre-trained adapters during few-shot adaptation. This establishes MHR-mu as an effective method for single-adapter fine-tuning.*.

</hfoption>
</hfoptions>

## PolyConfig

[[autodoc]] tuners.poly.config.PolyConfig

## PolyModel

[[autodoc]] tuners.poly.model.PolyModel
