<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# AdaLoRA

[AdaLoRA](https://hf.co/papers/2303.10512) (Adaptive LoRA) is a method for optimizing the number of trainable parameters to assign to weight matrices and layers, unlike LoRA, which distributes parameters evenly across all modules. More parameters are budgeted for important weight matrices and layers while less important ones receive fewer parameters. You can control the average desired *rank* or `r` of the matrices, and which modules to apply AdaLoRA to with `target_modules`. Other important parameters to set are `lora_alpha` (scaling factor), and `modules_to_save` (the modules apart from the AdaLoRA layers to be trained and saved). All of these parameters - and more - are found in the [`AdaLoraConfig`].

The abstract from the paper is:

*Fine-tuning large pre-trained language models on downstream tasks has become an important paradigm in NLP. However, common practice fine-tunes all of the parameters in a pre-trained model, which becomes prohibitive when a large number of downstream tasks are present. Therefore, many fine-tuning methods are proposed to learn incremental updates of pre-trained weights in a parameter efficient way, e.g., low-rank increments. These methods often evenly distribute the budget of incremental updates across all pre-trained weight matrices, and overlook the varying importance of different weight parameters. As a consequence, the fine-tuning performance is suboptimal. To bridge this gap, we propose AdaLoRA, which adaptively allocates the parameter budget among weight matrices according to their importance score. In particular, AdaLoRA parameterizes the incremental updates in the form of singular value decomposition. Such a novel approach allows us to effectively prune the singular values of unimportant updates, which is essentially to reduce their parameter budget but circumvent intensive exact SVD computations. We conduct extensive experiments with several pre-trained models on natural language processing, question answering, and natural language generation to validate the effectiveness of AdaLoRA. Results demonstrate that AdaLoRA manifests notable improvement over baselines, especially in the low budget settings. Our code is publicly available at https://github.com/QingruZhang/AdaLoRA*.

> [!WARNING]
> AdaLoRA has an [`~AdaLoraModel.update_and_allocate`] method that should be called at each training step to update the parameter budget and mask, otherwise the adaptation step is not performed. This requires writing a custom training loop or subclassing the [`~transformers.Trainer`] to incorporate this method. As an example, take a look at this [custom training loop](https://github.com/huggingface/peft/blob/912ad41e96e03652cabf47522cd876076f7a0c4f/examples/conditional_generation/peft_adalora_seq2seq.py#L120).

## How AdaLoRA works

AdaLoRA manages the parameter budget introduced from LoRA by allocating more parameters — in other words, a higher rank `r` — for important weight matrices that are better adapted for a task and pruning less important ones. The rank is controlled by a method similar to singular value decomposition (SVD). The $\Delta W$ is parameterized with two orthogonal matrices and a diagonal matrix which contains singular values. This parametrization method avoids iteratively applying SVD which is computationally expensive. Based on this method, the rank of $\Delta W$ is adjusted according to an importance score. $\Delta W$ is divided into triplets and each triplet is scored according to its contribution to model performance. Triplets with low importance scores are pruned and triplets with high importance scores are kept for finetuning.

### Importance measurement

The importance of each adapter parameter is measured using the **gradient-based sensitivity score**. For each parameter $p$ in the adapter matrices (lora_A, lora_B, lora_E), the raw importance is computed as:

$$\text{ipt}_t = |p \cdot \nabla_p \mathcal{L}|$$

where $\nabla_p \mathcal{L}$ is the gradient of the loss with respect to that parameter. This captures how much each parameter contributes to reducing the loss.

To reduce noise, the raw importance scores are smoothed using two exponential moving averages:

- **Sensitivity** (smoothed importance): $\bar{\text{ipt}}_t = \beta_1 \cdot \bar{\text{ipt}}_{t-1} + (1 - \beta_1) \cdot \text{ipt}_t$
- **Uncertainty** (variability): $\bar{\text{u}}_t = \beta_2 \cdot \bar{\text{u}}_{t-1} + (1 - \beta_2) \cdot |\text{ipt}_t - \bar{\text{ipt}}_t|$

The final importance score for each element is the product of sensitivity and uncertainty:

$$\text{score} = \bar{\text{ipt}} \cdot \bar{\text{u}}$$

This scoring mechanism ensures that parameters that are both important (high sensitivity) and consistent (low uncertainty) receive the highest scores, while parameters with noisy or diminishing contributions are deprioritized.

### Triplet scoring and budget allocation

Each adapted layer has a triplet of matrices: lora_A (right singular vectors), lora_E (singular values), and lora_B (left singular vectors). The importance scores for each triplet are combined by:

1. Summing the lora_A scores along the rank dimension
2. Summing the lora_B scores along the rank dimension
3. Adding the lora_E scores to get a single triplet score

During the budgeting phase, triplets with scores below a threshold (determined by the current budget) are masked — their lora_E values are set to 0, effectively removing those rank components from the adapter. The budget decreases over time according to a cubic scheduler, starting from the initial rank (`init_r`) and converging to the target rank (`target_r`).

### Training phases

Training with AdaLoRA has three phases:

1. **Init phase** (step ≤ `tinit`): No budgeting is applied. All ranks are active, allowing the model to learn initial adapter weights without constraint. This is a warmup period.

2. **Budgeting phase** (`tinit` < step ≤ `total_step` - `tfinal`): The budget decreases from `init_r` to `target_r` following a cubic schedule. Every `deltaT` steps, importance scores are updated and triplets are masked according to the current budget. This gradually prunes less important rank components.

3. **Final phase** (step > `total_step` - `tfinal`): The budget is fixed at `target_r`. The final rank pattern is applied and training continues with the reduced ranks to further improve performance. No further budgeting occurs.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=ADALORA"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

## Usage


```py
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_healthcheck()

[... training code ...]

model.update_and_allocate(step_idx)
```

# API

## AdaLoraConfig

[[autodoc]] tuners.adalora.config.AdaLoraConfig

## AdaLoraModel

[[autodoc]] tuners.adalora.model.AdaLoraModel
