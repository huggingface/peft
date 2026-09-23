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

[AdaLoRA](https://hf.co/papers/2303.10512) (Adaptive LoRA) allocates trainable parameters across weight matrices according to importance, unlike LoRA, which uses a fixed rank for every adapted module. More important matrices receive a higher rank; less important ones are pruned toward a lower rank under a global parameter budget. You control the target average rank with `target_r`, the initial rank with `init_r`, and which modules to adapt with `target_modules`. Other important parameters include `lora_alpha` (scaling factor), `modules_to_save` (modules trained and saved outside the AdaLoRA layers), and the schedule knobs `tinit`, `tfinal`, `deltaT`, and `total_step`. All of these — and more — are found in [`AdaLoraConfig`].

> [!WARNING]
> AdaLoRA has an [`~AdaLoraModel.update_and_allocate`] method that must be called at each training step to update the parameter budget and mask; otherwise adaptation and budget updates do not run. This requires a custom training loop or subclassing [`~transformers.Trainer`] to call the method. As an example, see this [custom training loop](https://github.com/huggingface/peft/blob/912ad41e96e03652cabf47522cd876076f7a0c4f/examples/conditional_generation/peft_adalora_seq2seq.py#L120).

## How AdaLoRA works

LoRA represents each weight update $\Delta W$ with a fixed-rank factorization. AdaLoRA instead parameterizes $\Delta W$ in an SVD-like form with two orthogonal factors and a diagonal of singular values. That lets the method prune singular values of less important updates — reducing their effective rank — without repeatedly computing a full SVD.

Importance is estimated from training dynamics: for each adapter parameter, AdaLoRA tracks a sensitivity score (magnitude of weight × gradient) and an uncertainty term, both smoothed with exponential moving averages (`beta1`, `beta2`). Scores are combined per singular-value “triplet” (the corresponding entries in the three SVD factors). Triplets with low importance are masked out; high-importance triplets keep their budget. Rank reallocation happens on a decreasing cubic budget schedule between the init and final phases, at intervals controlled by `deltaT`.

Note that LoRA’s `r` is **not** used by AdaLoRA (setting it emits a warning). Use `init_r` for the starting rank of each incremental matrix and `target_r` for the target average rank after budgeting. You must also set `total_step` to the total number of training steps before training.

## Training phases

AdaLoRA training has three phases, defined by `tinit`, `tfinal`, and `total_step` (handled inside [`~AdaLoraModel.update_and_allocate`]):

1. **Init phase** (steps ≤ `tinit`): No rank redistribution. Adapters train at `init_r` so they accumulate signal before pruning. Budgeting is not applied yet.

2. **Budgeting phase** (after `tinit`, until `total_step - tfinal`): The global rank budget shrinks on a cubic schedule toward `target_r`. Every `deltaT` steps, importance scores drive reallocation: more important adapters keep higher rank; less important ones are pruned. Call `update_and_allocate(global_step)` each training step so scores and masks stay up to date (reallocation itself occurs when `global_step % deltaT == 0`, and once more when entering the final phase).

3. **Final phase** (last `tfinal` steps, i.e. after `total_step - tfinal`): Budgeting has ended; ranks are fixed. Training may continue with the redistributed ranks to refine performance. The mask from the finalized rank pattern is still applied via `update_and_allocate`.

Example schedule: `tinit=10`, `tfinal=20`, `total_step=100` → 10 init steps, 70 budgeting steps, 20 final steps.

## When to use AdaLoRA

AdaLoRA is a good choice when:

- You want adaptive rank allocation under a parameter budget, giving more capacity to important layers than fixed-rank LoRA.
- You can run a custom training loop (or a Trainer subclass) that calls `update_and_allocate` every step.
- You are fine-tuning under a tight parameter budget and expect uneven importance across modules.

## When not to use AdaLoRA

AdaLoRA may not be the best choice when:

- You want simpler fixed-rank adapters and maximum ecosystem simplicity — prefer standard [LoRA](lora).
- You rely on the default [`~transformers.Trainer`] without custom step hooks; AdaLoRA will not update budgets unless `update_and_allocate` is called.
- You do not want to set and reason about the schedule (`total_step`, `tinit`, `tfinal`, `deltaT`).

The abstract from the paper is:

*Fine-tuning large pre-trained language models on downstream tasks has become an important paradigm in NLP. However, common practice fine-tunes all of the parameters in a pre-trained model, which becomes prohibitive when a large number of downstream tasks are present. Therefore, many fine-tuning methods are proposed to learn incremental updates of pre-trained weights in a parameter efficient way, e.g., low-rank increments. These methods often evenly distribute the budget of incremental updates across all pre-trained weight matrices, and overlook the varying importance of different weight parameters. As a consequence, the fine-tuning performance is suboptimal. To bridge this gap, we propose AdaLoRA, which adaptively allocates the parameter budget among weight matrices according to their importance score. In particular, AdaLoRA parameterizes the incremental updates in the form of singular value decomposition. Such a novel approach allows us to effectively prune the singular values of unimportant updates, which is essentially to reduce their parameter budget but circumvent intensive exact SVD computations. We conduct extensive experiments with several pre-trained models on natural language processing, question answering, and natural language generation to validate the effectiveness of AdaLoRA. Results demonstrate that AdaLoRA manifests notable improvement over baselines, especially in the low budget settings. Our code is publicly available at https://github.com/QingruZhang/AdaLoRA*.

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
    target_r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    total_step=5000,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 520,325 || all params: 87,614,722 || trainable%: 0.5938785036606062"

[... training code ...]

# Call after loss.backward() / optimizer.step(), before zero_grad() — see the WARNING above
model.base_model.update_and_allocate(global_step)
```

# API

## AdaLoraConfig

[[autodoc]] tuners.adalora.config.AdaLoraConfig

## AdaLoraModel

[[autodoc]] tuners.adalora.model.AdaLoraModel
