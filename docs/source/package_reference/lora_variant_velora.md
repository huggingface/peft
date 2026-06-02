<!--Copyright 2026-present The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

### VeLoRA

[VeLoRA](https://huggingface.co/papers/2405.17991) is a LoRA variant that reduces training memory by compressing the activations saved for the LoRA in the forward pass and then reconstructing them in the backwards pass to implement the update rules. In PEFT, VeLoRA is configured as a LoRA variant through the `velora_config` argument on [`LoraConfig`].

```py
from peft import LoraConfig, VeloraConfig

config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    velora_config=VeloraConfig(
        num_groups=64,
        scale=0.2,
        init_type="batch_average",
    ),
)
```

VeLoRA is applied to every LoRA layer selected by `target_modules`. `num_groups` controls how the input activation depth is split before compression. If the activation depth is not evenly divisible by `num_groups`, VeLoRA pads the grouped representation internally and removes the padding after reconstruction. `scale` rescales the reconstructed activations during the backward pass, and `init_type` chooses how the projection is initialized.

Use `batch_average_once` to initialize the projection from the first training batch, `batch_average` to update it from every training forward pass, or `random` to initialize it immediately from a random normalized vector.

Below are some results with the [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA).

| Variant | Training Loss | Max Memory (GiB) | Tokens/sec |
|---|---:|---:|---:|
| LoRA | 0.5427 | 27.69 | 2366.2 |
| LoRA + GC | 0.5426 | 13.17 | 1671.8 |
| LoRA+VeLoRA | 0.5427 | 19.94 | 2057.6 |

#### Caveats

- VeLoRA is currently supported on standard LoRA linear layers only.

