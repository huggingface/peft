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

# Weight-Decomposed Low-Rank Adaptation (DoRA)

> [!NOTE]
> This is a variant of LoRA and therefore everything that is possible with LoRA is valid for this method except otherwise stated on this page.

This technique decomposes the updates of the weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, especially at low ranks. For more information on DoRA, see  https://huggingface.co/papers/2402.09353.

```py
from peft import LoraConfig

config = LoraConfig(use_dora=True, ...)
```

If parts of the model or the DoRA adapter are offloaded to CPU you can get a significant speedup at the cost of some temporary (ephemeral) VRAM overhead by using `ephemeral_gpu_offload=True` in `config.runtime_config`.

```py
from peft import LoraConfig, LoraRuntimeConfig

config = LoraConfig(use_dora=True, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True), ...)
```

A `PeftModel` with a DoRA adapter can also be loaded with `ephemeral_gpu_offload=True` flag using the `from_pretrained` method as well as the `load_adapter` method.

```py
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, peft_model_id, ephemeral_gpu_offload=True)
```

## Optimization

DoRA is optimized (computes faster and takes less memory) for models in the evaluation mode, or when dropout is set to 0. We reuse the
base result at those times to get the speedup.
Running [dora finetuning](https://github.com/huggingface/peft/blob/main/examples/dora_finetuning/dora_finetuning.py)
with `CUDA_VISIBLE_DEVICES=0 ZE_AFFINITY_MASK=0 time python examples/dora_finetuning/dora_finetuning.py --quantize --lora_dropout 0 --batch_size 16 --eval_step 2 --use_dora` on a 4090 with gradient accumulation set to 2 and max step to 20 resulted with the following observations:

| | Without Optimization | With Optimization |
| :--: | :--: | :--: |
| train runtime (sec) | 359.7298 | **279.2676** |
| train samples per second | 1.779 | **2.292** |
| train steps per second | 0.056 | **0.072** |

Moreover, it is possible to further increase runtime performance of DoRA by using the [`DoraCaching`] helper context. This requires the model to be in `eval` mode:

```py
from peft.helpers import DoraCaching

model.eval()
with DoraCaching():
    output = model(inputs)
```

For [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B), the [DoRA caching benchmark script](https://github.com/huggingface/peft/blob/main/examples/dora_finetuning/dora-caching.py) shows that, compared to LoRA:

- DoRA without caching requires 139% more time
- DoRA without caching requires 4% more memory
- DoRA with caching requires 17% more time
- DoRA with caching requires 41% more memory

Caching can thus make inference with DoRA significantly faster but it also requires signficantly more memory. Ideally, if the use case allows it, just merge the DoRA adapter to avoid both memory and runtime overhead.

## Caveats

- DoRA only supports embedding, linear, and Conv2d layers at the moment.
- DoRA introduces a bigger overhead than pure LoRA, so it is recommended to merge weights for inference, see [`LoraModel.merge_and_unload`].
- DoRA should work with weights quantized with bitsandbytes ("QDoRA"). However, issues have been reported when using QDoRA with DeepSpeed Zero2.

