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

# PaCA: Partial Connection Adaptation

[PaCA](https://huggingface.co/papers/2503.01905) (ICLR 2025) is a PEFT method that targets training speed and activation
memory. Instead of adding adapter layers, it fine-tunes `r` randomly selected input columns ("partial connections") of
each targeted pretrained weight.

Adapter methods such as LoRA run a small adapter branch next to every pretrained layer. The branch needs few FLOPs, but
it is executed sequentially with the pretrained layer and it keeps its own copy of the input activation for the backward
pass. PaCA avoids both:

- **One GEMM per layer.** The adapted columns are written into the pretrained weight right before its matmul and restored
  right after it, so forward and backward each run a single matmul, exactly like the frozen base layer.
- **Partial activations.** The gradient of the adapted columns only depends on the matching `r` input channels, so only
  those are saved for the backward pass instead of the full input.

In terms of the resulting weights, a PaCA adapter is equivalent to a LoRA adapter of rank `r` whose `A` matrix is a fixed
column selector: `W' = W + delta @ S`, where `S` selects the `r` columns. It can therefore be converted to a LoRA adapter
of rank `r` without loss via [`~tuners.lora.conversion.convert_to_lora`], e.g. to serve it with a LoRA inference stack.

When to use PaCA:

- You want LoRA-level accuracy but faster fine-tuning steps and lower activation memory, e.g. to fit longer sequences or
  larger batches on the same GPU.
- You want an adapter that merges exactly into the base weights (the update is a plain column update).

Current constraints:

- Only `nn.Linear` layers (including quantized linear layers) are supported.
- PaCA runs the matmul itself instead of calling the base layer's `forward`, so forward hooks registered on the base layer
  are not called (a warning is emitted). Register hooks on the PaCA layer instead.
- Mixed adapter batches (`adapter_names` argument) are not supported.
- Training with sharded weights (FSDP, DeepSpeed ZeRO-3) has not been tested yet.
- `torch.compile` is supported: under compilation the column swap, matmul and restore run as opaque custom ops (`torch.ops.peft.paca_linear_forward`/`_backward`), so the compiler does not copy the base weights. This requires `torch.library.custom_op` (PyTorch >= 2.4); otherwise the eager implementation is traced.
- `paca_alpha / r` scales the column update and therefore acts as a multiplier of the effective learning rate.
- For inference, merge the adapter (`merge_and_unload()`); the unmerged forward still swaps the columns in and out on every call.

## Usage

```python
from transformers import AutoModelForCausalLM
from peft import PacaConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
config = PacaConfig(r=16, paca_alpha=16, target_modules="all-linear", random_seed=0)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

The abstract from the paper is:

> Prior parameter-efficient fine-tuning (PEFT) algorithms reduce memory usage and computational costs of fine-tuning
> large neural network models by training only a few additional adapter parameters, rather than the entire model.
> However, the reduction in computational costs due to PEFT does not necessarily translate to a reduction in training
> time; although the computational costs of the adapter layers are much smaller than the pretrained layers, it is well
> known that those two types of layers are processed sequentially on GPUs, resulting in significant latency overhead.
> LoRA and its variants avoid this latency overhead by merging the low-rank adapter matrices with the pretrained weights
> during inference. However, those layers cannot be merged during training since the pretrained weights must remain
> frozen while the low-rank adapter matrices are updated continuously over the course of training. Furthermore, LoRA
> and its variants do not reduce activation memory, as the first low-rank adapter matrix still requires the input
> activations to the pretrained weights to compute weight gradients. To mitigate this issue, we propose Partial
> Connection Adaptation (PaCA), which fine-tunes randomly selected partial connections within the pretrained weights
> instead of introducing adapter layers in the model. PaCA not only enhances training speed by eliminating the time
> overhead due to the sequential processing of the adapter and pretrained layers but also reduces activation memory
> since only partial activations, rather than full activations, need to be stored for gradient computation. Compared to
> LoRA, PaCA reduces training time by 22% and total memory usage by 16%, while maintaining comparable accuracy across
> various fine-tuning scenarios, such as fine-tuning on the MMLU dataset and instruction tuning on the Oasst1 dataset.
> PaCA can also be combined with quantization, enabling the fine-tuning of large models such as LLaMA3.1-70B. In
> addition, PaCA enables training with 23% longer sequence and improves throughput by 16% on both NVIDIA A100 GPU and
> INTEL Gaudi2 HPU compared to LoRA.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=PACA"
	frameborder="0"
	width="850"
	height="1000"
></iframe>

# API

## PacaConfig

[[autodoc]] tuners.paca.config.PacaConfig

## PacaModel

[[autodoc]] tuners.paca.model.PacaModel
