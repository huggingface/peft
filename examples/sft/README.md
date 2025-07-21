# Supervised Fine-tuning (SFT) with PEFT
In this example, we'll see how to use [PEFT](https://github.com/huggingface/peft) to perform SFT using PEFT on various distributed setups.

## Single GPU SFT with QLoRA
QLoRA uses 4-bit quantization of the base model to drastically reduce the GPU memory consumed by the base model while using LoRA for parameter-efficient fine-tuning. The command to use QLoRA is present at [run_peft.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft.sh).

Note: 
1. At present, `use_reentrant` needs to be `True` when using gradient checkpointing with QLoRA else QLoRA leads to high GPU memory consumption.


## Single GPU SFT with QLoRA using Unsloth
[Unsloth](https://github.com/unslothai/unsloth) enables finetuning Mistral/Llama 2-5x faster with 70% less memory. It achieves this by reducing data upcasting, using Flash Attention 2, custom Triton kernels for RoPE embeddings, RMS Layernorm & Cross Entropy Loss and manual clever autograd computation to reduce the FLOPs during QLoRA finetuning. Below is the list of the optimizations from the Unsloth blogpost [mistral-benchmark](https://unsloth.ai/blog/mistral-benchmark). The command to use QLoRA with Unsloth is present at [run_unsloth_peft.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_unsloth_peft.sh).

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/Unsloth.png"/>
</div>
<small>Optimization in Unsloth to speed up QLoRA finetuning while reducing GPU memory usage</small>

## Multi-GPU SFT with QLoRA
To speed up QLoRA finetuning when you have access to multiple GPUs, look at the launch command at [run_peft_multigpu.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_multigpu.sh). This example to performs DDP on 8 GPUs.

Note: 
1. At present, `use_reentrant` needs to be `False` when using gradient checkpointing with Multi-GPU QLoRA else it will lead to errors. However, this leads to huge GPU memory consumption. 

## Multi-GPU SFT with LoRA and DeepSpeed
When you have access to multiple GPUs, it would be better to use normal LoRA with DeepSpeed/FSDP. To use LoRA with DeepSpeed, refer to the docs at [PEFT with DeepSpeed](https://huggingface.co/docs/peft/accelerate/deepspeed).


## Multi-GPU SFT with LoRA and FSDP
When you have access to multiple GPUs, it would be better to use normal LoRA with DeepSpeed/FSDP. To use LoRA with FSDP, refer to the docs at [PEFT with FSDP](https://huggingface.co/docs/peft/accelerate/fsdp).


## Multi-GPU SFT with LoRA and FSDP for GPTQModel:
As in [Multi-GPU SFT with LoRA and FSDP](https://github.com/huggingface/peft/blob/main/examples/sft/README.md#multi-gpu-sft-with-lora-and-fsdp), we also support other quantization methods like GPTQModel. You may need to install [GPTQModel](https://github.com/ModelCloud/GPTQModel) > v3.0.0 or from source. Here is the launch command for reference: [run_peft_fsdp_gptq.sh]. For the `--model_name_or_path` argument, it is important to pass a model that is already quantized with GPTQModel, like `"hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"`.

Note: there is a bug in transformers v4.53.0 for this case, please skip this transformers version.

## Tip

Generally try to upgrade to the latest package versions for best results, especially when it comes to `bitsandbytes`, `accelerate`, `transformers`, `trl`, and `peft`.
