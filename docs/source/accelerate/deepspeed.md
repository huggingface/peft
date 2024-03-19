<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) is a library designed for speed and scale for distributed training of large models with billions of parameters. At its core is the Zero Redundancy Optimizer (ZeRO) that shards optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data parallel processes. This drastically reduces memory usage, allowing you to scale your training to billion parameter models. To unlock even more memory efficiency, ZeRO-Offload reduces GPU compute and memory by leveraging CPU resources during optimization.

Both of these features are supported in 🤗 Accelerate, and you can use them with 🤗 PEFT. 

## Compatibility with `bitsandbytes` quantization + LoRA

Below is a table that summarizes the compatibility between PEFT's LoRA, [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library and DeepSpeed Zero stages with respect to fine-tuning. DeepSpeed Zero-1 and 2 will have no effect at inference as stage 1 shards the optimizer states and stage 2 shards the optimizer states and gradients:

| DeepSpeed stage   | Is compatible? |
|---|---|
| Zero-1 |  🟢 |
| Zero-2   |  🟢 |
| Zero-3  |  🟢 |

For DeepSpeed Stage 3 + QLoRA, please refer to the section [Use PEFT QLoRA and DeepSpeed with ZeRO3 for finetuning large models on multiple GPUs](#use-peft-qlora-and-deepspeed-with-zero3-for-finetuning-large-models-on-multiple-gpus) below.

For confirming these observations, we ran the SFT (Supervised Fine-tuning) [offical example scripts](https://github.com/huggingface/trl/tree/main/examples) of the [Transformers Reinforcement Learning (TRL) library](https://github.com/huggingface/trl) using QLoRA + PEFT and the accelerate configs available [here](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs). We ran these experiments on a 2x NVIDIA T4 GPU.

Note DeepSpeed-Zero3 and `bitsandbytes` are currently **not** compatible.

# Use PEFT and DeepSpeed with ZeRO3 for finetuning large models on multiple devices and multiple nodes

This section of guide will help you learn how to use our DeepSpeed [training script](https://github.com/huggingface/peft/blob/main/examples/sft/train.py) for performing SFT. You'll configure the script to do SFT (supervised fine-tuning) of Llama-70B model with LoRA and ZeRO-3 on 8xH100 80GB GPUs on a single machine. You can configure it to scale to multiple machines by changing the accelerate config.

## Configuration

Start by running the following command to [create a DeepSpeed configuration file](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script) with 🤗 Accelerate. The `--config_file` flag allows you to save the configuration file to a specific location, otherwise it is saved as a `default_config.yaml` file in the 🤗 Accelerate cache.

The configuration file is used to set the default options when you launch the training script.

```bash
accelerate config --config_file deepspeed_config.yaml
```

You'll be asked a few questions about your setup, and configure the following arguments. In this example, you'll use ZeRO-3 so make sure you pick those options.

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them. Pass the same value as you would pass via cmd argument else you will encounter mismatch error.
`gradient_clipping`: Enable gradient clipping with value. Don't set this as you will be passing it via cmd arguments.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2. Set this as `none` as don't want to enable offloading.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3. Set this as `none` as don't want to enable offloading.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3. Set this to `True`.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3. Set this to `True`.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. Set this to `True`.
```

Once this is done, the corresponding config should look like below and you can find it in config folder at [deepspeed_config.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/deepspeed_config.yaml):

```yml
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 4
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Launch command

The launch command is available at [run_peft_deepspeed.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_deepspeed.sh) and it is also shown below:
```bash
accelerate launch --config_file "configs/deepspeed_config.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-lora-deepspeed" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False
```

Notice that we are using LoRA with  rank=8, alpha=16 and targeting all linear layers. We are passing the deepspeed config file and finetuning 70B Llama model on a subset of the ultrachat dataset.

## The important parts

Let's dive a little deeper into the script so you can see what's going on, and understand how it works.

The first thing to know is that the script uses DeepSpeed for distributed training as the DeepSpeed config has been passed. The `SFTTrainer` class handles all the heavy lifting of creating the PEFT model using the peft config that is passed. After that, when you call `trainer.train()`, `SFTTrainer` internally uses 🤗 Accelerate to prepare the model, optimizer and trainer using the DeepSpeed config to create DeepSpeed engine which is then trained. The main code snippet is below:

```python
# trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=data_args.packing,
    dataset_kwargs={
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    },
    dataset_text_field=data_args.dataset_text_field,
    max_seq_length=data_args.max_seq_length,
)
trainer.accelerator.print(f"{trainer.model}")

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
trainer.save_model()
```

## Memory usage

In the above example, the memory consumed per GPU is 64 GB (80%) as seen in the screenshot below:

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/peft_deepspeed_mem_usage.png"/>
</div>
<small>GPU memory usage for the training run</small>

## More resources
You can also refer this blog post [Falcon 180B Finetuning using 🤗 PEFT and DeepSpeed](https://medium.com/@sourabmangrulkar/falcon-180b-finetuning-using-peft-and-deepspeed-b92643091d99) on how to finetune 180B Falcon model on 16 A100 GPUs on 2 machines.


# Use PEFT QLoRA and DeepSpeed with ZeRO3 for finetuning large models on multiple GPUs

In this section, we will look at how to use QLoRA and DeepSpeed Stage-3 for finetuning 70B llama model on 2X40GB GPUs.
For this, we first need `bitsandbytes>=0.43.0`, `accelerate>=0.28.0`, `transformers>4.38.2`, `trl>0.7.11` and `peft>0.9.0`. We need to set `zero3_init_flag` to true when using Accelerate config. Below is the config which can be found at [deepspeed_config_z3_qlora.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/deepspeed_config_z3_qlora.yaml):

```yml
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Launch command is given below which is available at [run_peft_qlora_deepspeed_stage3.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_deepspeed.sh):
```
accelerate launch --config_file "configs/deepspeed_config_z3_qlora.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-qlora-dsz3" \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant True \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--bnb_4bit_quant_storage_dtype "bfloat16"
```

Notice the new argument being passed `bnb_4bit_quant_storage_dtype` which denotes the data type for packing the 4-bit parameters. For example, when it is set to `bfloat16`, **32/4 = 8** 4-bit params are packed together post quantization.

In terms of training code, the important code changes are: 

```diff
...

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.use_4bit_quantization,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.use_nested_quant,
+   bnb_4bit_quant_storage=quant_storage_dtype,
)

...

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
+   torch_dtype=quant_storage_dtype or torch.float32,
)
```

Notice that `torch_dtype` for `AutoModelForCausalLM` is same as the `bnb_4bit_quant_storage` data type. That's it. Everything else is handled by Trainer and TRL.

## Memory usage

In the above example, the memory consumed per GPU is **36.6 GB**. Therefore, what took 8X80GB GPUs with DeepSpeed Stage 3+LoRA and a couple of 80GB GPUs with DDP+QLoRA now requires 2X40GB GPUs. This makes finetuning of large models more accessible.

# Use PEFT and DeepSpeed with ZeRO3 and CPU Offloading for finetuning large models on a single GPU
This section of guide will help you learn how to use our DeepSpeed [training script](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py). You'll configure the script to train a large model for conditional generation with ZeRO-3 and CPU Offload.

<Tip>

💡 To help you get started, check out our example training scripts for [causal language modeling](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py) and [conditional generation](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py). You can adapt these scripts for your own applications or even use them out of the box if your task is similar to the one in the scripts.

</Tip>

## Configuration

Start by running the following command to [create a DeepSpeed configuration file](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script) with 🤗 Accelerate. The `--config_file` flag allows you to save the configuration file to a specific location, otherwise it is saved as a `default_config.yaml` file in the 🤗 Accelerate cache.

The configuration file is used to set the default options when you launch the training script.

```bash
accelerate config --config_file ds_zero3_cpu.yaml
```

You'll be asked a few questions about your setup, and configure the following arguments. In this example, you'll use ZeRO-3 along with CPU-Offload so make sure you pick those options.

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them.
`gradient_clipping`: Enable gradient clipping with value.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. 
```

An example [configuration file](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/accelerate_ds_zero3_cpu_offload_config.yaml) might look like the following. The most important thing to notice is that `zero_stage` is set to `3`, and `offload_optimizer_device` and `offload_param_device` are set to the `cpu`.

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
machine_rank: 0
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
```

## The important parts

Let's dive a little deeper into the script so you can see what's going on, and understand how it works.

Within the [`main`](https://github.com/huggingface/peft/blob/2822398fbe896f25d4dac5e468624dc5fd65a51b/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py#L103) function, the script creates an [`~accelerate.Accelerator`] class to initialize all the necessary requirements for distributed training.

<Tip>

💡 Feel free to change the model and dataset inside the `main` function. If your dataset format is different from the one in the script, you may also need to write your own preprocessing function. 

</Tip>

The script also creates a configuration for the 🤗 PEFT method you're using, which in this case, is LoRA. The [`LoraConfig`] specifies the task type and important parameters such as the dimension of the low-rank matrices, the matrices scaling factor, and the dropout probability of the LoRA layers. If you want to use a different 🤗 PEFT method, make sure you replace `LoraConfig` with the appropriate [class](../package_reference/tuners).

```diff
 def main():
+    accelerator = Accelerator()
     model_name_or_path = "facebook/bart-large"
     dataset_name = "twitter_complaints"
+    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
     )
```

Throughout the script, you'll see the [`~accelerate.Accelerator.main_process_first`] and [`~accelerate.Accelerator.wait_for_everyone`] functions which help control and synchronize when processes are executed.

The [`get_peft_model`] function takes a base model and the [`peft_config`] you prepared earlier to create a [`PeftModel`]:

```diff
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+ model = get_peft_model(model, peft_config)
```

Pass all the relevant training objects to 🤗 Accelerate's [`~accelerate.Accelerator.prepare`] which makes sure everything is ready for training:

```py
model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
)
```

The next bit of code checks whether the DeepSpeed plugin is used in the `Accelerator`, and if the plugin exists, then we check if we are using ZeRO-3. This conditional flag is used when calling `generate` function call during inference for syncing GPUs when the model parameters are sharded:

```py
is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
```

Inside the training loop, the usual `loss.backward()` is replaced by 🤗 Accelerate's [`~accelerate.Accelerator.backward`] which uses the correct `backward()` method based on your configuration:

```diff
  for epoch in range(num_epochs):
      with TorchTracemalloc() as tracemalloc:
          model.train()
          total_loss = 0
          for step, batch in enumerate(tqdm(train_dataloader)):
              outputs = model(**batch)
              loss = outputs.loss
              total_loss += loss.detach().float()
+             accelerator.backward(loss)
              optimizer.step()
              lr_scheduler.step()
              optimizer.zero_grad()
```

That is all! The rest of the script handles the training loop, evaluation, and even pushes it to the Hub for you.

## Train

Run the following command to launch the training script. Earlier, you saved the configuration file to `ds_zero3_cpu.yaml`, so you'll need to pass the path to the launcher with the `--config_file` argument like this:

```bash
accelerate launch --config_file ds_zero3_cpu.yaml examples/peft_lora_seq2seq_accelerate_ds_zero3_offload.py
```

You'll see some output logs that track memory usage during training, and once it's completed, the script returns the accuracy and compares the predictions to the labels:

```bash
GPU Memory before entering the train : 1916
GPU Memory consumed at the end of the train (end-begin): 66
GPU Peak Memory consumed during the train (max-begin): 7488
GPU Total Peak Memory consumed during the train (max): 9404
CPU Memory before entering the train : 19411
CPU Memory consumed at the end of the train (end-begin): 0
CPU Peak Memory consumed during the train (max-begin): 0
CPU Total Peak Memory consumed during the train (max): 19411
epoch=4: train_ppl=tensor(1.0705, device='cuda:0') train_epoch_loss=tensor(0.0681, device='cuda:0')
100%|████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:27<00:00,  3.92s/it]
GPU Memory before entering the eval : 1982
GPU Memory consumed at the end of the eval (end-begin): -66
GPU Peak Memory consumed during the eval (max-begin): 672
GPU Total Peak Memory consumed during the eval (max): 2654
CPU Memory before entering the eval : 19411
CPU Memory consumed at the end of the eval (end-begin): 0
CPU Peak Memory consumed during the eval (max-begin): 0
CPU Total Peak Memory consumed during the eval (max): 19411
accuracy=100.0
eval_preds[:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
```

# Caveats
1. Merging when using PEFT and DeepSpeed is currently unsupported and will raise error.
2. When using CPU offloading, the major gains from using PEFT to shrink the optimizer states and gradients to that of the adapter weights would be realized on CPU RAM and there won't be savings with respect to GPU memory.
3. DeepSpeed Stage 3 and qlora when used with CPU offloading leads to more GPU memory usage when compared to disabling CPU offloading. 
