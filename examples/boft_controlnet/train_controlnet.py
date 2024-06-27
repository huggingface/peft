#!/usr/bin/env python
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The implementation is based on "Parameter-Efficient Orthogonal Finetuning
# via Butterfly Factorization" (https://arxiv.org/abs/2311.06243) in ICLR 2024.

import itertools
import logging
import math
import os
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils.args_loader import (
    import_model_class_from_model_name_or_path,
    parse_args,
)
from utils.dataset import collate_fn, log_validation, make_dataset
from utils.light_controlnet import ControlNetModel
from utils.tracemalloc import TorchTracemalloc, b2mb
from utils.unet_2d_condition import UNet2DConditionNewModel

from peft import BOFTConfig, get_peft_model
from peft.peft_model import PeftModel


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = get_logger(__name__)

UNET_TARGET_MODULES = ["to_q", "to_v", "to_k", "query", "value", "key"]

TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]


@torch.no_grad()
def save_adaptor(accelerator, output_dir, nets_dict):
    for net_key in nets_dict.keys():
        net_model = nets_dict[net_key]
        unwarpped_net = accelerator.unwrap_model(net_model)

        if isinstance(unwarpped_net, PeftModel):
            unwarpped_net.save_pretrained(
                os.path.join(output_dir, net_key),
                state_dict=accelerator.get_state_dict(net_model),
                safe_serialization=True,
            )
        else:
            accelerator.save_model(
                unwarpped_net,
                os.path.join(output_dir, net_key),
                safe_serialization=True,
            )


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if args.report_to == "wandb":
        wandb_init = {
            "wandb": {
                "name": args.wandb_run_name,
                "mode": "online",
            }
        }

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionNewModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    controlnet = ControlNetModel()

    if args.controlnet_model_name_or_path != "":
        logger.info(f"Loading existing controlnet weights from {args.controlnet_model_name_or_path}")
        controlnet.load_state_dict(torch.load(args.controlnet_model_name_or_path))

    if args.use_boft:
        config = BOFTConfig(
            boft_block_size=args.boft_block_size,
            boft_block_num=args.boft_block_num,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=UNET_TARGET_MODULES,
            boft_dropout=args.boft_dropout,
            bias=args.boft_bias,
        )
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()

    vae.requires_grad_(False)
    controlnet.requires_grad_(True)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    unet.train()
    controlnet.train()

    if args.train_text_encoder and args.use_boft:
        config = BOFTConfig(
            boft_block_size=args.boft_block_size,
            boft_block_num=args.boft_block_num,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            boft_dropout=args.boft_dropout,
            bias=args.boft_bias,
        )
        text_encoder = get_peft_model(text_encoder, config, adapter_name=args.wandb_run_name)
        text_encoder.print_trainable_parameters()

    if args.train_text_encoder:
        text_encoder.train()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
            if args.train_text_encoder and not (args.use_lora or args.use_boft or args.use_oft):
                text_encoder.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder and not (args.use_lora or args.use_boft or args.use_oft):
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [param for param in controlnet.parameters() if param.requires_grad]
    params_to_optimize += [param for param in unet.parameters() if param.requires_grad]

    if args.train_text_encoder:
        params_to_optimize += [param for param in text_encoder.parameters() if param.requires_grad]

    # Optimizer creation
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Load the dataset
    train_dataset = make_dataset(args, tokenizer, accelerator, "train")
    val_dataset = make_dataset(args, tokenizer, accelerator, "test")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    if args.train_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_project_name, config=vars(args), init_kwargs=wandb_init)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            if "checkpoint-current" in dirs:
                path = "checkpoint-current"
                dirs = [d for d in dirs if d.startswith("checkpoint") and d.endswith("0")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))

            else:
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            if path.split("-")[1] == "current":
                global_step = int(dirs[-1].split("-")[1])
            else:
                global_step = int(path.split("-")[1])

            initial_global_step = global_step
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        if args.report_to == "wandb":
                            accelerator.print(progress_bar)
                    continue

                with accelerator.accumulate(controlnet), accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                    # Get the guided hint for the UNet (320 dim)
                    guided_hint = controlnet(
                        controlnet_cond=controlnet_image,
                    )

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        guided_hint=guided_hint,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(controlnet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else itertools.chain(
                                controlnet.parameters(),
                            )
                        )

                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.report_to == "wandb":
                        accelerator.print(progress_bar)
                    global_step += 1

                    step_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    if accelerator.is_main_process:
                        if global_step % args.validation_steps == 0 or global_step == 1:
                            logger.info(f"Running validation... \n Generating {args.num_validation_images} images.")
                            logger.info("Running validation... ")

                            with torch.no_grad():
                                log_validation(val_dataset, text_encoder, unet, controlnet, args, accelerator)

                        if global_step % args.checkpointing_steps == 0:
                            save_adaptor(accelerator, step_save_path, {"controlnet": controlnet, "unet": unet})

                            # save text_encoder if any
                            if args.train_text_encoder:
                                save_adaptor(accelerator, step_save_path, {"text_encoder": text_encoder})

                            accelerator.save_state(step_save_path)

                            logger.info(f"Saved {global_step} state to {step_save_path}")
                            logger.info(f"Saved current state to {step_save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
