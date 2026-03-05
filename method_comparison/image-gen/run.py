# Copyright 2026-present the HuggingFace Inc. team.
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

"""Main entry point for image generation method comparison experiments.

Based on https://github.com/huggingface/diffusers/blob/bbbcdd87bd9d960fa372663a50b9edbdcb1391c6/examples/dreambooth/train_dreambooth_lora_flux2_klein.py
"""

import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any, Optional

import torch
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    offload_models,
)
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import set_seed
from utils import (
    FILE_NAME_TRAIN_PARAMS,
    TrainConfig,
    TrainResult,
    TrainStatus,
    get_artifact_stem,
    get_base_model_info,
    get_dataset_info,
    get_dino_embeddings,
    get_dino_encoder,
    get_file_size,
    get_optimizer_and_scheduler,
    get_peft_branch,
    get_pipeline,
    get_sample_image_save_dir,
    get_torch_dtype,
    get_train_config,
    init_accelerator,
    log_results,
    validate_experiment_path,
)

from data import get_train_valid_test_datasets
from peft import PeftConfig
from peft.utils import CONFIG_NAME, infer_device


ACCELERATOR_EMPTY_CACHE_SCHEDULE = 25
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"


def get_sigmas(timesteps, noise_scheduler, n_dim, dtype):
    device = "cpu"
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class DummyGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def precompute_prompt_caches(
    pipeline, prompts: list[str], device_type: str, train_config: TrainConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds_cache = []
    text_ids_cache = []
    with torch.no_grad(), offload_models(pipeline.text_encoder, device=device_type, offload=True):
        for prompt in prompts:
            prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=train_config.max_sequence_length,
                text_encoder_out_layers=train_config.text_encoder_out_layers,
            )
            prompt_embeds_cache.append(prompt_embeds)
            text_ids_cache.append(text_ids)
    return torch.cat(prompt_embeds_cache, dim=0).to(device_type), torch.cat(text_ids_cache, dim=0).to(device_type)


def precompute_latent_cache(
    *,
    pipeline,
    vae,
    pixel_values: list[torch.Tensor],
    train_config: TrainConfig,
    device_type: str,
) -> torch.Tensor:
    latents_cache = []
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    with torch.no_grad(), offload_models(vae, device=device_type, offload=True):
        latents_bn_mean = latents_bn_mean.to(vae.device)
        latents_bn_std = latents_bn_std.to(vae.device)
        for i in range(0, len(pixel_values), train_config.batch_size):
            pixel_values_batch = torch.stack(pixel_values[i : i + train_config.batch_size]).to(
                device=vae.device, dtype=get_torch_dtype(train_config.dtype)
            )
            latents = vae.encode(pixel_values_batch).latent_dist.mode()
            latents = pipeline._patchify_latents(latents)
            latents = (latents - latents_bn_mean) / latents_bn_std
            latents_cache.append(latents.to(device_type))
    return torch.cat(latents_cache, dim=0)


@torch.inference_mode()
def evaluate(
    *,
    pipeline,
    ds_eval,
    processor,
    dino_model,
    config: TrainConfig,
) -> float:
    generated_images = []
    reference_images = []
    batch_size = config.batch_size_eval

    with offload_models(pipeline.text_encoder, pipeline.vae, device=pipeline.transformer.device, offload=True):
        seed = config.seed + 100_000  # don't use the same seed
        for i in range(0, len(ds_eval), batch_size):
            sliced = [ds_eval[j] for j in range(i, min(i + batch_size, len(ds_eval)))]
            prompts = [sample["prompt"] for sample in sliced]
            generator = torch.Generator(device=pipeline.transformer.device).manual_seed(seed + i)
            outputs = pipeline(
                prompt=prompts,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.resolution,  # hard-code square
                width=config.resolution,
                max_sequence_length=config.max_sequence_length,
                text_encoder_out_layers=config.text_encoder_out_layers,
                generator=generator,
                output_type="pil",
            )
            generated_images.extend(outputs.images)
            reference_images.extend([sample["raw_image"] for sample in sliced])
            if i + batch_size >= len(ds_eval):
                break

    generated_embeddings = get_dino_embeddings(generated_images, processor, dino_model, batch_size=batch_size)
    reference_embeddings = get_dino_embeddings(reference_images, processor, dino_model, batch_size=batch_size)
    cosine_sim = (generated_embeddings * reference_embeddings).sum(dim=-1)
    return cosine_sim.mean().item()


def train(
    *,
    pipeline,
    train_config: TrainConfig,
    accelerator_memory_init: int,
    is_adalora: bool,
    print_verbose: Callable[..., None],
) -> TrainResult:
    accelerator_memory_allocated_log = []
    accelerator_memory_reserved_log = []
    losses = []
    durations = []
    metrics = []
    total_samples = 0

    device_type = infer_device()
    train_dataset, valid_dataset, test_dataset = get_train_valid_test_datasets(
        train_config=train_config, print_fn=print_verbose
    )
    train_size_base = len(train_dataset["prompts"])
    gen = torch.Generator(device=device_type).manual_seed(train_config.seed)
    train_indices = torch.cat(
        [torch.randperm(train_size_base, generator=gen, device=device_type) for _ in range(train_dataset["repeats"])]
    )
    if train_config.max_steps > len(train_indices):
        raise ValueError(
            f"max_steps is too high ({train_config.max_steps}), there are only {len(train_indices)} training samples"
        )

    processor, dino_model = get_dino_encoder(train_config.dino_model_id, train_config.dino_image_size)

    torch_accelerator_module = getattr(torch, device_type, torch.cuda)
    if train_config.use_amp:
        grad_scaler: GradScaler | DummyGradScaler = GradScaler(device=device_type)
        autocast_ctx: Callable[[], AbstractContextManager[Any]] = partial(autocast, device_type=device_type)
    else:
        grad_scaler = DummyGradScaler()
        autocast_ctx = nullcontext

    vae = pipeline.vae  # CPU
    transformer = pipeline.transformer.to(device_type)
    noise_scheduler_copy = copy.deepcopy(pipeline.scheduler)  # prevent mutating it
    optimizer, lr_scheduler = get_optimizer_and_scheduler(
        transformer,
        optimizer_type=train_config.optimizer_type,
        max_steps=train_config.max_steps,
        lr_scheduler_arg=train_config.lr_scheduler,
        **train_config.optimizer_kwargs,
    )

    if hasattr(transformer, "get_nb_trainable_parameters"):
        num_trainable_params, num_params = transformer.get_nb_trainable_parameters()
    else:
        num_params = sum(param.numel() for param in transformer.parameters())
        num_trainable_params = sum(param.numel() for param in transformer.parameters() if param.requires_grad)
    print_verbose(
        f"trainable params: {num_trainable_params:,d} || all params: {num_params:,d} || "
        f"trainable: {100 * num_trainable_params / num_params:.4f}%"
    )

    status = TrainStatus.FAILED
    tic_train = time.perf_counter()
    eval_time = 0.0
    error_msg = ""

    # pre-compute, since they don't change during training and we can keep the text encoder and VAE offloaded
    prompt_embeds_cache, text_ids_cache = precompute_prompt_caches(
        pipeline, train_dataset["prompts"], device_type, train_config=train_config
    )
    latents_cache = precompute_latent_cache(
        pipeline=pipeline,
        vae=vae,
        pixel_values=train_dataset["pixel_values"],
        train_config=train_config,
        device_type=device_type,
    )

    try:
        pbar = tqdm(range(1, train_config.max_steps + 1))
        for step in pbar:
            tic = time.perf_counter()
            i_start = (step - 1) * train_config.batch_size
            i_stop = min(step * train_config.batch_size, len(train_indices))
            batch_indices = train_indices[i_start:i_stop].to(device=latents_cache.device, dtype=torch.long)
            latents = latents_cache.index_select(0, batch_indices)
            prompt_embeds = prompt_embeds_cache.index_select(0, batch_indices)
            text_ids = text_ids_cache.index_select(0, batch_indices)

            current_batch_size = latents.shape[0]
            total_samples += current_batch_size

            model_input_ids = pipeline._prepare_latent_ids(latents).to(latents.device)
            noise = torch.randn_like(latents, generator=gen)

            u = compute_density_for_timestep_sampling(
                weighting_scheme=train_config.weighting_scheme,
                batch_size=current_batch_size,
                logit_mean=train_config.logit_mean,
                logit_std=train_config.logit_std,
                mode_scale=train_config.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
            # Add noise according to flow matching. zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(timesteps, noise_scheduler_copy, n_dim=latents.ndim, dtype=latents.dtype).to(
                device_type
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            # [B, C, H, W] -> [B, H*W, C]
            packed_noisy_latents = pipeline._pack_latents(noisy_latents)

            # handle guidance
            if transformer.config.guidance_embeds:
                guidance = torch.full([1], train_config.guidance_scale, device=device_type)
                guidance = guidance.expand(current_batch_size)
            else:
                guidance = None

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                model_pred = transformer(
                    hidden_states=packed_noisy_latents,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=model_input_ids,  # B, image_seq_len, 4
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_latents.size(1)]
                model_pred = pipeline._unpack_latents_with_ids(model_pred, model_input_ids)
                # these weighting schemes use a uniform timestep sampling and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(train_config.weighting_scheme, sigmas=sigmas)
                target = noise - latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
                )
                loss = loss.mean()

            grad_scaler.scale(loss).backward()
            if train_config.grad_norm_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), train_config.grad_norm_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()

            if is_adalora:
                transformer.base_model.update_and_allocate(step)

            losses.append(loss)
            pbar.set_postfix({"loss": loss.item()})

            accelerator_memory_allocated_log.append(
                torch_accelerator_module.memory_allocated() - accelerator_memory_init
            )
            accelerator_memory_reserved_log.append(
                torch_accelerator_module.memory_reserved() - accelerator_memory_init
            )
            toc = time.perf_counter()
            durations.append(toc - tic)

            if step % train_config.eval_steps == 0:
                tic_eval = time.perf_counter()
                loss_avg = sum(losses[-train_config.eval_steps :]) / train_config.eval_steps
                loss_avg = loss_avg.item()
                memory_allocated_avg = (
                    sum(accelerator_memory_allocated_log[-train_config.eval_steps :]) / train_config.eval_steps
                )
                memory_reserved_avg = (
                    sum(accelerator_memory_reserved_log[-train_config.eval_steps :]) / train_config.eval_steps
                )
                dur_train = sum(durations[-train_config.eval_steps :])

                transformer.eval()
                valid_similarity = 555
                valid_similarity = evaluate(
                    pipeline=pipeline,
                    ds_eval=valid_dataset,
                    processor=processor,
                    dino_model=dino_model,
                    config=train_config,
                )
                transformer.train()

                toc_eval = time.perf_counter()
                dur_eval = toc_eval - tic_eval
                eval_time += dur_eval
                elapsed = time.perf_counter() - tic_train

                metrics.append(
                    {
                        "step": step,
                        "valid dino_similarity": valid_similarity,
                        "train loss": loss_avg,
                        "train samples": total_samples,
                        "train time": dur_train,
                        "eval time": dur_eval,
                        "mem allocated avg": memory_allocated_avg,
                        "mem reserved avg": memory_reserved_avg,
                        "elapsed time": elapsed,
                    }
                )

                log_dict = {
                    "step": f"{step:4d}",
                    "samples": f"{total_samples:5d}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "loss avg": f"{loss_avg:.4f}",
                    "valid sim": f"{valid_similarity:.4f}",
                    "train time": f"{dur_train:.1f}s",
                    "eval time": f"{dur_eval:.1f}s",
                    "mem allocated": f"{memory_allocated_avg:.0f}",
                    "mem reserved": f"{memory_reserved_avg:.0f}",
                    "elapsed time": f"{elapsed // 60:.0f}min {elapsed % 60:.0f}s",
                }
                print_verbose(json.dumps(log_dict))

            if step % ACCELERATOR_EMPTY_CACHE_SCHEDULE == 0:
                torch_accelerator_module.empty_cache()

        print_verbose(f"Training finished after {train_config.max_steps} steps, evaluation on test set follows.")
        transformer.eval()
        test_similarity = evaluate(
            pipeline=pipeline,
            ds_eval=test_dataset,
            processor=processor,
            dino_model=dino_model,
            config=train_config,
        )

        metrics.append(
            {
                "step": step,
                "test dino_similarity": test_similarity,
                "train loss": (sum(losses[-train_config.eval_steps :]) / train_config.eval_steps).item(),
                "train samples": total_samples,
            }
        )
        print_verbose(f"Test DINOv2 similarity: {test_similarity:.4f}")

    except KeyboardInterrupt:
        print_verbose("canceled training")
        status = TrainStatus.CANCELED
        error_msg = "manually canceled"
    except torch.OutOfMemoryError as exc:
        print_verbose("out of memory error encountered")
        status = TrainStatus.CANCELED
        error_msg = str(exc)
    except Exception as exc:
        print_verbose(f"encountered an error: {exc}")
        status = TrainStatus.CANCELED
        error_msg = str(exc)

    toc_train = time.perf_counter()
    train_time = toc_train - tic_train - eval_time

    if status != TrainStatus.CANCELED:
        status = TrainStatus.SUCCESS
    train_result = TrainResult(
        status=status,
        train_time=train_time,
        accelerator_memory_reserved_log=accelerator_memory_reserved_log,
        losses=losses,
        metrics=metrics,
        error_msg=error_msg,
        num_trainable_params=num_trainable_params,
        num_total_params=num_params,
    )
    return train_result


@torch.inference_mode()
def generate_sample_images(
    *,
    pipeline,
    train_config,
    sample_image_dir: str,
    file_stem: str,
    print_verbose: Callable[..., None],
) -> None:
    vae_device = pipeline.vae.device
    te_device = pipeline.text_encoder.device
    target_device = pipeline.transformer.device
    pipeline.vae.to(target_device)
    pipeline.text_encoder.to(target_device)
    try:
        for idx, prompt in enumerate(train_config.sample_image_prompts, start=1):
            generator = torch.Generator(device=target_device).manual_seed(train_config.seed + 100_000 + idx)
            image = pipeline(
                prompt=[prompt],
                num_inference_steps=train_config.num_inference_steps,
                guidance_scale=train_config.guidance_scale,
                height=train_config.resolution,
                width=train_config.resolution,
                max_sequence_length=train_config.max_sequence_length,
                text_encoder_out_layers=train_config.text_encoder_out_layers,
                generator=generator,
                output_type="pil",
            ).images[0]
            image_path = os.path.join(sample_image_dir, f"{file_stem}_{idx:02d}.png")
            image.save(image_path)
    finally:
        pipeline.vae.to(vae_device)
        pipeline.text_encoder.to(te_device)


def main(*, path_experiment: str, experiment_name: str, clean: bool) -> None:
    tic_total = time.perf_counter()
    start_date = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

    peft_branch = get_peft_branch()
    if peft_branch == "main":
        print_verbose("===== This experiment is categorized as a MAIN run because the PEFT branch is 'main' ======")
    else:
        print_verbose(
            f"===== This experiment is categorized as a TEST run because the PEFT branch is '{peft_branch}' ======"
        )

    peft_config: Optional[PeftConfig] = None
    if os.path.exists(os.path.join(path_experiment, CONFIG_NAME)):
        peft_config = PeftConfig.from_pretrained(path_experiment)
    else:
        print_verbose(f"Could not find PEFT config at {path_experiment}, performing FULL FINETUNING")

    path_train_config = os.path.join(path_experiment, FILE_NAME_TRAIN_PARAMS)
    train_config = get_train_config(path_train_config)
    accelerator_memory_init = init_accelerator()
    set_seed(train_config.seed)

    model_info = get_base_model_info(train_config.model_id)
    dataset_info = get_dataset_info(train_config.dataset_id)
    pipeline = get_pipeline(
        model_id=train_config.model_id,
        dtype=train_config.dtype,
        compile=train_config.compile,
        peft_config=peft_config,
        autocast_adapter_dtype=train_config.autocast_adapter_dtype,
    )
    print_verbose(pipeline.transformer)

    train_result = train(
        pipeline=pipeline,
        train_config=train_config,
        accelerator_memory_init=accelerator_memory_init,
        is_adalora=peft_config is not None and peft_config.peft_type == "ADALORA",
        print_verbose=print_verbose,
    )

    if train_result.status == TrainStatus.FAILED:
        print_verbose("Training failed, not logging results")
        sys.exit(1)

    file_size = get_file_size(pipeline.transformer, peft_config=peft_config, clean=clean, print_fn=print_verbose)

    time_total = time.perf_counter() - tic_total
    log_results(
        experiment_name=experiment_name,
        train_result=train_result,
        accelerator_memory_init=accelerator_memory_init,
        time_total=time_total,
        file_size=file_size,
        model_info=model_info,
        dataset_info=dataset_info,
        start_date=start_date,
        train_config=train_config,
        peft_config=peft_config,
        print_fn=print_verbose,
    )

    if (train_result.status == TrainStatus.SUCCESS) and train_config.sample_image_prompts:
        print_verbose("Generating sample images")
        try:
            sample_image_dir = get_sample_image_save_dir(train_status=train_result.status, peft_branch=peft_branch)
            file_stem = get_artifact_stem(experiment_name, start_date, sample_image_dir)
            generate_sample_images(
                pipeline=pipeline,
                train_config=train_config,
                sample_image_dir=sample_image_dir,
                file_stem=file_stem,
                print_verbose=print_verbose,
            )
            print_verbose(f"Stored sample images in {sample_image_dir}")
        except Exception as exc:
            print_verbose(f"Sample image generation failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("path_experiment", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete training artifacts after run finishes (logs are still saved)",
    )
    args = parser.parse_args()

    experiment_name = validate_experiment_path(args.path_experiment)

    if args.verbose:

        def print_verbose(*args, **kwargs) -> None:
            kwargs["file"] = sys.stderr
            print(*args, **kwargs)
    else:

        def print_verbose(*args, **kwargs) -> None:
            pass

    main(
        path_experiment=args.path_experiment,
        experiment_name=experiment_name,
        clean=args.clean,
    )
