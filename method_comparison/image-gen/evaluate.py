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

"""Evaluate an existing checkpoint from the image generation method comparison.

Loads a trained PEFT checkpoint on top of the same base model that was used for training and runs the same evaluation
as at the end of a training run (test set DINOv2 similarity, drift), then generates the sample images. The results and
sample images are always stored as temporary results.

Example:

python evaluate.py -v /path/to/checkpoint/

The checkpoint directory must contain the trained PEFT adapter (i.e. an adapter_config.json and the adapter weights).
This can e.g. be the temporary directory reported by run.py when called without the --clean flag or a checkpoint
downloaded from the Hugging Face Hub bucket. The training parameters are taken from default_training_params.json; if
the checkpoint was trained with different parameters, place the corresponding training_params.json into the checkpoint
directory.
"""

import argparse
import datetime as dt
import os
import sys
import time
from collections.abc import Callable

import torch
from run import evaluate, generate_sample_images, measure_drift
from transformers import set_seed
from utils import (
    FILE_NAME_TRAIN_PARAMS,
    RESULT_PATH_TEST,
    SAMPLE_IMAGE_PATH_TEST,
    TrainConfig,
    TrainResult,
    TrainStatus,
    get_artifact_stem,
    get_base_model_info,
    get_dataset_info,
    get_dino_encoder,
    get_file_size,
    get_pipeline,
    get_train_config,
    init_accelerator,
    log_results,
)

from data import get_train_valid_test_datasets
from peft import PeftConfig, PeftModel
from peft.utils import CONFIG_NAME, infer_device


def get_experiment_name(path_checkpoint: str) -> str:
    if not os.path.isdir(path_checkpoint):
        raise FileNotFoundError(f"Path {path_checkpoint} does not exist or is not a directory")
    return os.path.basename(os.path.normpath(path_checkpoint))


def evaluate_checkpoint(
    *,
    pipeline,
    train_config: TrainConfig,
    print_verbose: Callable[..., None],
) -> TrainResult:
    metrics = []
    device_type = infer_device()
    _, _, test_dataset = get_train_valid_test_datasets(train_config=train_config, print_fn=print_verbose)
    processor, dino_model = get_dino_encoder(train_config.dino_model_id, train_config.dino_image_size)

    torch_accelerator_module = getattr(torch, device_type, torch.cuda)
    transformer = pipeline.transformer.to(device_type)
    transformer.eval()

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
    error_msg = ""
    tic_eval_total = time.perf_counter()

    torch_accelerator_module.empty_cache()
    try:
        print_verbose("Evaluation on test set follows.")
        test_similarity = evaluate(
            pipeline=pipeline,
            ds_eval=test_dataset,
            processor=processor,
            dino_model=dino_model,
            config=train_config,
            num_repeats=3,
        )
        print_verbose("Calculating drift.")
        test_drift = measure_drift(pipeline=pipeline, processor=processor, dino_model=dino_model, config=train_config)
        metrics.append(
            {
                "test dino_similarity": test_similarity,
                "drift": test_drift,
                "eval time": time.perf_counter() - tic_eval_total,
            }
        )
        print_verbose(f"Test DINOv2 similarity: {test_similarity:.4f}")
        print_verbose(f"Test drift:             {test_drift:.4f}")

    except KeyboardInterrupt:
        print_verbose("canceled evaluation")
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

    if status != TrainStatus.CANCELED:
        status = TrainStatus.SUCCESS
    # the train-related attributes are set to empty/zero values, as no training is performed
    eval_result = TrainResult(
        status=status,
        train_time=0.0,
        accelerator_memory_reserved_log=[],
        accelerator_memory_max_train=0,
        losses=[],
        metrics=metrics,
        error_msg=error_msg,
        num_trainable_params=num_trainable_params,
        num_total_params=num_params,
    )
    return eval_result


def main(*, path_checkpoint: str, experiment_name: str) -> None:
    tic_total = time.perf_counter()
    start_date = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

    print_verbose("===== The results of this evaluation run are stored as temporary results ======")

    if not os.path.exists(os.path.join(path_checkpoint, CONFIG_NAME)):
        raise FileNotFoundError(
            f"Could not find a PEFT config at {path_checkpoint}. Note that evaluating full fine-tuning checkpoints is "
            "not supported."
        )
    peft_config = PeftConfig.from_pretrained(path_checkpoint)

    path_train_config = os.path.join(path_checkpoint, FILE_NAME_TRAIN_PARAMS)
    if not os.path.exists(path_train_config):
        print_verbose(
            f"Could not find {FILE_NAME_TRAIN_PARAMS} in {path_checkpoint}, using the default training parameters"
        )
    train_config = get_train_config(path_train_config)
    init_accelerator()
    set_seed(train_config.seed)

    model_info = get_base_model_info(train_config.model_id)
    dataset_info = get_dataset_info(train_config.dataset_id)
    # create the pipeline with the plain base model first, then load the trained adapter onto it; compilation, if
    # enabled, must come last, mirroring the order in get_pipeline
    pipeline = get_pipeline(
        model_id=train_config.model_id,
        dtype=train_config.dtype,
        compile=False,
        peft_config=None,
        autocast_adapter_dtype=train_config.autocast_adapter_dtype,
        use_gc=train_config.use_gc,
    )
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        path_checkpoint,
        is_trainable=True,  # to report the same number of trainable parameters as during training
        autocast_adapter_dtype=train_config.autocast_adapter_dtype,
    )
    if train_config.compile:
        pipeline.transformer = torch.compile(pipeline.transformer, dynamic=True)
    print_verbose(pipeline.transformer)

    eval_result = evaluate_checkpoint(
        pipeline=pipeline,
        train_config=train_config,
        print_verbose=print_verbose,
    )

    file_size = get_file_size(pipeline.transformer, peft_config=peft_config, clean=True, print_fn=print_verbose)

    time_total = time.perf_counter() - tic_total
    log_results(
        experiment_name=experiment_name,
        train_result=eval_result,
        time_total=time_total,
        file_size=file_size,
        model_info=model_info,
        dataset_info=dataset_info,
        start_date=start_date,
        train_config=train_config,
        peft_config=peft_config,
        print_fn=print_verbose,
        save_dir=RESULT_PATH_TEST,  # results of evaluation-only runs are always treated as temporary results
    )

    if (eval_result.status == TrainStatus.SUCCESS) and train_config.sample_image_prompts:
        print_verbose("Generating sample images")
        try:
            file_stem = get_artifact_stem(experiment_name, start_date, SAMPLE_IMAGE_PATH_TEST)
            generate_sample_images(
                pipeline=pipeline,
                train_config=train_config,
                sample_image_dir=SAMPLE_IMAGE_PATH_TEST,
                file_stem=file_stem,
            )
            print_verbose(f"Stored sample images in {SAMPLE_IMAGE_PATH_TEST}")
        except Exception as exc:
            print_verbose(f"Sample image generation failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "path_checkpoint", type=str, help="Path to the directory containing the trained PEFT checkpoint"
    )
    args = parser.parse_args()

    experiment_name = get_experiment_name(args.path_checkpoint)

    if args.verbose:

        def print_verbose(*args, **kwargs) -> None:
            kwargs["file"] = sys.stderr
            print(*args, **kwargs)
    else:

        def print_verbose(*args, **kwargs) -> None:
            pass

    main(path_checkpoint=args.path_checkpoint, experiment_name=experiment_name)
