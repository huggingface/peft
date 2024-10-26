# Copyright 2024-present the HuggingFace Inc. team.
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

import copy
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import datasets
import numpy as np
import torch
import torch.distributed
import transformers
from datasets import load_dataset
from transformers import BitsAndBytesConfig, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import BoneConfig, LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"
logger = logging.getLogger(__name__)

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # Bone setting
    use_bone: Optional[bool] = field(default=False)
    bone_r: Optional[int] = field(default=64)
    init_bone_weights: Literal[True, False] = field(default=True)
    # Lora or PiSSA setting
    use_lora: Optional[bool] = field(default=False)
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pre-initialized Bone adapter path; when this is not None, the following arguments are ignored."
            ),
        },
    )
    init_lora_weights: Literal[True, "pissa_niter_4", "olora"] = field(
        default=True,
        metadata={
            "help": ("True -> LoRA; `pissa` -> PiSSA"),
        },
    )
    target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=32.0)
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": ("Must be set to 0 when using PiSSA."),
        },
    )
    # Quantization setting
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    merge: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge the Bone adapter to the residual model or LoRA to the base model"},
    )
    bf16: Optional[bool] = field(default=True)
    run_name: str = field(default="None", metadata={"help": "Path to the training data."})


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, "completed"))
        if is_completed:
            return None  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + "-", "")))
        if max_step == 0:
            return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{max_step}")
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None  # first training


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return {"input_ids": input_ids, "labels": labels, "input_ids_lens": input_ids_lens, "labels_lens": labels_lens}


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = (_tokenize_fn(strings, tokenizer) for strings in (examples, sources))
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(self.tokenizer.pad_token_id)}


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map({"instruction": instruction}) for instruction in examples[query]]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def build_model(script_args, checkpoint_dir):
    # if not script_args.use_lora and not script_args.use_bone: assert script_args.bits in [16, 32]
    compute_dtype = torch.bfloat16 if script_args.bf16 else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        )
        if script_args.bits in [4, 8]
        else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    if compute_dtype == torch.float32 and script_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info("=" * 80)
            logger.info("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            logger.info("=" * 80)
    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)
    # Tokenizer

    if script_args.use_lora and script_args.bits < 16:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

    if script_args.use_lora:
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None:
            logger.info(
                f"Initilize adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}."
            )
            lora_config = LoraConfig.from_pretrained(
                script_args.model_name_or_path, subfolder=script_args.adapter_name_or_path
            )
            lora_config.lora_dropout = script_args.lora_dropout
            model = PeftModel.from_pretrained(
                model,
                script_args.model_name_or_path,
                subfolder=script_args.adapter_name_or_path,
                is_trainable=True,
                config=lora_config,
            )
        else:
            logger.info("Init LoRA modules...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(","),
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_lora_weights,
            )
            model = get_peft_model(model, peft_config)
    if script_args.use_bone:
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            bone_config = BoneConfig.from_pretrained(script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path)
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path, is_trainable=True, config=bone_config)
        else:
            logger.info(f'Init LoRA modules...')
        peft_config = BoneConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=script_args.target_modules.split(","),
            inference_mode=False,
            r=script_args.bone_r,
            init_weights=script_args.init_bone_weights,
        )
        model = get_peft_model(model, peft_config, adapter_name="weight")

    for name, module in model.named_modules():
        if "norm" in name or "gate" in name:
            module = module.to(torch.float32)
    return model


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if script_args.local_rank == 0:
        logger.info("=" * 100)
        logger.info(script_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    logger.info("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    logger.info("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if script_args.local_rank == 0:
        logger.info(f"Load tokenizer from {script_args.model_name_or_path} over.")

    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)

    raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)

    if script_args.local_rank > 0:
        torch.distributed.barrier()

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": script_args.dataset_field[0],
            "response": script_args.dataset_field[1],
        },
    )

    if script_args.local_rank == 0:
        torch.distributed.barrier()
        print(model)
        model.print_trainable_parameters()
        logger.info("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}."
            )
            logger.info(
                f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}."
            )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = {"train_dataset": train_dataset, "eval_dataset": None, "data_collator": data_collator}

    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    if script_args.use_lora:
        trainer.add_callback(SavePeftModelCallback)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_dir)
    trainer.save_state()
    if script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if not script_args.use_lora and not script_args.use_bone:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)


if __name__ == "__main__":
    train()
