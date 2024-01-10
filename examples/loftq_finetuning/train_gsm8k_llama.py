# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer


from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    full_precision:  bool = field(
        default=False,
        metadata={"help": "True: Use bitsandbytes Linear4bit, real quantization"
                          "False: Use quantization equivalent fp16/fp32 weights."
                          "Note: Set False for data parallel training"
                  },
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    bits: int = field(
        default=4,
        metadata={"help": "Bit of the backbone. LoftQ does not require this config. Used for QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    train_set = dataset['train']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
        )
    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model,
                                          model_args.adapter_name_or_path,
                                          is_trainable=True,
                                          token=model_args.token,
                                          )
    else:
        model = PeftModel.from_pretrained(model,
                                          model_args.model_name_or_path,
                                          subfolder='loftq_init',
                                          is_trainable=True,
                                          token=model_args.token,
                                          )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
