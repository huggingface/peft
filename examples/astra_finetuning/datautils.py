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

import os
import random

import numpy as np
import torch
from datasets import load_dataset


"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
doc https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000 * percent / 100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count())
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


llama_chat_format = """<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{instruction} [/INST] {response} </s>
"""


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    print(f" get_data_from: {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = f"cache/{name}_{model_id.replace('/', '_')}_{nsamples}_{seqlen}_{seed}.pt"
    traindataset = []
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        print(f"found data file: {cache_file}")
        traindataset = torch.load(cache_file)
        print("loaded ...")
        return traindataset
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
        )
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "traivia_qa":
        traindata = load_dataset("trivia_qa", "rc", split="train")
        tot_text = "\n\n".join(traindata["question"])
    elif name == "nqopen":
        traindata = load_dataset("nq_open", split="train")
        tot_text = "\n\n".join(traindata["question"])
    elif name == "alpaca":
        selected_data_dict = load_dataset("iboing/alpaca_data", split="train").shuffle(seed=seed).take(nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(instruction=example["instruction"], response=example["output"])
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "MetaMATH":
        selected_data_dict = load_dataset("iboing/MetaMathQA-395K", split="train").shuffle(seed=seed).take(nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(instruction=example["query"], response=example["response"])
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "codefeedback":
        selected_data_dict = (
            load_dataset("iboing/CodeFeedback-Filtered-Instruction", split="train").shuffle(seed=seed).take(nsamples)
        )
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(instruction=example["query"], response=example["answer"])
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    elif name == "WizLMinstruct":
        selected_data_dict = (
            load_dataset("iboing/WizardLM_evol_instruct_V2_143k", split="train").shuffle(seed=seed).take(nsamples)
        )
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(
                    instruction=example["conversation"][0]["human"], response=example["conversation"][0]["assistant"]
                )
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        print("example instruction:", s)
        torch.save(traindataset, cache_file)
        return traindataset
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError
