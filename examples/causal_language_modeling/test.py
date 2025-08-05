import random
from enum import Enum

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class SpecialTokens(str, Enum):
    begin_target = "<|begintarget|>"
    end_target = "<|endtarget|>"
    begin_context = "<|begincontext|>"
    end_context = "<|endcontext|>"
    system = "<|system|>"
    user = "<|user|>"
    begin_last_user_utterance = "<|beginlastuserutterance|>"
    end_last_user_utterance = "<|endlastuserutterance|>"
    begin_dsts = "<|begindsts|>"
    end_dsts = "<|enddsts|>"
    begin_dst = "<|begindst|>"
    end_dst = "<|enddst|>"
    begin_belief = "<|beginbelief|>"
    end_belief = "<|endbelief|>"
    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"
    begin_action = "<|beginaction|>"
    end_action = "<|endaction|>"
    begin_user_action = "<|beginuseraction|>"
    end_user_action = "<|enduseraction|>"
    sys_actions = "<|sysactions|>"
    begin_intent = "<|beginintent|>"
    end_intent = "<|endintent|>"
    begin_requested_slots = "<|beginrequestedslots|>"
    end_requested_slots = "<|endrequestedslots|>"
    pad_token = "<|pad|>"
    bos_token = "<|startoftext|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    pad_token=SpecialTokens.pad_token.value,
    bos_token=SpecialTokens.bos_token.value,
    eos_token=SpecialTokens.end_target.value,
    additional_special_tokens=SpecialTokens.list(),
)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
print(model.device)
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("smangrul/assistant_chatbot_dataset")
dataset = dataset["train"].train_test_split(0.2)
i = random.randint(0, len(dataset["test"]))
context = dataset["test"][i]["context"]

batch = tokenizer(context, return_tensors="pt")
batch = {k: v.to(device) for k, v in batch.items()}
model.eval()
output_tokens = model.generate(
    **batch,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
target_predicted = tokenizer.decode(output_tokens[0], skip_special_tokens=False).split("<|endcontext|>")[1]
target = dataset["test"][i]["target"]
print(f"{context=} \n\n {target_predicted=} \n\n {target=}")
