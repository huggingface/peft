import pdb
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_id = "ibm-granite/granite-3.1-8b-instruct"
default_model = AutoModelForCausalLM.from_pretrained(model_id)
assert default_model.config.tie_word_embeddings, "Model config does not have weight tying enabled"
assert (default_model.lm_head.weight != default_model.model.embed_tokens.weight).sum() == 0, "Embedding layer and LM head are not the same"

## CASE 1
print("Adding embed tokens as a modules_to_save")

config = LoraConfig(task_type="CAUSAL_LM", modules_to_save=["embed_tokens"], target_modules=["q_proj"])
peft_model = get_peft_model(default_model, config)

print(type(peft_model.base_model.model.model.embed_tokens))
# prints <class 'peft.utils.other.ModulesToSaveWrapper'>
print(type(peft_model.base_model.model.lm_head))
# prints <class 'torch.nn.modules.linear.Linear'>

del (peft_model)
del (default_model)

# ## CASE 2
print("Not adding embed tokens as a modules_to_save")
default_model = AutoModelForCausalLM.from_pretrained(model_id)
config = LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj"])
peft_model = get_peft_model(default_model, config)

print(type(peft_model.base_model.model.model.embed_tokens))
print(type(peft_model.base_model.model.lm_head))

del (peft_model)
del (default_model)

# ## CASE 3
print("Adding embed tokens as a target_modules")

default_model = AutoModelForCausalLM.from_pretrained(model_id)
config = LoraConfig(task_type="CAUSAL_LM", target_modules=["embed_tokens"])
peft_model = get_peft_model(default_model, config)

print(type(peft_model.base_model.model.model.embed_tokens))
print(type(peft_model.base_model.model.lm_head))

del (peft_model)
del (default_model)
