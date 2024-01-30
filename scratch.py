import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict

model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM", torch_dtype=torch.float16, device_map="auto")
torch.manual_seed(3000)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_type=torch.float32,
)
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    quantization_config=bnb_config,
    torch_dtype=torch.float32,
)
random_input = torch.LongTensor([[1, 0, 1, 0, 1, 0]]).to(model.device)
# compare outputs in probability space, because logits can have outliers
# and token ids are not precise enough
out_base = F.softmax(model(random_input).logits, dim=-1)

config = LoraConfig(
    r=8,
    init_lora_weights=False,
)
model = get_peft_model(model, config)

with torch.inference_mode():
    out_before_merge = F.softmax(model(random_input).logits, dim=-1)

model.merge_and_unload()
with torch.inference_mode():
    out_after_merge = F.softmax(model(random_input).logits, dim=-1)

# tolerances are pretty high because some deviations are expected with quantization
atol = 0.01
rtol = 10

print(type(model.base_model.model.model.decoder.layers[0].self_attn.q_proj))



print(type(model.base_model.model.model.decoder.layers[0].self_attn.q_proj.get_base_layer()))
# adapter_model = PeftModel.from_pretrained(model, "peft-internal-testing/tiny-OPTForCausalLM-lora", is_trainable=True)
# for name, param in adapter_model.named_parameters():
#     print(name, param.dtype) 

# """
# base_model.model.model.layers.31.self_attn.v_proj.weight torch.float16
# base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.weight torch.float32 (correct)
# base_model.model.model.layers.31.self_attn.v_proj.lora_B.default.weight torch.float32
# """

# import pdb; pdb.set_trace()

# adapter_model.load_adapter("peft-internal-testing/tiny-OPTForCausalLM-lora", "another_lora", is_trainable=True)
# for name, param in adapter_model.named_parameters():
#     print(name, param.dtype)