#!/usr/bin/env python
"""
Script to show-case how to offset quantization error with LoRA / LoftQ
when dealing with quantizations that both quantize weights and activations.
This is the case for bnb int8, for example, but also other quantizations
such as BitNet do this.

The math for how this works is explained in MyLinear8bitLt.forward and
can be seen quickly when defining W_q = W + E_W (quantized weight is the
sum of the original weights plus an error term) and the same for
x_q = x + e_x.

To demonstrate the effectiveness, we load a unquantized model, generate
logits for reference inputs and then do the same for a quantized model
and a quantized model with LoftQ and our mitigations applied. The
error between reference model logits and LoftQ logits is significantly
smaller compared to the logits produced by the quantized model without
mitigation.

Note: set llm_int8_threshold=0 in your BitsAndBytesConfig. The thresholding
enables dynamic fp16 quantization (for x values above that threshold).
The quantization error for the affected values is much lower and not
static anymore, LoftQ is not able to deal with this. While technically
possible, this script doesn't filter out these masked values and it
is probably not worth the effort.

Note: Some rudimentary testing showed that the LoftQ mitigation is still
more effective than tuning threshold values but YMMV.

Note: LoftQ is not doing the heavy lifting in this script's case. The error
between LoftQ and zeroed LoRA is only about two percent points (check this
yourself by using the `--no-loftq` flag). This effect is probably dependent
on the quantization strength.

Examples of experiments you can do:

- check the difference between applying no-op LoRA and LoftQ initialized
  LoRA by running the following two commands:
    * ./int8_correction.py --no-mitigation
    * ./int8_correction.py --no-mitigation --no-loftq

- check the effect of the mitigation vs. the static compenstation by LoftQ:
    * ./int8_correction.py
    * ./int8_correction.py --no-loftq

- check the rank contribution for LoftQ:
    * for r in 8 16 32 64 128; do ./int8_correction.py --rank $i --no-mitigation; done

"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

import peft.tuners.lora.bnb
from peft import LoftQConfig, LoraConfig, PeftModel, TaskType, get_peft_model


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MyLinear8bitLt(peft.tuners.lora.bnb.Linear8bitLt):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)

                # The premise of this is that when we quantize, we introduce an
                # error. This means that quantizing x to xq we can state that
                # x = x_q + e_x or x_q = x - e_x. The same goes for W: W = W_q + E_W
                # or W_q = W - E_W.
                #
                # LoftQ computes E_W, applies SVD and initializes LoRA's B and A with
                # self.r ranks of E_W, giving us ~E_W. For our forward this means:
                # y = W_q x + BAx = (W_q + BA) x = (W_q + ~E_W) x = W x
                # if ~E_W is close enough to E_W.
                #
                # This breaks down if x is also quantized (as is the case for bnb int8):
                # y = W_q x_q + BA x_q = (W_q + ~E_W) x_q = W x_q = W (x - e_x) = Wx - W e_x
                #
                # Since e_x is non-zero and W is relatively large, it is a non-negligible
                # error term. But we can offset this, since we can compute e_x and we can
                # approximate W with W_q - or - in the case of LoftQ with ~E_W. Both work.
                # I didn't see a difference empirically but with other quantizations this
                # might change. In any case, we can compute ex_mitigation = W_q e_x and
                # add it along with the LoRA to remove the W e_x term and be left with
                # y = Wx.
                #
                # This is the term yielding the most error correction gain. There's a
                # smaller gain to be had by passing x_q to the LoRA's layers. Let's
                # revisit the quantized base layer definition:
                # y = W_q x_q = (W - E_W) (x - e_x) = Wx - W e_x - E_W x + E_W e_x
                #
                # (W e_x) we handled before, (- E_W x) is what we approximate with (~E_W x)
                # and is subsequently removed as well but this leaves us (E_W e_x).
                # It turns out, if you pass x_q into the LoRA modules, you will end up
                # with (~E_W x - ~E_W e_x) - which removes this term as well.

                # Compute x_q (int8 quantized x) to pass it into the LoRA's forward.
                CB, SCB, _ = bnb.functional.int8_vectorwise_quant(x.half())
                CB = CB.reshape(-1, CB.shape[-1])
                x_q = bnb.functional.int8_vectorwise_dequant(CB, SCB).to(lora_A.weight.dtype)
                x_q = x_q.reshape(*x.shape)

                e_x = x - x_q
                W_dq = bnb.functional.int8_vectorwise_dequant(self.base_layer.state.CB, self.base_layer.state.SCB).to(
                    e_x.dtype
                )
                e_x_mitigation = e_x @ W_dq.T
                # e_x_mitigation = e_x @ (W_dq.T + (lora_B.weight @ lora_A.weight * scaling).T)

                output = lora_B(lora_A(dropout(x_q))) * scaling
                output += e_x_mitigation
                if requires_conversion:
                    output = output.to(expected_dtype)
                result = result + output

        return result


parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-loftq", action="store_true", default=False, help="Disable LoftQ initialization (LoRA no-op init instead)"
)
parser.add_argument(
    "--no-mitigation", action="store_true", default=False, help="Disable activation quantization mitigiation"
)
parser.add_argument(
    "--model",
    choices=["t5-small", "t5-base", "t5-large", "facebook/opt-125m"],
    default="t5-base",
    help="What model to test.",
)
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--int8-threshold", type=float, default=0.0, help="To demonstrate that int8 threshold > 0 doesn't work"
)

args = parser.parse_args()

device = args.device
qconf = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=args.int8_threshold,
)
input_texts = [
    "All I need",
    "All I want is",
    "Forever yours truly: ",
    "Translate French to German: Tu l'as lu?",
    "Translate German to French: Last du es?",
    (
        "Beautiful is better than ugly.\n"
        "Explicit is better than implicit.\n"
        "Simple is better than complex.\n"
        "Complex is better than complicated.\n"
    ),
]
bits = 8
loftq_iter = 1
rank = args.rank

model_id = args.model

if "t5" in args.model:
    target_modules = ["o", "k", "wi", "q", "v"]
    task_type = TaskType.SEQ_2_SEQ_LM
else:
    target_modules = "all-linear"
    task_type = TaskType.CAUSAL_LM

# ----


def get_logits(model, inputs):
    torch.manual_seed(0)
    if task_type == TaskType.CAUSAL_LM:
        return model(**inputs).logits

    with torch.inference_mode():
        return model(**inputs, labels=inputs["input_ids"]).logits


def mse(a, b, attention_mask=None):
    squared_error = torch.pow(a - b, 2)
    if attention_mask is not None:
        # attention_mask shape: [batch_size, seq_len]
        # squared_error shape: [batch_size, seq_len, vocab_size]
        # apply the mask (zeros out the squared error for padding tokens)
        mask = attention_mask.unsqueeze(-1).expand_as(squared_error)
        masked_squared_error = squared_error * mask
        return (masked_squared_error.sum() / mask.sum()).item()
    return squared_error.mean().item()


def get_model(*args, **kwargs):
    if task_type == TaskType.CAUSAL_LM:
        return AutoModelForCausalLM.from_pretrained(*args, **kwargs)
    return AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)


tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)

ref_model = get_model(model_id, dtype=torch.float32, device_map=device)
qref_model = get_model(model_id, quantization_config=qconf, dtype=torch.float32, device_map=device)

loftq_config = LoftQConfig(loftq_bits=bits, loftq_iter=loftq_iter)
lora_config = LoraConfig(
    task_type=task_type,
    r=rank,
    init_lora_weights=True if args.no_loftq else "loftq",
    loftq_config=loftq_config,
    target_modules=target_modules,
)

base_model = get_model(model_id, dtype=torch.float32, device_map=device)
loftq_model = get_peft_model(base_model, lora_config)

print("APPLYING SAVED ADAPTER TO QUANTIZED MODEL")
with TemporaryDirectory() as tmp_path:
    tmp_path = Path(tmp_path)
    loftq_model.base_model.peft_config["default"].init_lora_weights = True
    loftq_model.save_pretrained(tmp_path / "loftq_model")

    lora_config = LoraConfig.from_pretrained(tmp_path / "loftq_model")
    model_id = args.model

    if not args.no_mitigation:
        custom_module_mapping = {bnb.nn.Linear8bitLt: MyLinear8bitLt}
        lora_config._register_custom_module(custom_module_mapping)

    base_model = get_model(model_id, quantization_config=qconf, dtype=torch.float32, device_map=device)
    loftq_model = PeftModel.from_pretrained(
        base_model, tmp_path / "loftq_model", is_trainable=True, config=lora_config
    )


ref_logits = get_logits(ref_model, inputs)

qref_logits = get_logits(qref_model, inputs)
loftq_logits = get_logits(loftq_model, inputs)

mse_loftq = mse(ref_logits, loftq_logits, attention_mask=inputs["attention_mask"])
mse_qref = mse(ref_logits, qref_logits, attention_mask=inputs["attention_mask"])


print(f"{model_id=}{device=}")
print(f"{mse_qref=}, {mse_loftq=}")
assert mse_loftq < (mse_qref / 1.05), f"{mse_loftq} >= {mse_qref / 1.05}"
print(f"relative reduction of error: {(mse_qref - mse_loftq) / mse_qref * 100:.2f}%")
