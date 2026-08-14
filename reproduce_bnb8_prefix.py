"""
Minimal reproducer for BNB 8bit + prefix tuning CUDA illegal memory access.

Tests two scenarios:
1. PEFT prefix tuning + BNB 8bit (reproduces the CI failure)
2. No PEFT — BNB 8bit model with manually injected prefix KV cache
   (tests if the issue is PEFT-specific or BNB-specific)

Also runs a control: fp16 model + PEFT prefix tuning (should pass).
"""

import os
import sys
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DynamicCache

print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes: {bnb.__version__}")
except ImportError:
    print("bitsandbytes: not installed")

import transformers
print(f"transformers: {transformers.__version__}")

MODEL_ID = "peft-internal-testing/opt-125m"
NUM_VIRTUAL_TOKENS = 10

def get_model_config_info(model):
    """Extract config info needed for prefix tuning."""
    config = model.config
    return {
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
    }


def test_peft_prefix_tuning_8bit():
    """Reproduce the exact CI failure: PEFT prefix tuning on BNB 8bit model."""
    from peft import PrefixTuningConfig, TaskType, get_peft_model

    print("\n" + "=" * 70)
    print("TEST 1: PEFT prefix tuning + BNB 8bit (reproduces CI failure)")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_info = get_model_config_info(model)
    print(f"Model config: {config_info}")

    peft_config = PrefixTuningConfig(num_virtual_tokens=NUM_VIRTUAL_TOKENS, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    model.train()

    # Simple forward pass (no Trainer, no optimizer)
    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    print("Running forward pass...")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
        print(f"  Forward pass succeeded. Loss: {outputs.loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  Forward pass FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        del model
        torch.cuda.empty_cache()


def test_peft_prefix_tuning_fp16():
    """Control: PEFT prefix tuning on fp16 model (should work)."""
    from peft import PrefixTuningConfig, TaskType, get_peft_model

    print("\n" + "=" * 70)
    print("TEST 2: PEFT prefix tuning + fp16 (control — should work)")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_info = get_model_config_info(model)
    print(f"Model config: {config_info}")

    peft_config = PrefixTuningConfig(num_virtual_tokens=NUM_VIRTUAL_TOKENS, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    model.train()

    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    print("Running forward pass...")
    try:
        outputs = model(**inputs)
        print(f"  Forward pass succeeded. Loss: {outputs.loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  Forward pass FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        del model
        torch.cuda.empty_cache()


def test_bnb8_no_peft_prefix_kv():
    """
    No PEFT — BNB 8bit model with manually injected prefix KV cache.
    Tests if the issue is PEFT-specific or a BNB 8bit + DynamicCache interaction.
    """
    print("\n" + "=" * 70)
    print("TEST 3: BNB 8bit + manual prefix KV cache (no PEFT)")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_info = get_model_config_info(model)
    print(f"Model config: {config_info}")

    num_layers = config_info["num_hidden_layers"]
    num_heads = config_info["num_attention_heads"]
    head_dim = config_info["head_dim"]

    # Create a DynamicCache with prefix KV values (mimics what PEFT does)
    cache = DynamicCache()
    batch_size = 1

    # Generate random prefix KV values on GPU, same dtype as model
    prefix_kv_dtype = torch.float16  # PEFT uses base_model_torch_dtype
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, num_heads, NUM_VIRTUAL_TOKENS, head_dim, device="cuda", dtype=prefix_kv_dtype)
        value_states = torch.randn(batch_size, num_heads, NUM_VIRTUAL_TOKENS, head_dim, device="cuda", dtype=prefix_kv_dtype)
        cache_position = torch.arange(NUM_VIRTUAL_TOKENS, device="cuda")
        cache.update(key_states, value_states, layer_idx, cache_kwargs={"cache_position": cache_position})

    print(f"Cache: {len(cache)} layers, seq_len={cache.get_seq_length()}")

    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["past_key_values"] = cache

    print("Running forward pass with prefix KV cache...")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
        print(f"  Forward pass succeeded. Loss: {outputs.loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  Forward pass FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        del model
        torch.cuda.empty_cache()


def test_bnb8_no_peft_no_prefix():
    """Control: BNB 8bit model, no prefix KV cache (should work)."""
    print("\n" + "=" * 70)
    print("TEST 4: BNB 8bit, no prefix KV cache (control — should work)")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    print("Running forward pass (no prefix KV)...")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
        print(f"  Forward pass succeeded. Loss: {outputs.loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  Forward pass FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        del model
        torch.cuda.empty_cache()


def test_bnb8_no_peft_backward():
    """
    BNB 8bit model, no prefix, but with backward pass.
    Tests if the error is in backward (gradient computation).
    """
    print("\n" + "=" * 70)
    print("TEST 5: BNB 8bit, no prefix, with backward (tests backward path)")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    print("Running forward + backward (no prefix KV)...")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss
            print(f"  Forward succeeded. Loss: {loss.item():.4f}")
            loss.backward()
            print(f"  Backward succeeded.")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        del model
        torch.cuda.empty_cache()


def test_bnb8_prefix_kv_backward():
    """
    BNB 8bit model with manually injected prefix KV cache + backward.
    This most closely mirrors the CI test (which does trainer.train()).
    """
    print("\n" + "=" * 70)
    print("TEST 6: BNB 8bit + manual prefix KV + backward (closest to CI)")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_info = get_model_config_info(model)
    num_layers = config_info["num_hidden_layers"]
    num_heads = config_info["num_attention_heads"]
    head_dim = config_info["head_dim"]

    # Create DynamicCache with prefix KV
    cache = DynamicCache()
    batch_size = 1
    prefix_kv_dtype = torch.float16
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, num_heads, NUM_VIRTUAL_TOKENS, head_dim, device="cuda", dtype=prefix_kv_dtype, requires_grad=True)
        value_states = torch.randn(batch_size, num_heads, NUM_VIRTUAL_TOKENS, head_dim, device="cuda", dtype=prefix_kv_dtype, requires_grad=True)
        cache_position = torch.arange(NUM_VIRTUAL_TOKENS, device="cuda")
        cache.update(key_states, value_states, layer_idx, cache_kwargs={"cache_position": cache_position})

    inputs = tokenizer("Paris is the most beautiful city in the world.", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["past_key_values"] = cache

    print("Running forward + backward with prefix KV cache...")
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss
            print(f"  Forward succeeded. Loss: {loss.item():.4f}")
            loss.backward()
            print(f"  Backward succeeded.")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    results = {}

    # Run tests in order of increasing complexity
    results["test4_bnb8_no_prefix"] = test_bnb8_no_peft_no_prefix()
    results["test5_bnb8_backward"] = test_bnb8_no_peft_backward()
    results["test2_peft_fp16"] = test_peft_prefix_tuning_fp16()
    results["test3_bnb8_manual_kv"] = test_bnb8_no_peft_prefix_kv()
    results["test6_bnb8_prefix_backward"] = test_bnb8_prefix_kv_backward()
    results["test1_peft_bnb8"] = test_peft_prefix_tuning_8bit()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")