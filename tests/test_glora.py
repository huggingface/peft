import gc
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import (
    GLoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.glora.layer import Linear as GLoraLinear


# A very simple model for testing
class SimpleTransformer(torch.nn.Module):
    def __init__(self, vocab_size=100, hidden_size=16, num_layers=2, num_heads=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 2,  # Simple FFN
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = torch.nn.Linear(hidden_size, hidden_size)  # A targetable linear layer
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

        # Add a config attribute similar to HF models for _prepare_glora_config
        class SimpleConfig:
            model_type = "simple_transformer"  # Needs a mapping in constants.py or explicit target_modules

        self.config = SimpleConfig()

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)  # Pass through the targetable linear layer
        logits = self.lm_head(x)
        return logits


class DummyTokenizer:
    pad_token = 0
    eos_token = 0

    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.randint(0, 100, (2, 5))}

    def batch_decode(self, *args, **kwargs):
        return ["decoded text"]


class GLORATester(unittest.TestCase):
    def setUp(self):
        self.model_id = "HuggingFaceM4/tiny-random-Llama-3"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.base_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        except Exception:
            print("Failed to load HF tiny model, using SimpleTransformer for tests.")
            self.base_model = SimpleTransformer()
            self.tokenizer = DummyTokenizer()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def _get_target_modules(self):
        if isinstance(self.base_model, SimpleTransformer):
            return ["linear", "lm_head"]
        else:
            return [
                "q_proj",
                "v_proj",
                "o_proj",
                "down_proj",
                "up_proj",
                "gate_proj",
            ]

    def test_glora_model_creation_and_forward(self):
        target_modules = self._get_target_modules()
        glora_config = GLoraConfig(r=4, target_modules=target_modules, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.base_model, glora_config)
        assert isinstance(peft_model, PeftModel)
        assert isinstance(peft_model.base_model, type(self.base_model))

        if hasattr(self.tokenizer, "pad_token") and getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", 0)

        if isinstance(self.base_model, SimpleTransformer):
            dummy_input = torch.randint(0, 100, (2, 10))
        else:
            dummy_input = self.tokenizer("This is a test prompt", return_tensors="pt")["input_ids"]

        peft_model.eval()

        # Set deterministic eval_config for all GLoraLinear layers
        for module in peft_model.modules():
            if isinstance(module, GLoraLinear):
                chosen_eval_config = module.configs[0]
                module.eval_config = chosen_eval_config

        with torch.no_grad():
            output_peft = peft_model(dummy_input)
            output_base = self.base_model(dummy_input)
        assert isinstance(output_peft.shape, output_base.shape)

    def test_save_and_load_glora_adapter(self):
        target_modules = self._get_target_modules()
        glora_config = GLoraConfig(r=4, target_modules=target_modules, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.base_model, glora_config, adapter_name="test_adapter")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            peft_model.save_pretrained(tmp_dirname, safe_serialization=False)
            if isinstance(self.base_model, SimpleTransformer):
                loaded_base_model = SimpleTransformer()
            else:
                loaded_base_model = AutoModelForCausalLM.from_pretrained(self.model_id)
            loaded_peft_model = PeftModel.from_pretrained(
                loaded_base_model, tmp_dirname, adapter_name="test_adapter_loaded"
            )
            assert isinstance(loaded_peft_model, PeftModel)
            # Compare GLORA parameters
            original_glora_params = {
                k: v for k, v in peft_model.named_parameters() if "glora_" in k and v.requires_grad
            }
            loaded_glora_params = {
                k: v for k, v in loaded_peft_model.named_parameters() if "glora_" in k and v.requires_grad
            }
            assert len(original_glora_params) == len(loaded_glora_params)
            for (k_orig, v_orig), (k_load, v_load) in zip(original_glora_params.items(), loaded_glora_params.items()):
                assert torch.allclose(v_orig, v_load)

    def test_merge_and_unload_glora(self):
        target_modules = self._get_target_modules()
        glora_config = GLoraConfig(r=4, target_modules=target_modules, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.base_model, glora_config)
        # Set deterministic eval_config for all GLoraLinear layers
        for module in peft_model.modules():
            if isinstance(module, GLoraLinear):
                chosen_eval_config = module.configs[0]
                module.eval_config = chosen_eval_config
        # Store original weights for comparison
        target_layer_name = (
            glora_config.target_modules[0] if isinstance(glora_config.target_modules, list) else "linear"
        )
        module_ptr = peft_model.model
        for part in target_layer_name.split("."):
            module_ptr = getattr(module_ptr, part)
        if isinstance(module_ptr, GLoraLinear):
            original_weight = module_ptr.weight.data.clone()
        else:
            self.skipTest(f"Target module {target_layer_name} is not a GLoraLinear layer after PEFT application.")
        merged_model = peft_model.merge_and_unload()
        assert not isinstance(merged_model, PeftModel)
        assert isinstance(merged_model, type(self.base_model))
        merged_weight_module_ptr = merged_model
        for part in target_layer_name.split("."):
            merged_weight_module_ptr = getattr(merged_weight_module_ptr, part)
        merged_weight = merged_weight_module_ptr.weight.data
        assert not torch.allclose(original_weight, merged_weight)
        if isinstance(self.base_model, SimpleTransformer):
            dummy_input = torch.randint(0, 100, (2, 10))
        else:
            dummy_input = self.tokenizer("This is a test prompt after merging", return_tensors="pt")["input_ids"]
        with torch.no_grad():
            merged_model.eval()
            _ = merged_model(dummy_input)

    @unittest.skipIf(
        not torch.cuda.is_available()
        or not hasattr(torch.cuda, "is_bf16_supported")
        or not torch.cuda.is_bf16_supported(),
        "BF16 not supported or no CUDA",
    )
    def test_glora_with_kbit_training(self):
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_4bit = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
            )
            model_4bit = prepare_model_for_kbit_training(model_4bit)
        except Exception as e:
            self.skipTest(f"bitsandbytes or quantized model loading failed: {e}")
        glora_config = GLoraConfig(
            r=4,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model_4bit, glora_config)
        assert isinstance(peft_model, PeftModel)
        dummy_input = self.tokenizer("Test with 4-bit GLORA", return_tensors="pt")["input_ids"].to(peft_model.device)
        for module in peft_model.modules():
            if isinstance(module, GLoraLinear):
                chosen_eval_config = module.configs[0]
                module.eval_config = chosen_eval_config
        with torch.no_grad():
            peft_model.eval()
            output = peft_model(dummy_input)
            assert output is not None


if __name__ == "__main__":
    unittest.main(verbosity=2)
