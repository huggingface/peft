import gc
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import (
    GloraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.glora.layer import GloraLinear


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
                    dim_feedforward=hidden_size * 2,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

        class SimpleConfig(dict):
            model_type = "simple_transformer"

            def __init__(self):
                super().__init__()
                self["model_type"] = self.model_type

        self.config = SimpleConfig()

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.linear(x))
        return SimpleNamespace(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class DummyTokenizer:
    pad_token = 0
    eos_token = 0

    def __call__(self, text, return_tensors=None, **kwargs):
        return {"input_ids": torch.randint(0, 100, (2, 5))}

    def batch_decode(self, *args, **kwargs):
        return ["decoded text"]


class GLORATester(unittest.TestCase):
    def setUp(self):
        self.model_id = "HuggingFaceM4/tiny-random-Llama3ForCausalLM"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.base_model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.target_modules = ["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
            self.make_base_model = lambda: AutoModelForCausalLM.from_pretrained(self.model_id)
        except Exception:
            print("Failed to load HF tiny model, using SimpleTransformer for tests.")
            self.base_model = SimpleTransformer()
            self.tokenizer = DummyTokenizer()
            self.target_modules = ["linear", "lm_head"]
            self.make_base_model = SimpleTransformer

    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _make_dummy_input(self, prompt="This is a test prompt"):
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"]

    def test_glora_model_creation_and_forward(self):
        glora_config = GloraConfig(r=4, target_modules=self.target_modules, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.base_model, glora_config)
        assert isinstance(peft_model, PeftModel)
        assert hasattr(peft_model, "base_model")

        peft_model.eval()
        dummy_input = self._make_dummy_input()
        with torch.no_grad():
            output_peft = peft_model(dummy_input)
            output_base = self.base_model(dummy_input)
        assert output_peft.logits.shape == output_base.logits.shape

    def test_save_and_load_glora_adapter(self):
        glora_config = GloraConfig(r=4, target_modules=self.target_modules, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.base_model, glora_config)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            peft_model.save_pretrained(tmp_dirname, safe_serialization=False)
            loaded_peft_model = PeftModel.from_pretrained(self.make_base_model(), tmp_dirname, is_trainable=True)
            assert isinstance(loaded_peft_model, PeftModel)
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
        glora_config = GloraConfig(r=4, target_modules=self.target_modules, task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.base_model, glora_config)

        target_layer_name = self.target_modules[0]
        module_ptr = None
        module_path = None
        for name, module in peft_model.model.named_modules():
            if name.split(".")[-1] == target_layer_name and isinstance(module, GloraLinear):
                module_ptr = module
                module_path = name
                break
        if not isinstance(module_ptr, GloraLinear):
            self.skipTest(f"Target module {target_layer_name} is not a GloraLinear layer after PEFT application.")
        assert isinstance(module_ptr, GloraLinear)

        original_weight = module_ptr.weight.data.clone()
        if "default" in module_ptr.glora_Au:
            torch.nn.init.normal_(module_ptr.glora_Au["default"])
            torch.nn.init.normal_(module_ptr.glora_Bd["default"])
            torch.nn.init.normal_(module_ptr.glora_E["default"])

        merged_model = peft_model.merge_and_unload()
        assert hasattr(merged_model, "forward")
        merged_weight_module_ptr = next(
            (m for n, m in merged_model.named_modules() if n == module_path), None
        )
        assert merged_weight_module_ptr is not None, f"Could not find module at path '{module_path}' after merge"
        assert not torch.allclose(original_weight, merged_weight_module_ptr.weight.data)

        dummy_input = self._make_dummy_input("This is a test prompt after merging")
        peft_model.eval()
        merged_model.eval()
        with torch.no_grad():
            out_peft = peft_model(dummy_input)
            out_merged = merged_model(dummy_input)
        assert torch.allclose(out_peft.logits, out_merged.logits, atol=0.0001)

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
        glora_config = GloraConfig(
            r=4,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model_4bit, glora_config)
        assert isinstance(peft_model, PeftModel)
        dummy_input = self.tokenizer("Test with 4-bit GLORA", return_tensors="pt")["input_ids"].to(peft_model.device)
        with torch.no_grad():
            peft_model.eval()
            output = peft_model(dummy_input)
            assert output is not None


if __name__ == "__main__":
    unittest.main(verbosity=2)
