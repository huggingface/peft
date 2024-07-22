import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlamaForCausalLM
from peft import PrefixTuningConfig, get_peft_model
import unittest

class TestPastKV(unittest.TestCase):
    def test_past_kv(self):
        model_id = "trl-internal-testing/tiny-random-LlavaForConditionalGeneration"
        prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

        # prepare model and inputs
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        )
        processor = AutoProcessor.from_pretrained(model_id)
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(prompt, raw_image, return_tensors='pt').to(torch.float16)

        # get peft model
        peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
        model.language_model = get_peft_model(model.language_model, peft_config)

        output = model(**inputs, output_hidden_states = True)