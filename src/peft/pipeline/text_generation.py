# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import warnings

from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModel

from .base import BasePeftPipeline


class PeftTextGenerationPipeline(BasePeftPipeline):
    r"""
    Causal language model text generation should support LoRA, AdaLoRA, PrompTuning and all other PEFT architectures.
    """
    transformers_model_class = AutoModelForCausalLM
    transformers_processor_class = AutoTokenizer
    task_type = "text-generation"
    supported_extra_args = {"merge_model": bool, "adapter_name": str}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_model_and_processor(self):
        r"""
        Load the model and processor for the pipeline. This method first loads the peft config, infers the base model
        name or path from it and then loads the transformers model and processor.
        """
        if isinstance(self.model, str):
            peft_config = PeftConfig.from_pretrained(self.model)
            base_model_path = peft_config.base_model_name_or_path

            transformers_model = self.transformers_model_class.from_pretrained(
                base_model_path, **self.base_model_kwargs
            )

            try:
                self.processor = self.transformers_processor_class.from_pretrained(base_model_path)
            except EntryNotFoundError:
                raise ValueError(
                    f"We couldn't find a `transformers` tokenizer in {base_model_path} - make sure this is a correct path or url to a model checkpoint."
                )

            self.model = PeftModel.from_pretrained(
                transformers_model,
                self.model,
            )

            if self.device is None and not hasattr(self.model, "hf_device_map"):
                self.device = self.model.base_model.device

            if getattr(self, "merge_model", False):
                self.model = self.model.merge_and_unload()

        elif isinstance(self.model, PeftModel):
            self.processor = self.transformers_processor_class.from_pretrained(
                self.model.peft_config.base_model_name_or_path
            )
        else:
            raise ValueError(
                f"The model must be a path to a checkpoint or a PeftModel instance, got {self.model.__class__}"
            )

        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

        if hasattr(self, "adapter_name") and self.adapter_name is not None:
            self.model.set_adapter(self.adapter_name)

    def __call__(self, text, **kwargs):
        r"""
        Generate the text give text inputs using the underlying PEFT model. For LoRA models only it is possible to
        specify the adapter to use via the `adapter_name` keyword argument and potentiall overwrite the default adapter
        set at the pipeline creation.
        """
        adapter_name = kwargs.pop("adapter_name", None)
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        if adapter_name is not None:
            if self.adapter_name is not None:
                warnings.warn(
                    "You are setting an adapter name but the pipeline already has one set. This will overwrite the existing adapter."
                )

            self.model.set_adapter(adapter_name)

        encoded_text = self.processor(text, return_tensors="pt", padding=True).to(self.device)

        generate_output = self.model.generate(**encoded_text, **kwargs)
        batched_output = self.processor.batch_decode(generate_output, skip_special_tokens=skip_special_tokens)

        return [{"generated_text": output} for output in batched_output]
