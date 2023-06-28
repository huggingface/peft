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

from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModelForCausalLM

from .base import BasePeftPipeline


class PeftTextGenerationPipeline(BasePeftPipeline):
    """
    Peft text generation pipeline, can be used to perform text generation out of the box. Works for all supported PEFT
    methods as long as the finetuned adapter weights are available either locally or pushed on the Hub. This pipeline
    supports also switching between different adapters if using LoRA by specifying the adapter name when creating the
    pipeline or at forward pass.

    ```py
    >>> from peft import pipeline

    >>> pipe = pipeline("text-generation", "ybelkada/opt-350m-lora", adapter_name="default")
    >>> pipe("Hello world")
    [{"generated_text": "Hello world, how are you?"}]

    >>> pipe("Bonjour à tous", adapter_name="french_adapter")
    [{"generated_text": "Bonjour à tous, comment allez-vous?"}]
    ```

    For faster inference you can merge the adapters with the base model by setting `merge_model=True` when creating the
    pipeline.

    ```py
    >>> from peft import pipeline

    >>> pipe = pipeline("text-generation", "ybelkada/opt-350m-lora", merge_model=True)
    >>> pipe("Hello world")
    [{"generated_text": "Hello world, how are you?"}]
    ```
    Note that once the model has been merged, it is not possible to switch between adapters.

    Args:
        model ([`Union[str, PeftModel]`]): Base transformer model.
        processor ([`~transformers.PreTrainedTokenizer`]): Base model's tokenizer.
        device ([`Union[str, int, torch.device`]): Device to run inference on.
        base_model_kwargs (Dict[str, Any]):
            Additional kwargs to pass when loading the base model (e.g. `load_in_8bit`).
        merge_model (bool): Whether to merge the adapters with the base model or not. Defaults to `False`.
        adapter_name (str): Name of the adapter to use when using LoRA. Defaults to `default`.

    **Attributes**:
        - **transformers_model_class** ([`class`])-- Class of the transformers model to load.
        - **transformers_processor_class** ([`class`]) -- Class of the transformers processor to load.
        - **task_type** (`str`) -- Name of the task the pipeline is targeted at.
        - **model** ([`PeftModel`]) -- The associated PEFT model
        - **processor** ([`~transformers.PreTrainedTokenizer`]) -- Base model's tokenizer.
        - **device** ([`Union[str, int, torch.device`]) -- Device to run inference on.


    Example:

        ```py
        >>> from peft import pipeline

        >>> pipe = pipeline("text-generation", "ybelkada/opt-350m-lora")
        >>> pipe("Hello world")
        [{"generated_text": "Hello world, how are you?"}]
        ```
    """

    def __init__(self, *args, **kwargs):
        self.transformers_model_class = AutoModelForCausalLM
        self.transformers_processor_class = AutoTokenizer
        self.peft_model_class = PeftModelForCausalLM
        self.task_type = "text-generation"
        self.peft_task_type = "CAUSAL_LM"

        self.merge_model = kwargs.pop("merge_model", False)
        self.adapter_name = kwargs.pop("adapter_name", "default")

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

            if self.processor is None:
                try:
                    self.processor = self.transformers_processor_class.from_pretrained(base_model_path)
                except EntryNotFoundError:
                    raise ValueError(
                        f"We couldn't find a `transformers` tokenizer in {base_model_path} - make sure this is a correct path or url to a model checkpoint."
                    )
            elif not isinstance(self.processor, self.transformers_processor_class):
                raise ValueError(
                    f"The provided `tokenizer` does not match the associated model - got {self.processor.__class__} whereas it should be {self.transformers_processor_class}"
                )

            self.model = self.peft_model_class.from_pretrained(
                transformers_model,
                self.model,
                adapter_name=self.adapter_name,
            )

            if self.device is None and not hasattr(self.model, "hf_device_map"):
                self.device = self.model.base_model.device

            if getattr(self, "merge_model", False):
                self.model = self.model.merge_and_unload()
                self.merged_model = True
            else:
                self.merged_model = False

        elif isinstance(self.model, self.peft_model_class):
            self.processor = self.transformers_processor_class.from_pretrained(
                self.model.peft_config.base_model_name_or_path
            )
        else:
            raise ValueError(
                f"The model must be a path to a checkpoint or a {self.peft_model_class.__class__} instance, got {self.model.__class__}"
            )

        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

    def __call__(self, text, **kwargs):
        r"""
        Generate the text given text inputs using the underlying PEFT model. For LoRA models only it is possible to
        specify the adapter to use via the `adapter_name` keyword argument and potentially overwrite the default
        adapter set at the pipeline creation.
        """
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)
        encoded_text = self.processor(text, return_tensors="pt", padding=True).to(self.device)

        generate_output = self.model.generate(**encoded_text, **kwargs)
        batched_output = self.processor.batch_decode(generate_output, skip_special_tokens=skip_special_tokens)

        return [{"generated_text": output} for output in batched_output]
