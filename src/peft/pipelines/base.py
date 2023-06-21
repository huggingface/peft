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
from typing import Any, Union

import torch


class BasePeftPipeline(object):
    r"""
    Base skeleton class for all PEFT pipelines. This class cannot be used as it is and should be subclassed and the
    method `_load_model_and_processor` should be implemented.

    Attributes:
        transformers_model_class (`transformers.PreTrainedModel`):
            The class of the transformers model to use.
        transformers_processor_class (Union[`transformers.PreTrainedTokenizer`, `transformers.AutoProcessor`]):
            The class of the transformers processor to use.
        task_type (str):
            The task type of the pipeline - i.e. the name of the pipeline.
        device (Union[str, int, torch.device]):
            The device to run the pipeline on.
        supported_extra_args (tuple):
            The list of extra arguments supported by the pipeline.
    """
    transformers_model_class = None
    transformers_processor_class = None
    peft_model_class = None
    task_type: str = None
    device: Union[str, int, torch.device] = None
    supported_extra_args: dict = {}

    def __init__(self, model, processor=None, device=None, base_model_kwargs={}, **kwargs):
        self.model = model
        self.processor = processor
        self.device = device
        self.base_model_kwargs = base_model_kwargs

        self._setup_additional_args(**kwargs)
        self._load_model_and_processor()
        self._maybe_move_model_to_device()

    def _load_model_and_processor(self):
        raise NotImplementedError

    def _maybe_move_model_to_device(self):
        if not hasattr(self.model, "hf_device_map") and self.device is not None:
            self.model = self.model.to(self.device)
        elif hasattr(self.model, "hf_device_map") and self.device is None:
            self.device = list(set(self.model.hf_device_map.values()))[0]

    def _setup_additional_args(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.supported_extra_args.keys():
                if not isinstance(value, self.supported_extra_args[key]):
                    raise ValueError(f"Parameter {key} should be of type {self.supported_extra_args[key]}")
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Tried to pass non supported additional extra parameter: {key} - the supported parameters are {self.supported_extra_args}"
                )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
