# Copyright 2025-present the HuggingFace Inc. team.
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

from __future__ import annotations

import torch.nn as nn

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner
from peft.utils import _get_input_embeddings_name, _get_submodules

from .layer import TrainableTokensLayer


class TrainableTokensModel(BaseTuner):
    prefix: str = "trainable_tokens_"
    tuner_layer_cls = TrainableTokensLayer

    def _prepare_adapter_config(self, peft_config, model_config):
        # target_modules can be none which prompts us to infer the embedding layer name ourselves.
        if peft_config.target_modules is None:
            peft_config.target_modules = _get_input_embeddings_name(self.model, "embed_tokens")

        return peft_config

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ) -> None:
        super().inject_adapter(
            model=model,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )

        model_config = self.get_model_config(self)

        # In case of weight-tying we need to adapt the tied weights as well and use tie the embedding adapter.
        #
        # The TrainableTokensLayer supports being tied to another TrainableTokensLayer meaning that the layer will
        # not do any changes on its own but solely rely on the weights from the tied adapter. We will search for the
        # tied weights and put tied TrainableTokensLayer adapters on them, all tied to the adapter of the embedding
        # matrix.
        tied_weights_module_names = self._get_module_names_tied_with_embedding()

        if (
            tied_weights_module_names
            and model_config.get("tie_word_embeddings", False)
            and isinstance(self.model.get_input_embeddings(), TrainableTokensLayer)
        ):
            # disable removing of duplicates since we're essentially only dealing with duplicates (i.e. tied weights)
            for name, module in self.model.named_modules(remove_duplicate=False):
                matched_keys = [target_key for target_key in tied_weights_module_names if name.endswith(target_key)]
                if matched_keys:
                    parent, target, target_name = _get_submodules(model, name)

                    peft_config = self.peft_config[adapter_name].to_dict()
                    peft_config["tied_adapter"] = self.model.get_input_embeddings()

                    self._create_and_replace_dict(
                        peft_config,
                        adapter_name,
                        target,
                        target_name,
                        parent,
                        matched_keys[0],
                    )

    def _get_tied_target_modules(self, *args, **kwargs):
        # Normally this method would return the layers that target tied layers.
        #
        # We override this method since we explicitly support tied weights tied to the embedding layer.
        # Therefore, we don't need the warning issued by returning the modules here.
        return []

    def _create_and_replace_dict(
        self,
        peft_config: dict,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        The same as `_create_and_replace` but takes a dictionary instead of a peft config so that we can add keys that
        are not present in the config, such as `tied_adapter`.
        """
        kwargs = peft_config

        if isinstance(target, TrainableTokensLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(peft_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
        """
        kwargs = peft_config.to_dict()
        self._create_and_replace_dict(kwargs, adapter_name, target, target_name, parent, current_key)

    @staticmethod
    def _create_new_module(peft_config, adapter_name, target, **kwargs):
        new_module = TrainableTokensLayer(target, adapter_name, **kwargs)
        new_module.update_layer(
            adapter_name,
            init_weights=kwargs["init_weights"],
            token_indices=kwargs["token_indices"],
            tied_adapter=kwargs.get("tied_adapter", None),
        )

        return new_module
