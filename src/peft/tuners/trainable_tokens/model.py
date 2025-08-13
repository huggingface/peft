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

import warnings
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import AuxiliaryTrainingWrapper, _get_input_embeddings_name, _get_submodules

from .layer import TrainableTokensLayer


class TrainableTokensModel(BaseTuner):
    prefix: str = "trainable_tokens_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def _prepare_adapter_config(self, peft_config, model_config):
        # target_modules can be none which prompts us to infer the embedding layer name ourselves.
        if peft_config.target_modules is None:
            peft_config.target_modules = _get_input_embeddings_name(self.model, "embed_tokens")

        return peft_config

    def inject_adapter(
        self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False
    ) -> None:
        super().inject_adapter(
            model=model,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        model_config = self.get_model_config(self)

        # In case of weight-tying we need to adapt the tied weights as well and use tie the embedding adapter.
        #
        # The TrainableTokensLayer supports being tied to another TrainableTokensLayer meaning that the layer will
        # not do any changes on its own but solely rely on the weights from the tied adapter. We will search for the
        # tied weights and put tied TrainableTokensLayer adapters on them, all tied to the adapter of the embedding
        # matrix.
        if (
            model_config.get("tie_word_embeddings", False)
            # some models may be misconfigured to have weight tying enabled but don't define tied weights keys
            and self.model._tied_weights_keys is not None
            and isinstance(self.model.get_input_embeddings(), TrainableTokensLayer)
        ):
            module_keys = [".".join(n.split(".")[:-1]) for n in self.model._tied_weights_keys]
            # disable removing of duplicates since we're essentially only dealing with duplicates (i.e. tied weights)
            for name, module in self.model.named_modules(remove_duplicate=False):
                matched_keys = [target_key for target_key in module_keys if name.endswith(target_key)]
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

    def _check_target_module_exists(self, peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

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

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, TrainableTokensLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the trainable tokens modules without merging.
        """
        return self._unload_and_optionally_merge(merge=False)

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the trained tokens into the targeted embedding layer(s) of the base model. This is needed if
        someone wants to use the base model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "unload_and_optionally_merge_module"):
                    # if layers have special unloading method, like MultiheadAttention, use that
                    unloaded_module = target.unload_and_optionally_merge_module(
                        merge=merge, safe_merge=safe_merge, adapter_names=adapter_names
                    )
                    self._replace_module(parent, target_name, unloaded_module, target)
                elif hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model
