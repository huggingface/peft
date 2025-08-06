from __future__ import annotations

import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner
from peft.utils.osf_utils import (
    attach_gradient_hooks,
    auto_generate_target_osf_config,
    create_osf_model_class,
)


class OSFModel(BaseTuner):
    """A minimal tuner implementing Orthogonal Subspace Fine-tuning."""

    prefix: str = "osf_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _prepare_adapter_config(self, peft_config, model_config):
        return peft_config

    def inject_adapter(
        self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False
    ) -> None:
        svd_cfg = self.peft_config[adapter_name].target_svd_config
        if svd_cfg is None:
            svd_cfg = auto_generate_target_osf_config(model)
            self.peft_config[adapter_name].target_svd_config = svd_cfg
        OSFCls = create_osf_model_class(model.__class__)
        base_cfg = getattr(model, "config", None)
        osf_model = OSFCls(base_cfg, svd_config=svd_cfg, initialize_svd=False)
        osf_model.load_state_dict(model.state_dict())
        osf_model.reinitialize_svd()
        attach_gradient_hooks(osf_model)
        self.model = osf_model

    def _create_and_replace(self, *args, **kwargs):
        pass

    def _check_target_module_exists(self, *args, **kwargs) -> bool:
        return True

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if "svd_params" not in n and not n.endswith(("_U_low", "_S_low", "_V_low")):
                p.requires_grad = False

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        pass

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(True)

    def disable_adapter_layers(self) -> None:
        self._set_adapter_layers(False)

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name

    def unload(self):
        raise NotImplementedError("OSF models cannot be unloaded yet")

    def merge_adapter(self, *args, **kwargs):
        raise NotImplementedError("OSF models do not support merging")

    def unmerge_adapter(self, *args, **kwargs):
        raise NotImplementedError("OSF models do not support merging")

    def merge_and_unload(self, *args, **kwargs):
        raise NotImplementedError("OSF models do not support merging")