# model.py
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTuner

from .layer import NonlinearLoraLinear

class NonlinearLoraModel(BaseTuner):
    prefix = "nlora_"                 # unique prefix for state dict filtering
    tuner_layer_cls = NonlinearLoraLinear

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("NonlinearLoraConfig.target_modules must be set.")
        return peft_config

    def _create_and_replace(self, config, adapter_name, target, target_name, parent, **kwargs):
        # Only wrap Linear for now (extend: Conv1D, Embedding, etc.)
        if isinstance(target, nn.Linear):
            new_module = NonlinearLoraLinear(target)
            new_module.update_layer(
                adapter_name=adapter_name,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
                activation_fn=config.activation_fn,
            )
            setattr(parent, target_name, new_module)
    
    def _get_single_active_adapter(self) -> str:
        a = self.active_adapter
        if isinstance(a, list):
            if len(a) != 1:
                raise ValueError(f"Consolidation supports exactly 1 active adapter, got {a}")
            return a[0]
        return a

    @torch.no_grad()
    def consolidate(
        self,
        dataloader,
        *,
        adapter_name: str | None = None,
        lambda_: float | None = None,
        lr: float | None = None,
        offload_cpu: bool | None = None,
        accum_dtype: torch.dtype | None = None,
        scale_lambda_by_trace: bool | None = None,
        max_batches: int | None = None,
        inplace_disable_adapter: bool = False,
    ):
        """
        Data-dependent consolidation: fit ΔW per wrapped layer using ridge regression on calibration inputs,
        where targets are the adapter's current contribution.

        Call: peft_model.base_model.consolidate(calib_loader)
        """
        # TODO: support multiple adapters at once (currently requires separate calls or manual looping)
        if adapter_name is None:
            adapter_name = self._get_single_active_adapter()
        
        cfg = self.peft_config[adapter_name]

        if lambda_ is None:
            lambda_ = getattr(cfg, "consolidate_lambda", 1e-3)
        if lr is None:
            lr = getattr(cfg, "consolidate_lr", 1.0)
        if offload_cpu is None:
            offload_cpu = getattr(cfg, "consolidate_offload_cpu", True)
        if scale_lambda_by_trace is None:
            scale_lambda_by_trace = getattr(cfg, "consolidate_scale_lambda_by_trace", True)
        if max_batches is None:
            max_batches = getattr(cfg, "consolidate_batches", None)  # allow None = all

        if accum_dtype is None:
            dtype_str = getattr(cfg, "consolidate_dtype", "float32")
            accum_dtype = torch.float64 if dtype_str == "float64" else torch.float32

        layer_states: dict[NonlinearLoraLinear, dict] = {}
        hooks = []

        def make_hook(layer: NonlinearLoraLinear):
            def hook(module, inputs, output):
                x = inputs[0]
                layer.accumulate_consolidation_stats(
                    x=x,
                    adapter_name=adapter_name,
                    state=layer_states[layer],
                    off_load_to_cpu=offload_cpu,
                    accum_dtype=accum_dtype,
                )
            return hook

        # register hooks + init states
        layers = []
        for m in self.model.modules():
            if isinstance(m, NonlinearLoraLinear):
                layers.append(m)
                layer_states[m] = {}
                hooks.append(m.register_forward_hook(make_hook(m)))

        # accumulate stats
        self.model.eval()
        dev = next(self.model.parameters()).device

        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            if isinstance(batch, dict):
                batch = {k: v.to(dev) for k, v in batch.items()}
                _ = self.model(**batch)
            else:
                # if your dataloader yields (input_ids, attention_mask, labels) tuples etc.
                _ = self.model(*batch)

        for h in hooks:
            h.remove()

        # solve + merge per layer
        for layer in layers:
            layer.solve_and_merge(
                state=layer_states[layer],
                lambda_=lambda_,
                lr_=lr,
                adapter_name=adapter_name,
                inplace_disable_adapter=inplace_disable_adapter,
                scale_lambda_by_trace=scale_lambda_by_trace,
            )

        return layer_states
