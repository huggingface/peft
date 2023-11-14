import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from .config import PolyConfig
from .router import get_router


class PolyLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["poly_lora_A", "poly_lora_B", "poly_router"]

    def __init__(self, in_features: int, out_features: int, task_id_ptr: dict, **kwargs):
        self.r = {}
        self.n_tasks = {}
        self.n_skills = {}
        self.n_splits = {}
        self.poly_type = {}
        self.poly_router = nn.ModuleDict()
        self.poly_lora_A = nn.ParameterDict()
        self.poly_lora_B = nn.ParameterDict()

        self.in_features = in_features
        self.out_features = out_features
        self.task_id_ptr = task_id_ptr
        self.kwargs = kwargs

    @property
    def task_ids(self) -> torch.Tensor:
        return self.task_id_ptr["task_ids"]

    def update_layer(self, adapter_name, poly_config):
        if poly_config.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {poly_config.r}")

        self.r[adapter_name] = poly_config.r
        self.n_tasks[adapter_name] = poly_config.n_tasks
        self.n_skills[adapter_name] = poly_config.n_skills
        self.n_splits[adapter_name] = poly_config.n_splits
        self.poly_type[adapter_name] = poly_config.poly_type

        self.poly_lora_A[adapter_name] = nn.Parameter(
            torch.empty(
                poly_config.n_splits,
                poly_config.n_skills,
                self.in_features // poly_config.n_splits,
                poly_config.r,
            )
        )
        self.poly_lora_B[adapter_name] = nn.Parameter(
            torch.empty(
                poly_config.n_splits,
                poly_config.n_skills,
                poly_config.r,
                self.out_features // poly_config.n_splits,
            )
        )
        self.poly_router[adapter_name] = get_router(poly_config)

        if poly_config.init_poly_weights:
            self.reset_poly_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_poly_parameters(self, adapter_name):
        if adapter_name in self.poly_lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            n_splits, n_skills, d, r = self.poly_lora_A[adapter_name].shape
            for skill in range(n_skills):
                for split in range(n_splits):
                    param = torch.empty((r, d))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.poly_lora_A[adapter_name].data[split, skill, :, :] = param.T

            torch.nn.init.zeros_(self.poly_lora_B[adapter_name])

            # initialized router
            self.poly_router[adapter_name].reset()


class Linear(nn.Linear, PolyLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        poly_config: PolyConfig,
        task_id_ptr: dict,
        **kwargs: object,
    ) -> None:
        init_poly_weights = kwargs.pop("init_poly_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        PolyLayer.__init__(self, in_features=in_features, out_features=out_features, task_id_ptr=task_id_ptr)
        # Freezing the pre-trained weight matrix

        self.update_layer(adapter_name, poly_config)
        self.set_adapter(adapter_name)

    def _linear(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            result = self._linear(x)
        else:
            result = self._linear(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.poly_lora_A.keys():
                    continue

                r = self.r[active_adapter]
                poly_router = self.poly_router[active_adapter]
                poly_lora_A = self.poly_lora_A[active_adapter]
                poly_lora_B = self.poly_lora_B[active_adapter]

                task_ids = self.task_ids
                repeat = x.size(0) // task_ids.size(0)
                # this repeat follows the patten in `model.predict()` line 152
                if repeat:
                    task_ids = task_ids.repeat_interleave(repeat)
                mixing_weights = poly_router(task_ids=task_ids, input_ids=x).to(dtype=previous_dtype)
                bs, n_splits, n_skills = mixing_weights.size()

                # A is    n_splits, n_skills, D // n_splits, rank
                # we want bs,       n_splits, D // n_splits, rank
                A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, poly_lora_A))
                B = torch.einsum("bqs,qsrd->bqrd", (mixing_weights, poly_lora_B))

                A = A.reshape(bs, self.in_features, r)
                B = B.transpose(1, 2).reshape(bs, r, self.out_features)

                result += x.bmm(A).bmm(B) / r

        result = result.to(previous_dtype)
        return result
