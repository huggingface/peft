from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from .config import PolyConfig

EPS = 1e-12


def get_router(poly_config: PolyConfig) -> nn.Module:
    if poly_config.poly_type == "poly":
        return PolyRouter(poly_config)
    else:
        raise NotImplementedError


class Router(nn.Module, ABC):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def forward(self, task_ids: torch.Tensor, input_ids: torch.Tensor):
        ...


class PolyRouter(Router):
    def __init__(self, poly_config: PolyConfig):
        super().__init__()

        self.poly_type = poly_config.poly_type
        self.n_tasks = poly_config.n_tasks
        self.n_skills = poly_config.n_skills
        self.n_splits = poly_config.n_splits

        self.module_logits = nn.Parameter(torch.empty((self.n_tasks, self.n_splits * self.n_skills)))

    def reset(self):
        torch.nn.init.uniform_(self.module_logits, -1e-3, 1e-3)

    def forward(self, task_ids: torch.Tensor, input_ids: torch.Tensor):
        if task_ids is None:
            raise ValueError(f"task_ids should not be None.")
        if task_ids.max().item() >= self.n_tasks:
            raise ValueError(f"Only {self.n_tasks} tasks available. Found task id = {task_ids.max().item()}")

        # move task id to input's device
        task_ids = task_ids.to(self.module_logits.device)

        module_logits = self.module_logits[task_ids]
        module_logits = module_logits.view(-1, self.n_splits, self.n_skills)

        if self.training:
            module_logits = RelaxedBernoulli(temperature=1.0, logits=module_logits).rsample()
        else:
            module_logits = torch.sigmoid(module_logits)

        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)

        return module_weights
