# layer.py
import math
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

def make_phi(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unknown activation_fn={name}")

class NonlinearLoraLinear(nn.Module, BaseTunerLayer):
    """
    y = base(x) + scaling * ( phi(x @ A) @ B )
    A: [in, r], B: [r, out]
    """
    adapter_layer_names = ("nlora_A", "nlora_B")   # what PEFT saves
    other_param_names = ("r", "nlora_alpha", "scaling")

    def __init__(self, base_layer: nn.Linear):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("NonlinearLoraLinear supports nn.Linear only (extend as needed).")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.r = {}
        self.nlora_alpha = {}
        self.scaling = {}

        self.nlora_dropout = nn.ModuleDict()
        self.nlora_A = nn.ModuleDict()
        self.nlora_B = nn.ModuleDict()
        self.phi = nn.ModuleDict()

        self._disable_adapters = False
        self.merged_adapters = []

    def update_layer(self, adapter_name: str, r: int, alpha: int, dropout: float, activation_fn: str):
        self.r[adapter_name] = r
        self.nlora_alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / max(1, r)

        self.nlora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.nlora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.nlora_dropout[adapter_name] = nn.Dropout(dropout)
        self.phi[adapter_name] = make_phi(activation_fn)

        # init: A random, B zeros => starts as base model
        nn.init.kaiming_uniform_(self.nlora_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.nlora_B[adapter_name].weight)

        self.set_adapter(adapter_name)

    def forward(self, x):
        out = self.base_layer(x)
        if self._disable_adapters:
            return out

        active = self.active_adapters if isinstance(self.active_adapters, list) else [self.active_adapters]
        for name in active:
            if name not in self.nlora_A:
                continue
            z = self.nlora_dropout[name](x)
            z = self.nlora_A[name](z)
            z = self.phi[name](z)
            out = out + self.nlora_B[name](z) * self.scaling[name]
        return out
