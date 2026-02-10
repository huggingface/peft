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
    y = base(x) + scaling * ( phi(x @ V) @ U )
    V: [in, r], U: [r, out]
    """
    adapter_layer_names = ("nlora_V", "nlora_U")   # what PEFT saves
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
        self.nlora_V = nn.ModuleDict()
        self.nlora_U = nn.ModuleDict()
        self.phi = nn.ModuleDict()

        self._disable_adapters = False
        self.merged_adapters = []

    def update_layer(self, adapter_name: str, r: int, alpha: int, dropout: float, activation_fn: str):
        self.r[adapter_name] = r
        self.nlora_alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / max(1, r)

        self.nlora_V[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.nlora_U[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.nlora_dropout[adapter_name] = nn.Dropout(dropout)
        self.phi[adapter_name] = make_phi(activation_fn)

        # init: V random, U zeros => starts as base model
        nn.init.kaiming_uniform_(self.nlora_V[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.nlora_U[adapter_name].weight)

        self.set_adapter(adapter_name)

    def forward(self, x):
        out = self.base_layer(x)
        if self._disable_adapters:
            return out

        active = self.active_adapters if isinstance(self.active_adapters, list) else [self.active_adapters]
        for name in active:
            if name not in self.nlora_V:
                continue
            z = self.nlora_dropout[name](x)
            z = self.nlora_V[name](z)
            z = self.phi[name](z)
            out = out + self.nlora_U[name](z) * self.scaling[name]
        return out
    
    @torch.no_grad()
    def adapter_delta(self, x, adapter_name: str):
        z = self.nlora_V[adapter_name](self.nlora_dropout[adapter_name](x))
        z = self.phi[adapter_name](z)
        delta = self.nlora_U[adapter_name](z) * self.scaling[adapter_name]
        return delta

    @torch.no_grad()
    def accumulate_consolidation_stats(self, x, adapter_name: str, state: dict, off_load_to_cpu: bool = False, accum_dtype=torch.float32,):
        """
        Docstring for accumulate_consolidation_stats
        x: [*, d_in]
        the state should hold
            - "xxt": [d_in, d_in] sum of x_i x_i^T
            - "xzt": [d_in, r] sum of x_i z_i^T, where z_i is (phi(x @ V) @ U)_i
        :param self: Description
        :param x: Description
        :param adapter_name: Description
        :type adapter_name: str
        :param state: Description
        :type state: dict
        """
        if x.dim() == 3:
            x2 = x.reshape(-1, x.size(-1))
        else:
            x2 = x

        delta = self.adapter_delta(x, adapter_name)  # [*, d_out]
        if delta.dim() == 3:
            delta2 = delta.reshape(-1, delta.size(-1))
        else:
            delta2 = delta

        dev = torch.device('cpu' if off_load_to_cpu else delta2.device)
        xA = x2.to(dev, dtype=accum_dtype)
        rA = delta2.to(dev, dtype=accum_dtype)

        if "xxt" not in state:
            d = xA.size(1)
            m = rA.size(1)

            state["xxt"] = torch.zeros((d, d), device=dev, dtype=accum_dtype)
            state["xzt"] = torch.zeros((d, m), device=dev, dtype=accum_dtype)
        
        state["xxt"].add_(xA.t() @ xA)
        state["xzt"].add_(xA.t() @ rA)

    @torch.no_grad()
    def solve_dW(self, state: dict, lambda_: float, scale_lambda_by_trace=True):
        """
        Solve for optimal U given V and the accumulated stats.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        Returns dW of shape [r, out] which can be merged into base weights as base_w += (V @ dW).T
        """
        xxt = state["xxt"]  # [d, d]
        xzt = state["xzt"]  # [d, out]

        d = xxt.size(0)

        I = torch.eye(d, device=xxt.device, dtype=xxt.dtype)

        if scale_lambda_by_trace:
            # your stabilization heuristic, but now correctly applied
            lam = lambda_ * (torch.trace(xxt) / d).clamp_min(1e-6)
        else:
            lam = lambda_

        A = xxt + lam * I  # add scaled identity for numerical stability (and to prevent overfitting when data is limited)

        dW = torch.linalg.solve(A, xzt)  # [d, out]
        return dW

    @torch.no_grad()
    def solve_and_merge(self, state: dict, lambda_: float, lr_: float, adapter_name:str, inplace_disable_adapter=False,
                        scale_lambda_by_trace=True):
        """
        Solve for optimal U given V and the accumulated stats, then merge into base layer.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        """
        xxt = state["xxt"]  # [d, d]
        xzt = state["xzt"]  # [d, out]

        d = xxt.size(0)

        I = torch.eye(d, device=xxt.device, dtype=xxt.dtype)

        if scale_lambda_by_trace:
            # your stabilization heuristic, but now correctly applied
            lam = lambda_ * (torch.trace(xxt) / d).clamp_min(1e-6)
        else:
            lam = lambda_

        A = xxt + lam * I  # add scaled identity for numerical stability (and to prevent overfitting when data is limited)

        dW = torch.linalg.solve(A, xzt)  # [d, out]

        base_w = self.base_layer.weight.data  # [out, in]
        dW = dW.to(base_w.device)

        base_w.data.add_(dW.t() * lr_)

        if inplace_disable_adapter:
            self._disable_adapters = True
