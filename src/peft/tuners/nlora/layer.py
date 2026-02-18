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
    if name == "tanh":
        return nn.Tanh()
    if name == 'linear':
        return nn.Identity()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unknown activation_fn={name}")

# TODO: Implement zero-shift consolidation


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
    def inner_adapter_forward(self, x, adapter_name: str):
        z = self.nlora_V[adapter_name](x)
        z = self.phi[adapter_name](z)
        return z

    @torch.no_grad()
    def accumulate_consolidation_stats(self, x, adapter_name: str, state: dict, off_load_to_cpu: bool = False, accum_dtype=torch.float32,
                                       lambda_: float = 1e-3, scale_lambda_by_trace: bool = True, consolidate_rls: bool = False,
                                       zeroshift: bool = False):
        """
        Docstring for accumulate_consolidation_stats
        x: [*, d_in]
        the state should hold
            - "xxt": [d_in, d_in] sum of x_i x_i^T
            - "xzt": [d_in, r] sum of x_i z_i^T, where z_i is (phi(x @ V) @ U)_i
            - "zzt": [r, r] sum of z_i z_i^T (optional, only needed for some variants of RLS)
            - "uzt": [d_in, T] (optional) phi(x @ V) @ U for each sample, used for zero-shift consolidation
            - "zt": [r, T] (optional) x^T for each sample, used for zero-shift consolidation
            - "xt": [d_in, T] (optional) x for each sample, used for zero-shift consolidation
            - "pn": [d_in, d_in] precision matrix used for incremental updates (optional, can be computed on the fly from xxt)
            - "wn": [d_in, r] weight matrix used for incremental updates (optional, can be computed on the fly from xxt)
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
        d = xA.size(1)
        m = rA.size(1)
        if "xxt" not in state:
            if not consolidate_rls:
                state["xxt"] = torch.zeros((d, d), device=dev, dtype=accum_dtype)
                state["xzt"] = torch.zeros((d, m), device=dev, dtype=accum_dtype)
            else:
                state["wn"] = torch.zeros((d, m), device=dev, dtype=accum_dtype)
                lambda_scaled = lambda_

                if scale_lambda_by_trace:
                    lambda_scaled = lambda_ #* (torch.trace(xA.t() @ xA) / d).clamp_min(1e-6)
                else:
                    lambda_scaled = lambda_
                
                state["pn"] = (1 / lambda_scaled) * torch.eye(d, device=dev, dtype=accum_dtype)
        
        if consolidate_rls:
            S = torch.eye(xA.size(0), device=dev, dtype=accum_dtype) + xA @ state["pn"] @ xA.t()
            kn = state["pn"] @ xA.t() @ torch.linalg.solve(S, torch.eye(xA.size(0), device=dev, dtype=accum_dtype))

            wn = state["wn"] + kn @ (rA - xA @ state["wn"])
            pn = (state["pn"] - kn @ xA @ state["pn"])
            pn = 0.5 * (pn + pn.t())  # ensure symmetry

            state["wn"] = wn
            state["pn"] = pn
        else:
            state["xxt"].add_(xA.t() @ xA)
            state["xzt"].add_(xA.t() @ rA)

        z = self.inner_adapter_forward(x, adapter_name)  # [*, r]
        if z.dim() == 3:
            z2 = z.reshape(-1, z.size(-1))
        else:
            z2 = z
        
        zA = z2.to(dev, dtype=accum_dtype)

        state["uzt"] = rA.T
        state["zt"] = zA.T
        state["xt"] = xA.T
        state["zzt"] = zA.T @ zA


    @torch.no_grad()
    def solve_dU(self, adapter_name: str, dW, state: dict, lambda_: float, scale_lambda_by_trace=True):
        """
        Solve for optimal U given V and the accumulated stats.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        Returns U of shape [r, out] which can be merged into base weights as base_w += (V @ U).T
        """
        # b = (U @ phi(x @ V).T - dW x).T = (phi(x @ V) @ U - dW.T x.T), this is the regression residual we want to minimize
        dev = state["uzt"].device
        dW = dW.to(dev)
        zt = state["zt"]    # [T, r]
        zzt = state["zzt"]  # [r, r] # gram matrix
        xt = state["xt"]    # [T, d]
        alpha = self.scaling[adapter_name]
        target = -1/alpha * zt @ (xt.T @ dW) # [T, d]

        if scale_lambda_by_trace:
            lambda_scaled = lambda_ * (torch.trace(zzt) / zzt.size(0)).clamp_min(1e-6)
        else:
            lambda_scaled = lambda_

        A = zzt + lambda_scaled * torch.eye(zzt.size(0), device=dev, dtype=zzt.dtype)  # add regularization for numerical stability

        dU = torch.linalg.solve(A, target).T  # [r, out]
        return dU


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
    def solve_and_merge(self, state: dict, lr_: float, lambda_w: float, scale_by_lambda_: bool,
                        adapter_name:str, inplace_disable_adapter=False, consolidate_rls=False, 
                        zeroshift=False):
        """
        Solve for optimal U given V and the accumulated stats, then merge into base layer.
        This is equivalent to solving a ridge regression problem with Tikhonov regularization of strength lambda_.
        """
        if consolidate_rls:
            dW = state["wn"]  # [d, out]
        else:
            dW = self.solve_dW(state, lambda_=lambda_w, scale_lambda_by_trace=scale_by_lambda_)  # [d, out]

        base_w = self.base_layer.weight.data  # [out, in]
        dW = dW.to(base_w.device)

        base_w.data.add_(dW.t() * lr_)

        if zeroshift:
            dU = self.solve_dU(adapter_name, lr_ * dW, state, lambda_=lambda_w,
                               scale_lambda_by_trace=scale_by_lambda_)  # [r, out]
            dU = dU.to(self.nlora_U[adapter_name].weight.data.device)

            with torch.no_grad():
                self.nlora_U[adapter_name].weight.data += dU

        if inplace_disable_adapter:
            self._disable_adapters = True
