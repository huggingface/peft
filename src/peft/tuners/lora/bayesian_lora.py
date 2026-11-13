# Copyright 2026-present the HuggingFace Inc. team.
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
#
# Portions of this file are adapted from bayesianize (https://github.com/microsoft/bayesianize),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and from nflows
# (https://github.com/bayesiains/nflows), Copyright (c) 2020 Conor Durkan, Artur Bekasov, Iain Murray,
# George Papamakarios, licensed under the MIT License.
"""
Variational inducing-weight distribution used by Bayesian-LoRA (https://huggingface.co/papers/2601.21003).

A LoRA factor W of shape (out_features, in_features) is parametrized through a small matrix of inducing weights U of
shape (inducing_rows, inducing_cols). The approximate posterior over U is a diagonal Gaussian, optionally followed by a
masked autoregressive flow over the rows of U. W is then drawn from the conditional p(W | U) of a matrix-normal prior,
using Matheron's rule so that the full (out_features * in_features) covariance never has to be materialized.

Adapted from the official Bayesian-LoRA implementation (https://github.com/moulelin/Bayesian-LoRA), which builds on the
inducing-weight layers of bayesianize (https://github.com/microsoft/bayesianize, Ritter et al., "Sparse Uncertainty
Representation in Deep Learning with Inducing Weights", NeurIPS 2021) and on the masked autoregressive flow of nflows
(https://github.com/bayesiains/nflows).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


_JITTER = 1e-5
# MADE with two residual blocks: layer 0 is the input layer, layers 1-4 belong to the residual blocks (two per block),
# layer 5 is the output layer.
_NUM_FLOW_LAYERS = 6


def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def _jittered_cholesky(m: torch.Tensor) -> torch.Tensor:
    jitter = _JITTER * m.detach().diagonal().mean() * torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)
    # cholesky_ex avoids a device synchronization to check for errors; a failed factorization (e.g. because of
    # non-finite parameters) is turned into NaNs instead, which then show up in the outputs and the loss
    chol, info = torch.linalg.cholesky_ex(m + jitter)
    return torch.where(info == 0, chol, torch.full_like(chol, float("nan")))


def _made_degrees(features: int, hidden_features: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Input and output degrees of each layer of MADE (https://huggingface.co/papers/1502.03509)."""
    max_degree, min_degree = max(1, features - 1), min(1, features - 1)
    hidden_degrees = torch.arange(hidden_features) % max_degree + min_degree
    input_degrees = torch.arange(1, features + 1)
    # the output layer has a (scale, shift) pair per feature
    output_degrees = input_degrees.repeat_interleave(2)
    degrees = [(input_degrees, hidden_degrees)]
    degrees += [(hidden_degrees, hidden_degrees)] * (_NUM_FLOW_LAYERS - 2)
    degrees += [(hidden_degrees, output_degrees)]
    return degrees


class InducingWeight(nn.Module):
    """
    Variational distribution over a weight matrix of shape `(out_features, in_features)` based on inducing weights.

    Calling the module returns a weight matrix: a sample from the approximate posterior if `sample=True`, otherwise a
    deterministic point estimate obtained by propagating the posterior mean of the inducing weights without noise.
    After a sampling call, `kl_divergence()` returns a single-sample estimate of KL(q || p) for that sample.

    All parameters and buffers are direct attributes of this module (the flow is implemented functionally), so that
    the state dict keys follow the `<adapter_layer_name>.<adapter_name>.<param>` layout that PEFT expects.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        inducing_rows: int,
        inducing_cols: int,
        prior_sd: float,
        sqrt_width_scaling: bool,
        init_lambda: float,
        max_lambda: Optional[float],
        learn_lambda: bool,
        init_inducing_sd: float,
        max_inducing_sd: Optional[float],
        z_noise_scale: float,
        use_flow: bool,
        flow_hidden_features: int,
        flow_dropout: float,
        use_spectral_norm: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.inducing_rows = inducing_rows
        self.inducing_cols = inducing_cols
        self.prior_sd = prior_sd / math.sqrt(in_features) if sqrt_width_scaling else prior_sd
        self.max_lambda = max_lambda
        self.max_inducing_sd = max_inducing_sd
        self.z_noise_scale = z_noise_scale
        self.use_flow = use_flow
        self.flow_dropout = flow_dropout
        self.use_spectral_norm = use_flow and use_spectral_norm

        # 1D instead of a scalar, as FSDP does not support scalar parameters
        lamda = torch.tensor([_inverse_softplus(init_lambda)])
        if learn_lambda:
            self._lamda = nn.Parameter(lamda)
        else:
            self.register_buffer("_lamda", lamda)

        # variational posterior q(U) = N(inducing_mean, inducing_sd^2), optionally followed by a flow
        self.inducing_mean = nn.Parameter(torch.randn(inducing_rows, inducing_cols))
        self._inducing_sd = nn.Parameter(
            torch.full((inducing_rows, inducing_cols), _inverse_softplus(init_inducing_sd))
        )

        # prior parameters: inducing inputs Z and diagonal terms D of the row / column covariances
        self.z_row = nn.Parameter(torch.randn(inducing_rows, out_features) * inducing_rows**-0.5)
        self.z_col = nn.Parameter(torch.randn(inducing_cols, in_features) * inducing_cols**-0.5)
        self.z_row_rho = nn.Parameter(torch.full((inducing_rows,), _inverse_softplus(1e-3)))
        self.z_col_rho = nn.Parameter(torch.full((inducing_cols,), _inverse_softplus(1e-3)))
        self._d_row = nn.Parameter(torch.full((inducing_rows,), _inverse_softplus(1e-3)))
        self._d_col = nn.Parameter(torch.full((inducing_cols,), _inverse_softplus(1e-3)))

        if use_flow:
            self._init_flow(inducing_cols, flow_hidden_features)

        self._kl_cache: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------ flow

    def _init_flow(self, features: int, hidden_features: int) -> None:
        for i, (in_degrees, out_degrees) in enumerate(_made_degrees(features, hidden_features)):
            is_output = i == _NUM_FLOW_LAYERS - 1
            mask = (out_degrees[:, None] > in_degrees) if is_output else (out_degrees[:, None] >= in_degrees)
            linear = nn.Linear(len(in_degrees), len(out_degrees))
            if i in (2, 4):
                # the second layer of each residual block starts close to zero, so that the block is close to identity
                nn.init.uniform_(linear.weight, a=-1e-3, b=1e-3)
                nn.init.uniform_(linear.bias, a=-1e-3, b=1e-3)
            setattr(self, f"flow_weight_{i}", linear.weight)
            setattr(self, f"flow_bias_{i}", linear.bias)
            self.register_buffer(f"flow_mask_{i}", mask.float())
            if self.use_spectral_norm:
                # singular vector estimates for the power iteration of spectral normalization
                self.register_buffer(f"flow_sn_u_{i}", F.normalize(torch.randn(len(out_degrees)), dim=0, eps=1e-12))
                self.register_buffer(f"flow_sn_v_{i}", F.normalize(torch.randn(len(in_degrees)), dim=0, eps=1e-12))
        self.reset_spectral_norm()

    @torch.no_grad()
    def _power_iteration(self, i: int) -> None:
        # the buffers can have a different dtype than the weight, e.g. when the adapter weights are autocast to float32
        weight, u, v = (
            getattr(self, f"flow_weight_{i}"),
            getattr(self, f"flow_sn_u_{i}"),
            getattr(self, f"flow_sn_v_{i}"),
        )
        new_v = F.normalize(weight.t() @ u.to(weight.dtype), dim=0, eps=1e-12)
        new_u = F.normalize(weight @ new_v, dim=0, eps=1e-12)
        v.copy_(new_v)
        u.copy_(new_u)

    def reset_spectral_norm(self, n_iterations: int = 15) -> None:
        """
        Estimate the singular vectors of the current flow weights. Without this, the spectral norm estimate would be
        based on random vectors until the first training step, which can scale the weights arbitrarily.
        """
        if not self.use_spectral_norm or self.flow_weight_0.device.type == "meta":
            return
        for i in range(_NUM_FLOW_LAYERS):
            for _ in range(n_iterations):
                self._power_iteration(i)

    def _flow_linear(self, i: int, x: torch.Tensor, training_step: bool) -> torch.Tensor:
        weight = getattr(self, f"flow_weight_{i}")
        if self.use_spectral_norm:
            if training_step:
                self._power_iteration(i)
            # cast creates a copy, which is required as the buffers are updated in-place by later power iterations
            u = getattr(self, f"flow_sn_u_{i}").to(weight.dtype, copy=True)
            v = getattr(self, f"flow_sn_v_{i}").to(weight.dtype, copy=True)
            weight = weight / torch.dot(u, weight @ v)
        mask = getattr(self, f"flow_mask_{i}").to(weight.dtype)
        return F.linear(x, weight * mask, getattr(self, f"flow_bias_{i}"))

    def _flow(self, u0: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masked affine autoregressive flow, applied independently to each row of u0; the scale is `softplus(s) + 1e-3`.
        With a training step, dropout is applied and the spectral norm estimates are updated by one power iteration.
        """
        if not self.use_flow:
            return u0, u0.new_zeros(u0.shape[0])
        training_step = sample and self.training
        h = self._flow_linear(0, u0, training_step)
        for first in (1, 3):  # residual blocks
            t = self._flow_linear(first, F.relu(h), training_step)
            t = F.dropout(F.relu(t), p=self.flow_dropout, training=training_step)
            h = h + self._flow_linear(first + 1, t, training_step)
        params = self._flow_linear(_NUM_FLOW_LAYERS - 1, h, training_step).view(-1, self.inducing_cols, 2)
        scale = F.softplus(params[..., 0]) + 1e-3
        shift = params[..., 1]
        return scale * u0 + shift, scale.log().sum(-1)

    # ------------------------------------------------------------------ distribution

    @property
    def dtype(self) -> torch.dtype:
        return self.inducing_mean.dtype

    @torch.no_grad()
    def zero_init_(self) -> None:
        """Initialize such that the point estimate of the weight is zero."""
        nn.init.zeros_(self.inducing_mean)
        if self.use_flow:
            # the flow outputs (scale, shift) pairs; zero the shift so that the flow maps zero to zero
            last = _NUM_FLOW_LAYERS - 1
            nn.init.zeros_(getattr(self, f"flow_weight_{last}")[1::2])
            nn.init.zeros_(getattr(self, f"flow_bias_{last}")[1::2])
            self.reset_spectral_norm()

    def lamda(self) -> torch.Tensor:
        return F.softplus(self._lamda[0]).clamp(0.0, self.max_lambda)

    def inducing_sd(self) -> torch.Tensor:
        return F.softplus(self._inducing_sd).clamp(0.0, self.max_inducing_sd)

    def _scale_tril(self, z: torch.Tensor, z_rho: torch.Tensor, d: torch.Tensor, sample: bool) -> torch.Tensor:
        if sample:
            z = z + self.z_noise_scale * torch.rand_like(z) * F.softplus(z_rho).view(-1, 1)
        return _jittered_cholesky(z @ z.t() + F.softplus(d).pow(2).diag_embed())

    def forward(self, sample: bool = True) -> torch.Tensor:
        if sample:
            u0 = self.inducing_mean + self.inducing_sd() * torch.randn_like(self.inducing_mean)
        else:
            u0 = self.inducing_mean
        u, logdet = self._flow(u0, sample=sample)
        u = u.view(self.inducing_rows, self.inducing_cols)
        if sample:
            self._kl_cache = (u0, u, logdet)

        # The covariances are ill-conditioned (their diagonal terms are small) and Cholesky is not implemented for
        # half precision, so the conditional is computed in at least float32.
        compute_dtype = torch.promote_types(self.dtype, torch.float32)
        u = u.to(compute_dtype)
        z_row, z_col = self.z_row.to(compute_dtype), self.z_col.to(compute_dtype)
        _d_row, _d_col = self._d_row.to(compute_dtype), self._d_col.to(compute_dtype)
        row_chol = self._scale_tril(z_row, self.z_row_rho.to(compute_dtype), _d_row, sample)
        col_chol = self._scale_tril(z_col, self.z_col_rho.to(compute_dtype), _d_col, sample)
        # whitened parametrization: U = L_r @ V @ L_c^T with a standard normal prior over V
        u = row_chol @ u @ col_chol.t()

        row_transform = torch.cholesky_solve(z_row, row_chol).t()  # (out_features, inducing_rows)
        col_transform = torch.cholesky_solve(z_col, col_chol)  # (inducing_cols, in_features)
        mean = row_transform @ u @ col_transform
        if not sample:
            return (self.prior_sd * mean).to(self.dtype)

        # Matheron's rule: draw from the joint prior of (W, U), then correct towards the inducing weights
        e1 = torch.randn(self.out_features, self.in_features, device=u.device, dtype=compute_dtype)
        e2, e3, e4 = (torch.randn_like(u) for _ in range(3))
        d_row = F.softplus(_d_row).diag_embed()
        d_col = F.softplus(_d_col).diag_embed()
        u_bar = z_row @ e1 @ z_col.t() + row_chol @ e2 @ d_col + d_row @ e3 @ col_chol.t() + d_row @ e4 @ d_col
        noise = e1 - row_transform @ u_bar @ col_transform
        return (self.prior_sd * (mean + self.lamda() * noise)).to(self.dtype)

    def kl_divergence(self) -> torch.Tensor:
        if self._kl_cache is None:
            # no sample was drawn yet, draw one so that the estimate is well defined
            self(sample=True)
        u0, u, logdet = self._kl_cache
        log_q0 = torch.distributions.Normal(self.inducing_mean, self.inducing_sd()).log_prob(u0).sum()
        log_p = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u)).log_prob(u).sum()
        lamda = self.lamda()
        conditional_kl = self.in_features * self.out_features * (0.5 * lamda**2 - lamda.log() - 0.5)
        return log_q0 - logdet.sum() - log_p + conditional_kl
