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

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import nn


if TYPE_CHECKING:
    from .config import BayesianLoraConfig


# The values below are not exposed in `BayesianLoraConfig`, as there is rarely a reason to change them.
# relative jitter added to the diagonal of the covariances before the Cholesky factorization
_JITTER = 1e-5
# initial value of lambda, the scale of the conditional noise of the weights given the inducing weights
_INIT_LAMBDA = 1e-3
# initial standard deviation of the variational posterior over the inducing weights
_INIT_INDUCING_SD = 1e-3
# scale of the uniform noise added to the inducing inputs when sampling
_Z_NOISE_SCALE = 0.01
# dropout probability inside the autoregressive network of the flow
_FLOW_DROPOUT = 0.1
# The flow consists of `num_flow_layers` masked autoregressive transforms (L in the paper, 1 by default). The scale and
# shift of each transform are computed by a MADE network with two residual blocks, i.e. 6 linear layers: layer 0 is the
# input layer, layers 1-4 belong to the residual blocks (two per block), layer 5 is the output layer.
_NUM_MADE_BLOCKS = 2
_NUM_MADE_LAYERS = 2 * _NUM_MADE_BLOCKS + 2


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
    degrees += [(hidden_degrees, hidden_degrees)] * (_NUM_MADE_LAYERS - 2)
    degrees += [(hidden_degrees, output_degrees)]
    return degrees


class InducingWeight(nn.Module):
    """
    Variational distribution over a weight matrix of shape `(out_features, in_features)` based on inducing weights.

    Calling the module returns a weight matrix: a sample from the approximate posterior if `sample=True`, otherwise a
    deterministic point estimate obtained by propagating the posterior mean of the inducing weights without noise.
    After a sampling call, `kl_divergence()` returns a single-sample estimate of KL(q || p) for that sample.

    The flow is implemented functionally and all its parameters and buffers are direct attributes of this module
    (`flow_weight_0`, `flow_bias_0`, ...; use `_flow_tensor` to access them). The linear layers of all flow layers are
    numbered consecutively, i.e. the i-th linear layer of the l-th flow layer has the index `l * _NUM_MADE_LAYERS + i`. Nested containers such as
    `nn.ParameterList` or `BufferDict` cannot be used here: PEFT removes the adapter name from the state dict keys when
    saving, which is only supported for keys of the form `<adapter_layer_name>.<adapter_name>.<param>`.

    Args:
        in_features (`int`): Number of columns of the weight matrix.
        out_features (`int`): Number of rows of the weight matrix.
        config ([`BayesianLoraConfig`]): The Bayesian-LoRA configuration.
    """

    def __init__(self, in_features: int, out_features: int, config: BayesianLoraConfig):
        super().__init__()
        inducing_rows, inducing_cols = config.inducing_rows, config.inducing_cols
        self.in_features = in_features
        self.out_features = out_features
        self.inducing_rows = inducing_rows
        self.inducing_cols = inducing_cols
        self.prior_sd = config.prior_sd / math.sqrt(in_features) if config.sqrt_width_scaling else config.prior_sd
        self.max_lambda = config.max_lambda
        self.max_inducing_sd = config.max_inducing_sd
        self.use_flow = config.use_flow
        self.num_flow_layers = config.num_flow_layers if config.use_flow else 0

        # 1D instead of a scalar, as FSDP does not support scalar parameters
        self._lambda = nn.Parameter(torch.tensor([_inverse_softplus(_INIT_LAMBDA)]))

        # variational posterior q(U) = N(inducing_mean, inducing_sd^2), optionally followed by a flow
        self.inducing_mean = nn.Parameter(torch.randn(inducing_rows, inducing_cols))
        self._inducing_sd = nn.Parameter(
            torch.full((inducing_rows, inducing_cols), _inverse_softplus(_INIT_INDUCING_SD))
        )

        # prior parameters: inducing inputs Z and diagonal terms D of the row / column covariances
        self.z_row = nn.Parameter(torch.randn(inducing_rows, out_features) * inducing_rows**-0.5)
        self.z_col = nn.Parameter(torch.randn(inducing_cols, in_features) * inducing_cols**-0.5)
        self.z_row_rho = nn.Parameter(torch.full((inducing_rows,), _inverse_softplus(1e-3)))
        self.z_col_rho = nn.Parameter(torch.full((inducing_cols,), _inverse_softplus(1e-3)))
        self._d_row = nn.Parameter(torch.full((inducing_rows,), _inverse_softplus(1e-3)))
        self._d_col = nn.Parameter(torch.full((inducing_cols,), _inverse_softplus(1e-3)))

        if self.use_flow:
            self._init_flow(inducing_cols, config.flow_hidden_features)

        self._kl_cache: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------ flow

    def _init_flow(self, features: int, hidden_features: int) -> None:
        for flow_layer in range(self.num_flow_layers):
            for j, (in_degrees, out_degrees) in enumerate(_made_degrees(features, hidden_features)):
                i = flow_layer * _NUM_MADE_LAYERS + j
                is_output = j == _NUM_MADE_LAYERS - 1
                mask = (out_degrees[:, None] > in_degrees) if is_output else (out_degrees[:, None] >= in_degrees)
                linear = nn.Linear(len(in_degrees), len(out_degrees))
                if j in (2, 4):
                    # the second layer of each residual block starts close to zero, so that the block is close to
                    # identity
                    nn.init.uniform_(linear.weight, a=-1e-3, b=1e-3)
                    nn.init.uniform_(linear.bias, a=-1e-3, b=1e-3)
                # weight, bias and autoregressive mask of the linear layer
                self.register_parameter(f"flow_weight_{i}", linear.weight)
                self.register_parameter(f"flow_bias_{i}", linear.bias)
                self.register_buffer(f"flow_mask_{i}", mask.float())
                # singular vector estimates for the power iteration of spectral normalization, which keeps the flow
                # stable
                self.register_buffer(f"flow_sn_u_{i}", F.normalize(torch.randn(len(out_degrees)), dim=0, eps=1e-12))
                self.register_buffer(f"flow_sn_v_{i}", F.normalize(torch.randn(len(in_degrees)), dim=0, eps=1e-12))
        self.reset_spectral_norm()

    def _flow_tensor(self, name: str, i: int) -> torch.Tensor:
        """Return the parameter or buffer `name` (weight, bias, mask, sn_u or sn_v) of the i-th linear layer of the flow."""
        return getattr(self, f"flow_{name}_{i}")

    @torch.no_grad()
    def _power_iteration(self, i: int) -> None:
        # the buffers can have a different dtype than the weight, e.g. when the adapter weights are autocast to float32
        weight, u, v = (self._flow_tensor(name, i) for name in ("weight", "sn_u", "sn_v"))
        new_v = F.normalize(weight.t() @ u.to(weight.dtype), dim=0, eps=1e-12)
        new_u = F.normalize(weight @ new_v, dim=0, eps=1e-12)
        v.copy_(new_v)
        u.copy_(new_u)

    def reset_spectral_norm(self, n_iterations: int = 15) -> None:
        """
        Estimate the singular vectors of the current flow weights. Without this, the spectral norm estimate would be
        based on random vectors until the first training step, which can scale the weights arbitrarily.
        """
        if not self.use_flow or self._flow_tensor("weight", 0).device.type == "meta":
            return
        for i in range(self.num_flow_layers * _NUM_MADE_LAYERS):
            for _ in range(n_iterations):
                self._power_iteration(i)

    def _flow_linear(self, i: int, x: torch.Tensor, training_step: bool) -> torch.Tensor:
        weight = self._flow_tensor("weight", i)
        if training_step:
            self._power_iteration(i)
        # cast creates a copy, which is required as the buffers are updated in-place by later power iterations
        u = self._flow_tensor("sn_u", i).to(weight.dtype, copy=True)
        v = self._flow_tensor("sn_v", i).to(weight.dtype, copy=True)
        weight = weight / torch.dot(u, weight @ v)
        mask = self._flow_tensor("mask", i).to(weight.dtype)
        return F.linear(x, weight * mask, self._flow_tensor("bias", i))

    def _flow(self, u0: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masked affine autoregressive flow, applied independently to each row of u0; the scale is `softplus(s) + 1e-3`.
        With a training step, dropout is applied and the spectral norm estimates are updated by one power iteration.
        """
        if not self.use_flow:
            return u0, u0.new_zeros(u0.shape[0])
        training_step = sample and self.training
        u, logdet = u0, u0.new_zeros(u0.shape[0])
        for flow_layer in range(self.num_flow_layers):
            if flow_layer > 0:
                # Reverse the order of the features between two flow layers. In an autoregressive transform, the first
                # feature does not depend on any other feature, so without this, it would never be mixed with the rest.
                u = u.flip(-1)
            offset = flow_layer * _NUM_MADE_LAYERS
            h = self._flow_linear(offset, u, training_step)
            for first in range(offset + 1, offset + _NUM_MADE_LAYERS - 1, 2):  # residual blocks
                t = self._flow_linear(first, F.relu(h), training_step)
                t = F.dropout(F.relu(t), p=_FLOW_DROPOUT, training=training_step)
                h = h + self._flow_linear(first + 1, t, training_step)
            params = self._flow_linear(offset + _NUM_MADE_LAYERS - 1, h, training_step)
            params = params.view(-1, self.inducing_cols, 2)
            scale = F.softplus(params[..., 0]) + 1e-3
            shift = params[..., 1]
            u = scale * u + shift
            logdet = logdet + scale.log().sum(-1)
        return u, logdet

    # ------------------------------------------------------------------ distribution

    @property
    def dtype(self) -> torch.dtype:
        return self.inducing_mean.dtype

    @torch.no_grad()
    def zero_init_(self) -> None:
        """Initialize such that the point estimate of the weight is zero."""
        nn.init.zeros_(self.inducing_mean)
        if self.use_flow:
            # each flow layer outputs (scale, shift) pairs; zero the shifts so that the flow maps zero to zero
            for flow_layer in range(self.num_flow_layers):
                last = (flow_layer + 1) * _NUM_MADE_LAYERS - 1
                nn.init.zeros_(self._flow_tensor("weight", last)[1::2])
                nn.init.zeros_(self._flow_tensor("bias", last)[1::2])
            self.reset_spectral_norm()

    def lambda_(self) -> torch.Tensor:
        return F.softplus(self._lambda[0]).clamp(0.0, self.max_lambda)

    def inducing_sd(self) -> torch.Tensor:
        return F.softplus(self._inducing_sd).clamp(0.0, self.max_inducing_sd)

    def _scale_tril(self, z: torch.Tensor, z_rho: torch.Tensor, d: torch.Tensor, sample: bool) -> torch.Tensor:
        if sample:
            z = z + _Z_NOISE_SCALE * torch.rand_like(z) * F.softplus(z_rho).view(-1, 1)
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
        return (self.prior_sd * (mean + self.lambda_() * noise)).to(self.dtype)

    def kl_divergence(self) -> torch.Tensor:
        if self._kl_cache is None:
            # no sample was drawn yet, draw one so that the estimate is well defined
            self(sample=True)
        u0, u, logdet = self._kl_cache
        # the estimate belongs to a single forward pass, so do not keep the sample (and its graph) around
        self._kl_cache = None
        log_q0 = torch.distributions.Normal(self.inducing_mean, self.inducing_sd()).log_prob(u0).sum()
        log_p = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u)).log_prob(u).sum()
        lambda_ = self.lambda_()
        conditional_kl = self.in_features * self.out_features * (0.5 * lambda_**2 - lambda_.log() - 0.5)
        return log_q0 - logdet.sum() - log_p + conditional_kl
