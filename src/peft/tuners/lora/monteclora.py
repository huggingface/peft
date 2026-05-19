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

from typing import Optional

import torch
from torch import nn
from torch.distributions.wishart import Wishart

from peft.utils import infer_device


class BufferedMontecloraSampler:
    """
    A buffered sampler for Monteclora that pre-generates samples to improve training efficiency.
    """

    def __init__(self, model: MontecloraSampler, buffer_size: int = 150, device: Optional[str] = None) -> None:
        self.model = model
        self.device = device if device is not None else infer_device()
        self.buffer_size = buffer_size
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.index = 0

        # Pre-create Wishart sampler.
        # scale_tril must be a lower triangular matrix.
        self.wish_sampler = Wishart(
            df=self.model.out_features, scale_tril=torch.eye(self.model.out_features, device=self.device)
        )

        self._refill_buffer()

    def _refill_buffer(self) -> None:
        """Generates a batch of samples to refill the internal buffer."""
        # Note: We generate on self.device (which might be CPU initially).
        # We will cast to the correct accelerator in the forward pass of the main module.
        sample_dtype = self.model.dtype
        with torch.no_grad():
            z_mvn_bulk = torch.randn(
                (self.buffer_size, self.model.num_samples, self.model.in_features, self.model.out_features),
                device=self.device,
                dtype=sample_dtype,
            )

            z_dirichlet_bulk = torch.randn(
                (self.buffer_size, self.model.num_samples), device=self.device, dtype=sample_dtype
            )

            self.buffer = []
            for i in range(self.buffer_size):
                # Wishart sampling is numerically unstable in low precision; sample in fp32 then cast.
                z_wishart = self.wish_sampler._bartlett_sampling(torch.Size()).to(self.device)
                if sample_dtype is not None:
                    z_wishart = z_wishart.to(sample_dtype)

                sample = {"z_mvn": z_mvn_bulk[i], "z_wishart": z_wishart, "z_dirichlet": z_dirichlet_bulk[i]}
                self.buffer.append(sample)

        self.index = 0

    def get(self) -> dict[str, torch.Tensor]:
        """
        Retrieves a single sample set from the buffer. Refills buffer if empty.
        """
        if self.index >= self.buffer_size:
            self._refill_buffer()

        sample = self.buffer[self.index]
        self.index += 1
        return sample


class MontecloraSampler(nn.Module):
    """
    The main module responsible for maintaining the variational parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_samples: int,
        use_entropy: bool = True,
        dirichlet_prior: float = 1.0,
        sample_scaler: float = 3e-4,
        kl_loss_weight: float = 1e-5,
        buffer_size: int = 10,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples
        self.use_entropy = use_entropy
        # Initial device, might change if model.to() is called.
        self.device = device if device is not None else infer_device()
        self.dtype = dtype
        self.sample_scaler = sample_scaler
        self.kl_loss_weight = kl_loss_weight
        self.dirichlet_prior = dirichlet_prior

        # Variational Parameters. The dtype is matched to the parent LoRA layer dtype so that PEFT's
        # adapter dtype handling (e.g. `autocast_adapter_dtype`) treats these parameters consistently
        # with the rest of the LoRA adapter.
        self.std_prior = nn.Parameter(torch.rand(out_features, dtype=dtype))
        self.expert_weights_prior = nn.Parameter(torch.rand(num_samples, dtype=dtype))

        # Buffers for state tracking
        self.register_buffer("gaussian_var_prior", torch.eye(out_features, dtype=dtype))
        self.register_buffer("expert_weights", torch.ones(num_samples, dtype=dtype) / num_samples)

        self.sampler = BufferedMontecloraSampler(self, buffer_size=buffer_size, device=self.device)

        # Move to initial device
        self.to(self.device)

    def wishart_reparameterization(self, std: torch.Tensor, z_wishart: torch.Tensor) -> torch.Tensor:
        eps = 1e-3
        # z_wishart and std must be on the same device
        updated_var = std @ z_wishart @ std.T
        updated_var = torch.diag(torch.clip(updated_var.diag(), min=eps))
        return updated_var

    def multivariate_reparameterization(self, z_mvn: torch.Tensor, cov_matrix: torch.Tensor) -> torch.Tensor:
        eps = 1e-3
        # Cholesky decomposition is numerically unstable in low precision (fp16/bf16) and can produce NaNs or
        # negative-definite errors when autocast downcasts the inputs. We force the decomposition to run in fp32
        # and cast back to the original dtype afterwards.
        with torch.amp.autocast("cuda", enabled=False):
            L = torch.linalg.cholesky(cov_matrix.float())
        L = L.to(cov_matrix.dtype)

        varsum = torch.einsum("eio,op->eip", z_mvn, L)
        varsum = torch.nan_to_num(varsum, nan=eps)
        return self.sample_scaler * varsum

    def dirichlet_reparameterization(self, alpha: torch.Tensor, z_dirichlet: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        alpha = torch.clamp(alpha, min=eps)

        mu = torch.log(alpha) - torch.log(alpha).mean()
        # Dirichlet approximation via Normal distribution in log-space
        diag_term = 1 / alpha * (1 - 2 / self.num_samples) + 1 / (self.num_samples**2) * (1 / alpha).sum()
        diag_term = torch.clamp(diag_term, min=eps)
        sigma = torch.diag(diag_term)
        sigma = sigma + eps * torch.eye(len(alpha), device=self.expert_weights_prior.device)

        # As above, force Cholesky to run in fp32 to avoid NaNs / "matrix is not positive-definite" errors caused
        # by autocast downcasting `sigma` to fp16/bf16.
        with torch.amp.autocast("cuda", enabled=False):
            try:
                L = torch.linalg.cholesky(sigma.float())
            except torch.linalg.LinAlgError:
                # Fallback for numerical stability
                sigma = sigma + 1e-4 * torch.eye(len(alpha), device=self.expert_weights_prior.device)
                L = torch.linalg.cholesky(sigma.float())

        L = L.to(sigma.dtype)
        return L @ z_dirichlet + mu

    def dirichlet_kl(self, alpha2: torch.Tensor) -> torch.Tensor:
        current_device = self.expert_weights_prior.device
        alpha1 = torch.tensor([self.dirichlet_prior] * self.num_samples, device=current_device)

        def custom_gamma(x: torch.Tensor) -> torch.Tensor:
            return torch.lgamma(x).exp()

        return (
            torch.log(custom_gamma(alpha2.sum()) / custom_gamma(alpha1.sum()))
            + (torch.log(custom_gamma(alpha2) / custom_gamma(alpha1))).sum()
            + ((alpha2 - alpha1) * (torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()
        )

    def wishart_kl(self, std: torch.Tensor) -> torch.Tensor:
        var = std @ std.T
        var = torch.diag(var.diag())
        return 0.5 * (
            -torch.log(var).trace() * self.out_features + var.trace() * self.out_features - self.out_features**2
        )

    def multivariate_kl(self, var: torch.Tensor) -> torch.Tensor:
        var = torch.clamp(var, min=1e-6)
        return self.num_samples * 0.5 * (var.trace() - torch.log(var).trace() - self.out_features)

    def get_variational_loss(self) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """Calculates the KL divergence and entropy loss components."""
        if self.training:
            kl1 = self.dirichlet_kl(torch.exp(self.expert_weights_prior))
            kl2 = self.wishart_kl(torch.diag(torch.exp(self.std_prior)))
            kl3 = self.multivariate_kl(self.gaussian_var_prior)
            entropy = (self.expert_weights**2).sum() if self.use_entropy else 0

            return self.kl_loss_weight * (kl1 + kl2 + kl3), entropy
        return 0.0, 0.0

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: (variational_noise, expert_weights)
        """
        # Use std_prior (nn.Parameter) as the source of truth for the current device/dtype.
        current_device = self.std_prior.device
        current_dtype = self.std_prior.dtype

        sample = self.sampler.get()
        z_mvn = sample["z_mvn"].to(device=current_device, dtype=current_dtype)
        z_wishart = sample["z_wishart"].to(device=current_device, dtype=current_dtype)
        z_dirichlet = sample["z_dirichlet"].to(device=current_device, dtype=current_dtype)

        std = torch.diag(torch.exp(self.std_prior))

        gaussian_var = self.wishart_reparameterization(std, z_wishart)

        self.gaussian_var_prior = gaussian_var

        var = self.multivariate_reparameterization(z_mvn, gaussian_var)

        # Calculate Weights
        expert_weights_logits = self.dirichlet_reparameterization(torch.exp(self.expert_weights_prior), z_dirichlet)
        expert_weights = torch.softmax(expert_weights_logits, dim=-1)
        self.expert_weights = expert_weights

        return var, self.expert_weights
