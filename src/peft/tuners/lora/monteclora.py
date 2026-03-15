from __future__ import annotations

import torch
from torch import nn
from torch.distributions.wishart import Wishart


class BufferedMontecloraSampler:
    """
    A buffered sampler for Monteclora that pre-generates samples to improve training efficiency.
    """

    def __init__(self, model, buffer_size=150, device="cpu"):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

        # Pre-create Wishart sampler
        # scale_tril must be a lower triangular matrix
        self.wish_sampler = Wishart(
            df=self.model.out_features, scale_tril=torch.eye(self.model.out_features, device=self.device)
        )

        # Initialize buffer
        self._refill_buffer()

    def _refill_buffer(self):
        """Generates a batch of samples to refill the internal buffer."""
        # Note: We generate on self.device (which might be CPU initially).
        # We will cast to the correct GPU in the forward pass of the main module.
        with torch.no_grad():
            # Generate all z_mvn samples at once
            z_mvn_bulk = torch.randn(
                (self.buffer_size, self.model.monteclora_n, self.model.in_features, self.model.out_features),
                device=self.device,
            )

            # Generate all z_dirichlet samples at once
            z_dirichlet_bulk = torch.randn((self.buffer_size, self.model.monteclora_n), device=self.device)

            # Create buffer
            self.buffer = []
            for i in range(self.buffer_size):
                # Wishart sampling
                z_wishart = self.wish_sampler._bartlett_sampling(torch.Size()).to(self.device)

                sample = {"z_mvn": z_mvn_bulk[i], "z_wishart": z_wishart, "z_dirichlet": z_dirichlet_bulk[i]}
                self.buffer.append(sample)

        self.index = 0

    def get(self):
        """
        Retrieves a single sample set from the buffer. Refills buffer if empty.
        """
        try:
            if self.index >= self.buffer_size:
                self._refill_buffer()

            sample = self.buffer[self.index]
            self.index += 1
            return sample
        except Exception as e:
            print(f"Error getting sample from buffer. Returning None: {e}")
            return None

    def stop(self):
        pass


class MontecloraSampler(nn.Module):
    """
    The main module responsible for maintaining the variational parameters.
    """

    def __init__(
        self,
        in_features,
        out_features,
        monteclora_n,
        use_entropy=True,
        dirichlet_prior=1.0,
        sample_scaler=3e-4,
        kl_loss_weight=1e-5,
        buffer_size=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.monteclora_n = monteclora_n
        self.use_entropy = use_entropy
        self.device = device  # Initial device, might change if model.to() is called
        self.sample_scaler = sample_scaler
        self.kl_loss_weight = kl_loss_weight
        self.dirichlet_prior = dirichlet_prior

        # Variational Parameters
        self.std_prior = nn.Parameter(torch.rand(out_features))
        self.expert_weights_prior = nn.Parameter(torch.rand(monteclora_n))

        # Buffers for state tracking
        self.register_buffer("gaussian_var_prior", torch.eye(out_features))
        self.register_buffer("expert_weights", torch.ones(monteclora_n) / monteclora_n)

        # Initialize Sampler
        self.sampler = BufferedMontecloraSampler(self, buffer_size=150, device=device)

        # Move to initial device
        self.to(device)

    def wishart_reparameterization(self, std, z_wishart):
        eps = 1e-3
        # z_wishart and std must be on the same device
        updated_var = std @ z_wishart @ std.T
        updated_var = torch.diag(torch.clip(updated_var.diag(), min=eps))
        return updated_var

    def multivariate_reparameterization(self, z_mvn, cov_matrix):
        eps = 1e-3
        # Cholesky decomposition for reparameterization
        with torch.amp.autocast("cuda", enabled=False):
            L = torch.linalg.cholesky(cov_matrix.float())
        L = L.to(cov_matrix.dtype)

        varsum = torch.einsum("eio,op->eip", z_mvn, L)
        varsum = torch.nan_to_num(varsum, nan=eps)
        return self.sample_scaler * varsum

    def dirichlet_reparameterization(self, alpha, z_dirichlet):
        eps = 1e-6
        alpha = torch.clamp(alpha, min=eps)

        mu = torch.log(alpha) - torch.log(alpha).mean()
        # Dirichlet approximation via Normal distribution in log-space
        diag_term = 1 / alpha * (1 - 2 / self.monteclora_n) + 1 / (self.monteclora_n**2) * (1 / alpha).sum()
        diag_term = torch.clamp(diag_term, min=eps)
        sigma = torch.diag(diag_term)
        sigma = sigma + eps * torch.eye(len(alpha), device=self.expert_weights_prior.device)

        with torch.amp.autocast("cuda", enabled=False):
            try:
                L = torch.linalg.cholesky(sigma.float())
            except torch.linalg.LinAlgError:
                # Fallback for numerical stability
                sigma = sigma + 1e-4 * torch.eye(len(alpha), device=self.expert_weights_prior.device)
                L = torch.linalg.cholesky(sigma.float())

        L = L.to(sigma.dtype)
        return L @ z_dirichlet + mu

    def dirichlet_kl(self, alpha2):
        current_device = self.expert_weights_prior.device
        alpha1 = torch.tensor([self.dirichlet_prior] * self.monteclora_n, device=current_device)

        def custom_gamma(x):
            return torch.lgamma(x).exp()

        return (
            torch.log(custom_gamma(alpha2.sum()) / custom_gamma(alpha1.sum()))
            + (torch.log(custom_gamma(alpha2) / custom_gamma(alpha1))).sum()
            + ((alpha2 - alpha1) * (torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()
        )

    def wishart_kl(self, std):
        var = std @ std.T
        var = torch.diag(var.diag())
        return 0.5 * (
            -torch.log(var).trace() * self.out_features + var.trace() * self.out_features - self.out_features**2
        )

    def multivariate_kl(self, var):
        var = torch.clamp(var, min=1e-6)
        return self.monteclora_n * 0.5 * (var.trace() - torch.log(var).trace() - self.out_features)

    def get_variational_loss(self):
        """Calculates the KL divergence and entropy loss components."""
        if self.training:
            kl1 = self.dirichlet_kl(torch.exp(self.expert_weights_prior))
            kl2 = self.wishart_kl(torch.diag(torch.exp(self.std_prior)))
            kl3 = self.multivariate_kl(self.gaussian_var_prior)
            entropy = (self.expert_weights**2).sum() if self.use_entropy else 0

            return self.kl_loss_weight * (kl1 + kl2 + kl3), entropy
        return 0, 0

    def forward(self):
        """
        Returns:
            tuple: (variational_noise, expert_weights)
        """
        # DYNAMIC DEVICE DETECTION
        # Use std_prior (nn.Parameter) as the source of truth for the current device
        current_device = self.std_prior.device

        sample = self.sampler.get()

        if sample is not None:
            z_mvn = sample["z_mvn"].to(current_device)
            z_wishart = sample["z_wishart"].to(current_device)
            z_dirichlet = sample["z_dirichlet"].to(current_device)
        else:
            # Fallback if sampler fails
            z_mvn = torch.randn((self.monteclora_n, self.in_features, self.out_features), device=current_device)
            z_wishart = self.sampler.wish_sampler._bartlett_sampling(torch.Size()).to(current_device)
            z_dirichlet = torch.randn(self.monteclora_n, device=current_device)

        # Reparameterization steps
        std = torch.diag(torch.exp(self.std_prior))

        # This line caused the error before; now both inputs are guaranteed on current_device
        gaussian_var = self.wishart_reparameterization(std, z_wishart)

        # Update running buffer
        self.gaussian_var_prior = gaussian_var

        # Calculate Variations
        var = self.multivariate_reparameterization(z_mvn, gaussian_var)

        # Calculate Weights
        expert_weights_logits = self.dirichlet_reparameterization(torch.exp(self.expert_weights_prior), z_dirichlet)
        expert_weights = torch.softmax(expert_weights_logits, dim=-1)
        self.expert_weights = expert_weights

        return var, self.expert_weights

    def eval(self):
        if self.sampler:
            self.sampler.stop()
        super().eval()
