# model.py
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTuner
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,  # Reuse LoRA mapping
)

from .layer import FeRALayer, FeRALinear


class FrequencyEnergyIndicator(nn.Module):
    """Calculates Frequency-Energy Indicator (FEI) from latent z_t."""

    def __init__(self, num_bands=3):
        super().__init__()
        self.num_bands = num_bands
        self.scales = [2 ** (k) for k in range(num_bands)]

    def get_gaussian_kernel(self, kernel_size, sigma, channels, device, dtype):
        x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
        x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        variance = sigma**2.0
        kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(-torch.sum(xy_grid**2.0, dim=-1) / (2 * variance))
        kernel = kernel / torch.sum(kernel)
        return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1).to(dtype)

    def apply_gaussian_blur(self, x, sigma):
        if sigma <= 0:
            return x
        k_size = int(2 * 4 * sigma + 1) | 1
        kernel = self.get_gaussian_kernel(k_size, sigma, x.shape[1], x.device, x.dtype)
        padding = k_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def forward(self, z_t):
        # z_t: (B, C, H, W)
        B, C, H, W = z_t.shape
        kappa = min(H, W) / 128.0
        sigmas = [kappa * s for s in self.scales]

        gaussian_pyramid = [z_t]
        for s in sigmas:
            gaussian_pyramid.append(self.apply_gaussian_blur(z_t, s))

        band_components = []
        # Band 1 (Low)
        band_components.append(gaussian_pyramid[-1])
        # Mid
        for i in range(len(sigmas) - 1, 0, -1):
            band_components.append(gaussian_pyramid[i] - gaussian_pyramid[i + 1])
        # High
        band_components.append(gaussian_pyramid[0] - gaussian_pyramid[1])

        if len(band_components) > self.num_bands:
            band_components = band_components[: self.num_bands]

        energies = [torch.sum(b**2, dim=[1, 2, 3]) for b in band_components]
        energy_vec = torch.stack(energies, dim=1)  # (B, num_bands)
        e_t = energy_vec / (torch.sum(energy_vec, dim=1, keepdim=True) + 1e-8)
        return e_t, band_components


class SoftFrequencyRouter(nn.Module):
    """Maps FEI to Expert Weights."""

    def __init__(self, num_bands, num_experts, tau=0.7):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(nn.Linear(num_bands, 64), nn.ReLU(), nn.Linear(64, num_experts))

    def forward(self, e_t):
        logits = self.net(e_t)
        return F.softmax(logits / self.tau, dim=-1)


# --- Main Model Class ---


class FeRAModel(BaseTuner):
    """
    Creates FeRA model from a pretrained transformers model.
    """

    prefix: str = "fera_"
    tuner_layer_cls = FeRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        self.fei_indicator = FrequencyEnergyIndicator(num_bands=config.num_bands)
        self.router = SoftFrequencyRouter(config.num_bands, config.num_experts, config.router_tau)

    def _create_and_replace(
        self,
        fera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = {
            "rank": fera_config.rank,
            "lora_alpha": fera_config.lora_alpha,
            "dropout": fera_config.dropout,
            "num_experts": fera_config.num_experts,
            "init_weights": fera_config.init_lora_weights,
        }

        if isinstance(target, FeRALayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(fera_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(fera_config, adapter_name, target, **kwargs):
        if isinstance(target, torch.nn.Linear):
            return FeRALinear(target, adapter_name, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )

    def prepare_forward(self, z_t: torch.Tensor):
        device = z_t.device
        self.fei_indicator.to(device)
        self.router.to(device)

        e_t, _ = self.fei_indicator(z_t)

        routing_weights = self.router(e_t)  # (B, num_experts)

        count = 0
        for module in self.model.modules():
            if isinstance(module, FeRALinear):
                module.set_routing_weights(routing_weights)
                count += 1

        return routing_weights
