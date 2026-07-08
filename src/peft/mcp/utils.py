# Copyright 2023-present the HuggingFace Inc. team.
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

"""Utility functions and classes for PEFT MCP Server."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Optional

from peft.utils import PeftType


logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe cache for PEFT models with LRU eviction."""

    def __init__(self, max_size: int = 100):
        """Initialize model cache.

        Args:
            max_size: Maximum number of models to cache.
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()

    def get(self, model_id: str) -> Optional[Any]:
        """Get a model from cache.

        Args:
            model_id: Model identifier.

        Returns:
            Cached model or None if not found.
        """
        with self._lock:
            if model_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_id)
                logger.debug("Cache hit for model: %s", model_id)
                return self._cache[model_id]
            logger.debug("Cache miss for model: %s", model_id)
            return None

    def put(self, model_id: str, model: Any) -> None:
        """Put a model in cache.

        Args:
            model_id: Model identifier.
            model: Model object to cache.
        """
        with self._lock:
            if model_id in self._cache:
                # Update existing entry
                self._cache.move_to_end(model_id)
                self._cache[model_id] = model
            else:
                # Add new entry
                self._cache[model_id] = model
                # Evict oldest if over capacity
                if len(self._cache) > self._max_size:
                    oldest_key, _oldest_model = self._cache.popitem(last=False)
                    logger.info("Evicted model from cache: %s", oldest_key)

    def remove(self, model_id: str) -> bool:
        """Remove a model from cache.

        Args:
            model_id: Model identifier.

        Returns:
            True if model was removed, False if not found.
        """
        with self._lock:
            if model_id in self._cache:
                del self._cache[model_id]
                logger.info("Removed model from cache: %s", model_id)
                return True
            return False

    def clear(self) -> None:
        """Clear all models from cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared model cache")

    def __contains__(self, model_id: str) -> bool:
        """Check if model is in cache."""
        with self._lock:
            return model_id in self._cache

    def __len__(self) -> int:
        """Get number of cached models."""
        with self._lock:
            return len(self._cache)

    def keys(self) -> list[str]:
        """Get list of cached model IDs."""
        with self._lock:
            return list(self._cache.keys())


class ProgressCallback:
    """Progress callback manager for tracking training progress."""

    def __init__(self, task_id: str):
        """Initialize progress callback.

        Args:
            task_id: Unique task identifier.
        """
        self.task_id = task_id
        self._callbacks: list[Callable] = []
        self._progress: dict[str, Any] = {
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "metrics": {},
        }

    def add_callback(self, callback: Callable) -> None:
        """Add a progress callback function.

        Args:
            callback: Function to call on progress updates.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a progress callback function.

        Args:
            callback: Function to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def update(
        self,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update progress and notify callbacks.

        Args:
            current_step: Current training step.
            total_steps: Total number of training steps.
            current_epoch: Current epoch number.
            total_epochs: Total number of epochs.
            loss: Current loss value.
            metrics: Additional metrics dictionary.
        """
        if current_step is not None:
            self._progress["current_step"] = current_step
        if total_steps is not None:
            self._progress["total_steps"] = total_steps
        if current_epoch is not None:
            self._progress["current_epoch"] = current_epoch
        if total_epochs is not None:
            self._progress["total_epochs"] = total_epochs
        if loss is not None:
            self._progress["loss"] = loss
        if metrics is not None:
            self._progress["metrics"].update(metrics)

        # Notify all callbacks
        for callback in self._callbacks:
            try:
                callback(self.task_id, self._progress.copy())
            except (ValueError, TypeError) as e:
                logger.error("Error in progress callback: %s", e)

    def get_progress(self) -> dict[str, Any]:
        """Get current progress state.

        Returns:
            Dictionary containing current progress information.
        """
        return self._progress.copy()

    @property
    def percentage(self) -> float:
        """Get progress as percentage.

        Returns:
            Progress percentage (0.0 to 100.0).
        """
        if self._progress["total_steps"] > 0:
            return (self._progress["current_step"] / self._progress["total_steps"]) * 100.0
        return 0.0


class PEFTMethodRegistry:
    """Registry for PEFT methods with metadata and configuration."""

    # Common PEFT methods with their descriptions
    _METHOD_DESCRIPTIONS = {
        "LORA": "Low-Rank Adaptation - efficient fine-tuning via low-rank decomposition",
        "ADALORA": "Adaptive LoRA - dynamically adjusts rank allocation",
        "IA3": "Infused Adapter by Inhibiting and Amplifying Inner Activations",
        "PREFIX_TUNING": "Prefix-tuning - optimizes continuous prompts in attention",
        "PROMPT_TUNING": "Prompt tuning - learns soft prompts for language models",
        "P_TUNING": "P-tuning - uses trainable prompt embeddings",
        "LOHA": "LoHa - Hadamard product parameterization of low-rank matrices",
        "LOKR": "LoKr - Kronecker product parameterization of low-rank matrices",
        "OFT": "Orthogonal Fine-Tuning - maintains orthogonality during adaptation",
        "BOFT": "Butterfly Orthogonal Fine-Tuning - butterfly factorization based OFT",
        "VERA": "VeRA - Vector-based Random Matrix Adaptation",
        "FOURIERFT": "FourierFT - Fourier domain fine-tuning",
        "HRA": "Householder Rank Adaptation",
        "FROD": "Factorized Robust Distillation",
        "POLY": "Poly - polynomial-based parameter-efficient fine-tuning",
        "LN_TUNING": "LayerNorm tuning - only fine-tunes LayerNorm parameters",
        "BITFIT": "BitFit - bias-only fine-tuning",
        "ADAPTION_PROMPT": "Adaption prompt - adapter layers for vision-language models",
    }

    def __init__(self):
        """Initialize PEFT method registry."""
        self._methods: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._initialize_default_methods()

    def _initialize_default_methods(self) -> None:
        """Initialize registry with default PEFT methods."""
        with self._lock:
            # Register all available PEFT types
            for peft_type in PeftType:
                method_name = peft_type.value
                self._methods[method_name] = {
                    "name": method_name,
                    "description": self._METHOD_DESCRIPTIONS.get(method_name, f"{method_name} PEFT method"),
                    "peft_type": peft_type,
                    "available": True,
                }

    def get_method(self, method_name: str) -> Optional[dict[str, Any]]:
        """Get method information by name.

        Args:
            method_name: Name of the PEFT method.

        Returns:
            Method information dictionary or None if not found.
        """
        with self._lock:
            method_name_upper = method_name.upper()
            return self._methods.get(method_name_upper)

    def list_methods(self) -> list[dict[str, Any]]:
        """List all registered PEFT methods.

        Returns:
            List of method information dictionaries.
        """
        with self._lock:
            return [info.copy() for info in self._methods.values()]

    def get_available_methods(self) -> list[str]:
        """Get list of available method names.

        Returns:
            List of method names.
        """
        with self._lock:
            return [name for name, info in self._methods.items() if info.get("available", True)]

    def is_available(self, method_name: str) -> bool:
        """Check if a method is available.

        Args:
            method_name: Name of the PEFT method.

        Returns:
            True if method is available, False otherwise.
        """
        with self._lock:
            method_name_upper = method_name.upper()
            if method_name_upper in self._methods:
                return self._methods[method_name_upper].get("available", True)
            return False

    def register_method(
        self,
        name: str,
        description: Optional[str] = None,
        config_cls: Optional[type] = None,
        model_cls: Optional[type] = None,
    ) -> None:
        """Register a custom PEFT method.

        Args:
            name: Name of the method.
            description: Description of the method.
            config_cls: Configuration class for the method.
            model_cls: Model class for the method.
        """
        with self._lock:
            name_upper = name.upper()
            self._methods[name_upper] = {
                "name": name_upper,
                "description": description or f"{name_upper} PEFT method",
                "config_cls": config_cls,
                "model_cls": model_cls,
                "available": True,
            }
            logger.info("Registered PEFT method: %s", name_upper)

    def get_config_class(self, method_name: str) -> Optional[type]:
        """Get configuration class for a method.

        Args:
            method_name: Name of the PEFT method.

        Returns:
            Configuration class or None if not found.
        """
        with self._lock:
            method_name_upper = method_name.upper()
            if method_name_upper in self._methods:
                return self._methods[method_name_upper].get("config_cls")
            return None

    def get_model_class(self, method_name: str) -> Optional[type]:
        """Get model class for a method.

        Args:
            method_name: Name of the PEFT method.

        Returns:
            Model class or None if not found.
        """
        with self._lock:
            method_name_upper = method_name.upper()
            if method_name_upper in self._methods:
                return self._methods[method_name_upper].get("model_cls")
            return None


def generate_task_id() -> str:
    """Generate a unique task ID.

    Returns:
        UUID string for task identification.
    """
    return str(uuid.uuid4())


def count_trainable_parameters(model: Any) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except (AttributeError, TypeError) as e:
        logger.error("Error counting trainable parameters: %s", e)
        return 0


def count_total_parameters(model: Any) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of parameters.
    """
    try:
        return sum(p.numel() for p in model.parameters())
    except (AttributeError, TypeError) as e:
        logger.error("Error counting total parameters: %s", e)
        return 0
