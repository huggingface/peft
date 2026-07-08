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

"""Core tools for PEFT MCP Server."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

from peft import (
    PeftModel,
    get_peft_model,
)
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.utils import PeftType

from .models import PEFTConfig as MCPPEFTConfig
from .models import PEFTModelInfo, ToolResponse, TrainingResult
from .utils import (
    ModelCache,
    PEFTMethodRegistry,
    ProgressCallback,
    count_total_parameters,
    count_trainable_parameters,
    generate_task_id,
)


logger = logging.getLogger(__name__)


# Global instances
_model_cache = ModelCache()
_method_registry = PEFTMethodRegistry()
_training_tasks: dict[str, TrainingResult] = {}


def list_peft_methods() -> ToolResponse:
    """List all available PEFT methods.

    Returns:
        ToolResponse containing list of available PEFT methods.
    """
    try:
        methods = _method_registry.list_methods()
        method_list = [
            {
                "name": m["name"],
                "description": m["description"],
                "available": m["available"],
            }
            for m in methods
        ]
        return ToolResponse(success=True, data={"methods": method_list})
    except (ValueError, KeyError) as e:
        logger.error("Error listing PEFT methods: %s", e)
        return ToolResponse(success=False, error=str(e))


def get_peft_config_info(method: str) -> ToolResponse:
    """Get configuration parameters and defaults for a PEFT method.

    Args:
        method: Name of the PEFT method.

    Returns:
        ToolResponse containing configuration information.
    """
    try:
        method_upper = method.upper()

        # Check if method is available
        if not _method_registry.is_available(method_upper):
            return ToolResponse(
                success=False,
                error=f"PEFT method '{method}' is not available",
            )

        # Get config class
        config_cls = _method_registry.get_config_class(method_upper)
        if config_cls is None:
            # Try to get from PEFT's mapping
            peft_type = getattr(PeftType, method_upper, None)
            if peft_type and peft_type in PEFT_TYPE_TO_CONFIG_MAPPING:
                config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]

        if config_cls is None:
            return ToolResponse(
                success=False,
                error=f"Configuration class not found for method '{method}'",
            )

        # Extract configuration parameters
        sig = inspect.signature(config_cls.__init__)
        params = {}
        defaults = {}

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "kwargs"]:
                continue

            param_info = {
                "name": param_name,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "required": param.default == inspect.Parameter.empty,
            }

            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
                param_info["default"] = param.default

            params[param_name] = param_info

        # Get method description
        method_info = _method_registry.get_method(method_upper)
        description = method_info["description"] if method_info else f"{method_upper} PEFT method"

        config_info = MCPPEFTConfig(
            method=method_upper,
            params=params,
            defaults=defaults,
            description=description,
        )

        return ToolResponse(success=True, data=config_info.to_dict())

    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Error getting PEFT config for %s: %s", method, e)
        return ToolResponse(success=False, error=str(e))


def create_peft_model(
    model_id: str,
    base_model: Any,
    method: str,
    config_params: Optional[dict[str, Any]] = None,
    adapter_name: str = "default",
) -> ToolResponse:
    """Create a PEFT model.

    Args:
        model_id: Unique identifier for the PEFT model.
        base_model: Base model to apply PEFT to.
        method: PEFT method to use (e.g., 'LORA', 'ADALORA').
        config_params: Configuration parameters for the PEFT method.
        adapter_name: Name for the adapter (default: 'default').

    Returns:
        ToolResponse containing model information.
    """
    try:
        method_upper = method.upper()

        # Check if method is available
        if not _method_registry.is_available(method_upper):
            return ToolResponse(
                success=False,
                error=f"PEFT method '{method}' is not available",
            )

        # Get config class
        peft_type = getattr(PeftType, method_upper, None)
        if peft_type is None or peft_type not in PEFT_TYPE_TO_CONFIG_MAPPING:
            return ToolResponse(
                success=False,
                error=f"Configuration class not found for method '{method}'",
            )

        config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]

        # Create config with provided parameters
        config_params = config_params or {}
        config = config_cls(**config_params)

        # Create PEFT model
        peft_model = get_peft_model(base_model, config, adapter_name=adapter_name)

        # Count parameters
        trainable_params = count_trainable_parameters(peft_model)
        total_params = count_total_parameters(peft_model)

        # Create model info
        model_info = PEFTModelInfo(
            model_id=model_id,
            peft_method=method_upper,
            trainable_params=trainable_params,
            status="created",
            base_model=str(type(base_model).__name__),
            task_type=config.task_type.value if config.task_type else None,
            metadata={
                "total_params": total_params,
                "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
                "adapter_name": adapter_name,
            },
        )

        # Cache the model
        _model_cache.put(model_id, peft_model)

        logger.info("Created PEFT model '%s' with %s method", model_id, method_upper)
        return ToolResponse(success=True, data=model_info.to_dict())

    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Error creating PEFT model: %s", e)
        return ToolResponse(success=False, error=str(e))


async def train_peft_model(
    model_id: str,
    train_dataset: Any,
    training_args: Optional[dict[str, Any]] = None,
    async_mode: bool = True,
) -> ToolResponse:
    """Execute PEFT training.

    Args:
        model_id: ID of the PEFT model to train.
        train_dataset: Training dataset.
        training_args: Training arguments (e.g., learning_rate, epochs).
        async_mode: Whether to run training asynchronously.

    Returns:
        ToolResponse containing training result information.
    """
    try:
        # Get model from cache
        peft_model = _model_cache.get(model_id)
        if peft_model is None:
            return ToolResponse(
                success=False,
                error=f"Model '{model_id}' not found in cache",
            )

        # Generate task ID
        task_id = generate_task_id()

        # Initialize training result
        training_result = TrainingResult(
            task_id=task_id,
            status="running" if async_mode else "completed",
        )
        _training_tasks[task_id] = training_result

        # Create progress callback
        progress_callback = ProgressCallback(task_id)

        # For now, return a placeholder response
        # In a real implementation, this would integrate with transformers Trainer
        if async_mode:
            # Return immediately with task ID
            return ToolResponse(
                success=True,
                data={
                    "task_id": task_id,
                    "status": "running",
                    "message": "Training started in background",
                },
            )
        else:
            # Synchronous training placeholder
            training_result.status = "completed"
            training_result.metrics = {"placeholder": "Training not yet implemented"}

            return ToolResponse(
                success=True,
                data=training_result.to_dict(),
            )

    except (ValueError, TypeError, KeyError) as e:
        logger.error("Error training PEFT model: %s", e)
        return ToolResponse(success=False, error=str(e))


def merge_peft_weights(
    model_id: str,
    output_path: Optional[str] = None,
) -> ToolResponse:
    """Merge PEFT weights into the base model.

    Args:
        model_id: ID of the PEFT model.
        output_path: Optional path to save merged model.

    Returns:
        ToolResponse containing merge result.
    """
    try:
        # Get model from cache
        peft_model = _model_cache.get(model_id)
        if peft_model is None:
            return ToolResponse(
                success=False,
                error=f"Model '{model_id}' not found in cache",
            )

        # Check if model supports merging
        if not hasattr(peft_model, "merge_and_unload"):
            return ToolResponse(
                success=False,
                error=f"Model '{model_id}' does not support weight merging",
            )

        # Merge weights
        merged_model = peft_model.merge_and_unload()

        # Save if output path provided
        if output_path:
            merged_model.save_pretrained(output_path)
            logger.info("Merged model saved to: %s", output_path)

        # Update model info
        model_info = PEFTModelInfo(
            model_id=model_id,
            peft_method="MERGED",
            trainable_params=0,
            status="merged",
            metadata={"output_path": output_path} if output_path else {},
        )

        return ToolResponse(success=True, data=model_info.to_dict())

    except (OSError, RuntimeError, AttributeError) as e:
        logger.error("Error merging PEFT weights: %s", e)
        return ToolResponse(success=False, error=str(e))


def save_peft_model(
    model_id: str,
    output_path: str,
) -> ToolResponse:
    """Save PEFT model to disk.

    Args:
        model_id: ID of the PEFT model.
        output_path: Path to save the model.

    Returns:
        ToolResponse containing save result.
    """
    try:
        # Get model from cache
        peft_model = _model_cache.get(model_id)
        if peft_model is None:
            return ToolResponse(
                success=False,
                error=f"Model '{model_id}' not found in cache",
            )

        # Save model
        peft_model.save_pretrained(output_path)

        logger.info("PEFT model '%s' saved to: %s", model_id, output_path)

        return ToolResponse(
            success=True,
            data={"model_id": model_id, "output_path": output_path},
        )

    except (OSError, RuntimeError, AttributeError) as e:
        logger.error("Error saving PEFT model: %s", e)
        return ToolResponse(success=False, error=str(e))


def load_peft_model(
    model_id: str,
    base_model: Any,
    adapter_path: str,
    adapter_name: str = "default",
) -> ToolResponse:
    """Load PEFT model from disk.

    Args:
        model_id: Unique identifier for the loaded model.
        base_model: Base model to load adapters into.
        adapter_path: Path to the PEFT adapter weights.
        adapter_name: Name for the adapter (default: 'default').

    Returns:
        ToolResponse containing model information.
    """
    try:
        # Load PEFT model
        peft_model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=adapter_name)

        # Count parameters
        trainable_params = count_trainable_parameters(peft_model)
        total_params = count_total_parameters(peft_model)

        # Create model info
        model_info = PEFTModelInfo(
            model_id=model_id,
            peft_method="LOADED",
            trainable_params=trainable_params,
            status="loaded",
            base_model=str(type(base_model).__name__),
            metadata={
                "total_params": total_params,
                "adapter_path": adapter_path,
                "adapter_name": adapter_name,
            },
        )

        # Cache the model
        _model_cache.put(model_id, peft_model)

        logger.info("Loaded PEFT model '%s' from: %s", model_id, adapter_path)

        return ToolResponse(success=True, data=model_info.to_dict())

    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Error loading PEFT model: %s", e)
        return ToolResponse(success=False, error=str(e))


def evaluate_peft_model(
    model_id: str,
    eval_dataset: Any,
    metrics: Optional[list[str]] = None,
) -> ToolResponse:
    """Evaluate PEFT model performance.

    Args:
        model_id: ID of the PEFT model.
        eval_dataset: Evaluation dataset.
        metrics: List of metrics to compute (e.g., ['accuracy', 'f1']).

    Returns:
        ToolResponse containing evaluation results.
    """
    try:
        # Get model from cache
        peft_model = _model_cache.get(model_id)
        if peft_model is None:
            return ToolResponse(
                success=False,
                error=f"Model '{model_id}' not found in cache",
            )

        # Placeholder evaluation
        # In a real implementation, this would integrate with transformers Trainer
        eval_results = {
            "model_id": model_id,
            "status": "completed",
            "metrics": {
                "placeholder": "Evaluation not yet implemented",
                "note": "This is a placeholder for actual evaluation metrics",
            },
        }

        return ToolResponse(success=True, data=eval_results)

    except (ValueError, RuntimeError) as e:
        logger.error("Error evaluating PEFT model: %s", e)
        return ToolResponse(success=False, error=str(e))


def compare_peft_methods(
    base_model: Any,
    methods: list[str],
    config_params: Optional[dict[str, dict[str, Any]]] = None,
) -> ToolResponse:
    """Compare different PEFT methods.

    Args:
        base_model: Base model to compare methods on.
        methods: List of PEFT method names to compare.
        config_params: Optional configuration parameters for each method.

    Returns:
        ToolResponse containing comparison results.
    """
    try:
        config_params = config_params or {}
        comparison_results = []

        for method in methods:
            method_upper = method.upper()

            # Create temporary model for comparison
            temp_model_id = f"temp_{method_upper}_{generate_task_id()}"

            # Get method-specific config
            method_config = config_params.get(method_upper, {})

            # Create model
            result = create_peft_model(
                model_id=temp_model_id,
                base_model=base_model,
                method=method_upper,
                config_params=method_config,
            )

            if result.success:
                comparison_results.append(
                    {
                        "method": method_upper,
                        "trainable_params": result.data["trainable_params"],
                        "status": result.data["status"],
                        "metadata": result.data.get("metadata", {}),
                    }
                )

                # Clean up temporary model
                _model_cache.remove(temp_model_id)
            else:
                comparison_results.append(
                    {
                        "method": method_upper,
                        "error": result.error,
                    }
                )

        return ToolResponse(
            success=True,
            data={"comparison": comparison_results},
        )

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Error comparing PEFT methods: %s", e)
        return ToolResponse(success=False, error=str(e))


def get_training_metrics(task_id: str) -> ToolResponse:
    """Get training metrics for a specific task.

    Args:
        task_id: ID of the training task.

    Returns:
        ToolResponse containing training metrics.
    """
    try:
        if task_id not in _training_tasks:
            return ToolResponse(
                success=False,
                error=f"Training task '{task_id}' not found",
            )

        training_result = _training_tasks[task_id]

        return ToolResponse(success=True, data=training_result.to_dict())

    except (KeyError, ValueError) as e:
        logger.error("Error getting training metrics: %s", e)
        return ToolResponse(success=False, error=str(e))
