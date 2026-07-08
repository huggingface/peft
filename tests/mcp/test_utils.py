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

"""Tests for PEFT MCP utility functions and classes."""

from unittest.mock import MagicMock

from peft.mcp.utils import (
    ModelCache,
    PEFTMethodRegistry,
    ProgressCallback,
    count_total_parameters,
    count_trainable_parameters,
    generate_task_id,
)


class TestModelCache:
    """Test ModelCache class."""

    def test_model_cache_creation(self):
        """Test ModelCache creation with default size."""
        cache = ModelCache()
        assert len(cache) == 0

    def test_model_cache_custom_size(self):
        """Test ModelCache creation with custom size."""
        cache = ModelCache(max_size=50)
        assert len(cache) == 0

    def test_model_cache_put_and_get(self):
        """Test putting and getting models from cache."""
        cache = ModelCache()
        model = MagicMock()
        cache.put("model_1", model)
        retrieved = cache.get("model_1")
        assert retrieved is model

    def test_model_cache_get_missing(self):
        """Test getting non-existent model from cache."""
        cache = ModelCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_model_cache_contains(self):
        """Test checking if model is in cache."""
        cache = ModelCache()
        model = MagicMock()
        cache.put("model_1", model)
        assert "model_1" in cache
        assert "model_2" not in cache

    def test_model_cache_remove(self):
        """Test removing model from cache."""
        cache = ModelCache()
        model = MagicMock()
        cache.put("model_1", model)
        assert cache.remove("model_1") is True
        assert "model_1" not in cache
        assert len(cache) == 0

    def test_model_cache_remove_missing(self):
        """Test removing non-existent model from cache."""
        cache = ModelCache()
        result = cache.remove("nonexistent")
        assert result is False

    def test_model_cache_clear(self):
        """Test clearing all models from cache."""
        cache = ModelCache()
        cache.put("model_1", MagicMock())
        cache.put("model_2", MagicMock())
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_model_cache_keys(self):
        """Test getting list of cached model IDs."""
        cache = ModelCache()
        cache.put("model_1", MagicMock())
        cache.put("model_2", MagicMock())
        keys = cache.keys()
        assert "model_1" in keys
        assert "model_2" in keys
        assert len(keys) == 2

    def test_model_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ModelCache(max_size=2)
        model1 = MagicMock()
        model2 = MagicMock()
        model3 = MagicMock()

        cache.put("model_1", model1)
        cache.put("model_2", model2)
        assert len(cache) == 2

        # Adding third model should evict oldest
        cache.put("model_3", model3)
        assert len(cache) == 2
        assert "model_1" not in cache
        assert "model_2" in cache
        assert "model_3" in cache

    def test_model_cache_update_existing(self):
        """Test updating existing model in cache."""
        cache = ModelCache()
        model1 = MagicMock()
        model2 = MagicMock()

        cache.put("model_1", model1)
        assert cache.get("model_1") is model1

        cache.put("model_1", model2)
        assert cache.get("model_1") is model2
        assert len(cache) == 1

    def test_model_cache_lru_order(self):
        """Test LRU order is maintained on access."""
        cache = ModelCache(max_size=2)
        model1 = MagicMock()
        model2 = MagicMock()
        model3 = MagicMock()

        cache.put("model_1", model1)
        cache.put("model_2", model2)

        # Access model_1 to make it most recently used
        cache.get("model_1")

        # Adding model_3 should evict model_2 (least recently used)
        cache.put("model_3", model3)
        assert "model_1" in cache
        assert "model_2" not in cache
        assert "model_3" in cache


class TestProgressCallback:
    """Test ProgressCallback class."""

    def test_progress_callback_creation(self):
        """Test ProgressCallback creation."""
        callback = ProgressCallback(task_id="task_123")
        assert callback.task_id == "task_123"
        progress = callback.get_progress()
        assert progress["current_step"] == 0
        assert progress["total_steps"] == 0
        assert progress["current_epoch"] == 0
        assert progress["total_epochs"] == 0
        assert progress["loss"] is None
        assert progress["metrics"] == {}

    def test_progress_callback_update(self):
        """Test updating progress."""
        callback = ProgressCallback(task_id="task_123")
        callback.update(current_step=10, total_steps=100, loss=0.5)
        progress = callback.get_progress()
        assert progress["current_step"] == 10
        assert progress["total_steps"] == 100
        assert progress["loss"] == 0.5

    def test_progress_callback_update_partial(self):
        """Test partial progress update."""
        callback = ProgressCallback(task_id="task_123")
        callback.update(current_step=10, total_steps=100)
        callback.update(loss=0.5)
        progress = callback.get_progress()
        assert progress["current_step"] == 10
        assert progress["total_steps"] == 100
        assert progress["loss"] == 0.5

    def test_progress_callback_percentage(self):
        """Test progress percentage calculation."""
        callback = ProgressCallback(task_id="task_123")
        callback.update(current_step=25, total_steps=100)
        assert callback.percentage == 25.0

    def test_progress_callback_percentage_zero(self):
        """Test progress percentage with zero total steps."""
        callback = ProgressCallback(task_id="task_123")
        assert callback.percentage == 0.0

    def test_progress_callback_add_callback(self):
        """Test adding progress callback."""
        callback = ProgressCallback(task_id="task_123")
        received_updates = []

        def update_handler(task_id, progress):
            received_updates.append((task_id, progress))

        callback.add_callback(update_handler)
        callback.update(current_step=10, total_steps=100)

        assert len(received_updates) == 1
        assert received_updates[0][0] == "task_123"
        assert received_updates[0][1]["current_step"] == 10

    def test_progress_callback_remove_callback(self):
        """Test removing progress callback."""
        callback = ProgressCallback(task_id="task_123")
        received_updates = []

        def update_handler(task_id, progress):
            received_updates.append((task_id, progress))

        callback.add_callback(update_handler)
        callback.update(current_step=10, total_steps=100)
        assert len(received_updates) == 1

        callback.remove_callback(update_handler)
        callback.update(current_step=20, total_steps=100)
        assert len(received_updates) == 1  # No new update

    def test_progress_callback_multiple_callbacks(self):
        """Test multiple progress callbacks."""
        callback = ProgressCallback(task_id="task_123")
        updates1 = []
        updates2 = []

        def handler1(task_id, progress):
            updates1.append(progress)

        def handler2(task_id, progress):
            updates2.append(progress)

        callback.add_callback(handler1)
        callback.add_callback(handler2)
        callback.update(current_step=10, total_steps=100)

        assert len(updates1) == 1
        assert len(updates2) == 1

    def test_progress_callback_metrics_update(self):
        """Test updating metrics."""
        callback = ProgressCallback(task_id="task_123")
        callback.update(metrics={"accuracy": 0.95, "f1": 0.92})
        progress = callback.get_progress()
        assert progress["metrics"]["accuracy"] == 0.95
        assert progress["metrics"]["f1"] == 0.92

        # Update metrics again
        callback.update(metrics={"accuracy": 0.97})
        progress = callback.get_progress()
        assert progress["metrics"]["accuracy"] == 0.97
        assert progress["metrics"]["f1"] == 0.92  # Should still be there


class TestPEFTMethodRegistry:
    """Test PEFTMethodRegistry class."""

    def test_registry_creation(self):
        """Test registry creation with default methods."""
        registry = PEFTMethodRegistry()
        methods = registry.list_methods()
        assert len(methods) > 0
        # Should have at least LORA
        method_names = [m["name"] for m in methods]
        assert "LORA" in method_names

    def test_registry_get_method(self):
        """Test getting method information."""
        registry = PEFTMethodRegistry()
        method_info = registry.get_method("LORA")
        assert method_info is not None
        assert method_info["name"] == "LORA"
        assert "description" in method_info
        assert method_info["available"] is True

    def test_registry_get_method_case_insensitive(self):
        """Test getting method with case insensitivity."""
        registry = PEFTMethodRegistry()
        method_info = registry.get_method("lora")
        assert method_info is not None
        assert method_info["name"] == "LORA"

    def test_registry_get_method_missing(self):
        """Test getting non-existent method."""
        registry = PEFTMethodRegistry()
        method_info = registry.get_method("NONEXISTENT")
        assert method_info is None

    def test_registry_is_available(self):
        """Test checking method availability."""
        registry = PEFTMethodRegistry()
        assert registry.is_available("LORA") is True
        assert registry.is_available("NONEXISTENT") is False

    def test_registry_get_available_methods(self):
        """Test getting list of available methods."""
        registry = PEFTMethodRegistry()
        available = registry.get_available_methods()
        assert len(available) > 0
        assert "LORA" in available

    def test_registry_register_custom_method(self):
        """Test registering custom method."""
        registry = PEFTMethodRegistry()
        registry.register_method(
            name="CUSTOM_METHOD",
            description="Custom PEFT method",
            config_cls=MagicMock,
            model_cls=MagicMock,
        )
        method_info = registry.get_method("CUSTOM_METHOD")
        assert method_info is not None
        assert method_info["name"] == "CUSTOM_METHOD"
        assert method_info["description"] == "Custom PEFT method"

    def test_registry_get_config_class(self):
        """Test getting config class for method."""
        registry = PEFTMethodRegistry()
        config_cls = registry.get_config_class("LORA")
        # LORA should have a config class from default initialization
        # It might be None if not explicitly set, but should not raise error
        assert config_cls is None or callable(config_cls)

    def test_registry_get_model_class(self):
        """Test getting model class for method."""
        registry = PEFTMethodRegistry()
        model_cls = registry.get_model_class("LORA")
        # Similar to config class
        assert model_cls is None or callable(model_cls)

    def test_registry_list_methods(self):
        """Test listing all methods."""
        registry = PEFTMethodRegistry()
        methods = registry.list_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
        # Each method should have required fields
        for method in methods:
            assert "name" in method
            assert "description" in method
            assert "available" in method


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_generate_task_id(self):
        """Test generating unique task IDs."""
        id1 = generate_task_id()
        id2 = generate_task_id()
        assert id1 != id2
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_count_trainable_parameters(self):
        """Test counting trainable parameters."""
        model = MagicMock()
        param1 = MagicMock()
        param1.numel.return_value = 100
        param1.requires_grad = True

        param2 = MagicMock()
        param2.numel.return_value = 200
        param2.requires_grad = False

        param3 = MagicMock()
        param3.numel.return_value = 50
        param3.requires_grad = True

        model.parameters.return_value = [param1, param2, param3]

        count = count_trainable_parameters(model)
        assert count == 150  # Only param1 and param3

    def test_count_trainable_parameters_error(self):
        """Test counting trainable parameters with error."""
        model = MagicMock()
        model.parameters.side_effect = AttributeError("Error")
        count = count_trainable_parameters(model)
        assert count == 0

    def test_count_total_parameters(self):
        """Test counting total parameters."""
        model = MagicMock()
        param1 = MagicMock()
        param1.numel.return_value = 100

        param2 = MagicMock()
        param2.numel.return_value = 200

        model.parameters.return_value = [param1, param2]

        count = count_total_parameters(model)
        assert count == 300

    def test_count_total_parameters_error(self):
        """Test counting total parameters with error."""
        model = MagicMock()
        model.parameters.side_effect = AttributeError("Error")
        count = count_total_parameters(model)
        assert count == 0
