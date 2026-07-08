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

"""Tests for PEFT MCP tool functions."""

from unittest.mock import MagicMock, patch

import pytest

from peft.mcp.tools import (
    _model_cache,
    _training_tasks,
    compare_peft_methods,
    create_peft_model,
    evaluate_peft_model,
    get_peft_config_info,
    get_training_metrics,
    list_peft_methods,
    load_peft_model,
    merge_peft_weights,
    save_peft_model,
    train_peft_model,
)


class TestListPeftMethods:
    """Test list_peft_methods tool."""

    def test_list_peft_methods_success(self):
        """Test listing PEFT methods successfully."""
        result = list_peft_methods()
        assert result.success is True
        assert "methods" in result.data
        assert isinstance(result.data["methods"], list)
        assert len(result.data["methods"]) > 0

    def test_list_peft_methods_structure(self):
        """Test structure of returned methods."""
        result = list_peft_methods()
        methods = result.data["methods"]
        # Check first method has required fields
        if len(methods) > 0:
            method = methods[0]
            assert "name" in method
            assert "description" in method
            assert "available" in method

    def test_list_peft_methods_includes_lora(self):
        """Test that LORA is in the list."""
        result = list_peft_methods()
        method_names = [m["name"] for m in result.data["methods"]]
        assert "LORA" in method_names


class TestGetPeftConfigInfo:
    """Test get_peft_config_info tool."""

    def test_get_peft_config_info_lora(self):
        """Test getting LORA config info."""
        result = get_peft_config_info("LORA")
        assert result.success is True
        assert result.data is not None
        assert result.data["method"] == "LORA"
        assert "params" in result.data
        assert "defaults" in result.data
        assert "description" in result.data

    def test_get_peft_config_info_case_insensitive(self):
        """Test getting config info with lowercase method name."""
        result = get_peft_config_info("lora")
        assert result.success is True
        assert result.data["method"] == "LORA"

    def test_get_peft_config_info_unavailable_method(self):
        """Test getting config info for unavailable method."""
        result = get_peft_config_info("NONEXISTENT_METHOD")
        assert result.success is False
        assert "not available" in result.error

    def test_get_peft_config_info_structure(self):
        """Test structure of returned config info."""
        result = get_peft_config_info("LORA")
        if result.success:
            config_data = result.data
            assert "method" in config_data
            assert "params" in config_data
            assert "defaults" in config_data
            assert "description" in config_data
            assert isinstance(config_data["params"], dict)
            assert isinstance(config_data["defaults"], dict)


class TestCreatePeftModel:
    """Test create_peft_model tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()

    @patch("peft.mcp.tools.get_peft_model")
    @patch("peft.mcp.tools.count_trainable_parameters")
    @patch("peft.mcp.tools.count_total_parameters")
    def test_create_peft_model_success(self, mock_count_total, mock_count_trainable, mock_get_peft):
        """Test creating PEFT model successfully."""
        # Mock the PEFT model
        mock_model = MagicMock()
        mock_get_peft.return_value = mock_model
        mock_count_trainable.return_value = 1000
        mock_count_total.return_value = 100000

        # Mock base model
        base_model = MagicMock()
        base_model.__class__.__name__ = "MockModel"

        result = create_peft_model(
            model_id="test_model",
            base_model=base_model,
            method="LORA",
            config_params={"r": 8},
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["model_id"] == "test_model"
        assert result.data["peft_method"] == "LORA"
        assert result.data["trainable_params"] == 1000
        assert result.data["status"] == "created"

    def test_create_peft_model_unavailable_method(self):
        """Test creating model with unavailable method."""
        base_model = MagicMock()
        result = create_peft_model(
            model_id="test_model",
            base_model=base_model,
            method="NONEXISTENT",
        )
        assert result.success is False
        assert "not available" in result.error

    @patch("peft.mcp.tools.get_peft_model")
    def test_create_peft_model_error(self, mock_get_peft):
        """Test error handling when creating model."""
        mock_get_peft.side_effect = ValueError("Model creation failed")
        base_model = MagicMock()

        result = create_peft_model(
            model_id="test_model",
            base_model=base_model,
            method="LORA",
        )

        assert result.success is False
        assert "Model creation failed" in result.error

    @patch("peft.mcp.tools.get_peft_model")
    @patch("peft.mcp.tools.count_trainable_parameters")
    @patch("peft.mcp.tools.count_total_parameters")
    def test_create_peft_model_with_adapter_name(self, mock_count_total, mock_count_trainable, mock_get_peft):
        """Test creating model with custom adapter name."""
        mock_model = MagicMock()
        mock_get_peft.return_value = mock_model
        mock_count_trainable.return_value = 1000
        mock_count_total.return_value = 100000

        base_model = MagicMock()
        base_model.__class__.__name__ = "MockModel"

        result = create_peft_model(
            model_id="test_model",
            base_model=base_model,
            method="LORA",
            adapter_name="custom_adapter",
        )

        assert result.success is True
        # Verify adapter_name is in metadata
        assert "metadata" in result.data
        assert result.data["metadata"]["adapter_name"] == "custom_adapter"


class TestTrainPeftModel:
    """Test train_peft_model tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()
        _training_tasks.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()
        _training_tasks.clear()

    @pytest.mark.asyncio
    async def test_train_peft_model_not_found(self):
        """Test training non-existent model."""
        result = await train_peft_model(
            model_id="nonexistent",
            train_dataset=MagicMock(),
        )
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_train_peft_model_async_mode(self):
        """Test async training mode."""
        # Add model to cache
        mock_model = MagicMock()
        _model_cache.put("test_model", mock_model)

        result = await train_peft_model(
            model_id="test_model",
            train_dataset=MagicMock(),
            async_mode=True,
        )

        assert result.success is True
        assert "task_id" in result.data
        assert result.data["status"] == "running"

    @pytest.mark.asyncio
    async def test_train_peft_model_sync_mode(self):
        """Test synchronous training mode."""
        mock_model = MagicMock()
        _model_cache.put("test_model", mock_model)

        result = await train_peft_model(
            model_id="test_model",
            train_dataset=MagicMock(),
            async_mode=False,
        )

        assert result.success is True
        assert result.data["status"] == "completed"


class TestMergePeftWeights:
    """Test merge_peft_weights tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()

    def test_merge_peft_weights_not_found(self):
        """Test merging non-existent model."""
        result = merge_peft_weights(model_id="nonexistent")
        assert result.success is False
        assert "not found" in result.error

    def test_merge_peft_weights_no_merge_support(self):
        """Test merging model that doesn't support it."""
        mock_model = MagicMock(spec=[])  # No merge_and_unload method
        _model_cache.put("test_model", mock_model)

        result = merge_peft_weights(model_id="test_model")
        assert result.success is False
        assert "does not support" in result.error

    def test_merge_peft_weights_success(self):
        """Test successful weight merging."""
        mock_model = MagicMock()
        mock_merged = MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged
        _model_cache.put("test_model", mock_model)

        result = merge_peft_weights(model_id="test_model")
        assert result.success is True
        assert result.data["status"] == "merged"
        mock_model.merge_and_unload.assert_called_once()

    def test_merge_peft_weights_with_output_path(self):
        """Test merging with output path."""
        mock_model = MagicMock()
        mock_merged = MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged
        _model_cache.put("test_model", mock_model)

        result = merge_peft_weights(
            model_id="test_model",
            output_path="/tmp/merged_model",
        )
        assert result.success is True
        mock_merged.save_pretrained.assert_called_once_with("/tmp/merged_model")


class TestSavePeftModel:
    """Test save_peft_model tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()

    def test_save_peft_model_not_found(self):
        """Test saving non-existent model."""
        result = save_peft_model(
            model_id="nonexistent",
            output_path="/tmp/model",
        )
        assert result.success is False
        assert "not found" in result.error

    def test_save_peft_model_success(self):
        """Test successful model saving."""
        mock_model = MagicMock()
        _model_cache.put("test_model", mock_model)

        result = save_peft_model(
            model_id="test_model",
            output_path="/tmp/model",
        )
        assert result.success is True
        assert result.data["model_id"] == "test_model"
        assert result.data["output_path"] == "/tmp/model"
        mock_model.save_pretrained.assert_called_once_with("/tmp/model")


class TestLoadPeftModel:
    """Test load_peft_model tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()

    @patch("peft.mcp.tools.PeftModel.from_pretrained")
    @patch("peft.mcp.tools.count_trainable_parameters")
    @patch("peft.mcp.tools.count_total_parameters")
    def test_load_peft_model_success(self, mock_count_total, mock_count_trainable, mock_from_pretrained):
        """Test loading PEFT model successfully."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        mock_count_trainable.return_value = 1000
        mock_count_total.return_value = 100000

        base_model = MagicMock()
        base_model.__class__.__name__ = "MockModel"

        result = load_peft_model(
            model_id="loaded_model",
            base_model=base_model,
            adapter_path="/tmp/adapter",
        )

        assert result.success is True
        assert result.data["model_id"] == "loaded_model"
        assert result.data["status"] == "loaded"
        assert "loaded_model" in _model_cache

    @patch("peft.mcp.tools.PeftModel.from_pretrained")
    def test_load_peft_model_error(self, mock_from_pretrained):
        """Test error handling when loading model."""
        mock_from_pretrained.side_effect = OSError("Load failed")
        base_model = MagicMock()

        result = load_peft_model(
            model_id="loaded_model",
            base_model=base_model,
            adapter_path="/tmp/adapter",
        )

        assert result.success is False
        assert "Load failed" in result.error


class TestEvaluatePeftModel:
    """Test evaluate_peft_model tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()

    def test_evaluate_peft_model_not_found(self):
        """Test evaluating non-existent model."""
        result = evaluate_peft_model(
            model_id="nonexistent",
            eval_dataset=MagicMock(),
        )
        assert result.success is False
        assert "not found" in result.error

    def test_evaluate_peft_model_success(self):
        """Test successful model evaluation."""
        mock_model = MagicMock()
        _model_cache.put("test_model", mock_model)

        result = evaluate_peft_model(
            model_id="test_model",
            eval_dataset=MagicMock(),
        )
        assert result.success is True
        assert result.data["model_id"] == "test_model"
        assert result.data["status"] == "completed"
        assert "metrics" in result.data


class TestComparePeftMethods:
    """Test compare_peft_methods tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()

    @patch("peft.mcp.tools.create_peft_model")
    def test_compare_peft_methods_success(self, mock_create):
        """Test comparing PEFT methods successfully."""
        # Mock create_peft_model to return success
        mock_create.return_value = MagicMock(
            success=True,
            data={
                "trainable_params": 1000,
                "status": "created",
                "metadata": {},
            },
        )

        base_model = MagicMock()
        result = compare_peft_methods(
            base_model=base_model,
            methods=["LORA", "ADALORA"],
        )

        assert result.success is True
        assert "comparison" in result.data
        assert len(result.data["comparison"]) == 2

    @patch("peft.mcp.tools.create_peft_model")
    def test_compare_peft_methods_with_config(self, mock_create):
        """Test comparing methods with custom configs."""
        mock_create.return_value = MagicMock(
            success=True,
            data={
                "trainable_params": 1000,
                "status": "created",
                "metadata": {},
            },
        )

        base_model = MagicMock()
        config_params = {
            "LORA": {"r": 8},
            "ADALORA": {"init_r": 12},
        }

        result = compare_peft_methods(
            base_model=base_model,
            methods=["LORA", "ADALORA"],
            config_params=config_params,
        )

        assert result.success is True
        # Verify create_peft_model was called with correct configs
        assert mock_create.call_count == 2


class TestGetTrainingMetrics:
    """Test get_training_metrics tool."""

    def setup_method(self):
        """Set up test fixtures."""
        _training_tasks.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _training_tasks.clear()

    def test_get_training_metrics_not_found(self):
        """Test getting metrics for non-existent task."""
        result = get_training_metrics(task_id="nonexistent")
        assert result.success is False
        assert "not found" in result.error

    def test_get_training_metrics_success(self):
        """Test getting training metrics successfully."""
        from peft.mcp.models import TrainingResult

        task_id = "test_task"
        training_result = TrainingResult(
            task_id=task_id,
            status="completed",
            metrics={"loss": 0.5, "accuracy": 0.95},
        )
        _training_tasks[task_id] = training_result

        result = get_training_metrics(task_id=task_id)
        assert result.success is True
        assert result.data["task_id"] == task_id
        assert result.data["status"] == "completed"
        assert result.data["metrics"]["loss"] == 0.5
