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

"""Tests for PEFT MCP data models."""

from peft.mcp.models import PEFTConfig, PEFTModelInfo, ToolResponse, TrainingResult


class TestPEFTModelInfo:
    """Test PEFTModelInfo dataclass."""

    def test_peft_model_info_creation(self):
        """Test PEFTModelInfo creation with required fields."""
        model_info = PEFTModelInfo(
            model_id="test_model",
            peft_method="LORA",
            trainable_params=1000,
        )
        assert model_info.model_id == "test_model"
        assert model_info.peft_method == "LORA"
        assert model_info.trainable_params == 1000
        assert model_info.status == "initialized"
        assert model_info.base_model is None
        assert model_info.task_type is None
        assert model_info.metadata == {}

    def test_peft_model_info_with_optional_fields(self):
        """Test PEFTModelInfo creation with all fields."""
        model_info = PEFTModelInfo(
            model_id="test_model",
            peft_method="LORA",
            trainable_params=1000,
            status="trained",
            base_model="bert-base-uncased",
            task_type="CAUSAL_LM",
            metadata={"key": "value"},
        )
        assert model_info.status == "trained"
        assert model_info.base_model == "bert-base-uncased"
        assert model_info.task_type == "CAUSAL_LM"
        assert model_info.metadata == {"key": "value"}

    def test_peft_model_info_to_dict(self):
        """Test PEFTModelInfo to_dict method."""
        model_info = PEFTModelInfo(
            model_id="test_model",
            peft_method="LORA",
            trainable_params=1000,
            status="created",
            base_model="gpt2",
            task_type="CAUSAL_LM",
            metadata={"total_params": 10000},
        )
        result = model_info.to_dict()
        assert isinstance(result, dict)
        assert result["model_id"] == "test_model"
        assert result["peft_method"] == "LORA"
        assert result["trainable_params"] == 1000
        assert result["status"] == "created"
        assert result["base_model"] == "gpt2"
        assert result["task_type"] == "CAUSAL_LM"
        assert result["metadata"] == {"total_params": 10000}

    def test_peft_model_info_to_dict_minimal(self):
        """Test PEFTModelInfo to_dict with minimal fields."""
        model_info = PEFTModelInfo(
            model_id="test_model",
            peft_method="LORA",
            trainable_params=1000,
        )
        result = model_info.to_dict()
        assert result["model_id"] == "test_model"
        assert result["base_model"] is None
        assert result["task_type"] is None
        assert result["metadata"] == {}


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_training_result_creation(self):
        """Test TrainingResult creation with required fields."""
        result = TrainingResult(task_id="task_123")
        assert result.task_id == "task_123"
        assert result.status == "pending"
        assert result.metrics == {}
        assert result.error is None
        assert result.model_info is None

    def test_training_result_with_metrics(self):
        """Test TrainingResult with metrics."""
        result = TrainingResult(
            task_id="task_123",
            status="completed",
            metrics={"loss": 0.5, "accuracy": 0.95},
        )
        assert result.status == "completed"
        assert result.metrics == {"loss": 0.5, "accuracy": 0.95}

    def test_training_result_with_error(self):
        """Test TrainingResult with error."""
        result = TrainingResult(
            task_id="task_123",
            status="failed",
            error="Training failed due to OOM",
        )
        assert result.status == "failed"
        assert result.error == "Training failed due to OOM"

    def test_training_result_with_model_info(self):
        """Test TrainingResult with model info."""
        model_info = PEFTModelInfo(
            model_id="test_model",
            peft_method="LORA",
            trainable_params=1000,
        )
        result = TrainingResult(
            task_id="task_123",
            status="completed",
            model_info=model_info,
        )
        assert result.model_info is not None
        assert result.model_info.model_id == "test_model"

    def test_training_result_to_dict(self):
        """Test TrainingResult to_dict method."""
        result = TrainingResult(
            task_id="task_123",
            status="completed",
            metrics={"loss": 0.5},
            error=None,
        )
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["task_id"] == "task_123"
        assert result_dict["status"] == "completed"
        assert result_dict["metrics"] == {"loss": 0.5}
        assert result_dict["error"] is None
        assert "model_info" not in result_dict

    def test_training_result_to_dict_with_model_info(self):
        """Test TrainingResult to_dict with model info."""
        model_info = PEFTModelInfo(
            model_id="test_model",
            peft_method="LORA",
            trainable_params=1000,
        )
        result = TrainingResult(
            task_id="task_123",
            status="completed",
            model_info=model_info,
        )
        result_dict = result.to_dict()
        assert "model_info" in result_dict
        assert result_dict["model_info"]["model_id"] == "test_model"


class TestPEFTConfig:
    """Test PEFTConfig dataclass."""

    def test_peft_config_creation(self):
        """Test PEFTConfig creation with required fields."""
        config = PEFTConfig(method="LORA")
        assert config.method == "LORA"
        assert config.params == {}
        assert config.defaults == {}
        assert config.description is None

    def test_peft_config_with_params(self):
        """Test PEFTConfig with parameters."""
        config = PEFTConfig(
            method="LORA",
            params={"r": 8, "lora_alpha": 16},
            defaults={"r": 8, "lora_alpha": 16},
            description="LoRA configuration",
        )
        assert config.params == {"r": 8, "lora_alpha": 16}
        assert config.defaults == {"r": 8, "lora_alpha": 16}
        assert config.description == "LoRA configuration"

    def test_peft_config_to_dict(self):
        """Test PEFTConfig to_dict method."""
        config = PEFTConfig(
            method="LORA",
            params={"r": 8},
            defaults={"r": 8},
            description="Test config",
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["method"] == "LORA"
        assert result["params"] == {"r": 8}
        assert result["defaults"] == {"r": 8}
        assert result["description"] == "Test config"


class TestToolResponse:
    """Test ToolResponse dataclass."""

    def test_tool_response_success(self):
        """Test ToolResponse for success case."""
        response = ToolResponse(success=True, data={"result": "ok"})
        assert response.success is True
        assert response.data == {"result": "ok"}
        assert response.error is None

    def test_tool_response_error(self):
        """Test ToolResponse for error case."""
        response = ToolResponse(success=False, error="Something went wrong")
        assert response.success is False
        assert response.data is None
        assert response.error == "Something went wrong"

    def test_tool_response_to_dict_success(self):
        """Test ToolResponse to_dict for success."""
        response = ToolResponse(success=True, data={"key": "value"})
        result = response.to_dict()
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["data"] == {"key": "value"}
        assert "error" not in result

    def test_tool_response_to_dict_error(self):
        """Test ToolResponse to_dict for error."""
        response = ToolResponse(success=False, error="Error occurred")
        result = response.to_dict()
        assert result["success"] is False
        assert result["error"] == "Error occurred"
        assert "data" not in result

    def test_tool_response_to_dict_minimal(self):
        """Test ToolResponse to_dict with minimal fields."""
        response = ToolResponse(success=True)
        result = response.to_dict()
        assert result["success"] is True
        assert "data" not in result
        assert "error" not in result
