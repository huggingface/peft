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

"""Data models for PEFT MCP Server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PEFTModelInfo:
    """Information about a PEFT model."""

    model_id: str
    peft_method: str
    trainable_params: int
    status: str = "initialized"
    base_model: Optional[str] = None
    task_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "peft_method": self.peft_method,
            "trainable_params": self.trainable_params,
            "status": self.status,
            "base_model": self.base_model,
            "task_type": self.task_type,
            "metadata": self.metadata,
        }


@dataclass
class TrainingResult:
    """Result of a PEFT training operation."""

    task_id: str
    status: str = "pending"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    model_info: Optional[PEFTModelInfo] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "status": self.status,
            "metrics": self.metrics,
            "error": self.error,
        }
        if self.model_info:
            result["model_info"] = self.model_info.to_dict()
        return result


@dataclass
class PEFTConfig:
    """Configuration for a PEFT method."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "params": self.params,
            "defaults": self.defaults,
            "description": self.description,
        }


@dataclass
class ToolResponse:
    """Response from an MCP tool."""

    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        response = {"success": self.success}
        if self.data is not None:
            response["data"] = self.data
        if self.error:
            response["error"] = self.error
        return response
