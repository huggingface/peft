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

"""Integration tests for PEFT MCP Server."""

from unittest.mock import MagicMock, patch

import pytest

from peft.mcp import (
    ModelCache,
    PEFTMCPServer,
    PEFTMethodRegistry,
    ProgressCallback,
)
from peft.mcp.tools import (
    _model_cache,
    _training_tasks,
    create_peft_model,
    load_peft_model,
    merge_peft_weights,
    save_peft_model,
    train_peft_model,
)


class TestPEFTMCPServer:
    """Test PEFTMCPServer class."""

    def test_server_creation(self):
        """Test server creation with default parameters."""
        server = PEFTMCPServer()
        assert server.name == "peft-mcp-server"
        assert server.version is not None

    def test_server_creation_custom_name(self):
        """Test server creation with custom name."""
        server = PEFTMCPServer(name="custom-server")
        assert server.name == "custom-server"

    def test_server_creation_custom_version(self):
        """Test server creation with custom version."""
        server = PEFTMCPServer(version="1.0.0")
        assert server.version == "1.0.0"

    def test_server_standalone_mode(self):
        """Test server in standalone mode (without fastmcp)."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", False):
            server = PEFTMCPServer()
            assert hasattr(server, "_tools")
            assert "list_peft_methods" in server._tools
            assert "create_peft_model" in server._tools

    @pytest.mark.asyncio
    async def test_server_handle_request_standalone(self):
        """Test handling JSON-RPC request in standalone mode."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", False):
            server = PEFTMCPServer()
            request = {
                "jsonrpc": "2.0",
                "method": "list_peft_methods",
                "params": {},
                "id": 1,
            }
            response = await server.handle_request(request)
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 1
            assert "result" in response
            assert response["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_server_handle_request_method_not_found(self):
        """Test handling request for non-existent method."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", False):
            server = PEFTMCPServer()
            request = {
                "jsonrpc": "2.0",
                "method": "nonexistent_method",
                "params": {},
                "id": 1,
            }
            response = await server.handle_request(request)
            assert "error" in response
            assert response["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_server_handle_request_with_params(self):
        """Test handling request with parameters."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", False):
            server = PEFTMCPServer()
            request = {
                "jsonrpc": "2.0",
                "method": "get_peft_config",
                "params": {"method": "LORA"},
                "id": 1,
            }
            response = await server.handle_request(request)
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 1
            assert "result" in response


class TestEndToEndWorkflow:
    """Test end-to-end PEFT workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()
        _training_tasks.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()
        _training_tasks.clear()

    @patch("peft.mcp.tools.get_peft_model")
    @patch("peft.mcp.tools.count_trainable_parameters")
    @patch("peft.mcp.tools.count_total_parameters")
    def test_create_train_merge_workflow(self, mock_count_total, mock_count_trainable, mock_get_peft):
        """Test complete workflow: create -> train -> merge."""
        # Mock PEFT model
        mock_model = MagicMock()
        mock_merged = MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged
        mock_get_peft.return_value = mock_model
        mock_count_trainable.return_value = 1000
        mock_count_total.return_value = 100000

        base_model = MagicMock()
        base_model.__class__.__name__ = "MockModel"

        # Step 1: Create PEFT model
        create_result = create_peft_model(
            model_id="workflow_model",
            base_model=base_model,
            method="LORA",
            config_params={"r": 8},
        )
        assert create_result.success is True
        assert create_result.data["model_id"] == "workflow_model"
        assert "workflow_model" in _model_cache

        # Step 2: Train model (async mode)
        train_result = (
            pytest.importorskip("asyncio")
            .get_event_loop()
            .run_until_complete(
                train_peft_model(
                    model_id="workflow_model",
                    train_dataset=MagicMock(),
                    async_mode=True,
                )
            )
        )
        assert train_result.success is True
        assert train_result.data["status"] == "running"

        # Step 3: Merge weights
        merge_result = merge_peft_weights(model_id="workflow_model")
        assert merge_result.success is True
        assert merge_result.data["status"] == "merged"
        mock_model.merge_and_unload.assert_called_once()

    @patch("peft.mcp.tools.get_peft_model")
    @patch("peft.mcp.tools.count_trainable_parameters")
    @patch("peft.mcp.tools.count_total_parameters")
    def test_create_save_load_workflow(self, mock_count_total, mock_count_trainable, mock_get_peft):
        """Test workflow: create -> save -> load."""
        # Mock PEFT model
        mock_model = MagicMock()
        mock_get_peft.return_value = mock_model
        mock_count_trainable.return_value = 1000
        mock_count_total.return_value = 100000

        base_model = MagicMock()
        base_model.__class__.__name__ = "MockModel"

        # Step 1: Create PEFT model
        create_result = create_peft_model(
            model_id="save_load_model",
            base_model=base_model,
            method="LORA",
        )
        assert create_result.success is True

        # Step 2: Save model
        save_result = save_peft_model(
            model_id="save_load_model",
            output_path="/tmp/test_model",
        )
        assert save_result.success is True
        mock_model.save_pretrained.assert_called_once_with("/tmp/test_model")

        # Step 3: Load model (mock the load)
        with patch("peft.mcp.tools.PeftModel.from_pretrained") as mock_from_pretrained:
            mock_loaded_model = MagicMock()
            mock_from_pretrained.return_value = mock_loaded_model

            load_result = load_peft_model(
                model_id="loaded_model",
                base_model=base_model,
                adapter_path="/tmp/test_model",
            )
            assert load_result.success is True
            assert load_result.data["status"] == "loaded"
            assert "loaded_model" in _model_cache


class TestMultipleModelManagement:
    """Test managing multiple models simultaneously."""

    def setup_method(self):
        """Set up test fixtures."""
        _model_cache.clear()

    def teardown_method(self):
        """Clean up after tests."""
        _model_cache.clear()

    @patch("peft.mcp.tools.get_peft_model")
    @patch("peft.mcp.tools.count_trainable_parameters")
    @patch("peft.mcp.tools.count_total_parameters")
    def test_multiple_models_in_cache(self, mock_count_total, mock_count_trainable, mock_get_peft):
        """Test managing multiple models in cache."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_get_peft.side_effect = [mock_model1, mock_model2]
        mock_count_trainable.return_value = 1000
        mock_count_total.return_value = 100000

        base_model = MagicMock()
        base_model.__class__.__name__ = "MockModel"

        # Create first model
        result1 = create_peft_model(
            model_id="model_1",
            base_model=base_model,
            method="LORA",
        )
        assert result1.success is True

        # Create second model with LORA (different config)
        result2 = create_peft_model(
            model_id="model_2",
            base_model=base_model,
            method="LORA",
            config_params={"r": 16},  # Different rank
        )
        assert result2.success is True

        # Both should be in cache
        assert "model_1" in _model_cache
        assert "model_2" in _model_cache
        assert len(_model_cache) == 2

        # Can retrieve both
        assert _model_cache.get("model_1") is mock_model1
        assert _model_cache.get("model_2") is mock_model2


class TestProgressCallbackIntegration:
    """Test ProgressCallback integration."""

    def test_progress_callback_with_training(self):
        """Test progress callback during training simulation."""
        callback = ProgressCallback(task_id="test_task")
        updates = []

        def handler(task_id, progress):
            updates.append(progress.copy())

        callback.add_callback(handler)

        # Simulate training progress
        for step in range(0, 100, 10):
            callback.update(
                current_step=step,
                total_steps=100,
                loss=1.0 - (step / 100),
            )

        assert len(updates) == 10
        assert updates[-1]["current_step"] == 90
        assert updates[-1]["total_steps"] == 100
        assert callback.percentage == 90.0

    def test_progress_callback_multiple_handlers(self):
        """Test multiple progress callbacks."""
        callback = ProgressCallback(task_id="test_task")
        updates1 = []
        updates2 = []

        def handler1(task_id, progress):
            updates1.append(progress)

        def handler2(task_id, progress):
            updates2.append(progress)

        callback.add_callback(handler1)
        callback.add_callback(handler2)

        callback.update(current_step=50, total_steps=100)

        assert len(updates1) == 1
        assert len(updates2) == 1
        assert updates1[0]["current_step"] == 50
        assert updates2[0]["current_step"] == 50


class TestModelCacheIntegration:
    """Test ModelCache integration scenarios."""

    def test_cache_with_lru_eviction(self):
        """Test cache behavior with LRU eviction."""
        cache = ModelCache(max_size=3)

        # Add models
        cache.put("model_1", MagicMock())
        cache.put("model_2", MagicMock())
        cache.put("model_3", MagicMock())
        assert len(cache) == 3

        # Access model_1 to make it recently used
        cache.get("model_1")

        # Add model_4, should evict model_2 (least recently used)
        cache.put("model_4", MagicMock())
        assert len(cache) == 3
        assert "model_1" in cache
        assert "model_2" not in cache
        assert "model_3" in cache
        assert "model_4" in cache

    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        import threading

        cache = ModelCache()
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    model_id = f"model_{thread_id}_{i}"
                    cache.put(model_id, MagicMock())
                    cache.get(model_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(cache) == 50  # 5 threads * 10 models each


class TestPEFTMethodRegistryIntegration:
    """Test PEFTMethodRegistry integration."""

    def test_registry_with_all_methods(self):
        """Test registry contains expected methods."""
        registry = PEFTMethodRegistry()
        methods = registry.list_methods()

        # Check for common methods
        method_names = [m["name"] for m in methods]
        assert "LORA" in method_names
        assert "ADALORA" in method_names
        assert "IA3" in method_names

    def test_registry_method_availability(self):
        """Test checking method availability."""
        registry = PEFTMethodRegistry()

        # Available methods
        assert registry.is_available("LORA") is True
        assert registry.is_available("ADALORA") is True

        # Non-existent methods
        assert registry.is_available("NONEXISTENT") is False

    def test_registry_custom_method_registration(self):
        """Test registering custom method."""
        registry = PEFTMethodRegistry()

        class CustomConfig:
            pass

        class CustomModel:
            pass

        registry.register_method(
            name="CUSTOM",
            description="Custom PEFT method",
            config_cls=CustomConfig,
            model_cls=CustomModel,
        )

        method_info = registry.get_method("CUSTOM")
        assert method_info is not None
        assert method_info["name"] == "CUSTOM"
        assert method_info["description"] == "Custom PEFT method"

        config_cls = registry.get_config_class("CUSTOM")
        assert config_cls is CustomConfig

        model_cls = registry.get_model_class("CUSTOM")
        assert model_cls is CustomModel


class TestToolResponseIntegration:
    """Test ToolResponse in integration scenarios."""

    def test_tool_response_chain(self):
        """Test chaining tool responses."""
        from peft.mcp.models import ToolResponse

        # Simulate a chain of operations
        response1 = ToolResponse(success=True, data={"step": 1})
        assert response1.success is True

        response2 = ToolResponse(success=True, data={"step": 2})
        assert response2.success is True

        # Convert to dict for serialization
        dict1 = response1.to_dict()
        dict2 = response2.to_dict()

        assert dict1["success"] is True
        assert dict1["data"]["step"] == 1
        assert dict2["success"] is True
        assert dict2["data"]["step"] == 2

    def test_error_propagation(self):
        """Test error propagation through tool responses."""
        from peft.mcp.models import ToolResponse

        # Simulate error in workflow
        response = ToolResponse(success=False, error="Operation failed")
        assert response.success is False
        assert response.error == "Operation failed"

        response_dict = response.to_dict()
        assert response_dict["success"] is False
        assert response_dict["error"] == "Operation failed"


class TestServerRunMethods:
    """Test server run methods."""

    def test_server_run_stdio(self):
        """Test run method with stdio transport."""
        server = PEFTMCPServer()
        with patch.object(server, "run_stdio") as mock_run_stdio:
            server.run(transport="stdio")
            mock_run_stdio.assert_called_once()

    def test_server_run_sse(self):
        """Test run method with SSE transport."""
        server = PEFTMCPServer()
        with patch.object(server, "run_sse") as mock_run_sse:
            server.run(transport="sse", host="localhost", port=9000)
            mock_run_sse.assert_called_once_with(host="localhost", port=9000)

    def test_server_run_http(self):
        """Test run method with HTTP transport."""
        server = PEFTMCPServer()
        with patch.object(server, "run_http") as mock_run_http:
            server.run(transport="http", host="localhost", port=9000)
            mock_run_http.assert_called_once_with(host="localhost", port=9000)

    def test_server_run_unknown_transport(self):
        """Test run method with unknown transport."""
        server = PEFTMCPServer()
        with patch("sys.exit") as mock_exit:
            server.run(transport="unknown")
            mock_exit.assert_called_once_with(1)

    def test_server_run_sse_without_fastmcp(self):
        """Test run_sse without fastmcp."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", False):
            server = PEFTMCPServer()
            with patch("sys.exit") as mock_exit:
                server.run_sse()
                mock_exit.assert_called_once_with(1)

    def test_server_run_http_without_fastmcp(self):
        """Test run_http without fastmcp."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", False):
            server = PEFTMCPServer()
            with patch("sys.exit") as mock_exit:
                server.run_http()
                mock_exit.assert_called_once_with(1)


class TestServerFastMCPMode:
    """Test server in FastMCP mode."""

    def test_server_with_fastmcp(self):
        """Test server creation with FastMCP available."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", True):
            with patch("peft.mcp.server.FastMCP") as mock_fastmcp:
                mock_mcp_instance = MagicMock()
                mock_fastmcp.return_value = mock_mcp_instance

                server = PEFTMCPServer()
                assert server._mcp is mock_mcp_instance
                mock_fastmcp.assert_called_once_with("peft-mcp-server")

    @pytest.mark.asyncio
    async def test_server_handle_request_with_fastmcp(self):
        """Test handling request with FastMCP."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", True):
            with patch("peft.mcp.server.FastMCP") as mock_fastmcp:
                mock_mcp_instance = MagicMock()
                # Use AsyncMock to return an awaitable
                from unittest.mock import AsyncMock

                mock_mcp_instance.handle_request = AsyncMock(return_value={"result": "success"})
                mock_fastmcp.return_value = mock_mcp_instance

                server = PEFTMCPServer()
                request = {"method": "test", "params": {}}
                response = await server.handle_request(request)
                assert response == {"result": "success"}


class TestServerErrorHandling:
    """Test server error handling."""

    @pytest.mark.asyncio
    async def test_handle_request_exception(self):
        """Test handling request that raises exception."""
        with patch("peft.mcp.server.FASTMCP_AVAILABLE", False):
            server = PEFTMCPServer()
            # Replace a tool with one that raises exception
            server._tools["test_method"] = MagicMock(side_effect=ValueError("Test error"))

            request = {
                "jsonrpc": "2.0",
                "method": "test_method",
                "params": {},
                "id": 1,
            }
            response = await server.handle_request(request)
            assert "error" in response
            assert response["error"]["code"] == -32000
            assert "Test error" in response["error"]["message"]


class TestMainFunction:
    """Test main function."""

    def test_main_with_default_args(self):
        """Test main function with default arguments."""
        from peft.mcp.server import main

        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                transport="stdio",
                host="0.0.0.0",
                port=8000,
                log_level="INFO",
            )
            with patch("peft.mcp.server.PEFTMCPServer") as mock_server_class:
                mock_server = MagicMock()
                mock_server_class.return_value = mock_server

                main()

                mock_server_class.assert_called_once()
                mock_server.run.assert_called_once_with(transport="stdio", host="0.0.0.0", port=8000)

    def test_main_with_custom_args(self):
        """Test main function with custom arguments."""
        from peft.mcp.server import main

        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                transport="http",
                host="localhost",
                port=9000,
                log_level="DEBUG",
            )
            with patch("peft.mcp.server.PEFTMCPServer") as mock_server_class:
                mock_server = MagicMock()
                mock_server_class.return_value = mock_server

                main()

                mock_server.run.assert_called_once_with(transport="http", host="localhost", port=9000)
