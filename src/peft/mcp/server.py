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

"""MCP Server main class for PEFT operations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Optional

from . import __version__
from .tools import (
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


logger = logging.getLogger(__name__)


# Try to import fastmcp; provide a fallback if not available
try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None


class PEFTMCPServer:
    """PEFT MCP Server - Model Context Protocol server for PEFT operations.

    This server exposes PEFT functionality through the Model Context Protocol,
    allowing AI assistants and other MCP clients to create, train, evaluate,
    and manage PEFT models.

    Supports three transport modes:
    - stdio: Standard input/output communication
    - SSE: Server-Sent Events over HTTP
    - HTTP: Standard HTTP transport
    """

    def __init__(self, name: str = "peft-mcp-server", version: Optional[str] = None):
        """Initialize PEFT MCP Server.

        Args:
            name: Server name.
            version: Server version (defaults to module version).
        """
        self.name = name
        self.version = version or __version__
        self._mcp: Optional[Any] = None
        self._setup_server()

    def _setup_server(self) -> None:
        """Set up the MCP server and register tools."""
        if FASTMCP_AVAILABLE:
            self._mcp = FastMCP(self.name)
            self._register_tools_fastmcp()
        else:
            logger.warning(
                "fastmcp is not installed. Install with: pip install fastmcp. "
                "Running in standalone mode with JSON-RPC interface."
            )
            self._register_tools_standalone()

    def _register_tools_fastmcp(self) -> None:
        """Register all tools with fastmcp."""
        mcp = self._mcp

        @mcp.tool()
        def mcp_list_peft_methods() -> dict[str, Any]:
            """List all available PEFT methods.

            Returns a comprehensive list of all PEFT methods supported by this server,
            including their names, descriptions, and availability status.

            Returns:
                Dictionary containing list of available PEFT methods.
            """
            result = list_peft_methods()
            return result.to_dict()

        @mcp.tool()
        def mcp_get_peft_config(method: str) -> dict[str, Any]:
            """Get configuration parameters and defaults for a PEFT method.

            Args:
                method: Name of the PEFT method (e.g., 'LORA', 'ADALORA', 'IA3').

            Returns:
                Dictionary containing configuration parameters and their defaults.
            """
            result = get_peft_config_info(method)
            return result.to_dict()

        @mcp.tool()
        def mcp_create_peft_model(
            model_id: str,
            base_model_name_or_path: str,
            method: str,
            config_params: Optional[dict[str, Any]] = None,
            adapter_name: str = "default",
        ) -> dict[str, Any]:
            """Create a PEFT model.

            Args:
                model_id: Unique identifier for the PEFT model.
                base_model_name_or_path: Name or path of the base model.
                method: PEFT method to use (e.g., 'LORA', 'ADALORA').
                config_params: Configuration parameters for the PEFT method.
                adapter_name: Name for the adapter.

            Returns:
                Dictionary containing model information.
            """
            # Load base model
            try:
                from transformers import AutoModelForCausalLM

                base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
            except (OSError, ImportError) as e:
                return {"success": False, "error": f"Failed to load base model: {e}"}

            result = create_peft_model(
                model_id=model_id,
                base_model=base_model,
                method=method,
                config_params=config_params,
                adapter_name=adapter_name,
            )
            return result.to_dict()

        @mcp.tool()
        async def mcp_train_peft_model(
            model_id: str,
            dataset_name_or_path: str,
            training_args: Optional[dict[str, Any]] = None,
            async_mode: bool = True,
        ) -> dict[str, Any]:
            """Execute PEFT training.

            Args:
                model_id: ID of the PEFT model to train.
                dataset_name_or_path: Name or path of the training dataset.
                training_args: Training arguments (e.g., learning_rate, epochs).
                async_mode: Whether to run training asynchronously.

            Returns:
                Dictionary containing training result information.
            """
            # Placeholder - in real implementation, load dataset
            result = await train_peft_model(
                model_id=model_id,
                train_dataset=dataset_name_or_path,
                training_args=training_args,
                async_mode=async_mode,
            )
            return result.to_dict()

        @mcp.tool()
        def mcp_merge_peft_weights(
            model_id: str,
            output_path: Optional[str] = None,
        ) -> dict[str, Any]:
            """Merge PEFT weights into the base model.

            Args:
                model_id: ID of the PEFT model.
                output_path: Optional path to save merged model.

            Returns:
                Dictionary containing merge result.
            """
            result = merge_peft_weights(model_id=model_id, output_path=output_path)
            return result.to_dict()

        @mcp.tool()
        def mcp_save_peft_model(
            model_id: str,
            output_path: str,
        ) -> dict[str, Any]:
            """Save PEFT model to disk.

            Args:
                model_id: ID of the PEFT model.
                output_path: Path to save the model.

            Returns:
                Dictionary containing save result.
            """
            result = save_peft_model(model_id=model_id, output_path=output_path)
            return result.to_dict()

        @mcp.tool()
        def mcp_load_peft_model(
            model_id: str,
            base_model_name_or_path: str,
            adapter_path: str,
            adapter_name: str = "default",
        ) -> dict[str, Any]:
            """Load PEFT model from disk.

            Args:
                model_id: Unique identifier for the loaded model.
                base_model_name_or_path: Name or path of the base model.
                adapter_path: Path to the PEFT adapter weights.
                adapter_name: Name for the adapter.

            Returns:
                Dictionary containing model information.
            """
            try:
                from transformers import AutoModelForCausalLM

                base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
            except (OSError, ImportError) as e:
                return {"success": False, "error": f"Failed to load base model: {e}"}

            result = load_peft_model(
                model_id=model_id,
                base_model=base_model,
                adapter_path=adapter_path,
                adapter_name=adapter_name,
            )
            return result.to_dict()

        @mcp.tool()
        def mcp_evaluate_peft_model(
            model_id: str,
            dataset_name_or_path: str,
            metrics: Optional[list[str]] = None,
        ) -> dict[str, Any]:
            """Evaluate PEFT model performance.

            Args:
                model_id: ID of the PEFT model.
                dataset_name_or_path: Name or path of the evaluation dataset.
                metrics: List of metrics to compute.

            Returns:
                Dictionary containing evaluation results.
            """
            result = evaluate_peft_model(
                model_id=model_id,
                eval_dataset=dataset_name_or_path,
                metrics=metrics,
            )
            return result.to_dict()

        @mcp.tool()
        def mcp_compare_peft_methods(
            base_model_name_or_path: str,
            methods: list[str],
            config_params: Optional[dict[str, dict[str, Any]]] = None,
        ) -> dict[str, Any]:
            """Compare different PEFT methods.

            Args:
                base_model_name_or_path: Name or path of the base model.
                methods: List of PEFT method names to compare.
                config_params: Optional configuration parameters for each method.

            Returns:
                Dictionary containing comparison results.
            """
            try:
                from transformers import AutoModelForCausalLM

                base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
            except (OSError, ImportError) as e:
                return {"success": False, "error": f"Failed to load base model: {e}"}

            result = compare_peft_methods(
                base_model=base_model,
                methods=methods,
                config_params=config_params,
            )
            return result.to_dict()

        @mcp.tool()
        def mcp_get_training_metrics(task_id: str) -> dict[str, Any]:
            """Get training metrics for a specific task.

            Args:
                task_id: ID of the training task.

            Returns:
                Dictionary containing training metrics.
            """
            result = get_training_metrics(task_id=task_id)
            return result.to_dict()

    def _register_tools_standalone(self) -> None:
        """Register tools for standalone JSON-RPC mode (fallback when fastmcp is unavailable)."""
        self._tools = {
            "list_peft_methods": list_peft_methods,
            "get_peft_config": get_peft_config_info,
            "create_peft_model": create_peft_model,
            "train_peft_model": train_peft_model,
            "merge_peft_weights": merge_peft_weights,
            "save_peft_model": save_peft_model,
            "load_peft_model": load_peft_model,
            "evaluate_peft_model": evaluate_peft_model,
            "compare_peft_methods": compare_peft_methods,
            "get_training_metrics": get_training_metrics,
        }

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC style request in standalone mode.

        Args:
            request: JSON-RPC request dictionary.

        Returns:
            JSON-RPC response dictionary.
        """
        if FASTMCP_AVAILABLE and self._mcp:
            # Delegate to fastmcp
            return await self._mcp.handle_request(request)

        # Standalone JSON-RPC handling
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method not in self._tools:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        try:
            tool_func = self._tools[method]
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**params)
            else:
                result = tool_func(**params)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result.to_dict(),
            }
        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error handling request: %s", e)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(e)},
            }

    def run_stdio(self) -> None:
        """Run the server using stdio transport."""
        if FASTMCP_AVAILABLE and self._mcp:
            logger.info("Starting PEFT MCP Server (stdio transport)")
            self._mcp.run(transport="stdio")
        else:
            logger.info("Starting PEFT MCP Server (stdio JSON-RPC mode)")
            self._run_stdio_jsonrpc()

    def _run_stdio_jsonrpc(self) -> None:
        """Run standalone JSON-RPC server over stdio."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = asyncio.run(self.handle_request(request))
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                print(json.dumps(error_response), flush=True)
            except (ValueError, TypeError, KeyError) as e:
                logger.error("Error processing request: %s", e)

    def run_sse(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the server using SSE transport.

        Args:
            host: Host to bind to.
            port: Port to listen on.
        """
        if FASTMCP_AVAILABLE and self._mcp:
            logger.info("Starting PEFT MCP Server (SSE transport) on %s:%s", host, port)
            self._mcp.run(transport="sse", host=host, port=port)
        else:
            logger.error("SSE transport requires fastmcp. Install with: pip install fastmcp")
            sys.exit(1)

    def run_http(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the server using HTTP transport.

        Args:
            host: Host to bind to.
            port: Port to listen on.
        """
        if FASTMCP_AVAILABLE and self._mcp:
            logger.info("Starting PEFT MCP Server (HTTP transport) on %s:%s", host, port)
            self._mcp.run(transport="streamable-http", host=host, port=port)
        else:
            logger.error("HTTP transport requires fastmcp. Install with: pip install fastmcp")
            sys.exit(1)

    def run(self, transport: str = "stdio", host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the server with the specified transport.

        Args:
            transport: Transport type ('stdio', 'sse', or 'http').
            host: Host to bind to (for SSE/HTTP).
            port: Port to listen on (for SSE/HTTP).
        """
        if transport == "stdio":
            self.run_stdio()
        elif transport == "sse":
            self.run_sse(host=host, port=port)
        elif transport == "http":
            self.run_http(host=host, port=port)
        else:
            logger.error("Unknown transport: %s", transport)
            sys.exit(1)


def main() -> None:
    """Command-line entry point for the PEFT MCP Server."""
    parser = argparse.ArgumentParser(
        description="PEFT MCP Server - Model Context Protocol server for PEFT operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to for SSE/HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on for SSE/HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run server
    server = PEFTMCPServer()
    server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
