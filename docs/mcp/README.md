<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PEFT MCP Server

The PEFT MCP (Model Context Protocol) server exposes PEFT functionality through the Model Context Protocol, allowing AI assistants and other MCP clients to create, train, evaluate, and manage PEFT models through natural language interactions.

## Installation

Install PEFT with MCP dependencies:

```bash
pip install peft[mcp]
```

This installs `fastmcp` along with the core PEFT package. If `fastmcp` is not available, the server falls back to a standalone JSON-RPC mode.

## Starting the Server

The server supports three transport modes:

### stdio (default)

Standard input/output communication, ideal for local AI assistant integration:

```bash
python -m peft.mcp.server stdio
```

### SSE (Server-Sent Events)

HTTP-based streaming transport for web clients:

```bash
python -m peft.mcp.server sse --host 0.0.0.0 --port 8000
```

### HTTP

Standard HTTP transport:

```bash
python -m peft.mcp.server http --host 0.0.0.0 --port 8000
```

### Additional Options

```bash
# Set logging level
python -m peft.mcp.server stdio --log-level DEBUG

# Check version
python -m peft.mcp.server --version
```

## Available Tools

The MCP server provides 10 tools:

| Tool | Description |
|------|-------------|
| `list_peft_methods` | List all available PEFT methods |
| `get_peft_config` | Get configuration parameters for a PEFT method |
| `create_peft_model` | Create a PEFT model from a base model |
| `train_peft_model` | Execute PEFT training |
| `merge_peft_weights` | Merge PEFT weights into the base model |
| `save_peft_model` | Save PEFT model to disk |
| `load_peft_model` | Load PEFT model from disk |
| `evaluate_peft_model` | Evaluate PEFT model performance |
| `compare_peft_methods` | Compare different PEFT methods |
| `get_training_metrics` | Get training metrics for a task |

## AI Agent Integration

### Claude Desktop

Add the PEFT MCP server to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "peft": {
      "command": "python",
      "args": ["-m", "peft.mcp.server", "stdio"]
    }
  }
}
```

After restarting Claude Desktop, you can ask it to perform PEFT operations directly:

```
Create a LoRA model from meta-llama/Llama-2-7b-hf with rank 16.
```

### Claude Code

In Claude Code, start the server and connect:

```bash
# Terminal 1: Start the server
python -m peft.mcp.server stdio
```

Then in Claude Code, use the MCP tools to interact with PEFT:

```
> List all available PEFT methods
> Get the configuration for AdaLoRA
> Create a LoRA model from Qwen/Qwen2.5-3B-Instruct with r=8
```

## Command-Line Usage

You can also interact with the server programmatically via JSON-RPC over stdio:

```bash
# Send a request
echo '{"jsonrpc":"2.0","id":1,"method":"list_peft_methods","params":{}}' | python -m peft.mcp.server stdio
```

Expected response:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "success": true,
    "data": {
      "methods": [
        {
          "name": "LORA",
          "description": "Low-Rank Adaptation - efficient fine-tuning via low-rank decomposition",
          "available": true
        },
        ...
      ]
    }
  }
}
```

## Python API Usage

You can use the MCP server classes directly in Python:

```python
from peft.mcp import PEFTMCPServer

# Create server instance
server = PEFTMCPServer()

# Run with stdio transport
server.run(transport="stdio")

# Or with SSE transport
server.run(transport="sse", host="0.0.0.0", port=8000)
```

You can also use the individual tool functions directly:

```python
from peft.mcp.tools import (
    list_peft_methods,
    get_peft_config_info,
    create_peft_model,
    save_peft_model,
)

# List available methods
result = list_peft_methods()
print(result.to_dict())

# Get LoRA configuration
result = get_peft_config_info("LORA")
print(result.to_dict())
```

## Complete Fine-Tuning Workflow

Below is a complete end-to-end workflow demonstrating the full fine-tuning lifecycle using the MCP tools.

### Step 1: Explore Available Methods

```python
# Ask your AI assistant:
# "What PEFT methods are available?"

# Or call directly:
result = list_peft_methods()
# Returns all methods with descriptions and availability status
```

### Step 2: Get Configuration Details

```python
# Ask your AI assistant:
# "What configuration parameters does LoRA support?"

# Or call directly:
result = get_peft_config_info("LORA")
# Returns parameter names, types, defaults, and required status
```

### Step 3: Create a PEFT Model

```python
# Ask your AI assistant:
# "Create a LoRA model from Qwen/Qwen2.5-3B-Instruct with rank 16 and alpha 32"

# Or call directly:
result = create_peft_model(
    model_id="my-lora-model",
    base_model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    method="LORA",
    config_params={"r": 16, "lora_alpha": 32, "task_type": "CAUSAL_LM"},
    adapter_name="default"
)
# Returns model info including trainable parameter count
```

### Step 4: Train the Model

```python
# Ask your AI assistant:
# "Train the model on the twitter_complaints dataset"

# Or call directly:
result = train_peft_model(
    model_id="my-lora-model",
    dataset_name_or_path="ought/raft/twitter_complaints",
    training_args={"learning_rate": 1e-4, "num_train_epochs": 3},
    async_mode=True
)
# Returns a task_id for tracking training progress
```

### Step 5: Monitor Training Progress

```python
# Ask your AI assistant:
# "How is the training going for task <task_id>?"

# Or call directly:
result = get_training_metrics(task_id="<task_id>")
# Returns current step, loss, epoch, and other metrics
```

### Step 6: Evaluate the Model

```python
# Ask your AI assistant:
# "Evaluate the model on the test set"

# Or call directly:
result = evaluate_peft_model(
    model_id="my-lora-model",
    dataset_name_or_path="ought/raft/twitter_complaints",
    metrics=["accuracy", "f1"]
)
# Returns evaluation metrics
```

### Step 7: Save the Adapter

```python
# Ask your AI assistant:
# "Save the trained adapter to ./my-lora-adapter"

# Or call directly:
result = save_peft_model(
    model_id="my-lora-model",
    output_path="./my-lora-adapter"
)
# Returns success status and output path
```

### Step 8: Merge Weights (Optional)

```python
# Ask your AI assistant:
# "Merge the adapter weights into the base model and save to ./merged-model"

# Or call directly:
result = merge_peft_weights(
    model_id="my-lora-model",
    output_path="./merged-model"
)
# Returns merged model info
```

### Step 9: Load a Saved Model

```python
# Ask your AI assistant:
# "Load the adapter from ./my-lora-adapter into Qwen/Qwen2.5-3B-Instruct"

# Or call directly:
result = load_peft_model(
    model_id="loaded-model",
    base_model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    adapter_path="./my-lora-adapter",
    adapter_name="default"
)
# Returns loaded model info
```

### Step 10: Compare Methods

```python
# Ask your AI assistant:
# "Compare LoRA and AdaLoRA on Qwen/Qwen2.5-3B-Instruct"

# Or call directly:
result = compare_peft_methods(
    base_model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    methods=["LORA", "ADALORA"],
    config_params={
        "LORA": {"r": 16},
        "ADALORA": {"init_r": 16}
    }
)
# Returns comparison of trainable parameters for each method
```

## Comparing PEFT Methods

The `compare_peft_methods` tool is useful for understanding the trade-offs between different PEFT methods before committing to one:

```python
result = compare_peft_methods(
    base_model_name_or_path="meta-llama/Llama-2-7b-hf",
    methods=["LORA", "ADALORA", "IA3"],
)

# Example output:
# {
#   "comparison": [
#     {"method": "LORA", "trainable_params": 4194304, "status": "created"},
#     {"method": "ADALORA", "trainable_params": 4194304, "status": "created"},
#     {"method": "IA3", "trainable_params": 262144, "status": "created"},
#   ]
# }
```
