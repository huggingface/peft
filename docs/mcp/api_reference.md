<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MCP Server API Reference

This document provides detailed API reference for all PEFT MCP Server tools.

## Tool Response Format

All tools return a standardized response format:

```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

On error:

```json
{
  "success": false,
  "data": null,
  "error": "Error message describing what went wrong"
}
```

---

## list_peft_methods

List all available PEFT methods with their descriptions and availability status.

### Parameters

None

### Returns

```json
{
  "success": true,
  "data": {
    "methods": [
      {
        "name": "LORA",
        "description": "Low-Rank Adaptation - efficient fine-tuning via low-rank decomposition",
        "available": true
      },
      {
        "name": "ADALORA",
        "description": "Adaptive LoRA - dynamically adjusts rank allocation",
        "available": true
      }
    ]
  }
}
```

### Example

```python
result = list_peft_methods()
methods = result.data["methods"]
for method in methods:
    print(f"{method['name']}: {method['description']}")
```

---

## get_peft_config

Get configuration parameters and defaults for a specific PEFT method.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `method` | string | Yes | Name of the PEFT method (e.g., 'LORA', 'ADALORA', 'IA3') |

### Returns

```json
{
  "success": true,
  "data": {
    "method": "LORA",
    "description": "Low-Rank Adaptation - efficient fine-tuning via low-rank decomposition",
    "params": {
      "r": {
        "name": "r",
        "type": "int",
        "default": 8,
        "required": false
      },
      "lora_alpha": {
        "name": "lora_alpha",
        "type": "int",
        "default": 8,
        "required": false
      },
      "target_modules": {
        "name": "target_modules",
        "type": "Optional[Union[list[str], str]]",
        "default": null,
        "required": false
      }
    },
    "defaults": {
      "r": 8,
      "lora_alpha": 8,
      "target_modules": null
    }
  }
}
```

### Example

```python
result = get_peft_config(method="LORA")
params = result.data["params"]
for param_name, param_info in params.items():
    print(f"{param_name}: {param_info['type']} = {param_info.get('default')}")
```

### Errors

- `PEFT method 'XYZ' is not available` - The specified method is not supported
- `Configuration class not found for method 'XYZ'` - Internal configuration error

---

## create_peft_model

Create a PEFT model from a base model with specified configuration.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | Unique identifier for the PEFT model |
| `base_model_name_or_path` | string | Yes | Name or path of the base model (e.g., 'meta-llama/Llama-2-7b-hf') |
| `method` | string | Yes | PEFT method to use (e.g., 'LORA', 'ADALORA') |
| `config_params` | dict | No | Configuration parameters for the PEFT method |
| `adapter_name` | string | No | Name for the adapter (default: 'default') |

### Returns

```json
{
  "success": true,
  "data": {
    "model_id": "my-lora-model",
    "peft_method": "LORA",
    "trainable_params": 4194304,
    "status": "created",
    "base_model": "LlamaForCausalLM",
    "task_type": "CAUSAL_LM",
    "metadata": {
      "total_params": 6738415616,
      "trainable_percentage": 0.0622,
      "adapter_name": "default"
    }
  }
}
```

### Example

```python
result = create_peft_model(
    model_id="my-lora-model",
    base_model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    method="LORA",
    config_params={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    },
    adapter_name="default"
)

print(f"Created model with {result.data['trainable_params']} trainable parameters")
print(f"Trainable percentage: {result.data['metadata']['trainable_percentage']:.4f}%")
```

### Errors

- `Failed to load base model: <error>` - Could not load the base model
- `PEFT method 'XYZ' is not available` - The specified method is not supported
- `Configuration class not found for method 'XYZ'` - Invalid method configuration

---

## train_peft_model

Execute PEFT training on a model.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | ID of the PEFT model to train |
| `dataset_name_or_path` | string | Yes | Name or path of the training dataset |
| `training_args` | dict | No | Training arguments (e.g., learning_rate, epochs) |
| `async_mode` | bool | No | Whether to run training asynchronously (default: true) |

### Returns

For async mode:

```json
{
  "success": true,
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "running",
    "message": "Training started in background"
  }
}
```

For sync mode:

```json
{
  "success": true,
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "metrics": {
      "train_loss": 0.452,
      "train_runtime": 120.5
    }
  }
}
```

### Example

```python
# Async training
result = train_peft_model(
    model_id="my-lora-model",
    dataset_name_or_path="ought/raft/twitter_complaints",
    training_args={
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4
    },
    async_mode=True
)

task_id = result.data["task_id"]
print(f"Training started with task ID: {task_id}")

# Check progress later
metrics = get_training_metrics(task_id=task_id)
```

### Errors

- `Model 'XYZ' not found in cache` - The model_id does not exist
- Training errors from the underlying training framework

---

## get_training_metrics

Get training metrics and progress for a specific training task.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_id` | string | Yes | ID of the training task |

### Returns

```json
{
  "success": true,
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "running",
    "metrics": {
      "current_step": 150,
      "total_steps": 500,
      "current_epoch": 1,
      "total_epochs": 3,
      "loss": 0.452
    },
    "error": null
  }
}
```

### Example

```python
result = get_training_metrics(task_id="550e8400-e29b-41d4-a716-446655440000")

if result.data["status"] == "running":
    metrics = result.data["metrics"]
    progress = (metrics["current_step"] / metrics["total_steps"]) * 100
    print(f"Training progress: {progress:.1f}%")
    print(f"Current loss: {metrics['loss']:.4f}")
elif result.data["status"] == "completed":
    print("Training completed!")
    print(f"Final metrics: {result.data['metrics']}")
```

### Errors

- `Training task 'XYZ' not found` - The task_id does not exist

---

## merge_peft_weights

Merge PEFT adapter weights into the base model.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | ID of the PEFT model |
| `output_path` | string | No | Optional path to save the merged model |

### Returns

```json
{
  "success": true,
  "data": {
    "model_id": "my-lora-model",
    "peft_method": "MERGED",
    "trainable_params": 0,
    "status": "merged",
    "base_model": null,
    "task_type": null,
    "metadata": {
      "output_path": "./merged-model"
    }
  }
}
```

### Example

```python
# Merge and save to disk
result = merge_peft_weights(
    model_id="my-lora-model",
    output_path="./merged-model"
)

if result.success:
    print(f"Merged model saved to: {result.data['metadata']['output_path']}")

# Merge without saving (in-memory only)
result = merge_peft_weights(model_id="my-lora-model")
```

### Errors

- `Model 'XYZ' not found in cache` - The model_id does not exist
- `Model 'XYZ' does not support weight merging` - The model type doesn't support merging

---

## save_peft_model

Save PEFT model adapter weights to disk.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | ID of the PEFT model |
| `output_path` | string | Yes | Path to save the model |

### Returns

```json
{
  "success": true,
  "data": {
    "model_id": "my-lora-model",
    "output_path": "./my-lora-adapter"
  }
}
```

### Example

```python
result = save_peft_model(
    model_id="my-lora-model",
    output_path="./my-lora-adapter"
)

if result.success:
    print(f"Model saved to: {result.data['output_path']}")
```

### Errors

- `Model 'XYZ' not found in cache` - The model_id does not exist
- File system errors (permission denied, disk full, etc.)

---

## load_peft_model

Load a PEFT model from disk.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | Unique identifier for the loaded model |
| `base_model_name_or_path` | string | Yes | Name or path of the base model |
| `adapter_path` | string | Yes | Path to the PEFT adapter weights |
| `adapter_name` | string | No | Name for the adapter (default: 'default') |

### Returns

```json
{
  "success": true,
  "data": {
    "model_id": "loaded-model",
    "peft_method": "LOADED",
    "trainable_params": 4194304,
    "status": "loaded",
    "base_model": "LlamaForCausalLM",
    "task_type": null,
    "metadata": {
      "total_params": 6738415616,
      "adapter_path": "./my-lora-adapter",
      "adapter_name": "default"
    }
  }
}
```

### Example

```python
result = load_peft_model(
    model_id="loaded-model",
    base_model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    adapter_path="./my-lora-adapter",
    adapter_name="default"
)

if result.success:
    print(f"Loaded model with {result.data['trainable_params']} trainable parameters")
```

### Errors

- `Failed to load base model: <error>` - Could not load the base model
- File not found or invalid adapter path

---

## evaluate_peft_model

Evaluate PEFT model performance on a dataset.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | ID of the PEFT model |
| `dataset_name_or_path` | string | Yes | Name or path of the evaluation dataset |
| `metrics` | list | No | List of metrics to compute (e.g., ['accuracy', 'f1']) |

### Returns

```json
{
  "success": true,
  "data": {
    "model_id": "my-lora-model",
    "status": "completed",
    "metrics": {
      "accuracy": 0.923,
      "f1": 0.915,
      "eval_runtime": 45.2
    }
  }
}
```

### Example

```python
result = evaluate_peft_model(
    model_id="my-lora-model",
    dataset_name_or_path="ought/raft/twitter_complaints",
    metrics=["accuracy", "f1"]
)

if result.success:
    metrics = result.data["metrics"]
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
```

### Errors

- `Model 'XYZ' not found in cache` - The model_id does not exist
- Evaluation errors from the underlying framework

---

## compare_peft_methods

Compare different PEFT methods on the same base model.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `base_model_name_or_path` | string | Yes | Name or path of the base model |
| `methods` | list | Yes | List of PEFT method names to compare |
| `config_params` | dict | No | Optional configuration parameters for each method |

### Returns

```json
{
  "success": true,
  "data": {
    "comparison": [
      {
        "method": "LORA",
        "trainable_params": 4194304,
        "status": "created",
        "metadata": {
          "total_params": 6738415616,
          "trainable_percentage": 0.0622
        }
      },
      {
        "method": "ADALORA",
        "trainable_params": 4194304,
        "status": "created",
        "metadata": {
          "total_params": 6738415616,
          "trainable_percentage": 0.0622
        }
      }
    ]
  }
}
```

### Example

```python
result = compare_peft_methods(
    base_model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    methods=["LORA", "ADALORA", "IA3"],
    config_params={
        "LORA": {"r": 16},
        "ADALORA": {"init_r": 16},
        "IA3": {}
    }
)

if result.success:
    for comparison in result.data["comparison"]:
        method = comparison["method"]
        params = comparison["trainable_params"]
        percentage = comparison["metadata"]["trainable_percentage"]
        print(f"{method}: {params:,} parameters ({percentage:.4f}%)")
```

### Errors

- `Failed to load base model: <error>` - Could not load the base model
- Errors for individual methods are included in the comparison results

---

## Error Handling

All tools return a standardized error response when something goes wrong:

```json
{
  "success": false,
  "data": null,
  "error": "Descriptive error message"
}
```

### Common Error Types

1. **Model Not Found**: `Model 'XYZ' not found in cache`
   - The model_id does not exist in the server's cache
   - Solution: Create or load the model first

2. **Method Not Available**: `PEFT method 'XYZ' is not available`
   - The specified PEFT method is not supported
   - Solution: Use `list_peft_methods()` to see available methods

3. **Base Model Load Failed**: `Failed to load base model: <error>`
   - Could not load the specified base model
   - Solution: Check model name/path and ensure it's accessible

4. **Task Not Found**: `Training task 'XYZ' not found`
   - The task_id does not exist
   - Solution: Use the task_id returned from `train_peft_model()`

5. **Merge Not Supported**: `Model 'XYZ' does not support weight merging`
   - The model type doesn't support the merge operation
   - Solution: Use a model type that supports merging (e.g., LoRA)

### Error Handling Example

```python
result = create_peft_model(
    model_id="test",
    base_model_name_or_path="invalid-model",
    method="LORA"
)

if not result.success:
    print(f"Error: {result.error}")
    # Handle the error appropriately
```

---

## Best Practices

### 1. Use Descriptive Model IDs

Choose meaningful model_id values to track your models:

```python
# Good
model_id = "lora-qwen2.5-3b-twitter-complaints"

# Avoid
model_id = "model1"
```

### 2. Check Method Availability First

Before creating a model, verify the method is available:

```python
methods = list_peft_methods()
available = [m["name"] for m in methods.data["methods"] if m["available"]]

if "LORA" in available:
    create_peft_model(...)
```

### 3. Monitor Training Progress

For long-running training tasks, periodically check progress:

```python
task_id = train_peft_model(..., async_mode=True).data["task_id"]

while True:
    metrics = get_training_metrics(task_id=task_id)
    status = metrics.data["status"]
    
    if status == "completed":
        break
    elif status == "failed":
        print(f"Training failed: {metrics.data['error']}")
        break
    
    # Check progress
    current = metrics.data["metrics"]["current_step"]
    total = metrics.data["metrics"]["total_steps"]
    print(f"Progress: {current}/{total}")
    
    time.sleep(60)  # Check every minute
```

### 4. Save Models Regularly

Save your models after training to avoid losing work:

```python
# Train
train_peft_model(model_id="my-model", ...)

# Save immediately
save_peft_model(
    model_id="my-model",
    output_path="./checkpoints/my-model-epoch-1"
)
```

### 5. Compare Methods Before Committing

Use `compare_peft_methods` to understand trade-offs:

```python
comparison = compare_peft_methods(
    base_model_name_or_path="meta-llama/Llama-2-7b-hf",
    methods=["LORA", "ADALORA", "IA3"]
)

# Choose the method with the best balance of parameters and performance
for result in comparison.data["comparison"]:
    print(f"{result['method']}: {result['trainable_params']:,} params")
```

### 6. Handle Errors Gracefully

Always check the `success` field and handle errors:

```python
result = some_tool(...)

if result.success:
    # Use result.data
    pass
else:
    # Handle result.error
    print(f"Operation failed: {result.error}")
```

### 7. Clean Up Resources

When done with a model, you can remove it from cache (if implementing custom cleanup):

```python
# This would need to be implemented in your application
model_cache.remove(model_id="old-model")
```
