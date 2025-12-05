#!/bin/bash

ADAPTER_PATH="example_bd_lora_adapter"

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B \
    --enable-lora \
    --lora-modules lora1=$ADAPTER_PATH lora2=$ADAPTER_PATH \
    --tensor-parallel-size 2 \
    --block_diagonal_sharded_loras \
    --max-lora-rank 128 \
    --enforce-eager \
    --port 8000
