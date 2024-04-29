#!/usr/bin/env python3

# coding=utf-8
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
import copy
import importlib
import os
import unittest

import torch
import torch.nn.init as init

from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict


def is_megatron_available() -> bool:
    return importlib.util.find_spec("megatron") is not None


if is_megatron_available():
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    world_size = 1
    rank = 0

    def initialize_distributed():
        print(f"Initializing torch.distributed with rank: {rank}, world_size: {world_size}")
        torch.cuda.set_device(0)
        init_method = "tcp://"
        master_ip = os.getenv("MASTER_ADDR", "localhost")
        master_port = os.getenv("MASTER_PORT", "6001")
        init_method += master_ip + ":" + master_port
        torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank, init_method=init_method)

    def destroy_model_parallel():
        parallel_state.destroy_model_parallel()
        torch.distributed.barrier()

    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None,
    ):
        parallel_state.destroy_model_parallel()
        if not torch.distributed.is_initialized():
            initialize_distributed()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
        )

    class DummyModule(MegatronModule):
        def __init__(self, config: TransformerConfig):
            super().__init__(config)
            self.linear = tensor_parallel.ColumnParallelLinear(
                input_size=10,
                output_size=10,
                config=config,
                init_method=init.xavier_normal_,
                bias=False,
                gather_output=False,
            )
            self.lm_head = tensor_parallel.RowParallelLinear(
                input_size=10,
                output_size=10,
                config=config,
                init_method=init.xavier_normal_,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
            )

        def forward(self, input):
            x = self.linear(input)[0]
            x = self.lm_head(x)[0]
            return x

    class TestMegatronLora(unittest.TestCase):
        def setUp(self):
            initialize_model_parallel(1, 1)
            model_parallel_cuda_manual_seed(123)
            transformer_config = {
                "num_layers": 2,
                "hidden_size": 12,
                "num_attention_heads": 4,
                "use_cpu_initialization": True,
            }
            config = TransformerConfig(**transformer_config)
            self.megatron_module = DummyModule(config=config).cuda()
            self.dummy_module = copy.deepcopy(self.megatron_module).cuda()

            lora_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                target_modules=["linear", "lm_head"],
                megatron_config=config,
                megatron_core="megatron.core",
            )
            self.megatron_module = get_peft_model(self.megatron_module, lora_config)

        def tearDown(self):
            destroy_model_parallel()

        def test_megatron_lora_module(self):
            megatron_module = self.megatron_module
            assert isinstance(megatron_module, PeftModel)

            for name, module in megatron_module.named_modules():
                if name.endswith("linear"):
                    assert hasattr(module, "lora_A")
                    assert hasattr(module, "lora_B")
                if name.endswith("linear.lora_A.default"):
                    assert isinstance(module, torch.nn.Linear)
                if name.endswith("linear.lora_B.default"):
                    assert isinstance(module, tensor_parallel.ColumnParallelLinear)

                if name.endswith("lm_head.lora_A.default"):
                    assert isinstance(module, tensor_parallel.RowParallelLinear)
                if name.endswith("lm_head.lora_B.default"):
                    assert isinstance(module, torch.nn.Linear)

        def test_forward(self):
            x = torch.ones((2, 4, 10)).cuda()
            megatron_module_result = self.megatron_module(x)
            dummt_module_result = self.dummy_module(x)

            # Because lora_B is initialized with 0, the forward results of two models should be equal before backward.
            assert megatron_module_result.equal(dummt_module_result)

        def test_backward(self):
            optimizer = torch.optim.AdamW(self.megatron_module.parameters())
            loss_fn = torch.nn.CrossEntropyLoss()

            x = torch.randn(2, 4, 10, requires_grad=True).cuda()
            label = torch.randint(10, (2 * 4,)).cuda()

            output = self.megatron_module(x)
            output = output.reshape(2 * 4, 10)
            loss = loss_fn(output, label)

            loss.backward()
            optimizer.step()

        def test_get_peft_model_state_dict(self):
            peft_state_dict = get_peft_model_state_dict(self.megatron_module)

            for key in peft_state_dict.keys():
                assert "lora" in key
