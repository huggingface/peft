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

import torch


# needed for prefix-tuning of bloom model
def bloom_model_postprocess_past_key_value(past_key_values):
    past_key_values = torch.cat(past_key_values)
    total_layers, batch_size, num_attention_heads, num_virtual_tokens, head_dim = past_key_values.shape
    keys = past_key_values[: total_layers // 2]
    keys = keys.transpose(2, 3).reshape(
        total_layers // 2, batch_size * num_attention_heads, head_dim, num_virtual_tokens
    )
    values = past_key_values[total_layers // 2 :]
    values = values.reshape(total_layers // 2, batch_size * num_attention_heads, num_virtual_tokens, head_dim)

    return tuple(zip(keys, values))


TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {
    "bloom": bloom_model_postprocess_past_key_value,
}


# copied from transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _set_trainable(model):
    if model.modules_to_save is not None:
        for name, param in model.named_parameters():
            if any(module_name in name for module_name in model.modules_to_save):
                param.requires_grad = True


def fsdp_auto_wrap_policy(model):
    import functools
    import os

    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from ..tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            FullyShardedDataParallelPlugin.get_module_class_from_name(
                model, os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP", "")
            ),
        ),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight
