import torch


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
