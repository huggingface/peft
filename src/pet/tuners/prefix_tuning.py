from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ..utils import PromptLearningConfig


@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a :class:`~pet.PrefixEncoder`.

    Args:
        encoder_hidden_size (:obj: int): The hidden size of the prompt encoder.
        prefix_projection (:obj: bool): Whether to project the prefix embeddings.
        postprocess_past_key_value_function (:
            obj: Optional[Callable]): The function to postprocess the past key value.
    """

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )
    postprocess_past_key_value_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "The function to postprocess the past key value"},
    )


# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
class PrefixEncoder(torch.nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Args:
        config (:class:`PrefixTuningConfig`): The configuration of the prefix encoder.

    Example::

        >>> from pet import PrefixEncoder, PrefixTuningConfig >>> config = PrefixTuningConfig(
                pet_type="PREFIX_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, token_dim=768,
                num_transformer_submodules=1, num_attention_heads=12, num_layers=12, encoder_hidden_size=768
            )
        >>> prefix_encoder = PrefixEncoder(config)


    Attributes:
        embedding (:obj:`torch.nn.Embedding`): The embedding layer of the prefix encoder. trans
        (:obj:`torch.nn.Sequential`): The two-layer MLP to transform the prefix embeddings
            if :obj:`prefix_projection` is :obj:`True`.
        prefix_projection (:obj:`bool`): Whether to project the prefix embeddings.

    Input shape: (batch_size, num_virtual_tokens)

    Output shape: (batch_size, num_virtual_tokens, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
