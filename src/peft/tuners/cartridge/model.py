# Copyright 2025-present the HuggingFace Inc. team.
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

from __future__ import annotations

import torch


class CartridgeEncoder(torch.nn.Module):
    """
    A parameterized prefix KV cache.

    The parameters are stored in the same flattened layout as `PrefixEncoder` output: `[num_virtual_tokens, num_layers
    * 2 * token_dim]`, where `token_dim` is per-head hidden size times number of heads (after any GQA adjustment
    performed by `_prepare_prompt_learning_config`).

    If `num_frozen_tokens > 0`, the first `num_frozen_tokens` virtual tokens are stored as a non-trainable parameter,
    and the remaining tokens are trainable.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        num_virtual_tokens = config.num_virtual_tokens
        hidden = config.num_layers * 2 * config.token_dim
        num_frozen_tokens = int(config.num_frozen_tokens)
        if num_frozen_tokens < 0 or num_frozen_tokens > num_virtual_tokens:
            raise ValueError(
                f"`num_frozen_tokens` must be in [0, num_virtual_tokens], got {num_frozen_tokens} for "
                f"num_virtual_tokens={num_virtual_tokens}."
            )

        self.num_frozen_tokens = num_frozen_tokens
        self.num_trainable_tokens = num_virtual_tokens - num_frozen_tokens

        if self.num_frozen_tokens:
            frozen = torch.empty(self.num_frozen_tokens, hidden)
            self.frozen_embedding = torch.nn.Parameter(frozen, requires_grad=False)
        else:
            self.frozen_embedding = None

        trainable = torch.empty(self.num_trainable_tokens, hidden)
        self.trainable_embedding = torch.nn.Parameter(trainable, requires_grad=not config.inference_mode)

        self.reset_parameters()

    @property
    def embedding(self):
        """
        Expose a prefix-encoder compatible interface (`.embedding.weight`) for PEFT internals.
        """

        class _Proxy(torch.nn.Module):
            def __init__(self, parent: CartridgeEncoder):
                super().__init__()
                self._parent = parent

            @property
            def weight(self):
                return self._parent.weight

        return _Proxy(self)

    @property
    def weight(self) -> torch.Tensor:
        if self.frozen_embedding is None:
            return self.trainable_embedding
        return torch.cat([self.frozen_embedding, self.trainable_embedding], dim=0)

    def reset_parameters(self):
        # Match `torch.nn.Embedding` initialization (normal with std=1).
        with torch.no_grad():
            if self.frozen_embedding is not None:
                torch.nn.init.normal_(self.frozen_embedding)
            torch.nn.init.normal_(self.trainable_embedding)

    def load_prompt_embeddings(self, prompt_embeddings: torch.Tensor) -> None:
        """
        Load the flattened prompt embeddings saved by PEFT (`prompt_embeddings`).

        PEFT saves prompt-learning adapters as a single `prompt_embeddings` tensor. For CARTRIDGE, we split that tensor
        into frozen and trainable segments according to `self.num_frozen_tokens`.
        """
        if prompt_embeddings.ndim != 2 or prompt_embeddings.shape[0] != (
            self.num_frozen_tokens + self.num_trainable_tokens
        ):
            raise ValueError(
                "Invalid `prompt_embeddings` shape. Expected "
                f"({self.num_frozen_tokens + self.num_trainable_tokens}, hidden), got {tuple(prompt_embeddings.shape)}."
            )
        with torch.no_grad():
            if self.frozen_embedding is not None:
                self.frozen_embedding.copy_(
                    prompt_embeddings[: self.num_frozen_tokens].to(self.frozen_embedding.device)
                )
                trainable_part = prompt_embeddings[self.num_frozen_tokens :]
            else:
                trainable_part = prompt_embeddings
            self.trainable_embedding.copy_(trainable_part.to(self.trainable_embedding.device))

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = prefix_tokens.shape[0]
        # Ignore token ids; they exist for prompt-learning uniformity.
        return self.weight.unsqueeze(0).expand(batch_size, -1, -1)
