# Copyright 2024-present the HuggingFace Inc. team.
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

import torch
from torch.nn import CrossEntropyLoss

from peft.utils.integrations import gather_params_ctx


class CPTEmbedding(torch.nn.Module):
    """
    CPTEmbedding is a custom embedding layer designed for Context-aware Prompt Tuning (CPT) in PEFT. It initializes
    embeddings, applies prompt-specific projections, and computes loss using label masks.
    """

    def __init__(self, config, word_embeddings):
        """
        Initializes the CPTEmbedding module.

        Args:
            config (Namespace):
                Configuration object containing model hyperparameters and CPT-specific settings.
            word_embeddings (torch.nn.Embedding):
                The base word embedding layer used to initialize CPT embeddings.
        """
        super().__init__()
        self.config = copy.deepcopy(config)
        num_virtual_tokens = config.num_virtual_tokens

        # Initialize embeddings with virtual token dimensions
        self.embedding = torch.nn.Embedding(num_virtual_tokens, config.token_dim)

        # Initialize embeddings using text-based prompt tuning, if configured
        if not config.inference_mode:
            assert config.num_virtual_tokens == len(config.cpt_token_ids)

            init_token_ids = torch.LongTensor(config.cpt_token_ids).to(word_embeddings.weight.device)
            with gather_params_ctx(word_embeddings.parameters()):
                word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

        # Initialize delta embedding with zero weights
        self.delta_embedding = torch.nn.Embedding(num_virtual_tokens, config.token_dim)
        self.delta_embedding.weight.data = torch.zeros_like(self.delta_embedding.weight).to(torch.float32)

        # Apply hook for backward gradient updates
        self.set_updated_tokens()

    def forward(self, indices):
        """
        Computes the prompt embeddings and applies delta adjustments.

        Args:
            indices (torch.Tensor):
                Indices of the tokens to be embedded.

        Returns:
            torch.Tensor:
                Sum of prompt embeddings and delta embeddings.
        """
        with torch.no_grad():
            prompt_embeddings = self.embedding(indices)

        self.delta_embedding.weight.data = self.get_projection()  # Apply epsilon-based projection

        delta_prompt_embeddings = self.delta_embedding(indices)

        return prompt_embeddings + delta_prompt_embeddings

    def set_updated_tokens(self):
        """
        Sets up a backward hook to selectively update token gradients based on the CPT token type mask.
        """
        tensor_ICL_mask = torch.Tensor(self.config.cpt_tokens_type_mask).long()
        mask_input_template = torch.remainder(tensor_ICL_mask, 4) == 1
        mask_input = torch.remainder(tensor_ICL_mask, 4) == 2
        mask_output_template = torch.remainder(tensor_ICL_mask, 4) == 3
        mask = mask_input_template | mask_input | mask_output_template
        mask = mask.view(-1, 1)

        def backward_hook(grad):
            grad = grad * mask.to(grad.device)  # Apply mask to gradients
            return grad

        self.delta_embedding.weight.register_hook(backward_hook)

    def get_epsilon(self):
        cpt_tokens_type_mask = self.config.cpt_tokens_type_mask

        MIN_VALUE = 1e-10

        # Calculate normalized epsilon values for input, output, and format tokens
        normalized_format_eps = self.config.opt_projection_format_epsilon * torch.sqrt(
            torch.Tensor([self.config.token_dim / 2048])
        )
        normalized_input_eps = self.config.opt_projection_epsilon * torch.sqrt(
            torch.Tensor([self.config.token_dim / 2048])
        )

        epsilon = torch.ones_like(torch.Tensor(cpt_tokens_type_mask)).to(torch.float32) * MIN_VALUE
        cpt_tokens_type_mask = torch.Tensor(cpt_tokens_type_mask).long()

        epsilon[(cpt_tokens_type_mask > 0) & (torch.remainder(cpt_tokens_type_mask, 4) == 1)] = normalized_format_eps
        epsilon[(cpt_tokens_type_mask > 0) & (torch.remainder(cpt_tokens_type_mask, 4) == 3)] = normalized_format_eps
        epsilon[(cpt_tokens_type_mask > 0) & (torch.remainder(cpt_tokens_type_mask, 4) == 2)] = normalized_input_eps

        return epsilon

    def get_projection(self):
        """
        Applies epsilon-based projection to the delta embeddings to control their norm.
        """

        # Apply projection to control delta embedding norm
        with torch.no_grad():
            new_embeddings_weights = self.delta_embedding.weight.clone().to(self.delta_embedding.weight.device)
            token_norm = torch.norm(new_embeddings_weights, p=2, dim=1)

            projection_mask = token_norm > 0
            if torch.any(projection_mask):
                epsilon = self.get_epsilon().to(self.delta_embedding.weight.device)
                new_embeddings_weights[projection_mask] *= (
                    epsilon[projection_mask] / (token_norm[projection_mask].clamp(min=epsilon[projection_mask]))
                ).view(-1, 1)
            return new_embeddings_weights

    @staticmethod
    def calculate_loss(base_model_output, labels, cpt_type_mask, config):
        """
        Computes the loss for CPT models with optional exponential decay.

        Args:
            base_model_output (ModelOutput):
                Output from the base model containing logits.
            labels (torch.Tensor):
                Ground-truth labels for the input tokens.
            cpt_type_mask (torch.Tensor):
                Token type mask used for filtering valid loss terms.
            config (Namespace):
                Configuration object containing loss-related hyperparameters.

        Returns:
            ModelOutput:
                The base model output with computed loss.
        """

        device = base_model_output.logits.device

        lm_logits = base_model_output.logits
        labels = labels.to(device)

        # Shift logits and labels for token prediction
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_cpt_type_mask = cpt_type_mask[..., 1:].contiguous()

        shift_labels_bool = (shift_labels.clone().detach() != -100).bool()
        batch_size, seq_length, vocab_size = shift_logits.shape

        # Compute cross-entropy loss
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
        )
        loss = loss.view(batch_size, seq_length)
        # Apply exponential decay weights to the loss
        shift_labels_weights = shift_labels_bool.clone().detach().float()

        for i in range(batch_size):
            idx_labels = (shift_cpt_type_mask[i] > 0) & (shift_cpt_type_mask[i] % 4 == 0)
            labels_ids = shift_cpt_type_mask[i][idx_labels].unique()

            exponential_decay = torch.ones_like(shift_cpt_type_mask[i]).to(device=device).float()
            decay_value = 1
            for label_mask_idx in torch.flip(labels_ids, [0]):
                exponential_decay[shift_cpt_type_mask[i] == label_mask_idx] = decay_value
                decay_value *= config.opt_loss_decay_factor
            if config.opt_weighted_loss_type == "decay":
                shift_labels_weights[i] *= exponential_decay

        # Compute the weighted mean loss
        loss = (loss[shift_labels_bool] * shift_labels_weights[shift_labels_bool]).mean()

        base_model_output.loss = loss

        return base_model_output
