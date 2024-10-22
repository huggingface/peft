import copy

import torch
from torch.nn import CrossEntropyLoss

from peft.utils.integrations import gather_params_ctx

from .config import PromptTuningInit


class CPTEmbedding(torch.nn.Module):
    """
    CPTEmbedding is a custom embedding layer designed for Context-aware Prompt Tuning (CPT) in PEFT.
    It initializes embeddings, applies prompt-specific projections, and computes loss using label masks.
    """

    def __init__(self, config, word_embeddings):
        """
        Initializes the CPTEmbedding module.

        Args:
            config (Namespace): Configuration object containing model hyperparameters and CPT-specific settings.
            word_embeddings (torch.nn.Embedding): The base word embedding layer used to initialize CPT embeddings.
        """
        super().__init__()
        self.config = copy.deepcopy(config)
        self.check_config()
        num_virtual_tokens = config.num_virtual_tokens

        # Initialize embeddings with virtual token dimensions
        self.embedding = torch.nn.Embedding(num_virtual_tokens, config.token_dim)

        # Initialize embeddings using text-based prompt tuning, if configured
        if config.CPT_prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
            assert config.num_virtual_tokens == len(config.CPT_token_ids)

            init_token_ids = torch.LongTensor(config.CPT_token_ids).to(word_embeddings.weight.device)
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
            indices (torch.Tensor): Indices of the tokens to be embedded.

        Returns:
            torch.Tensor: Sum of prompt embeddings and delta embeddings.
        """
        with torch.no_grad():
            prompt_embeddings = self.embedding(indices)

        self.projection()  # Apply epsilon-based projection
        delta_prompt_embeddings = self.delta_embedding(indices)

        return prompt_embeddings + delta_prompt_embeddings

    def set_updated_tokens(self):
        """
        Sets up a backward hook to selectively update token gradients based on the CPT token type mask.
        """
        if self.config.CPT_prompt_tuning_init == PromptTuningInit.TEXT:
            tensor_ICL_mask = torch.Tensor(self.config.CPT_tokens_type_mask).long()
            mask_input_template = torch.remainder(tensor_ICL_mask, 4) == 1
            mask_input = torch.remainder(tensor_ICL_mask, 4) == 2
            mask_output_template = torch.remainder(tensor_ICL_mask, 4) == 3
            mask = mask_input_template | mask_input | mask_output_template
            mask = mask.view(-1, 1)
        elif self.config.CPT_prompt_tuning_init == PromptTuningInit.RANDOM:
            mask = torch.ones((self.config.num_virtual_tokens, 1)).long()

        def backward_hook(grad):
            grad = grad * mask.to(grad.device)  # Apply mask to gradients
            return grad

        self.delta_embedding.weight.register_hook(backward_hook)

    def get_epsilon(self):
        if self.config.CPT_prompt_tuning_init == "TEXT":
            CPT_tokens_type_mask = self.config.CPT_tokens_type_mask
        else:
            CPT_tokens_type_mask = [2] * self.config.num_virtual_tokens

        MIN_VALUE = 1e-10

        # Calculate normalized epsilon values for input, output, and format tokens
        normalized_format_eps = (
            self.config.opt_projection_format_epsilon
            * torch.sqrt(
                torch.Tensor([self.config.token_dim / 2048])
            )
        )
        normalized_input_eps = self.config.opt_projection_epsilon * torch.sqrt(
            torch.Tensor([self.config.token_dim / 2048])
        )

        epsilon = torch.ones_like(torch.Tensor(CPT_tokens_type_mask)).to(torch.float32) * MIN_VALUE
        CPT_tokens_type_mask = torch.Tensor(CPT_tokens_type_mask).long()

        epsilon[(CPT_tokens_type_mask > 0) & (torch.remainder(CPT_tokens_type_mask, 4) == 1)] = normalized_format_eps
        epsilon[(CPT_tokens_type_mask > 0) & (torch.remainder(CPT_tokens_type_mask, 4) == 3)] = normalized_format_eps
        epsilon[(CPT_tokens_type_mask > 0) & (torch.remainder(CPT_tokens_type_mask, 4) == 2)] = normalized_input_eps

        return epsilon

    def projection(self):
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
                self.delta_embedding.weight.data = new_embeddings_weights

    @staticmethod
    def calculate_loss(base_model_output, labels, CPT_type_mask, config):
        """
        Computes the loss for CPT models with optional exponential decay.

        Args:
            base_model_output (ModelOutput): Output from the base model containing logits.
            labels (torch.Tensor): Ground-truth labels for the input tokens.
            CPT_type_mask (torch.Tensor): Token type mask used for filtering valid loss terms.
            config (Namespace): Configuration object containing loss-related hyperparameters.

        Returns:
            ModelOutput: The base model output with computed loss.
        """

        if config.opt_weighted_loss_type in ["decay"]:
            device = base_model_output.logits.device

            lm_logits = base_model_output.logits
            labels = labels.to(device)

            # Shift logits and labels for token prediction
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_CPT_type_mask = CPT_type_mask[..., 1:].contiguous()

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
                idx_labels = (shift_CPT_type_mask[i] > 0) & (shift_CPT_type_mask[i] % 4 == 0)
                labels_ids = shift_CPT_type_mask[i][idx_labels].unique()

                exponential_decay = torch.ones_like(shift_CPT_type_mask[i]).to(device=device).float()
                decay_value = 1
                for label_mask_idx in torch.flip(labels_ids, [0]):
                    exponential_decay[shift_CPT_type_mask[i] == label_mask_idx] = decay_value
                    decay_value *= config.opt_loss_decay_factor
                shift_labels_weights[i] *= exponential_decay

            # Compute the weighted mean loss
            loss = (loss[shift_labels_bool] * shift_labels_weights[shift_labels_bool]).mean()
            base_model_output.loss = loss
        elif config.opt_weighted_loss_type not in ["none"]:
            raise NotImplementedError(f"Loss type '{config.opt_weighted_loss_type}' not implemented.")

        return base_model_output

    def check_config(self):
        if self.config.CPT_prompt_tuning_init == PromptTuningInit.TEXT:
            assert self.config.CPT_token_ids is not None
            assert self.config.CPT_mask is not None
            assert self.config.CPT_tokens_type_mask is not None
            assert (
                len(self.config.CPT_token_ids)
                == len(self.config.CPT_mask)
                == len(self.config.CPT_tokens_type_mask)
                == self.config.num_virtual_tokens
            )
        elif self.config.CPT_prompt_tuning_init == PromptTuningInit.RANDOM:
            assert self.config.CPT_token_ids is None
            assert self.config.CPT_mask is None
            assert self.config.CPT_tokens_type_mask is None
            assert self.config.num_virtual_tokens > 0
        else:
            raise NotImplementedError(f" was not implemented for {self.config.CPT_prompt_tuning_init}")
