import enum
import math

import torch


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


class PromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config["num_virtual_tokens"]
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config["token_dim"])
        if config["prompt_encoder_config"]["prompt_tuning_init"] == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config["prompt_encoder_config"]["tokenizer_name_or_path"])
            self.init_text = config["prompt_encoder_config"]["prompt_tuning_text"]
            init_token_ids = self.tokenizer(self.init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]

            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
