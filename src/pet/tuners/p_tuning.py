import enum

import torch


class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"


# Based on https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/modules/common/prompt_encoder.py
# with some refactor
class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.
    """

    def __init__(self, config):
        super().__init__()
        self.token_dim = config["token_dim"]
        self.input_size = config["token_dim"]
        self.output_size = config["token_dim"]
        self.hidden_size = config["prompt_hidden_size"]
        self.total_virtual_tokens = config["num_virtual_tokens"] * config["num_transformer_submodules"]
        self.encoder_type = config["prompt_encoder_config"]["prompt_reparam_type"]

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not config.get("inference_mode", False):
            if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
                if "dropout" not in config["prompt_encoder_config"]:
                    lstm_dropout = 0.0
                else:
                    lstm_dropout = config["prompt_encoder_config"]["dropout"]

                if "num_layers" not in config["prompt_encoder_config"]:
                    num_layers = 2
                else:
                    num_layers = config["prompt_encoder_config"]["num_layers"]
                # LSTM
                self.lstm_head = torch.nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    bidirectional=True,
                    batch_first=True,
                )

                self.mlp_head = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size * 2, self.output_size),
                )

            elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
                layers = [
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                ]
                layers.extend(
                    [
                        torch.nn.Linear(self.hidden_size, self.hidden_size),
                        torch.nn.ReLU(),
                    ]
                )
                layers.append(torch.nn.Linear(self.hidden_size, self.output_size))
                self.mlp_head = torch.nn.Sequential(*layers)

            else:
                raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds
