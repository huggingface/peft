import torch


# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
class PrefixEncoder(torch.nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config["prompt_encoder_config"]["prefix_projection"]
        if self.prefix_projection and not config.get("inference_mode", False):
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config["num_virtual_tokens"], config["token_dim"])
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config["token_dim"], config["prompt_hidden_size"]),
                torch.nn.Tanh(),
                torch.nn.Linear(
                    config["prompt_hidden_size"],
                    config["num_layers"] * 2 * config["token_dim"],
                ),
            )
        else:
            self.embedding = torch.nn.Embedding(
                config["num_virtual_tokens"],
                config["num_layers"] * 2 * config["token_dim"],
            )

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
