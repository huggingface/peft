import enum
import warnings
from collections import OrderedDict

import torch
from accelerate.state import AcceleratorState
from transformers import PreTrainedModel

from tuners.p_tuning import PromptEncoder
from tuners.prefix_tuning import PrefixEncoder
from tuners.prompt_tuning import PromptEmbedding


class PromptEncoderType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"


class ParameterEfficientTuningModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prompt_learning_config = model.config.prompt_learning_config

        modules = list(self.model._modules)

        for module in modules:
            if isinstance(self.model.get_submodule(module), PreTrainedModel):
                transformer_backbone = self.model.get_submodule(module)
                break

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        # Make sure to freeze Tranformers model
        for param in transformer_backbone.parameters():
            param.requires_grad = False

        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(self.prompt_learning_config, self.word_embeddings)
        elif self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.P_TUNING:
            prompt_encoder = PromptEncoder(self.prompt_learning_config)
        elif self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(self.prompt_learning_config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder = prompt_encoder
        self.prompt_tokens = torch.arange(self.prompt_learning_config["num_virtual_tokens"]).long()

    def get_prompt(self, batch_size):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.transformer_backbone.device)
        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PREFIX_TUNING:
            if self.prompt_learning_config.get("inference_mode", False):
                past_key_values = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = self.prompt_encoder(prompt_tokens)
            past_key_values = past_key_values.view(
                batch_size,
                self.prompt_learning_config["num_virtual_tokens"],
                self.prompt_learning_config["num_layers"] * 2,
                self.prompt_learning_config["num_attention_heads"],
                self.prompt_learning_config["token_dim"] // self.prompt_learning_config["num_attention_heads"],
            )
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            return past_key_values
        else:
            if self.prompt_learning_config.get("inference_mode", False):
                prompts = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = self.prompt_encoder(prompt_tokens)
            return prompts

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        No frozen model parameters are stored in the state dict.
        """
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(1, -1).to(self.model.device)
        prompt_embeddings = self.prompt_encoder(prompt_tokens).detach().cpu()
        if destination is None:
            state_dict_ = OrderedDict()
        else:
            state_dict_ = destination
        state_dict_["prompt_embeddings"] = prompt_embeddings[0]
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder parameters. Matching load method
        for this class' custom state dict method.
        """
        self.prompt_encoder.embedding.load_state_dict({"weight": state_dict["prompt_embeddings"]}, strict)


class ParameterEfficientTuningModelForSequenceClassification(ParameterEfficientTuningModel):
    def __init__(self, model):
        super().__init__(model)
        self.config = self.model.config
        self.modules_to_save = ("prompt_encoder", "classifier")

        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        # concat prompt attention mask
        prefix_attention_mask = torch.ones(batch_size, self.prompt_learning_config["num_virtual_tokens"]).to(
            self.model.device
        )
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs["token_type_ids"] is not None:
            kwargs["token_type_ids"] = torch.cat(
                (
                    torch.ones(batch_size, self.prompt_learning_config["num_virtual_tokens"]).to(self.model.device),
                    kwargs["token_type_ids"],
                ),
                dim=1,
            )

        if kwargs["position_ids"] is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None

        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size=batch_size)

            return self.model(
                input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            raw_embedding = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)

            return self.model(
                # input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
                # past_key_values=past_key_values,
            )

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        No frozen model parameters are stored in the state dict.
        """
        if destination is None:
            state_dict_ = OrderedDict()
        else:
            state_dict_ = destination
        state_dict_["prompt_encoder"] = super().state_dict()
        state_dict_["classifier"] = self.model.classifier.state_dict()
        if AcceleratorState().fsdp_plugin is not None:
            state_dict_["_flat_param"] = None
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder parameters. Matching load method
        for this class' custom state dict method.
        """
        super().load_state_dict(state_dict["prompt_encoder"], strict)
        self.model.classifier.load_state_dict(state_dict["classifier"], strict)

    def clean_state_dict(self, state_dict):
        if AcceleratorState().fsdp_plugin is not None:
            new_state_dict = OrderedDict()
            for key in self.modules_to_save:
                new_state_dict[key] = state_dict[key].copy()
            state_dict = new_state_dict
        return state_dict
