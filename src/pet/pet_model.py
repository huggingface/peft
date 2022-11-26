import enum
import warnings
import inspect
from collections import OrderedDict

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from accelerate.state import AcceleratorState
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .tuners import PromptEncoder
from .tuners import PrefixEncoder
from .tuners import PromptEmbedding


class PromptEncoderType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"


class PETModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prompt_learning_config = model.config.prompt_learning_config

        num_transformer_submodules = 0
        transformer_backbone = None
        for name, module in self.model.named_children():
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
                num_transformer_submodules += 1
        self.prompt_learning_config["num_transformer_submodules"] = num_transformer_submodules

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
        self.prompt_tokens = torch.arange(
            self.prompt_learning_config["num_virtual_tokens"]
            * self.prompt_learning_config["num_transformer_submodules"]
        ).long()

    def get_prompt(self, batch_size):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.prompt_learning_config["num_virtual_tokens"]]
            if self.prompt_learning_config.get("inference_mode", False):
                past_key_values = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = self.prompt_encoder(prompt_tokens)
            past_key_values = past_key_values.view(
                batch_size,
                self.prompt_learning_config["num_virtual_tokens"],
                self.prompt_learning_config["num_layers"]
                * self.prompt_learning_config["num_transformer_submodules"]
                * 2,
                self.prompt_learning_config["num_attention_heads"],
                self.prompt_learning_config["token_dim"] // self.prompt_learning_config["num_attention_heads"],
            )
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                self.prompt_learning_config["num_transformer_submodules"] * 2
            )
            if "postprocess_past_key_value_function" in self.prompt_learning_config["prompt_encoder_config"]:
                post_process_fn = self.prompt_learning_config["prompt_encoder_config"][
                    "postprocess_past_key_value_function"
                ]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if self.prompt_learning_config.get("inference_mode", False):
                prompts = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = self.prompt_encoder(prompt_tokens)
            return prompts


class PETModelForSequenceClassification(PETModel):
    def __init__(self, model):
        super().__init__(model)
        self.config = self.model.config

        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Linear):
                self.cls_layer_name = name
                break

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
        if attention_mask is not None and self.prompt_learning_config["prompt_encoder_type"] != PromptEncoderType.LORA:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.prompt_learning_config["num_virtual_tokens"]).to(
                self.model.device
            )
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PREFIX_TUNING:
            return self.prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, self.prompt_learning_config["num_virtual_tokens"]).to(
                            self.model.device
                        ),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.model(inputs_embeds=inputs_embeds, **kwargs)

    def prefix_tuning_forward(
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
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.model.named_children())]:
                pooled_output = self.model.dropout(pooled_output)
            logits = self.model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PETModelForCausalLM(PETModel):
    def __init__(self, model):
        super().__init__(model)
        self.config = self.model.config

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        if self.prompt_learning_config["prompt_encoder_type"] != PromptEncoderType.LORA:
            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, self.prompt_learning_config["num_virtual_tokens"]).to(
                    self.model.device
                )
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.prompt_learning_config["num_virtual_tokens"]), -100).to(
                    self.device
                )
                labels = torch.cat((prefix_labels, labels), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.model(inputs_embeds=inputs_embeds, **kwargs)


class PETModelForSeq2SeqLM(PETModel):
    def __init__(self, model):
        super().__init__(model)
        self.config = self.model.config

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
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]

        if self.prompt_learning_config["prompt_encoder_type"] != PromptEncoderType.LORA:
            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, self.prompt_learning_config["num_virtual_tokens"]).to(
                    self.model.device
                )
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                if decoder_attention_mask is not None:
                    decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.prompt_learning_config["num_virtual_tokens"]), -100).to(
                    self.device
                )
                labels = torch.cat((prefix_labels, labels), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None:
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat(
                (prompts[:, : self.prompt_learning_config["num_virtual_tokens"]], inputs_embeds), dim=1
            )
            decoder_inputs_embeds = torch.cat(
                (prompts[:, self.prompt_learning_config["num_virtual_tokens"] :], decoder_inputs_embeds), dim=1
            )
            return self.model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs)
