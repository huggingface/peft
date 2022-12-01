import inspect
import warnings

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .tuners import LoRAModel, PrefixEncoder, PromptEmbedding, PromptEncoder
from .utils import PETConfig, PETType, TaskType, shift_tokens_right, _set_trainable


class PETModel(torch.nn.Module):
    """
    Parameter Efficient Tuning Model. Base model encompassing various PET methods.

    Args:
        model (:obj:`PreTrainedModel`): The base transformer model used for PET.
        pet_config (:obj:`PETConfig`): The configuration of the PET model.


    Attributes:
        base_model (:obj:`PreTrainedModel`): The base transformer model used for PET.
        pet_config (:obj:`PETConfig`): The configuration of the PET model.
        modules_to_save (:obj:`list` of :obj:`str`): The list of sub-module names to save when saving the model.
        prompt_encoder (:obj:`PromptEncoder`): The prompt encoder used for PET if `pet_config.pet_type != PETType.LORA`.
        prompt_tokens (:obj:`torch.Tensor`): The virtual prompt tokens used for PET if `pet_config.pet_type != PETType.LORA`.
        transformer_backbone_name (:obj:`str`): The name of the transformer backbone in the base model
            if `pet_config.pet_type != PETType.LORA`.
        word_embeddings (:obj:`torch.nn.Embedding`): The word embeddings of the transformer backbone
            in the base model if `pet_config.pet_type != PETType.LORA`.
    """

    def __init__(self, model, pet_config: PETConfig):
        super().__init__()
        self.pet_config = pet_config
        self.base_model = model
        self.modules_to_save = None
        if pet_config.pet_type != PETType.LORA:
            self._setup_prompt_encoder()
        else:
            self.base_model = LoRAModel(pet_config, model)

    def _setup_prompt_encoder(self):
        num_transformer_submodules = 0
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                for param in module.parameters():
                    param.requires_grad = False
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
                num_transformer_submodules += 1
        self.pet_config.num_transformer_submodules = 2 if self.pet_config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if self.pet_config.pet_type == PETType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(self.pet_config, self.word_embeddings)
        elif self.pet_config.pet_type == PETType.P_TUNING:
            prompt_encoder = PromptEncoder(self.pet_config)
        elif self.pet_config.pet_type == PETType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(self.pet_config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder = prompt_encoder
        self.prompt_tokens = torch.arange(
            self.pet_config.num_virtual_tokens * self.pet_config.num_transformer_submodules
        ).long()

    def get_prompt_embedding_to_save(self):
        """
        Returns the prompt embedding to save when saving the model.
        Only applocable when `pet_config.pet_type != PETType.LORA`.
        """
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(1, -1).to(self.base_model.device)
        if self.pet_config.pet_type == PETType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.pet_config.num_virtual_tokens]
        prompt_embeddings = self.prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size):
        """
        Returns the virtual prompts to use for PET.
        Only applocable when `pet_config.pet_type != PETType.LORA`.
        """
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.base_model.device)
        if self.pet_config.pet_type == PETType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.pet_config.num_virtual_tokens]
            if self.pet_config.inference_mode:
                past_key_values = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = self.prompt_encoder(prompt_tokens)
            past_key_values = past_key_values.view(
                batch_size,
                self.pet_config.num_virtual_tokens,
                self.pet_config.num_layers * 2,
                self.pet_config.num_attention_heads,
                self.pet_config.token_dim // self.pet_config.num_attention_heads,
            )
            if self.pet_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                self.pet_config.num_transformer_submodules * 2
            )
            if self.pet_config.postprocess_past_key_value_function is not None:
                post_process_fn = self.pet_config.postprocess_past_key_value_function
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if self.pet_config.inference_mode:
                prompts = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = self.prompt_encoder(prompt_tokens)
            return prompts

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


class PETModelForSequenceClassification(PETModel):
    """
    PET model for sequence classification tasks.

    Args:
        model (:obj:`PreTrainedModel`): Base transformer model
        pet_config (:obj:`PETConfig`): PET config.

    Attributes:
        config (:obj:`PretrainedConfig`): The configuration object of the base model.
        cls_layer_name (:obj:`str`): The name of the classification layer.

    Example::

        >>> from transformers import AutoModelForSequenceClassification
        >>> from pet import PETModelForSequenceClassification, get_pet_config
        >>> config = {
                'pet_type': 'PREFIX_TUNING',
                'task_type': 'SEQ_CLS',
                'inference_mode': False,
                'num_virtual_tokens': 20,
                'token_dim': 768,
                'num_transformer_submodules': 1,
                'num_attention_heads': 12,
                'num_layers': 12,
                'encoder_hidden_size': 768,
                'prefix_projection': False,
                'postprocess_past_key_value_function': None
            }
        >>> pet_config = get_pet_config(config)
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        >>> pet_model = PETModelForSequenceClassification(model, pet_config)
        >>> pet_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
    """

    def __init__(self, model, pet_config: PETConfig):
        super().__init__(model, pet_config)
        self.config = self.base_model.config
        self.modules_to_save = ["classifier"]

        for name, module in self.base_model.named_children():
            if isinstance(module, torch.nn.Linear):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self.base_model)

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

        if self.pet_config.pet_type == PETType.LORA:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.pet_config.num_virtual_tokens).to(
                self.base_model.device
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

        if self.pet_config.pet_type == PETType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, self.pet_config.num_virtual_tokens).to(self.base_model.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
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
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
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
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
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
    """
    PET model for Causal LM

    Args:
        model (:obj:`PreTrainedModel`): Base transformer model
        pet_config (:obj:`PETConfig`): PET config.


    Example::

        >>> from transformers import AutoModelForCausalLM
        >>> from pet import PETModelForCausalLM, get_pet_config
        >>> config = {
                'pet_type': 'PREFIX_TUNING',
                'task_type': 'CAUSAL_LM',
                'inference_mode': False,
                'num_virtual_tokens': 20,
                'token_dim': 1280,
                'num_transformer_submodules': 1,
                'num_attention_heads': 20,
                'num_layers': 36,
                'encoder_hidden_size': 1280,
                'prefix_projection': False,
                'postprocess_past_key_value_function': None
            }
        >>> pet_config = get_pet_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> pet_model = PETModelForCausalLM(model, pet_config)
        >>> pet_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
    """

    def __init__(self, model, pet_config: PETConfig):
        super().__init__(model, pet_config)

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
        if self.pet_config.pet_type == PETType.LORA:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.pet_config.num_virtual_tokens).to(
                self.base_model.device
            )
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

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

        if self.pet_config.pet_type == PETType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.pet_config.num_virtual_tokens), -100).to(
                    self.base_model.device
                )
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


class PETModelForSeq2SeqLM(PETModel):
    """
    PET model for Seq2Seq LM

    Args:
        model (:obj:`PreTrainedModel`): Base transformer model
        pet_config (:obj:`PETConfig`): PET config.


    Example::

        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from pet import PETModelForSeq2SeqLM, get_pet_config
        >>> config = {
                'pet_type': 'LORA',
                'task_type': 'SEQ_2_SEQ_LM',
                'inference_mode': False,
                'r': 8,
                'target_modules': ['q', 'v'],
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'merge_weights': False,
                'fan_in_fan_out': False,
                'enable_lora': None,
                'bias': 'none'
            }
        >>> pet_config = get_pet_config(config)
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> pet_model = PETModelForSeq2SeqLM(model, pet_config)
        >>> pet_model.print_trainable_parameters()
        trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
    """

    def __init__(self, model, pet_config: PETConfig):
        super().__init__(model, pet_config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if self.pet_config.pet_type == PETType.LORA:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.pet_config.num_virtual_tokens).to(
                self.base_model.device
            )
            decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

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

        if self.pet_config.pet_type == PETType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, self.pet_config.num_virtual_tokens).to(
                    self.base_model.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.pet_config.num_virtual_tokens), -100).to(
                    self.base_model.device
                )
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat((prompts[:, : self.pet_config.num_virtual_tokens], inputs_embeds), dim=1)
            decoder_inputs_embeds = torch.cat(
                (prompts[:, self.pet_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
            )
            return self.base_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs)
