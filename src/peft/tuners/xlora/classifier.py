import builtins
import json
import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy
import torch
import torch.nn as nn
from transformers.modeling_outputs import (  # type: ignore
    ModelOutput,
)

from .config import xLoRAConfig


Number = Union[builtins.int, builtins.float, builtins.bool]


class TemperatureScaledSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits):
        # Scale logits by the temperature
        scaled_logits = logits / self.temperature
        # Apply softmax to the scaled logits
        return self.softmax(scaled_logits)


@dataclass
class InhibitorFlagPayload:
    batch_size: int
    seq_len: int
    override_scaling_pass_value: Number


class xLoRAClassifier(nn.Module):
    """
    A classifier to select LoRA layers for xLoRA.
    """

    def __init__(
        self,
        model: nn.Module,  # PeftModel
        config: xLoRAConfig,
        n_classes: int,
        n_layers: int,
    ):
        super().__init__()

        # To avoid registering this with nn.Module
        self.__dict__["model"] = model
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.config = config
        self.log_scalings: List[torch.Tensor] = []
        self.softmax = TemperatureScaledSoftmax(temperature=self.config.softmax_temperature)
        self.override_scaling_pass_value: Number = config.scaling_pass_value

        self.n_predictions_lifetime = 0
        self.scalings_logging = False

        dtype = next(model.parameters()).dtype
        bias_flag = config.use_bias

        self.inner: nn.ModuleList = nn.ModuleList([])
        if self.config.xlora_depth == 1:
            if config.layerwise_scalings:  # bias=False if we have just one layer
                self.last = (
                    nn.Linear(config.hidden_size, n_classes * n_layers, bias=bias_flag).to(config.device).to(dtype)
                )
            else:
                self.last = nn.Linear(config.hidden_size, n_classes, bias=bias_flag).to(config.device).to(dtype)
        elif self.config.xlora_depth == 2:
            self.inner.append(
                nn.Linear(config.hidden_size, config.xlora_size, bias=bias_flag).to(config.device).to(dtype)
            )

            if config.enable_relu_and_dropout:
                self.inner.append(nn.ReLU())
                self.inner.append(nn.Dropout(p=config.xlora_dropout_p))

            if config.layerwise_scalings:
                self.last = (
                    nn.Linear(config.xlora_size, n_classes * n_layers, bias=bias_flag).to(config.device).to(dtype)
                )
            else:
                self.last = nn.Linear(config.xlora_size, n_classes, bias=bias_flag).to(config.device).to(dtype)
        else:
            assert self.config.xlora_depth > 0
            self.inner.append(
                nn.Linear(config.hidden_size, config.xlora_size, bias=bias_flag).to(config.device).to(dtype)
            )

            if config.enable_relu_and_dropout:
                self.inner.append(nn.ReLU())
                self.inner.append(nn.Dropout(p=config.xlora_dropout_p))

            for _ in range(config.xlora_depth - 2):
                self.inner.append(
                    nn.Linear(config.xlora_size, config.xlora_size, bias=bias_flag).to(config.device).to(dtype)
                )

                if config.enable_relu_and_dropout:
                    self.inner.append(nn.ReLU())
                    self.inner.append(nn.Dropout(p=config.xlora_dropout_p))

            if config.layerwise_scalings:
                self.last = (
                    nn.Linear(config.xlora_size, n_classes * n_layers, bias=bias_flag).to(config.device).to(dtype)
                )
            else:
                self.last = nn.Linear(config.xlora_size, n_classes, bias=bias_flag).to(config.device).to(dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Using the hidden states of the model, predict `n_classes` LoRA alpha values. Sets the scalings.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = typing.cast(torch.FloatTensor, inputs_embeds).shape[0]

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        else:
            seq_len = typing.cast(torch.FloatTensor, inputs_embeds).shape[1]

        # For type checking
        model: nn.Module = self.model  # type: ignore
        with torch.no_grad():
            with model.disable_adapter():
                # TODO(EricLBuehler): Pending removal following analysis
                """
                for module in model.base_model.modules():
                    if isinstance(module.forward.__self__, xLoRALayer):
                        inst = module.forward.__self__
                        inst.disabled = True  # Disable it
                """

                kwargs["output_hidden_states"] = True
                kwargs["return_dict"] = True

                result: ModelOutput = model.forward(
                    *args,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    _xlora_classifier_inhibitor_flag=InhibitorFlagPayload(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        override_scaling_pass_value=self.override_scaling_pass_value,
                    ),
                    **kwargs,
                )

                # TODO(EricLBuehler): Pending removal following analysis
                """
                # Enable the xLoRALayers
                for module in model.base_model.modules():
                    if isinstance(module.forward.__self__, xLoRALayer):
                        inst = module.forward.__self__
                        inst.disabled = False  # Disable it
                """

        hidden_states = result.hidden_states  # type: ignore

        assert hidden_states is not None
        hidden_state = hidden_states[-1]  # Get the last hidden state

        ### Calculate the sequence lengths

        # TODO(all): Pending removal following analysis
        """
        # hidden_state=[batch_size, seq_len, hidden_size]
        if self.config.stop_token_id is None:  # Calculate via attention mask
            if input_ids is not None:
                assert attention_mask is not None, (
                    "Stop token id was not provided, so sequence length calculation via attention mask was attempted"
                    + "but the attention mask was not given"
                )
                sequence_lengths: Union[int, torch.Tensor] = torch.eq(attention_mask, 0).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_state.device)  # type: ignore
            else:
                sequence_lengths = -1
        else:  # Calculate via stop token id
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.stop_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_state.device)  # type: ignore
            else:
                sequence_lengths = -1

        # AFTER THIS: hidden_state=[batch_size, hidden_size]
        if self.config.use_mean_pool:
            assert isinstance(sequence_lengths, torch.Tensor)
            max_length = hidden_state.shape[1]
            mask = torch.arange(max_length).expand(len(sequence_lengths), max_length).to(
                hidden_state.device
            ) < sequence_lengths.unsqueeze(1)

            # Mask the hidden_states
            masked_hidden_state = hidden_state * mask.unsqueeze(-1)

            # Sum across the sequence length and divide by actual sequence length
            summed = torch.sum(masked_hidden_state, dim=1)
            hidden_state = summed / sequence_lengths.unsqueeze(1)
        else:
            # Get it for the last token
            hidden_state = hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]
        """

        ### Classifier run
        # hidden_state=[batch_size, seq_len, hidden_size]
        for layer in self.inner:
            hidden_state = layer.forward(hidden_state)

        logits = self.last.forward(hidden_state)

        ### Repeat to make layerwise scalings if the classifier layer does not
        if not self.config.layerwise_scalings:
            logits = logits.unsqueeze(2)
            logits = logits.expand(-1, -1, self.n_layers, -1)

        ### Classifier run

        scalings = logits.reshape(batch_size, seq_len, self.n_layers, self.n_classes)
        # scalings = [batch_size, seq_len, n_layers, n_classes]

        if self.config.enable_softmax:
            scalings = self.softmax(scalings)

        if self.n_predictions_lifetime > 0:
            print(f"Scaling predictions: {scalings}")
            self.n_predictions_lifetime -= 1

        if self.scalings_logging:
            self.log_scalings.append(scalings)

        return scalings

    def get_nb_trainable_parameters(self):
        # https://github.com/huggingface/peft/blob/main/src/peft/mixed_model.py#L156
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel  # type: ignore

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    @staticmethod
    def _save_scalings(file: str, scalings: List[torch.Tensor]):
        result = torch.cat(scalings, dim=0)
        npy = result.numpy()
        numpy.save(file, npy)

    def flush_log_scalings(self, path: str):
        if not self.scalings_logging:
            raise Exception("Scalings logging is disabled!")

        if len(self.log_scalings) == 0:
            raise ValueError("No log scalings to flush.")

        seqlens_map: Dict[int, Tuple[List[int], List[torch.Tensor]]] = {}
        for i, scaling in enumerate(self.log_scalings):
            seq_len = scaling.shape[0]
            if seq_len not in seqlens_map:
                seqlens_map[seq_len] = ([i], [scaling])
            else:
                seqlens_map[seq_len][0].append(i)
                seqlens_map[seq_len][1].append(scaling)

        if len(seqlens_map) == 1:
            self._save_scalings(path, [scaling.unsqueeze(0) for scaling in self.log_scalings])
        else:
            indices_map: Dict[str, List[int]] = {}
            for seq_len, (indices, scalings_list) in seqlens_map.items():
                indices_map[f"{path}-{seq_len}.npy"] = indices

                self._save_scalings(f"{path}-{seq_len}", [scaling.unsqueeze(0) for scaling in scalings_list])

            with open(f"{path}-mapping.json", "w") as f:
                f.write(json.dumps(indices_map))

        self.log_scalings = []

    def set_override_scaling_pass_value(self, value: Union[Number, None]):
        if value is None:
            self.override_scaling_pass_value = 1 / self.n_classes
        else:
            self.override_scaling_pass_value = value
        self.config.scaling_pass_value = self.override_scaling_pass_value
