import enum
import torch
import math
import os

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
    ModuleWrapPolicy,
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
    _or_policy,
)
from collections import OrderedDict


class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"


class PromptEncoderType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING_V1 = "P_TUNING_V1"
    P_TUNING_V2 = "P_TUNING_V2"


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual
    token embeddings for p-tuning.
    """

    def __init__(self, config):
        super().__init__()
        self.token_dim = config["token_dim"]
        self.input_size = config["token_dim"]
        self.output_size = config["token_dim"]
        self.hidden_size = config["prompt_hidden_size"]
        self.total_virtual_tokens = config["num_virtual_tokens"]
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
                layers = [torch.nn.Linear(self.input_size, self.hidden_size), torch.nn.ReLU()]
                layers.extend([torch.nn.Linear(self.hidden_size, self.hidden_size), torch.nn.ReLU()])
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
                torch.nn.Linear(config["prompt_hidden_size"], config["num_layers"] * 2 * config["token_dim"]),
            )
        else:
            self.embedding = torch.nn.Embedding(
                config["num_virtual_tokens"], config["num_layers"] * 2 * config["token_dim"]
            )

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


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
        # Just get embeddings and dropout
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings


class PromptModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.prompt_learning_config = model.config.prompt_learning_config

        modules = list(model._modules)

        for module in modules:
            if isinstance(model.get_submodule(module), PreTrainedModel):
                self.transformer_backbone = model.get_submodule(module)
                break

        for named_param, value in list(self.transformer_backbone.named_parameters()):
            if value.shape[0] == model.config.vocab_size:
                self.word_embeddings = self.transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        # Make sure to freeze Tranformers model
        for param in self.transformer_backbone.parameters():
            param.requires_grad = False

        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(self.prompt_learning_config, self.word_embeddings)
        elif self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.P_TUNING_V1:
            prompt_encoder = PromptEncoder(self.prompt_learning_config)
        elif self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.P_TUNING_V2:
            prompt_encoder = PrefixEncoder(self.prompt_learning_config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder = prompt_encoder
        self.prompt_tokens = torch.arange(self.prompt_learning_config["num_virtual_tokens"]).long()

    def get_prompt(self, batch_size):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.transformer_backbone.device)
        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.P_TUNING_V2:
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
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(1, -1).to(self.transformer_backbone.device)
        prompt_embeddings = self.prompt_encoder(prompt_tokens).detach().cpu()
        if destination is None:
            state_dict_ = OrderedDict()
        else:
            state_dict_ = destination
        state_dict_["prompt_embeddings"] = prompt_embeddings[0]
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method.
        """
        self.prompt_encoder.embedding.load_state_dict({"weight": state_dict["prompt_embeddings"]}, strict)


class PromptModelForSequenceClassification(PromptModel):
    def __init__(self, model):
        super().__init__(model)
        if "dropout" in [name for name, _ in model.named_children()]:
            self.dropout = model.dropout
        else:
            self.dropout = torch.nn.Dropout(model.config.hidden_dropout_prob)
        self.classifier = model.classifier
        self.num_labels = model.num_labels
        self.config = model.config
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
            self.transformer_backbone.device
        )
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if self.prompt_learning_config["prompt_encoder_type"] == PromptEncoderType.P_TUNING_V2:
            past_key_values = self.get_prompt(batch_size=batch_size)

            outputs = self.transformer_backbone(
                input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                **kwargs,
            )

            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
        else:
            raw_embedding = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)

            outputs = self.transformer_backbone(
                # input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # **kwargs,
                # past_key_values=past_key_values,
            )

            sequence_output = outputs[0]
            sequence_output = sequence_output[:, self.prompt_learning_config["num_virtual_tokens"] :, :].contiguous()
            pooled_output = sequence_output[:, 0]

            if (
                "pooler" in [name for name, _ in self.transformer_backbone.named_children()]
                and self.transformer_backbone.pooler is not None
            ):
                pooled_output = self.transformer_backbone.pooler.dense(pooled_output)
                pooled_output = self.transformer_backbone.pooler.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        No frozen model parameters are stored in the state dict.
        """
        if destination is None:
            state_dict_ = OrderedDict()
        else:
            state_dict_ = destination
        state_dict_["prompt_encoder"] = super().state_dict()
        state_dict_["classifier"] = self.classifier.state_dict()
        if AcceleratorState().fsdp_plugin is not None:
            state_dict_["_flat_param"] = None
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method.
        """
        super().load_state_dict(state_dict["prompt_encoder"], strict)
        self.classifier.load_state_dict(state_dict["classifier"], strict)

    def clean_state_dict(self, state_dict):
        if AcceleratorState().fsdp_plugin is not None:
            new_state_dict = OrderedDict()
            for key in self.modules_to_save:
                new_state_dict[key] = state_dict[key].copy()
            state_dict = new_state_dict
        return state_dict


model_type_to_prompt_model_mapping = {"SequenceClassification": PromptModelForSequenceClassification}
num_virtual_tokens = 30
model_name_or_path = "roberta-large"
tokenizer_name_or_path = "roberta-large"

prompt_tuning_config = {
    "num_virtual_tokens": num_virtual_tokens,
    "prompt_encoder_type": "PROMPT_TUNING",
    "prompt_encoder_config": {
        "prompt_tuning_init": "TEXT",
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "prompt_tuning_text": "Output is true or false. Task requires to recognize"
        " whether the meaning of one text is entailed (can be inferred) from the other text.",
    },
}


p_tuning_v1_mlp_config = {
    "num_virtual_tokens": num_virtual_tokens,
    "prompt_encoder_type": "P_TUNING_V1",
    "prompt_encoder_config": {"prompt_reparam_type": "MLP"},
}

p_tuning_v1_lstm_config = {
    "num_virtual_tokens": num_virtual_tokens,
    "prompt_encoder_type": "P_TUNING_V1",
    "prompt_encoder_config": {"prompt_reparam_type": "LSTM"},
}

p_tuning_v2_no_proj_config = {
    "num_virtual_tokens": num_virtual_tokens,
    "prompt_encoder_type": "P_TUNING_V2",
    "prompt_encoder_config": {"prefix_projection": False},
}

p_tuning_v2_proj_config = {
    "num_virtual_tokens": num_virtual_tokens,
    "prompt_encoder_type": "P_TUNING_V2",
    "prompt_encoder_config": {"prefix_projection": True},
}


def prepare_prompt_model(model, prompt_learning_config):
    config = model.config.to_dict()
    if "num_layers" not in prompt_learning_config:
        if "num_hidden_layers" in config:
            num_layers = config["num_hidden_layers"]
        elif "num_layers" in config:
            num_layers = config["num_layers"]
        else:
            raise ValueError("Please specify `num_layers` in `prompt_learning_config`")
        prompt_learning_config["num_layers"] = num_layers

    if "token_dim" not in prompt_learning_config:
        if "hidden_size" in config:
            token_dim = config["hidden_size"]
        elif "n_embd" in config:
            token_dim = config["n_embd"]
        elif "d_model" in config:
            token_dim = config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `prompt_learning_config`")
        prompt_learning_config["token_dim"] = token_dim

    if "num_attention_heads" not in prompt_learning_config:
        if "num_attention_heads" in config:
            num_attention_heads = config["num_attention_heads"]
        elif "n_head" in config:
            num_attention_heads = config["n_head"]
        elif "num_heads" in config:
            num_attention_heads = config["num_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `prompt_learning_config`")
        prompt_learning_config["num_attention_heads"] = num_attention_heads

    if "prompt_hidden_size" not in prompt_learning_config:
        prompt_learning_config["prompt_hidden_size"] = token_dim

    model.config.prompt_learning_config = prompt_learning_config
    model_type = model.__class__.__name__.split("For")
    if len(model_type) < 2:
        raise ValueError("Model Type not supported")
    model_cls = model_type_to_prompt_model_mapping[model_type[1]]
    prompt_model = model_cls(model)
    return prompt_model


def fsdp_auto_wrap_policy(model):
    def wrap_layers_with_required_grads(module):
        if (
            len(list(module.children())) == 0
            and len(list(module.named_parameters())) > 0
            and module.weight.requires_grad
        ):
            return True
        return False

    transformer_cls_to_wrap = {
        PrefixEncoder,
        PromptEmbedding,
        PromptEncoder,
        PromptModel,
        FullyShardedDataParallelPlugin.get_module_class_from_name(
            model, os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP", "")
        ),
    }
    policy_1 = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,
    )
    policy_2 = functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=wrap_layers_with_required_grads,
    )
    auto_wrap_policy = functools.partial(_or_policy, policies=[policy_1, policy_2])
    return auto_wrap_policy


def main():
    accelerator = Accelerator()
    task = "rte"
    batch_size = 16
    lr = 5e-3
    num_epochs = 100
    device = "cuda"
    seed = 11
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model = prepare_prompt_model(
        model, p_tuning_v2_no_proj_config
    )  # p_tuning_v2_proj_config)#p_tuning_v2_no_proj_config)
    # model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    datasets = load_dataset("glue", task)
    metric = evaluate.load("glue", task)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )

    # Instantiate optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        accelerator.print(f"epoch {epoch}:", eval_metric)
        accelerator.print(f"epoch {epoch} train loss:", total_loss / len(train_dataloader))

    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        BackwardPrefetch,
        CPUOffload,
        FullStateDictConfig,
        ShardingStrategy,
        StateDictType,
    )

    FSDP.set_state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    )
    state_dict = model.state_dict()
    state_dict = model.clean_state_dict(state_dict)
    accelerator.print(state_dict)

    torch.save(state_dict, "p_tuning_v2.pt")


if __name__ == "__main__":
    main()
