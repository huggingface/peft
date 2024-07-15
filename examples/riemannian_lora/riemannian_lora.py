import torch
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from peft import LoraConfig, TaskType, create_riemannian_optimizer, get_peft_model


model_checkpoint = "microsoft/deberta-v3-base"
dataset = load_dataset("glue", "cola")
metric = load_metric("glue", "cola")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
task_to_keys = {"cola": ("sentence", None)}
sentence1_key, sentence2_key = task_to_keys["cola"]


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=4,
    lora_alpha=8,
    lora_dropout=0.01,
    target_modules=["query_proj", "key_proj", "value_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
metric_name = "matthews_correlation"

args = TrainingArguments(
    "glue_tune", save_strategy="epoch", per_device_train_batch_size=8, num_train_epochs=3, logging_steps=10, seed=0
)

optim_config = {"lr": 1e-5, "eps": 1e-6, "betas": (0.9, 0.999), "weight_decay": 0.01}

optimizer = create_riemannian_optimizer(
    model=model, optimizer_cls=torch.optim.AdamW, optimizer_kwargs=optim_config, reg=1e-2
)
trainer = Trainer(
    model, args, train_dataset=encoded_dataset["train"], tokenizer=tokenizer, optimizers=[optimizer, None]
)
trainer.train()
