import gc
import math

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft.helpers import find_kappa_target_modules


# ==========================================
# 1. Data Preparation
# ==========================================
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token


def format_gsm8k(example):
    return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}


print("Loading and preprocessing datasets...")
gsm8k_ds = load_dataset("gsm8k", "main", split="train[:1000]").train_test_split(test_size=0.1)
gsm8k_tokenized = gsm8k_ds.map(format_gsm8k).map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=256),
    batched=True,
    remove_columns=["question", "answer", "text"],
)

wiki_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:400]")
wiki_tokenized = wiki_ds.filter(lambda x: len(x["text"]) > 20).map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=256),
    batched=True,
    remove_columns=wiki_ds.column_names,
)


# ==========================================
# 2. Experiment Engine
# ==========================================
def evaluate_perplexity(model, dataset, name="Dataset"):
    model.eval()
    total_loss = 0
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=data_collator)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch, use_cache=False)
            total_loss += outputs.loss.item()
            if i >= 40:
                break
    return math.exp(total_loss / (i + 1))


def run_experiment(method_name):
    print(f"\n{'=' * 40}\n>>> EXPERIMENT: {method_name}\n{'=' * 40}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, trust_remote_code=True, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # Configure PEFT based on method
    if method_name == "LoRA_Global":
        Target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(r=256, target_modules=Target_modules, task_type=TaskType.CAUSAL_LM, lora_dropout=0.05)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        LR = 2e-4
        STP = 40

    elif method_name == "KappaTune_LoRA":
        print(" [KappaTune] Selecting target modules using PEFT KappaTuneSelector...")

        # Relative selection‚ works on any architecture
        stable_modules_dic = find_kappa_target_modules(model, top_p=0.2)

        lora_config = LoraConfig(
            r=85,
            target_modules=stable_modules_dic["target_modules"],
            target_parameters=stable_modules_dic["target_parameters"]
            if stable_modules_dic["target_parameters"]
            else None,
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.05,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainable = [(n, p.shape, p.numel()) for n, p in model.named_parameters() if p.requires_grad]

        print(f"#trainable tensors: {len(trainable)}")
        print(f"#trainable params: {sum(x[2] for x in trainable):,}")

        LR = 2e-4
        STP = 40  # or whatever step count you prefer for fair comparison

    if method_name != "Baseline":
        args = TrainingArguments(
            output_dir=f"./{method_name}_out",
            per_device_train_batch_size=40,
            gradient_accumulation_steps=4,
            learning_rate=LR,
            num_train_epochs=STP,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=gsm8k_tokenized["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()

    t_ppl_test = evaluate_perplexity(model, gsm8k_tokenized["test"], "gsm8k")
    t_ppl_train = evaluate_perplexity(model, gsm8k_tokenized["train"], "gsm8k")
    f_ppl = evaluate_perplexity(model, wiki_tokenized, "WikiText")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return t_ppl_test, t_ppl_train, f_ppl


# ==========================================
# 3. Results (same table as paper)
# ==========================================
results = {}

results["KappaTune"] = run_experiment("KappaTune_LoRA")
results["Baseline"] = run_experiment("Baseline")
results["LoRA_Global"] = run_experiment("LoRA_Global")

print("\n" + "=" * 70)
print(
    f"{'METHOD':<15} | {'gsm8k PPL (Task train)':<18} |  {'gsm8k PPL (Task test)':<18} | {'Wiki PPL (General/control)':<18}"
)
print("-" * 70)
for m, (tpte, tptr, fp) in results.items():
    print(f"{m:<15} | {tptr:<18.4f} | {tpte:<18.4f} | {fp:<18.4f}")
print("=" * 70)
