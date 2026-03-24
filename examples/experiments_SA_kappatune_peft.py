import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from peft.utils.target_selection import KappaTuneSelector   # ← NEW: PEFT selector
import bitsandbytes as bnb
import gc
import math

# ==========================================
# 1. Kappa-Selection using OFFICIAL PEFT KappaTuneSelector
#    (exact paper logic: lowest κ on experts only)
# ==========================================
def get_stable_expert_names(model, budget_k=300):
    print(f" [KappaTune] Identifying {budget_k} most stable expert modules using PEFT KappaTuneSelector...")
    
    selector = KappaTuneSelector(model)                     # ← uses the new PEFT class
    
    # Get top candidates (more than needed)
    all_low_kappa = selector.get_best_targets(num_modules=budget_k * 2)
    
    # Keep ONLY expert modules (exact same filtering as the paper)
    expert_modules = [name for name in all_low_kappa if "experts" in name]
    
    selected = expert_modules[:budget_k]
    print(f" → Selected {len(selected)} expert modules with lowest κ")
    return selected

# ==========================================
# 2. Data Preparation
# ==========================================
MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_imdb(example):
    val = example.get('label', example.get('sentiment', 0))
    label_text = "Positive" if val == 1 else "Negative"
    return {"text": f"Review: {example['text'][:512]}\n\nSentiment: {label_text}"}

print("Loading and preprocessing datasets...")
imdb_ds = load_dataset("imdb", split="train[:1000]").train_test_split(test_size=0.1)
imdb_tokenized = imdb_ds.map(format_imdb).map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=256), 
    batched=True, remove_columns=imdb_ds["train"].column_names
)

wiki_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:400]")
wiki_tokenized = wiki_ds.filter(lambda x: len(x["text"]) > 20).map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=256), 
    batched=True, remove_columns=wiki_ds.column_names
)

# ==========================================
# 3. Experiment Engine
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
            if i >= 40: break 
    return math.exp(total_loss / (i + 1))

def run_experiment(method_name):
    print(f"\n{'='*40}\n>>> EXPERIMENT: {method_name}\n{'='*40}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, trust_remote_code=True, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # Configure PEFT based on method
    if method_name == "LoRA_Global":
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "up_proj", "down_proj"], 
            task_type=TaskType.CAUSAL_LM, lora_dropout=0.05
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        LR=2e-4
        STP=10

    elif method_name == "KappaTune_LoRA":
        # ← NOW USES PEFT SELECTOR
        stable_modules = get_stable_expert_names(model, budget_k=300)
        lora_config = LoraConfig(
            r=190,
            target_modules=stable_modules, 
            task_type=TaskType.CAUSAL_LM, lora_dropout=0.05
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        LR=2e-4
        STP=35

    if method_name != "Baseline":
        args = TrainingArguments(
            output_dir=f"./{method_name}_out", per_device_train_batch_size=40, 
            gradient_accumulation_steps=4, learning_rate=LR, num_train_epochs=STP, 
            bf16=True, logging_steps=5, save_strategy="no", report_to="none"
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=imdb_tokenized["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        trainer.train()
    
    t_ppl_test = evaluate_perplexity(model, imdb_tokenized["test"], "IMDB")
    t_ppl_train = evaluate_perplexity(model, imdb_tokenized["train"], "IMDB")
    f_ppl = evaluate_perplexity(model, wiki_tokenized, "WikiText")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return t_ppl_test, t_ppl_train, f_ppl

# ==========================================
# 4. Results (same table as paper)
# ==========================================
results = {}
results["Baseline"]     = run_experiment("Baseline")
results["LoRA_Global"]  = run_experiment("LoRA_Global")
results["KappaTune"]    = run_experiment("KappaTune_LoRA")

print("\n" + "="*70)
print(f"{'METHOD':<15} | {'IMDB PPL (Task train)':<18} |  {'IMDB PPL (Task test)':<18} | {'Wiki PPL (General/control)':<18}")
print("-" * 70)
for m, (tpte,tptr,fp) in results.items():
    print(f"{m:<15} | {tptr:<18.4f} | {tpte:<18.4f} | {fp:<18.4f}")
print("="*70)
