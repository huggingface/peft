# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig, Trainer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from utils import load_all_lr_layers, apply_lr_weights, compare_weights
from eval_peft import load_model_and_tokenizer


IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def find_all_linear_names(model) -> List[str]:
    """
    Finds all linear layer names in a model, excluding the lm_head.
    This is a robust way to automatically select target_modules for LoRA.
    """
    linear_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Exclude the output layer and embeddings from LoRA training
            if "lm_head" not in name and "embed_tokens" not in name:
                linear_layer_names.append(name)
    print(f"✅ Automatically discovered {len(linear_layer_names)} linear layers to target.")
    return linear_layer_names

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: list[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    dataloader_num_proc: int = field(default=16, metadata={"help": "Number of processes to load dataset"})
    bits: int = field(default=4, metadata={"help": "Number of bits to quantize the model. Default is 4."})
    adapter_path: str = field(
        default=None,
        metadata={"help": "Pfad zum gespeicherten LoRA-Adapter mit LR-Gewichten."},
    )
    dataloader_batch_size: int = field(
        default=3000,
        metadata={
            "help": "batch size to load dataset. To set the batch size for training, you should pass --batch_size argument instead."
        },
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    training_mode: str = field(
        default="lora",
        metadata={
            "help": "Training mode: 'full' for full finetuning, 'lora' for LoRA, 'qalora' for QA-LoRA, 'pissa' for PiSSA, 'corda' for CorDA"
        },
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "The rank of LoRA adapter. Used for lora, qalora, and pissa modes."},
    )
    qalora_group_size: int = field(
        default=32,
        metadata={"help": "Group size for QA-LoRA quantization."},
    )
    pissa_niter: int = field(
        default=4,
        metadata={"help": "Number of iterations for PiSSA initialization."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    # Keep legacy flags for backwards compatibility
    corda_mode: bool = field(default=None, metadata={"help": "Deprecated: use --training_mode=corda instead"})
    use_qalora: bool = field(default=None, metadata={"help": "Deprecated: use --training_mode=qalora instead"})
    outlier_percentage: float = field(
        default=0.1,
        metadata={"help": "Percentage of outliers to identify for Outlier-Aware QA-LoRA."},
    )   
    report_to: str = field(
        default="wandb",
        metadata={"help": "The integration to report the results and logs to."},
    )
    skip_training: bool = field(
        default=None
    )
    calibration_dataset: str = field(
        default="c4",
        metadata={"help": "GPTQ calibration_dataset"},
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = (_tokenize_fn(strings, tokenizer) for strings in (examples, sources))
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


def load_or_quantize_model(
    model_or_path,
    tokenizer,
    qalora_group_size=32,
    bits: int = 4,
    cache_dir: str = "./quantized_models",
    cache_key: str = None,
    calibration_dataset: str = "c4",
) -> AutoModelForCausalLM:
    is_model_object = isinstance(model_or_path, torch.nn.Module)
    os.makedirs(cache_dir, exist_ok=True)

    if is_model_object:
        key = cache_key or "in_memory_model"
        quantized_model_path = os.path.join(cache_dir, f"{key}_gptq_{bits}bit_groupsize_{qalora_group_size}")
        if os.path.exists(os.path.join(quantized_model_path, "config.json")):
            print(f"Cache hit: {quantized_model_path}")
            return AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
        raise ValueError("Quantizing an in-memory model is not implemented here. Pass a model name/path string.")

    base_model = model_or_path
    model_id = base_model.replace("/", "_").replace("\\", "_")
    quantized_model_path = os.path.join(cache_dir, f"{model_id}_gptq_{bits}bit_groupsize_{qalora_group_size}_calibration_dataset_{calibration_dataset}")

    if os.path.exists(os.path.join(quantized_model_path, "config.json")):
        print(f"Cache hit: {quantized_model_path}")
        return AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )

    print(f"Quantizing base model {base_model} with {bits}-bit, group_size={qalora_group_size}")
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        group_size=qalora_group_size,
        desc_act=False,
        sym=False,
        backend="auto_trainable",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", quantization_config=gptq_config, torch_dtype=torch.float16, trust_remote_code=True
    )
    model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)
    print(f"✅ Cached GPTQ model at {quantized_model_path}")
    return model


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [
        PROMPT.format_map(
            {
                "instruction": instruction,
            }
        )
        for instruction in examples[query]
    ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def compare_models(model1, model2, model1_name="Model 1", model2_name="Model 2", tolerance=1e-9):
    """
    Vergleicht zwei PyTorch-Modelle Parameter für Parameter und gibt detaillierte Unterschiede aus.
    """
    print(f"\n{'='*80}")
    print(f"🔬 Vergleiche '{model1_name}' mit '{model2_name}'")
    print(f"{'='*80}")

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # 1. Struktureller Vergleich: Haben beide Modelle die gleichen Parameter-Namen?
    keys1 = set(params1.keys())
    keys2 = set(params2.keys())

    if keys1 != keys2:
        print("❌ STRUKTURELLER UNTERSCHIED!")
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            print(f"   - Parameter nur in '{model1_name}' gefunden: {list(missing_in_2)[:5]}")
        if missing_in_1:
            print(f"   - Parameter nur in '{model2_name}' gefunden: {list(missing_in_1)[:5]}")
        return False

    # 2. Detaillierter Vergleich der einzelnen Parameter
    mismatched_params = []
    for name, param1 in params1.items():
        param2 = params2[name]

        # Form-Vergleich
        if param1.shape != param2.shape:
            mismatched_params.append(f"  - '{name}': Form-Mismatch! {param1.shape} vs {param2.shape}")
            continue

        # Datentyp-Vergleich
        if param1.dtype != param2.dtype:
            mismatched_params.append(f"  - '{name}': Dtype-Mismatch! {param1.dtype} vs {param2.dtype}")
            continue

        # Werte-Vergleich
        # torch.equal ist sehr strikt. Wir verwenden allclose für numerische Stabilität.
        if not torch.allclose(param1, param2, atol=tolerance):
            diff = torch.abs(param1 - param2).max().item()
            mismatched_params.append(f"  - '{name}': Werte-Mismatch! Maximale Differenz: {diff:.6e}")
            continue

    # 3. Ergebnis ausgeben
    if not mismatched_params:
        print("✅ Modelle sind identisch!")
        print(f"   - Alle {len(params1)} Parameter stimmen überein.")
        return True
    else:
        print(f"❌ UNTERSCHIEDE GEFUNDEN! ({len(mismatched_params)} von {len(params1)} Parametern)")
        # Gib die ersten 10 Unterschiede aus, um die Konsole nicht zu überfluten
        for mismatch in mismatched_params[:10]:
            print(mismatch)
        if len(mismatched_params) > 10:
            print(f"  ... und {len(mismatched_params) - 10} weitere.")
        return False


def ensure_gptq_artifact(model_path, model_to_quantize, calibration_set, tokenizer, bits, group_size):
    """
    Ensure a GPTQ-quantized artifact exists at `model_path`. If it doesn't, quantize
    the model found at `model_to_quantize` and save it to `model_path`.

    Args:
        model_path: Target directory for the quantized artifact.
        model_to_quantize: Source model (usually a directory) to quantize.
        tokenizer: Tokenizer to save alongside the model.
        bits: Quantization bit width.
        group_size: Group size for GPTQ.

    Returns:
        None. Writes artifacts to disk.
    """
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Cache hit: {model_path} – skip quantization")
        return

    os.makedirs(model_path, exist_ok=True)
    print(f"Quantizing {model_to_quantize} -> {model_path} with {bits}-bit, group_size={group_size}")
    gptq_cfg = GPTQConfig(
        bits=bits,
        # dataset="alpaca-cleaned",
        dataset=calibration_set,
        tokenizer=tokenizer,
        group_size=group_size,
        desc_act=False,
        sym=False,
        # backend="auto_trainable",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_to_quantize,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=gptq_cfg,
        trust_remote_code=True,
    )
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"✅ Saved quantized artifact to {model_path}")


def load_residual_with_adapter(
    residual_or_quantized_path: str,
    adapter_path: str,
    *,
    is_trainable: bool = False,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
):
    """
    Load a residual (FP) or GPTQ-quantized residual model and attach the matching LoRA adapter.
    """
    tok = transformers.AutoTokenizer.from_pretrained(
        residual_or_quantized_path, use_fast=True, padding_side="right", trust_remote_code=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        residual_or_quantized_path, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=is_trainable)
    if is_trainable:
        # ✅ CRITICAL: Set model to training mode
        model.train()
        
        # ✅ Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # ✅ CRITICAL: Enable input gradients for PEFT + gradient checkpointing
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
            
        # ✅ Disable cache during training (required for gradient computation)
        model.config.use_cache = False
        
        # ✅ Verify trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # ✅ Debug: Verify adapter parameters are trainable
        adapter_params = [name for name, param in model.named_parameters() if param.requires_grad]
        if adapter_params:
            print(f"✅ Found {len(adapter_params)} trainable adapter parameters")
        else:
            print("❌ NO TRAINABLE PARAMETERS FOUND!")
            
    else:
        model.eval()
    return model, tok

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    # --- ADD THIS BLOCK ---
    # Set seed before initializing model.
    print(f"Setting random seed to {script_args.seed}")
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(script_args.seed)
        # --- END OF BLOCK ---
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Training mode selectiowave_moden
    if script_args.training_mode == "corda":
        print("🔧 Setting up CorDA training...")
        res_model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(
            res_model, script_args.model_name_or_path, subfolder="corda_init", is_trainable=True
        )
    elif script_args.training_mode == "qalora":
        print("🔧 Setting up QA-LoRA training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
            calibration_dataset=script_args.calibration_dataset
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            # target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "gptq_lora":
        print("🔧 Setting up QA-LoRA training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            # target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "outlier_aware_qalora": # Empfehlung: Eigener Modus-Name
        print("🚀 Setting up Outlier-Aware QA-LoRA training...")
        model_name_clean = script_args.model_name_or_path.replace("/", "_").replace("\\", "_")

        # =======================================================================
        # SCHRITT 1 & 2: Hochpräzises Modell laden & Outlier analysieren
        # =======================================================================
        print("Step 1/6: Loading high-precision base model for Hessian analysis...")
        outlier_percentage_str = str(script_args.outlier_percentage).replace(".", "p")
        cache_dir = "./hessian_cache"
        os.makedirs(cache_dir, exist_ok=True)
        outlier_cache_path = os.path.join(cache_dir, f"{model_name_clean}_outliers_{outlier_percentage_str}pct.pt")

        if os.path.exists(outlier_cache_path):
            print(f"✅ Loading cached Hessian outliers from: {outlier_cache_path}")
            outlier_data = torch.load(outlier_cache_path, map_location="cpu")
            print(f"Found outliers in {len(outlier_data)} layers.")
        else:
            print("🔥 No cached outlier data found. Starting Hessian calculation...")
            # =======================================================================
            # SCHRITT 1 & 2: Hochpräzises Modell laden & Outlier analysieren
            # =======================================================================
            print("Step 1/6: Loading high-precision base model for Hessian analysis...")
            # Wichtig: Laden Sie das Modell in voller Präzision (z.B. bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                torch_dtype=torch.bfloat16, # Oder float16
                device_map="auto"
            )

            print("Step 2/6: Calculating Hessian saliency to find outliers...")
            from utils import calculate_hessian_outliers
            outlier_data = calculate_hessian_outliers(model, tokenizer, outlier_percentage=script_args.outlier_percentage)

            # from utils import spqr_extract_outliers, get_c4_calibration_data

            print("Step 2/6: Calculating Hessian saliency to find outliers...")

            # calibration_samples = get_c4_calibration_data(
            #     tokenizer,
            #     n_samples= 128,
            #     seq_len=2048,
            # )

            # calibration_batches = [dict(sample) for sample in calibration_samples]

            # outlier_data = spqr_extract_outliers(
            #     model,
            #     calibration_batches=calibration_batches,
            #     aaaaaaaaaaaaaaaaaaaaaaaaaaaaa=[
            #         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            #         "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            #     ],
            #     # Entweder globaler Anteil …
            #     # outlier_percentage=script_args.outlier_percentage,  # z.B. 0.001 für 0.1%
            #     # … oder feste Schwelle wie in SpQR (dann obige Zeile auskommentieren):
            #     # outlier_threshold=script_args.outlier_threshold,  # z.B. 0.2
            #     bits=2,
            #     group_size=32,
            #     percdamp=1.0,    # leichte Dämpfung für Stabilität
            # )
            print(f"Found outliers in {len(outlier_data)} layers.")

            # Speichere die berechneten Outlier für das nächste Mal in den Cache
            print(f"💾 Caching Hessian outliers to: {outlier_cache_path}")
            torch.save(outlier_data, outlier_cache_path)

            # Wichtig: Geben Sie den Speicher des hochpräzisen Modells frei
            del model
            torch.cuda.empty_cache()

        # =======================================================================
        # SCHRITT 3 & 4: Modell bereinigen und dann quantisieren
        # =======================================================================
        print("Step 3/6: Loading model again to create a cleaned version for quantization...")
        # Wir laden das Modell erneut, um sicherzugehen, dass wir einen sauberen Start haben
        model_for_quant = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print("Step 4/6: Zeroing out outliers before quantization...")
        for name, module in model_for_quant.named_modules():
            if name in outlier_data:
                # Holen Sie nur die Indizes, wir brauchen die Gewichte hier nicht
                _, outlier_indices = outlier_data[name]
                # Stellen Sie sicher, dass das Attribut 'weight' existiert und veränderbar ist
                if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                    with torch.no_grad():
                        # Setze die Gewichte an den Outlier-Positionen auf Null
                        module.weight.data.view(-1)[outlier_indices] = 0.0

        print("Step 5/6: Quantizing the cleaned model...")
        # Nutzen Sie Ihre bestehende Funktion, um das *bereinigte* Modell zu quantisieren.
        # Diese Funktion ersetzt die nn.Linear-Schichten durch Ihre QuantLinear-Schichten.
        cache_key = f"{model_name_clean}_outlier_aware_{outlier_percentage_str}_pct_gptq_{script_args.bits}bit_gs{script_args.qalora_group_size}"
        quantized_model = load_or_quantize_model(
            model_for_quant, # Übergeben Sie das modifizierte Modell
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
            cache_key=cache_key,
            calibration_dataset=script_args.calibration_dataset
            # Wichtig: Stellen Sie sicher, dass diese Funktion das Modell nicht neu von der Festplatte lädt,
            # sondern das übergebene Objekt verwendet.
        )

        # Geben Sie den Speicher des bereinigten Modells frei
        del model_for_quant
        torch.cuda.empty_cache()

        from gptqmodel.nn_modules.qlinear import BaseQuantLinear
        # =======================================================================
        # SCHRITT 5: Outlier als Buffer in das quantisierte Modell injizieren
        # =======================================================================
        print("Step 6/6: Injecting high-precision outliers into the quantized model...")
        for name, module in quantized_model.named_modules():
            # Stellen Sie sicher, dass Sie den richtigen Klassentyp Ihrer quantisierten Schicht prüfen
            if isinstance(module, BaseQuantLinear): # z.B. GPTQLinear, QuantLinear, etc.
                if name in outlier_data:
                    outlier_weights, outlier_indices = outlier_data[name]

                    # Registrieren der Outlier als nicht-trainierbare Buffer
                    module.register_buffer('outlier_weights', outlier_weights.to(torch.bfloat16))
                    module.register_buffer('outlier_indices', outlier_indices.to(torch.long))

        # =======================================================================
        # SCHRITT 6: PEFT anwenden (wie bei normalem QA-LoRA)
        # =======================================================================
        print("🔧 Applying PEFT with QA-LoRA config...")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        # Hier passiert die Magie: PEFT wird auf das vorbereitete, hybride Modell angewendet.
        # Ihre modifizierte `forward`-Methode in QALoraLinearVariant wird den Rest erledigen.
        model = get_peft_model(quantized_model, lora_config)

        print("\n✅ Outlier-Aware QA-LoRA model is ready for training!")
    elif script_args.training_mode == "pissa":
        print("🔧 Setting up PiSSA training...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            # init_lora_weights=f"pissa_niter_{script_args.pissa_niter}",  # PiSSA initialization
        )

        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "lora":
        print("🔧 Setting up LoRA training...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            init_lora_weights=True,
        )

        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "qlora":
        print("🔧 Setting up QLoRA training...")

        # Configure 4-bit quantization for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model with 4-bit quantization
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Prepare model for k-bit training (essential for QLoRA)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # Configure LoRA for QLoRA
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,  # Slightly higher dropout for QLoRA
            bias="none",
            init_lora_weights=True,
        )

        model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        print(f"✅ QLoRA model loaded with 4-bit quantization")
    elif script_args.training_mode == "pissa_rank_analysis":
        print("🔧 Setting up rank analysis with multiple quantization configurations...")

        # --- CACHING LOGIC START ---
        # Phase 1: Definiere Pfade und prüfe auf existierende Artefakte
        model_name_clean = script_args.model_name_or_path.replace("/", "_").replace("\\", "_")
        base_output_dir = os.path.join(script_args.output_dir, f"quantized_residuals_r{script_args.lora_r}")
        os.makedirs(base_output_dir, exist_ok=True)

        adapter_name = f"daniel_adapter_r{script_args.lora_r}_{model_name_clean}"
        adapter_path = os.path.join(base_output_dir, adapter_name)

        full_precision_residual_path = os.path.join(script_args.output_dir, f"{model_name_clean}_residual_base_r{script_args.lora_r}_fp16")

        # Prüfe, ob sowohl der Adapter als auch das Residual-Modell bereits existieren
        if os.path.exists(adapter_path) and os.path.exists(full_precision_residual_path):
            print(f"⏭️  Found cached adapter at: {adapter_path}")
            print(f"⏭️  Found cached residual model at: {full_precision_residual_path}")
            print("    Skipping model initialization and residual extraction.")

            # Lade die target_modules aus der Adapter-Konfiguration für die spätere Verwendung
            from peft import PeftConfig
            try:
                config = PeftConfig.from_pretrained(adapter_path)
                target_modules = config.target_modules
                print(f"    Loaded target_modules from adapter config: {len(target_modules)} modules.")
            except Exception as e:
                print(f"    Could not load adapter config ({e}), using empty target_modules list.")
                target_modules = []
        else:
            print("🔥 No cached artifacts found. Starting full initialization process...")

            # Phase 2: Lade das Originalmodell und richte PEFT ein
            print("Phase 2: Loading original model and setting up PEFT...")
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float32,
            )

            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                use_qalora=True,
                qalora_group_size=script_args.qalora_group_size,
                r=script_args.lora_r,
                lora_alpha=2 * script_args.lora_r,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                init_lora_weights="daniel",
            )

            peft_model = get_peft_model(model, lora_config)
            print("✅ PEFT model with daniel initialization complete")

            # Phase 3: Extrahiere und speichere den Adapter und das Residual-Modell
            print(f"Phase 3: Saving adapter to: {adapter_path}")
            peft_model.save_pretrained(adapter_path)

            print("🚀 Extracting residual weights using State Dict method...") # why is that so complicated? We need to clean up the "base_layer" substring model.layers.21.mlp.down_proj.base_layer.weight -> model.layers.21.mlp.down_proj.weight
            peft_base_model = peft_model.get_base_model().to(torch.float32)
            base_state_dict = peft_base_model.state_dict()
            clean_state_dict = {}

            for key, value in base_state_dict.items():
                if "base_layer.weight" in key:
                    clean_key = key.replace(".base_layer.weight", ".weight")
                    clean_state_dict[clean_key] = value.clone()
                elif "weight" in key and "lora" not in key and "base_layer" not in key:
                    clean_state_dict[key] = value.clone()

            # Füge den lm_head manuell hinzu, da get_base_model() ihn oft auslässt
            if "lm_head.weight" in peft_model.state_dict():
                print("🧠 Adding lm_head.weight to the clean state_dict.")
                clean_state_dict["lm_head.weight"] = peft_model.state_dict()["lm_head.weight"].clone()

            print(f"📋 Extracted: {len(clean_state_dict)} clean weights")

            residual_model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path, torch_dtype=torch.float32
            )

            residual_model.load_state_dict(clean_state_dict, strict=False)

            print(f"💾 Saving clean residual model to: {full_precision_residual_path}")
            residual_model.save_pretrained(full_precision_residual_path)
            tokenizer.save_pretrained(full_precision_residual_path)
            print("✅ Clean residual model saved.")

            # Verifiziere den Speicher-/Ladevorgang
            reloaded_model_loaded = AutoModelForCausalLM.from_pretrained(full_precision_residual_path, torch_dtype=torch.float32)
            compare_models(
                residual_model, reloaded_model_loaded, 
                model1_name="Extracted Residual Model", 
                model2_name="Reloaded Model"
            )
            model = reloaded_model_loaded # residual model is in FP16 still

        print("\n🧹 Clear memory before quantizing...")

        if 'peft_model' in locals():
            del peft_model
        # if 'model' in locals():
        #     del model
        if 'peft_base_model' in locals():
            del peft_base_model
        if 'base_state_dict' in locals():
            del base_state_dict
        if 'clean_state_dict' in locals():
            del clean_state_dict
        if 'residual_model' in locals():
            del residual_model
        if 'reloaded_model_loaded' in locals():
            del reloaded_model_loaded

        # Leere den CUDA-Cache, um den Speicher wirklich freizugeben
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Speicherbereinigung abgeschlossen. Starte Quantisierung...")

        # --- CACHING LOGIC END ---
        # Vorher: Mehrfach-Quantisierung (quantization_configs) -> entfernen.
        # Jetzt: Optional genau EINE Quantisierung auf Basis von script_args.bits und script_args.qalora_group_size
        bits = script_args.bits
        group_size = script_args.qalora_group_size
        print(f"\n{'=' * 60}")
        print(f"Quantizing residual once: {bits}-bit, group_size={group_size}")
        print(f"{'=' * 60}")

        calibration_dataset = script_args.calibration_dataset

        quantized_name = (
            f"w_res_{model_name_clean}_r{script_args.lora_r}_daniel_{bits}bit_gs{group_size}_{calibration_dataset}"
        )
        quantized_path = os.path.join(base_output_dir, quantized_name)

        ensure_gptq_artifact(
            quantized_path,
            full_precision_residual_path,
            calibration_dataset,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
        )

        print(f"✅ Residual quantized (or loaded from cache): {quantized_path}")

        model, tok = load_residual_with_adapter(
            residual_or_quantized_path=quantized_path,
            # residual_or_quantized_path=full_precision_residual_path,
            adapter_path=adapter_path,
            is_trainable=True,
            dtype=torch.float16,  # oder torch.bfloat16, falls so gespeichert
        )
        print(f"\n{'=' * 60}")
        print("🎉 QUANTIZATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Base model: {script_args.model_name_or_path}")
        print(f"LoRA rank: {script_args.lora_r}")
        print(f"Adapter saved to: {adapter_path}")

    elif script_args.training_mode == "full":
        print("🔧 Setting up full finetuning...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif script_args.training_mode == "daniel_old":
        print("🔧 Setting up QA-LoRA training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            init_lora_weights="daniel",  # PiSSA initialization
        )

        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "daniel":
        print("🔧 Setting up QA-LoRA training with PiSSA initialization...")

        # =================================================================================
        # PHASE 1: Sichern der originalen FP32-Gewichte
        # =================================================================================
        print("Phase 1: Lade das originale FP32-Modell zum Cachen der Gewichte...")
        # Lade das hochpräzise Originalmodell
        fp32_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)

        # Definiere die Zielmodule hier, damit wir wissen, welche Gewichte wir speichern müssen
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

        # Erstelle eine Map, die Layernamen auf ihre FP32-Gewichte abbildet
        fp32_weights_map = {
            name: module.weight.clone().to(torch.float32)
            for name, module in fp32_model.named_modules()
            # Speichere nur die Gewichte, die wir für PiSSA brauchen werden
            if any(target_module in name for target_module in target_modules)
        }

        print(f"  -> {len(fp32_weights_map)} FP32-Gewichtsmatrizen für Ziel-Layer gecached.")

        # Gib den Speicher des großen FP32-Modells sofort frei
        del fp32_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # =================================================================================
        # PHASE 2: Modell laden/quantisieren und Gewichte anheften
        # =================================================================================
        print("\nPhase 2: Lade oder quantisiere das Hauptmodell...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
        )
        print("  -> Modell geladen/quantisiert.")

        print("\nPhase 3: Hänge die gecachten FP32-Gewichte an die quantisierten Layer an...")
        # Iteriere durch die Layer des *quantisierten* Modells
        for name, module in model.named_modules():
            if name in fp32_weights_map:
                # Hänge das gecachte FP32-Gewicht als neues Attribut an den Layer an.
                # Ihre `daniel_init` Funktion wird nach diesem Attribut suchen.
                module.original_weight_fp32 = fp32_weights_map[name].to(module.qweight.device)
                print(f"  - Originalgewicht für '{name}' erfolgreich angehängt.")

        # =================================================================================
        # PHASE 3: PEFT-Setup mit PiSSA (Ihr ursprünglicher Code)
        # =================================================================================
        print("\nPhase 4: Richte PEFT mit PiSSA-Initialisierung ein...")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=target_modules,  # Wiederverwendung der oben definierten Liste
            lora_dropout=0.05,
            bias="none",
            init_lora_weights="daniel_niter_5",  # WICHTIG: Stellen Sie sicher, dass Sie hier die Anzahl der Iterationen angeben
        )

        model = get_peft_model(model, lora_config)
        print("  -> PEFT-Modell erfolgreich erstellt. PiSSA-Initialisierung wird beim Trainingsstart ausgelöst.")
    elif script_args.training_mode == "post_quantization":
        print("🔧 Setting up post_quantization training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
            # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        # model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        # raise ValueError(f"Unknown training mode: {script_args.training_mode}")

    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=script_args.dataloader_batch_size,
        num_proc=script_args.dataloader_num_proc,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": script_args.dataset_field[0],
            "response": script_args.dataset_field[1],
        },
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"❌ Gradient disabled for parameter: {name}")
    import wandb
    wandb.init()
    if not script_args.skip_training: 
        trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
        trainer.train()
        # trainer.save_state()
        model.save_pretrained(os.path.join(script_args.output_dir, "ft/adapter"))
        tokenizer.save_pretrained(os.path.join(script_args.output_dir, "ft/adapter"))

    if script_args.training_mode == "post_quantization":
        print("Applying GPTQ post-quantization...")
        # Configure 4-bit quantization for QLoRA
        from peft.tuners.lora.gptq import merge_gptq_lora_to_linear

        # This will dequantize-and-merge in-place (returns the same object for convenience)
        model = merge_gptq_lora_to_linear(model, adapter_names=None, dtype=torch.bfloat16)

        # model = model.dequantize()
        # model = model.merge_and_unload()
        temp_model_path = "./temp_merged_model"

        model.save_pretrained(temp_model_path)

        gptq_config = GPTQConfig(
            bits=script_args.bits,
            dataset=script_args.calibration_dataset,
            tokenizer=tokenizer,
            group_size=script_args.qalora_group_size,
            desc_act=False,
            sym=False,
        )
        # post quantization gptq
        model = AutoModelForCausalLM.from_pretrained(
            temp_model_path,
            quantization_config=gptq_config,
            device_map=None,
            torch_dtype=torch.bfloat16,
        )  
        if torch.cuda.is_available():
            model.to("cuda") 

    # --- START: Generation and Evaluation Logic ---
    print("\n🚀 Starting Alpaca evaluation...")
    # Put the model in evaluation mode
    model.eval()

    file_name = f"mode_{script_args.training_mode}_bits_{script_args.bits}_rank_{script_args.lora_r}_group_{script_args.qalora_group_size}_training_skip_{script_args.skip_training}_calibration_dataset_{script_args.calibration_dataset}"
    output_dir_eval = os.path.join(script_args.output_dir, "eval_results")
    os.makedirs(output_dir_eval, exist_ok=True)

    from eval_peft import generate_alpaca_response, run_lm_harness_and_print_results
    generate_alpaca_response(model, tokenizer, script_args.training_mode, script_args.lora_r, output_dir_eval, file_name)
    tasks="wikitext,piqa,tinyArc,tinyHellaswag,tinyGSM8k,tinyMMLU"
    # tasks = "wikitext"
    run_lm_harness_and_print_results(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        num_fewshot=1,
        limit=100,
        per_device_eval_batch_size=2,
        output_dir=output_dir_eval,
        file_name=file_name,
    )
    print("eval lm_harness")

if __name__ == "__main__":
    train()
