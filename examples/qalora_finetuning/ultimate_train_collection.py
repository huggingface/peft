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
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPTQConfig, Trainer
from transformers.trainer_callback import ProgressCallback
from peft import LoraConfig, PeftModel, get_peft_model
import datasets

is_training_on_cluster = os.environ.get("TRAIN_MODE", "").lower() == "cluster"
if is_training_on_cluster:
    datasets.disable_progress_bar()

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
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(['train', 'test', 'eval']):"})
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
        default=False,
        metadata={"help": "If true, skip the training phase and proceed to evaluation."}
    )
    skip_evaluation: bool = field(
        default=False,
        metadata={"help": "If true, skip the final evaluation phase."}
    )
    calibration_dataset: str = field(
        default="c4",
        metadata={"help": "GPTQ calibration_dataset"},
    )
    save_strategy: str = field(
        default="no",
        metadata={"help": "no"},
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

    os.makedirs(script_args.output_dir, exist_ok=True)
    config_path = os.path.join(script_args.output_dir, "run_config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(script_args), f, indent=4)
    print(f"✅ Run-Konfiguration gespeichert in: {config_path}")

    print(f"Setting random seed to {script_args.seed}")
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(script_args.seed)
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

    if script_args.training_mode == "qalora":
        print("🔧 Setting up QA-LoRA training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
            calibration_dataset=script_args.calibration_dataset
        )
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            target_modules=target_modules,
            lora_dropout=0,
            bias="none"
        )

        model = get_peft_model(model, lora_config)
        torch.cuda.empty_cache()
        print("🧹 Original model freed from memory")
    elif script_args.training_mode == "qalora_svd_error":
        print("🔧 Setting up QA-LoRA training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
            calibration_dataset=script_args.calibration_dataset
        )
        
        print("📥 Loading original model for error-svd initialization...")
        og_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Erstelle eine Map mit Layer-Name -> Gewicht
        original_weights_map = {}
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        
        for name, module in og_model.named_modules():
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                # Speichere nur die relevanten Layer
                if any(target in name for target in target_modules):
                    original_weights_map[name] = module.weight.data.clone().to(torch.float32)
        
        print(f"📊 Gespeichert: {len(original_weights_map)} Original-Gewichte für error-svd")

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            target_modules=target_modules,
            lora_dropout=0,
            bias="none",
            init_lora_weights={
                "method": "error-svd", 
                "original_weights_map": original_weights_map,
                "group_size": script_args.qalora_group_size
            }
        )
        del original_weights_map
        model = get_peft_model(model, lora_config)

        adapter_name = model.active_adapter
        config = model.peft_config[adapter_name]

        # Überprüfen, ob die Initialisierungsmethode verwendet wurde und die problematischen Daten enthält
        if hasattr(config, "init_lora_weights") and isinstance(config.init_lora_weights, dict):
            # Entfernen Sie den Tensor oder das gesamte Dictionary.
            # Beides ist eine gute Lösung. Das Ersetzen durch einen einfachen Wert ist oft am sichersten.
            print("Entferne nicht serialisierbare Tensor-Daten aus der Lora-Konfiguration vor dem Speichern...")
            if "original_weights_map" in config.init_lora_weights:
                del config.init_lora_weights["original_weights_map"]
            if "W_orig" in config.init_lora_weights:
                del config.init_lora_weights["W_orig"]
        
        # Cleanup
        del og_model
        torch.cuda.empty_cache()
        print("🧹 Original model freed from memory")
    elif script_args.training_mode == "qalora_svd_error_two_adapter":
        print("✅ 1. Originalmodell laden, um 'Spickzettel' zu erstellen...")
        original_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.float32,
        )

        original_weights_map = {
            name: module.weight.cpu().clone()
            for name, module in original_model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        del original_model

        print("🔧 2. Basismodell quantisieren...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
            calibration_dataset=script_args.calibration_dataset
        )

        print("🏗️ 3. Fehlerkorrektur-Adapter ('error_correction') hinzufügen...")

        init_config_error = {
            "method": "error-svd",
            "original_weights_map": original_weights_map
        }

        error_config = LoraConfig(
            r=1,
            lora_alpha=1,
            use_qalora=False,
            init_lora_weights=init_config_error,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        model = get_peft_model(model, error_config, adapter_name="error_correction")
        print("✅ 'error_correction' Adapter hinzugefügt und per SVD initialisiert.")

        adapter_name = model.active_adapter
        config = model.peft_config["error_correction"]

        if hasattr(config, "init_lora_weights") and isinstance(config.init_lora_weights, dict):
            print("Entferne nicht serialisierbare Tensor-Daten aus der Lora-Konfiguration vor dem Speichern...")
            if "original_weights_map" in config.init_lora_weights:
                del config.init_lora_weights["original_weights_map"]
            if "W_orig" in config.init_lora_weights:
                del config.init_lora_weights["W_orig"]
        
        # Cleanup
        print("❄️ 4. Fehlerkorrektur-Adapter einfrieren...")
        for name, param in model.named_parameters():
            if "error_correction" in name:
                param.requires_grad = False

        # ==============================================================================
        # SCHRITT 3: TASK-ADAPTER HINZUFÜGEN (GAUSS-INIT, TRAINABLE)
        # ==============================================================================
        print("🎨 5. Trainierbaren Task-Adapter ('task_adapter') hinzufügen...")

        train_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
        )

        model.add_adapter("task_adapter", train_config)
        print("✅ 'task_adapter' hinzugefügt und zufällig initialisiert.")

        print("🚀 6. 'task_adapter' als aktiv für das Training setzen...")
        model.set_adapter("task_adapter")

        print("\n--- Finales Modell-Setup ---")
        model.print_trainable_parameters()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            lora_dropout=0,
            bias="none",
            init_lora_weights="pissa"
        )

        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "pissa_rank_analysis":
        print("🔧 Setting up rank analysis with multiple quantization configurations...")

        model_name_clean = script_args.model_name_or_path.replace("/", "_").replace("\\", "_")
        base_output_dir = os.path.join(script_args.output_dir, f"quantized_residuals_r{script_args.lora_r}")
        os.makedirs(base_output_dir, exist_ok=True)

        adapter_name = f"daniel_adapter_r{script_args.lora_r}_{model_name_clean}"
        adapter_path = os.path.join(base_output_dir, adapter_name)

        full_precision_residual_path = os.path.join(script_args.output_dir, f"{model_name_clean}_residual_base_r{script_args.lora_r}_fp16")

        if os.path.exists(adapter_path) and os.path.exists(full_precision_residual_path):
            print(f"⏭️  Found cached adapter at: {adapter_path}")
            print(f"⏭️  Found cached residual model at: {full_precision_residual_path}")
            print("    Skipping model initialization and residual extraction.")

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
                lora_alpha=script_args.lora_r,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0,
                bias="none",
                init_lora_weights="daniel",
            )

            peft_model = get_peft_model(model, lora_config)
            print("✅ PEFT model with daniel initialization complete")

            print(f"Phase 3: Saving adapter to: {adapter_path}")
            peft_model.save_pretrained(adapter_path)

            print("🚀 Extracting residual weights using State Dict method...")
            peft_base_model = peft_model.get_base_model().to(torch.float32)
            base_state_dict = peft_base_model.state_dict()
            clean_state_dict = {}

            for key, value in base_state_dict.items():
                if "base_layer.weight" in key:
                    clean_key = key.replace(".base_layer.weight", ".weight")
                    clean_state_dict[clean_key] = value.clone()
                elif "weight" in key and "lora" not in key and "base_layer" not in key:
                    clean_state_dict[key] = value.clone()

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

            reloaded_model_loaded = AutoModelForCausalLM.from_pretrained(full_precision_residual_path, torch_dtype=torch.float32)
            compare_models(
                residual_model, reloaded_model_loaded, 
                model1_name="Extracted Residual Model", 
                model2_name="Reloaded Model"
            )
            model = reloaded_model_loaded

        print("\n🧹 Clear memory before quantizing...")

        if 'peft_model' in locals():
            del peft_model
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Speicherbereinigung abgeschlossen. Starte Quantisierung...")

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
            adapter_path=adapter_path,
            is_trainable=True,
            dtype=torch.float16,
        )
        print(f"\n{'=' * 60}")
        print("🎉 QUANTIZATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Base model: {script_args.model_name_or_path}")
        print(f"LoRA rank: {script_args.lora_r}")
        print(f"Adapter saved to: {adapter_path}")

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

    # import wandb
    # wandb.init()
    
    training_metrics = {}

    if not script_args.skip_training: 
        trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
        if is_training_on_cluster:
            trainer.remove_callback(ProgressCallback)
            transformers.utils.logging.set_verbosity_error() 

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        trainer.train()
        
        end_time = time.time()
        
        training_duration_min = (end_time - start_time) / 60
        if torch.cuda.is_available():
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            peak_vram_gb = peak_vram_bytes / (1024**3)
        else:
            peak_vram_gb = 0.0

        final_loss = trainer.state.log_history[-1].get('loss') if trainer.state.log_history else None
        
        training_metrics = {
            "peak_vram_gb": round(peak_vram_gb, 2),
            "training_time_min": round(training_duration_min, 2),
            "final_loss": final_loss,
        }
        
        adapter_output_dir = os.path.join(script_args.output_dir, "adapter")
        model.save_pretrained(adapter_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_output_dir)
        print(f"✅ Adapter gespeichert in: {adapter_output_dir}")
    else:
        print("⏭️ Training übersprungen, wie angegeben.")
    
    if script_args.training_mode == "post_quantization":
        print("Applying GPTQ post-quantization...")
        from peft.tuners.lora.gptq import merge_gptq_lora_to_linear

        model = merge_gptq_lora_to_linear(model, adapter_names=None, dtype=torch.bfloat16)
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
        model = AutoModelForCausalLM.from_pretrained(
            temp_model_path,
            quantization_config=gptq_config,
            device_map=None,
            torch_dtype=torch.bfloat16,
        )  
        if torch.cuda.is_available():
            model.to("cuda") 

    if not script_args.skip_evaluation:
        print("\n🚀 Starte Evaluation...")
        model.eval()

        evaluation_dir = os.path.join(script_args.output_dir, "evaluation")
        os.makedirs(evaluation_dir, exist_ok=True)

        from eval_peft import run_lm_harness_and_print_results
        tasks = "wikitext,piqa,tinyArc,tinyHellaswag,tinyGSM8k,tinyMMLU"
        harness_file_name = "lm_harness_results"
        run_lm_harness_and_print_results(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            num_fewshot=1,
            limit=100,
            per_device_eval_batch_size=2,
            output_dir=evaluation_dir,
            file_name=harness_file_name,
        )
        print(f"✅ LM-Harness Ergebnisse gespeichert in: {evaluation_dir}")

        from eval_peft import generate_alpaca_response
        alpaca_file_name = "alpaca_eval_results"
        generate_alpaca_response(model, tokenizer, script_args.training_mode, script_args.lora_r, evaluation_dir, alpaca_file_name)
        print(f"✅ AlpacaEval Ergebnisse gespeichert in: {evaluation_dir}")

        metrics_path = os.path.join(evaluation_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=4)
        print(f"✅ Trainingsmetriken gespeichert in: {metrics_path}")
    else:
        print("⏭️ Evaluation übersprungen, wie angegeben.")


if __name__ == "__main__":
    train()
