#!/usr/bin/env python3
"""
Evaluation script for pre-quantized residual connection models
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from eval_peft import load_model_and_tokenizer, evaluate_with_lm_eval, print_results
from typing import Dict, Optional 


def parse_residual_model_info(model_path: str) -> Dict[str, Any]:
    """Extract rank, bits, and group_size from residual model path"""
    path_name = os.path.basename(model_path)
    
    # Pattern 1: w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_4bit_gs32 (quantized residual)
    pattern = r'w_res_(.+)_r(\d+)_daniel_(\d+)bit_gs(\d+)'
    match = re.match(pattern, path_name)
    
    if match:
        base_model_part, rank, bits, group_size = match.groups()
        return {
            "base_model_name": base_model_part,
            "rank": int(rank),
            "bits": int(bits),
            "group_size": int(group_size),
            "model_type": "residual_quantized"
        }
    
    # Pattern 2: temp_residual_base_r256_fp16 (unquantized residual base)
    fp16_pattern = r'temp_residual_base_r(\d+)_fp16'
    match = re.match(fp16_pattern, path_name)
    
    if match:
        rank = match.group(1)
        return {
            "base_model_name": "HuggingFaceTB/SmolLM2-1.7B",  # Default base model
            "rank": int(rank),
            "bits": 16,  # FP16
            "group_size": None,
            "model_type": "residual_fp16_base"
        }
    
    # Skip standalone adapters - they are only used as components
    adapter_pattern = r'daniel_adapter_r(\d+)_(.+)'
    match = re.match(adapter_pattern, path_name)
    
    if match:
        # Don't return this as a standalone model
        return None
    
    return None


def discover_residual_models(base_dir: str) -> List[Dict[str, Any]]:
    """Discover all residual models in the directory"""
    models = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Skip JSON files
        if item.endswith('.json'):
            continue
            
        if os.path.isdir(item_path):
            # Check if it's a valid model directory
            adapter_config = os.path.join(item_path, "adapter_config.json")
            config_json = os.path.join(item_path, "config.json")
            
            # Check if it's either an adapter or a base model
            if os.path.exists(adapter_config) or os.path.exists(config_json):
                model_info = parse_residual_model_info(item)
                if model_info:
                    model_info["path"] = item_path
                    models.append(model_info)
                    print(f"🔍 Found model: {item} -> Type: {model_info['model_type']}, Rank: {model_info['rank']}, Bits: {model_info['bits']}")
    
    return sorted(models, key=lambda x: (x["rank"], x["bits"], x.get("group_size", 0)))


def find_corresponding_quantized_model(base_model_name: str, bit: str, group_size: str ) -> str:
    """Find the corresponding adapter for a quantized base model"""
    quantized_model = f"/home/nudel/Documents/peft/quantized_models/{base_model_name}_gptq_{bit}bit_groupsize_{group_size}"
    if os.path.exists(quantized_model):
        return quantized_model
    return None


def find_corresponding_adapter(base_model_path: str, base_dir: str) -> str:
    """Find the corresponding adapter for a quantized base model"""
    base_name = os.path.basename(base_model_path)
    
    # Extract rank from base model name: w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_4bit_gs32
    match = re.search(r'_r(\d+)_', base_name)
    if not match:
        return None
    
    rank = match.group(1)
    
    # Extract model name part: HuggingFaceTB_SmolLM2-1.7B
    model_match = re.search(r'w_res_(.+)_r\d+_daniel', base_name)
    if not model_match:
        return None
    
    model_part = model_match.group(1)
    
    # Look for corresponding adapter: daniel_adapter_r256_HuggingFaceTB_SmolLM2-1.7B
    adapter_name = f"daniel_adapter_r{rank}_{model_part}"
    adapter_path = os.path.join(base_dir, adapter_name)
    
    if os.path.exists(adapter_path):
        return adapter_path
    
    return None

def find_corresponding_adapter_for_fp16(base_model_path: str, base_dir: str) -> str:
    """Find the corresponding adapter for an FP16 base model"""
    base_name = os.path.basename(base_model_path)
    
    # Extract rank from FP16 base model name: temp_residual_base_r256_fp16
    match = re.search(r'temp_residual_base_r(\d+)_fp16', base_name)
    if not match:
        return None
    
    rank = match.group(1)
    
    # For FP16 base models, we need to find the adapter based on the directory structure
    # Look in parent directory for adapters with same rank
    parent_dir = os.path.dirname(base_model_path)
    
    # Pattern: daniel_adapter_r256_*
    for item in os.listdir(parent_dir):
        adapter_pattern = f"daniel_adapter_r{rank}_"
        if item.startswith(adapter_pattern) and os.path.isdir(os.path.join(parent_dir, item)):
            adapter_path = os.path.join(parent_dir, item)
            return adapter_path
    
    return None

def evaluate_residual_model(
    model_info: Dict[str, Any],
    base_dir: str,
    tasks: str,
    num_fewshot: int = 5,
    limit: int = None,
    per_device_eval_batch_size: int = 1
) -> Dict[str, Any]:
    """Evaluate a single residual model"""
    
    model_path = model_info["path"]
    
    print(f"\n🔍 Evaluating: {os.path.basename(model_path)}")
    print(f"   Rank: {model_info['rank']}, Bits: {model_info['bits']}, Group Size: {model_info.get('group_size', 'N/A')}")
    
    try:
        # Determine base model and adapter paths
        if model_info["model_type"] == "residual_quantized":
            # This is a quantized base model, find its adapter
            base_model_path = model_path
            adapter_path = find_corresponding_adapter(model_path, base_dir)
            quantized_model_path = find_corresponding_quantized_model(base_model_name=model_info["base_model_name"], bit=model_info["bits"], group_size=model_info["group_size"] )
            
            if not adapter_path:
                raise ValueError(f"Could not find corresponding adapter for {os.path.basename(model_path)}")
            
            print(f"   Residualmodel: {os.path.basename(base_model_path)}")
            print(f"   Adapter: {os.path.basename(adapter_path)}")
        elif model_info["model_type"] == "residual_fp16_base":
            # This is an unquantized FP16 base model, find its adapter
            base_model_path = model_path
            adapter_path = find_corresponding_adapter_for_fp16(model_path, base_dir)
            
            if not adapter_path:
                raise ValueError(f"Could not find corresponding adapter for {os.path.basename(model_path)}")
            
            print(f"   Base model (FP16): {os.path.basename(base_model_path)}")
            print(f"   Adapter: {os.path.basename(adapter_path)}") 
        elif model_info["model_type"] == "original_adapter":
            # This is an adapter, we need the original base model (unquantized)
            adapter_path = model_path
            base_model_path = model_info["base_model_name"]  # This should be the HF model name
            
            print(f"   Base model: {base_model_path}")
            print(f"   Adapter: {os.path.basename(adapter_path)}")
        
        else:
            raise ValueError(f"Unknown model type: {model_info['model_type']}")
        
        # Load model
        quantised_residual_with_LR_model, tokenizer = load_model_and_tokenizer(adapter_path, base_model_path)
        quantized_model, _ = load_model_and_tokenizer(None, quantized_model_path)
        
        prefix = "/home/nudel/Documents/peft/visualizations"
        errors = calculate_reconstruction_errors(adapter_path, quantised_residual_with_LR_model, quantized_model=quantized_model, visualization_path=f"{prefix}/layer_heatmaps_rank_128", svd_visualization_path=f"{prefix}/svd_heatmaps_rank_128")
        
        del quantized_model
        torch.cuda.empty_cache()

        svd_error = errors.get('svd_error', 0.0)
        quant_error = errors.get('quant_error', 0.0)
        total_error = errors.get('total_error', 0.0)

        print("\n---[ Analyse der Rekonstruktionsfehler ]---")
        print(f"  🇸 SVD-Fehler:       {svd_error:>8.4%}")
        print(f"     (Fehler zwischen W_original und dem reinen LoRA-Adapter)")
        print("")
        print(f"  📉 Quant.-Fehler:    {quant_error:>8.4%}")
        print(f"     (Fehler, der durch die Quantisierung des Residuals entsteht)")
        print("-" * 45)
        print(f"  🎯 Gesamtfehler:      {total_error:>8.4%}")
        print(f"     (Endgültige Abweichung des kombinierten Modells von W_original)")
        print("-------------------------------------------")

        limit = 100
        # Run evaluation

        results = evaluate_with_lm_eval(
            model=quantised_residual_with_LR_model,
            tokenizer=tokenizer,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            per_device_eval_batch_size=per_device_eval_batch_size
        )
        
        # Clean up
        del quantised_residual_with_LR_model
        torch.cuda.empty_cache()

        print_results(results)

        return {
            "model_info": model_info,
            "evaluation_results": results["results"],
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {model_path}: {e}")
        return {
            "model_info": model_info,
            "evaluation_results": None,
            "status": "failed",
            "error": str(e)
        }

def create_residual_performance_table(results: List[Dict], output_dir: str):
    """Create LaTeX table matching Overleaf format with WikiText column"""
    
    # Sort and group by rank
    sorted_results = sorted([r for r in results if r["status"] == "success"],
                          key=lambda x: (x["model_info"]["rank"], x["model_info"]["bits"], x["model_info"].get("group_size", 0)))
    
    rank_groups = {}
    for r in sorted_results:
        rank = r["model_info"]["rank"]
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(r)
    
    # Generate LaTeX content
    latex = []
    
    # SmolLM2 1.7B Block with multirow for all ranks
    total_rows = sum(len(models) for models in rank_groups.values())
    latex.append(f"\\multirow{{{total_rows}}}{{*}}{{\\shortstack{{SmolLM2 \\\\ 1.7B}}}}")
    
    sorted_ranks = sorted(rank_groups.keys())
    
    for rank_idx, rank in enumerate(sorted_ranks):
        models = rank_groups[rank]
        
        # Sort models: FP16 first, then by bits and group_size
        models.sort(key=lambda x: (0 if x["model_info"]["bits"] == 16 else 1, 
                                   x["model_info"]["bits"], 
                                   x["model_info"].get("group_size", 0)))
        
        for model_idx, result in enumerate(models):
            info = result["model_info"]
            eval_res = result["evaluation_results"]
            
            # Rank column with multirow
            if model_idx == 0:
                rank_col = f"\\multirow{{{len(models)}}}{{*}}{{{rank}}}"
            else:
                rank_col = ""
            
            # Quantization description
            if info["bits"] == 16:
                quant_desc = "16 (FP16)"
            elif info["bits"] == 4 and info.get("group_size"):
                quant_desc = f"4-bit (Gs {info['group_size']})"
            else:
                quant_desc = f"{info['bits']}-bit"
            
            # Extract metrics
            row_data = ["&", rank_col, "&", quant_desc]
            
            # Standard evaluation tasks
            for task in ["arc_challenge", "arc_easy", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande"]:
                if task in eval_res:
                    tr = eval_res[task]
                    acc = tr.get("acc_norm,none", tr.get("acc,none"))
                    stderr = tr.get("acc_norm_stderr,none", tr.get("acc_stderr,none"))
                    if acc is not None and stderr is not None:
                        row_data.append(f"& {acc*100:.1f} ± {stderr*100:.1f}")
                    else:
                        row_data.append("& ")
                else:
                    row_data.append("& ")
            
            # WikiText perplexity (lower is better, no error bars)
            if "wikitext" in eval_res:
                wt = eval_res["wikitext"]
                perplexity = wt.get("word_perplexity,none")
                if perplexity is not None:
                    row_data.append(f"& {perplexity:.1f}")
                else:
                    row_data.append("& ")
            else:
                row_data.append("& ")
            
            row_data.append("\\\\")
            latex.append(" ".join(row_data))
        
        # Add cmidrule after each rank group (except the last one)
        if rank_idx < len(sorted_ranks) - 1:
            latex.append("\\cmidrule{2-12}")  # Updated for 12 columns
    
    # Add final midrule
    latex.append("\\midrule")
    
    # Save to file
    latex_file = os.path.join(output_dir, "residual_quantization_analysis.tex")
    with open(latex_file, "w") as f:
        f.write("\n".join(latex))
    
    print(f"📄 LaTeX saved to: {latex_file}")
    print(f"\n{'='*60}\n" + "\n".join(latex) + f"\n{'='*60}")

def _get_lora_target_modules(peft_model: PeftModel) -> Dict[str, torch.nn.Module]:
    """Helper to find all LoRA layers in a PeftModel."""
    lora_layers = {}
    for name, module in peft_model.named_modules():
        # The LoraLayer class is a common parent for LoRA-adapted layers
        if "lora" in module.__class__.__name__.lower() and hasattr(module, "lora_A"):
            lora_layers[name] = module
    return lora_layers

def calculate_reconstruction_errors(
    adapter_path: str,
    peft_model: PeftModel,
    quantized_model: AutoModelForCausalLM = None,
    visualization_path: Optional[str] = None,
    svd_visualization_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Berechnet die relativen SVD-, Quantisierungs- und Gesamtfehler für ein PEFT-Modell
    mit einer quantisierten residualen Basis im Vergleich zum wahren Originalmodell.

    Args:
        true_original_model: Das ursprüngliche, unberührte Foundation Model (Ground Truth).
        peft_model: Das PeftModel, das aus einer quantisierten residualen Basis und einem LoRA-Adapter besteht.

    Returns:
        Ein Dictionary, das die berechneten relativen Fehler für 'svd', 'quant' und 'total' enthält.
    """
    if visualization_path:
        os.makedirs(visualization_path, exist_ok=True)
        print(f"Heatmaps werden in '{visualization_path}' gespeichert.")
        # Check if this is a PEFT model
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print(f"Loading PEFT model from {adapter_path}")

        # Read adapter config
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        # Get base model name
        # hier muss das basemodel mit dem residual model austausgetauscht werden
        base_model_name = adapter_config.get("base_model_name_or_path")
    else:
        raise ValueError(f"Adapter config not found at {adapter_config_path}")
    
    true_original_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    if svd_visualization_path:
        os.makedirs(svd_visualization_path, exist_ok=True)
        print(f"SVD-Analysen werden in '{svd_visualization_path}' gespeichert.")
        
    with torch.no_grad():
        original_modules = dict(true_original_model.named_modules())
        quantized_modules = dict(quantized_model.named_modules())
        lora_layers = _get_lora_target_modules(peft_model)

        if not lora_layers:
            print("Warnung: Keine LoRA-Layer im peft_model gefunden.")
            return {"svd_error": 0.0, "quant_error": 0.0, "total_error": 0.0}

        total_norm_W_original_sq = 0.0
        total_norm_svd_error_sq = 0.0
        total_norm_quant_error_sq = 0.0
        total_norm_total_error_sq = 0.0

        print(f"Analysiere {len(lora_layers)} LoRA-Layer...")
        for peft_layer_name, peft_layer in lora_layers.items():
            # --- NEUE LOGIK: Korrigiere den Layernamen ---
            # PeftModel fügt Präfixe hinzu. Wir entfernen sie, um den Namen im Originalmodell zu finden.
            # Beispiel: "base_model.model.model.layers.5.mlp.up_proj" -> "model.layers.5.mlp.up_proj"
            original_layer_name = peft_layer_name.replace("base_model.model.", "", 1)

            if original_layer_name not in original_modules:
                print(f"Warnung: Layer '{original_layer_name}' (abgeleitet von '{peft_layer_name}') nicht im true_original_model gefunden. Überspringe.")
                continue
            
            W_original = original_modules[original_layer_name].weight.data.clone().to(torch.float32)
            W_quantized_model = quantized_modules[original_layer_name].dequantize_weight().data.clone().to(torch.float32)
            W_quantized_model = W_quantized_model.T
            R_quant = peft_layer.get_base_layer().dequantize_weight().data.clone().to(torch.float32)
            R_quant = R_quant.T 

            adapter_name = peft_layer.active_adapters[0]
            lora_A = peft_layer.lora_A[adapter_name].weight.data.clone().to(torch.float32)
            lora_B = peft_layer.lora_B[adapter_name].weight.data.clone().to(torch.float32)
            scaling = peft_layer.scaling[adapter_name]
            W_svd = scaling * (lora_B @ lora_A)

            R_true = W_original - W_svd
            W_reconstructed = R_quant + W_svd

            if False:
                rank = peft_layer.r[adapter_name]  # vorhandener LoRA-Rang

                create_and_save_layer_heatmaps(
                    layer_name=peft_layer_name,
                    output_path=visualization_path,
                    w_original_tensor=W_original,
                    w_quantized_model=W_quantized_model,
                    r_true_tensor=R_true,
                    r_quant_tensor=R_quant,
                    rank=rank  # sorgt für automatische W_svd-Berechnung/Visualisierung
                )
                coverage_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

                rank = peft_layer.r[adapter_name]  # falls du die Heatmaps auch willst
                visualize_svd_components(
                    layer_name=peft_layer_name,
                    output_path=svd_visualization_path,
                    w_original_tensor=W_original,
                    rank=rank,                      # oder None, wenn nur die zwei SVD-Plots gewünscht sind
                    coverage_ranks=coverage_ranks
                )

                # visualize_svd_coverage(
                #     layer_name=peft_layer_name,
                #     output_path=svd_visualization_path,
                #     w_original_tensor=W_original,
                #     ranks=coverage_ranks
                # )

            svd_error_tensor = W_original - W_svd
            quant_error_tensor = R_true - R_quant
            total_error_tensor = W_original - W_reconstructed

            total_norm_W_original_sq += torch.linalg.norm(W_original).pow(2).item()
            total_norm_svd_error_sq += torch.linalg.norm(svd_error_tensor).pow(2).item()
            total_norm_quant_error_sq += torch.linalg.norm(quant_error_tensor).pow(2).item()
            total_norm_total_error_sq += torch.linalg.norm(total_error_tensor).pow(2).item()

        if total_norm_W_original_sq == 0:
            return {"svd_error": 0.0, "quant_error": 0.0, "total_error": 0.0}

        norm_W_original_total = total_norm_W_original_sq ** 0.5
        
        error_svd = (total_norm_svd_error_sq ** 0.5) / norm_W_original_total
        error_quant = (total_norm_quant_error_sq ** 0.5) / norm_W_original_total
        error_total = (total_norm_total_error_sq ** 0.5) / norm_W_original_total

        return {
            "svd_error": error_svd,
            "quant_error": error_quant,
            "total_error": error_total,
        }

def create_and_save_layer_heatmaps(
    layer_name: str,
    output_path: str,
    w_original_tensor: torch.Tensor,
    w_quantized_model: torch.Tensor,
    r_true_tensor: torch.Tensor,
    r_quant_tensor: torch.Tensor,
    log_scale_threshold: float = 1e-3,  # Schwellenwert für den linearen Bereich um Null
    w_svd_tensor: Optional[torch.Tensor] = None,
    rank: Optional[int] = None
):
    """
    Erstellt und speichert Heatmaps mit SymLogNorm:
      - W_original
      - W_svd (Top-r SVD-Approx., wenn w_svd_tensor übergeben oder rank gesetzt ist)
      - W_res (wahres Residuum)
      - Q(W_res) (quantisiertes Residuum)
      - Q(W_res) + W_svd (rekonstruierte Matrix)

    Wenn w_svd_tensor fehlt und rank gesetzt ist, wird W_svd aus W_original per
    truncierter SVD auf der CPU berechnet.

    Args:
        layer_name: Layer-Name für Titel/Dateiname.
        output_path: Zielordner.
        w_original_tensor: Original-Gewichtsmatrix (torch.Tensor).
        r_true_tensor: Wahres Residuum.
        r_quant_tensor: Quantisiertes Residuum.
        log_scale_threshold: Linearer Bereich um 0 für SymLogNorm.
        w_svd_tensor: Optional bereits berechnete W_svd-Matrix.
        rank: Optionaler Rang für die SVD-Approximation (nur genutzt, wenn w_svd_tensor None ist).
    """
    # --- Imports sind hier gekapselt ---
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    plt.switch_backend('Agg')
    os.makedirs(output_path, exist_ok=True)

    # --- Optional: W_svd berechnen, falls nicht übergeben ---
    W_svd_np = None
    if w_svd_tensor is None and rank is not None and rank > 0:
        with torch.no_grad():
            W_cpu = w_original_tensor.cpu()
            # Volle SVD (ökonomisch), dann Top-r Rekonstruktion
            U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
            r = min(int(rank), S.shape[0])
            U_r = U[:, :r]
            Vh_r = Vh[:r, :]
            S_r = S[:r].unsqueeze(1)  # (r,1)
            w_svd_tensor = U_r @ (S_r * Vh_r)

    if w_svd_tensor is not None:
        W_svd_np = w_svd_tensor.detach().cpu().numpy()

    # --- Daten für Matplotlib vorbereiten ---
    original_matrix = w_original_tensor.detach().cpu().numpy()
    quantized_matrix = w_quantized_model.detach().cpu().numpy()
    true_residual_matrix = r_true_tensor.detach().cpu().numpy()
    quant_residual_matrix = r_quant_tensor.detach().cpu().numpy()
    reconstructed_matrix_np = None
    if W_svd_np is not None:
        reconstructed_matrix_np = quant_residual_matrix + W_svd_np

    # --- Gemeinsame Skala ermitteln (inkl. aller Matrizen) ---
    candidates = [
        np.abs(original_matrix).max(),
        np.abs(quantized_matrix).max(),
        np.abs(true_residual_matrix).max(),
        np.abs(quant_residual_matrix).max()
    ]
    if W_svd_np is not None:
        candidates.append(np.abs(W_svd_np).max())
    if reconstructed_matrix_np is not None:
        candidates.append(np.abs(reconstructed_matrix_np).max())

    vmax = float(np.max(candidates)) if candidates else 1.0
    vmin = -vmax

    # SymLogNorm für bessere Struktur-Sichtbarkeit
    log_norm = SymLogNorm(linthresh=log_scale_threshold, vmin=vmin, vmax=vmax, base=10)

    # --- Plot erstellen: 2x3 Layout (5 Charts + 1 leer) ---
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    fig.suptitle(f"Analyse der Gewichte & Residuen für Layer: {layer_name} (SymLog-Skala)", fontsize=18)
    cmap = 'coolwarm'

    # 1) W_original
    im0 = axes[0, 0].imshow(original_matrix, cmap=cmap, norm=log_norm)
    axes[0, 0].set_title("1. Originalgewichte (W_original)")
    axes[0, 0].grid(False)

    # 2) W_svd (falls vorhanden)
    if W_svd_np is not None:
        axes[0, 1].imshow(W_svd_np, cmap=cmap, norm=log_norm)
        suffix = f" (Top-{rank} Approx.)" if rank is not None else ""
        axes[0, 1].set_title(f"2. SVD-Approximation W_svd{suffix}")
    else:
        axes[0, 1].imshow(np.zeros_like(original_matrix), cmap=cmap, norm=log_norm)
        axes[0, 1].set_title("2. SVD-Approximation W_svd (nicht verfügbar)")
    axes[0, 1].grid(False)

    # 3) Wahres Residuum
    axes[1, 0].imshow(true_residual_matrix, cmap=cmap, norm=log_norm)
    axes[1, 0].set_title("3. Wahres Residuum (W_res)")
    axes[1, 0].grid(False)

    # 4) Quantisiertes Residuum
    axes[1, 1].imshow(quant_residual_matrix, cmap=cmap, norm=log_norm)
    axes[1, 1].set_title("4. Quantisiertes Residuum Q(W_res)")
    axes[1, 1].grid(False)
    
    # 5) Rekonstruktion Q(W_res) + W_svd
    if reconstructed_matrix_np is not None:
        axes[0, 2].imshow(reconstructed_matrix_np, cmap=cmap, norm=log_norm)
        axes[0, 2].set_title("5. Rekonstruktion (Q(W_res) + W_svd)")
    else:
        axes[0, 2].set_title("5. Rekonstruktion (nicht verfügbar)")
        axes[0, 2].axis('off')  # Ausblenden, wenn nicht berechenbar
    axes[0, 2].grid(False)
    
    if reconstructed_matrix_np is not None:
        axes[1, 2].imshow(quantized_matrix, cmap=cmap, norm=log_norm)
        axes[1, 2].set_title("6. Quantisierte Matrix. Der Konkurrent")
    else:
        axes[1, 2].set_title("6. (Rekonstruktion nicht verfügbar)")
        axes[1, 2].axis('off')  # Ausblenden, wenn nicht berechenbar
    axes[1, 2].grid(False)       
    
    # Leeren 6. Plot ausblenden
    # axes[1, 2].axis('off')

    # Gemeinsame Farbleiste
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.8, label="Gewichtswert")

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])

    # --- Datei speichern ---
    safe_filename = layer_name.replace('.', '_') + "_full_analysis.png"
    full_save_path = os.path.join(output_path, safe_filename)
    plt.savefig(full_save_path, dpi=150)
    plt.close(fig)

def visualize_svd_coverage(
    layer_name: str,
    output_path: str,
    w_original_tensor: torch.Tensor,
    ranks: list[int]
):
    """
    Visualisiert die SVD-Energieabdeckung (kumulative Energie) für mehrere Ranks.
    Speichert eine PNG-Grafik und eine CSV-Tabelle mit den %-Werten.

    Args:
        layer_name: Layername (für Titel/Dateiname)
        output_path: Zielordner
        w_original_tensor: Gewichts-Tensor (torch.Tensor)
        ranks: Liste von Ranks (z.B. [1,2,4,8,16,32,64,128,256,512])
    """
    # --- Imports lokal halten ---
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.switch_backend('Agg')
    os.makedirs(output_path, exist_ok=True)

    # Nur die Singulärwerte berechnen (billiger als volle SVD)
    W_cpu = w_original_tensor.cpu()
    s = torch.linalg.svdvals(W_cpu)  # 1D: absteigend sortiert
    s_np = s.numpy()

    s_squared = s_np ** 2
    total_energy = float(np.sum(s_squared))
    if total_energy <= 0.0:
        total_energy = 1e-12
    cum_energy = np.cumsum(s_squared) / total_energy  # [0..1]

    # Valide Ränge beschränken
    max_rank = s_np.shape[0]
    ranks = [int(r) for r in ranks if 1 <= int(r) <= max_rank]
    if not ranks:
        return

    # Prozentwerte je Rank sammeln
    coverage_pct = []
    for r in ranks:
        coverage_pct.append(float(cum_energy[r - 1] * 100.0))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, max_rank + 1), cum_energy * 100.0, color='gray', alpha=0.5, label='Cumulative energy')
    ax.scatter(ranks, coverage_pct, color='tab:blue', label='Requested ranks')
    for r, p in zip(ranks, coverage_pct):
        ax.annotate(f"{p:.1f}%", (r, p), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_title(f"SVD Coverage vs. Rank: {layer_name}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Coverage (%)")
    ax.set_xscale("log")  # log hilft die Liste RANKS=(1,2,4,8,...) zu visualisieren
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

    safe_prefix = layer_name.replace('.', '_')
    png_path = os.path.join(output_path, f"{safe_prefix}_svd_coverage.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close(fig)

    # CSV schreiben
    import csv
    csv_path = os.path.join(output_path, f"{safe_prefix}_svd_coverage.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "coverage_percent"])
        for r, p in zip(ranks, coverage_pct):
            w.writerow([r, f"{p:.6f}"])
    
    # Konsolen-Log
    print(f"SVD coverage for {layer_name}: " + ", ".join([f"r={r}:{p:.2f}%" for r, p in zip(ranks, coverage_pct)]))

def visualize_svd_components(
    layer_name: str,
    output_path: str,
    w_original_tensor: torch.Tensor,
    rank: Optional[int] = None,
    coverage_ranks: list[int] = None
):
    """
    Visualisiert SVD als 4-Chart-Layout:
      Zeile 1: Singularwerte (log) über beide Spalten, farbige Marker für alle coverage_ranks.
      Zeile 2: Singularwerte (linear) über beide Spalten, farbige Marker für alle coverage_ranks.
      Zeile 3: Heatmaps U_r (links) und Vh_r (rechts), nur wenn rank gesetzt ist.
    In der Legende steht für jeden Rank der Coverage-%-Wert.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.switch_backend('Agg')
    os.makedirs(output_path, exist_ok=True)

    if coverage_ranks is None:
        coverage_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # SVD (CPU)
    W_cpu = w_original_tensor.cpu()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    s_np = S.numpy()
    n_sv = s_np.shape[0]
    idx = np.arange(n_sv)

    # Coverage
    s2 = s_np ** 2
    tot = float(np.sum(s2)) or 1e-12
    cum = np.cumsum(s2) / tot

    cov_ranks = [int(r) for r in coverage_ranks if 1 <= int(r) <= n_sv]
    cov_pcts = [float(cum[r - 1] * 100.0) for r in cov_ranks]

    # Farben
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(cov_ranks))]

    def plot_singular(ax, yscale: Optional[str], title: str):
        # Grundbalken
        ax.bar(idx, s_np, color='lightgray', label=f'Alle SVs (n={n_sv})')
        # Optional: Top-r hervorheben
        if rank is not None and rank >= 1:
            ax.bar(idx[:rank], s_np[:rank], color='cornflowerblue', alpha=0.7, label=f'Behaltene SVs (≤ {rank})')
        # RANKS farbig markieren
        for (r, p, c) in zip(cov_ranks, cov_pcts, colors):
            ax.axvline(x=r - 0.5, color=c, linestyle='--', linewidth=1.8)
            ax.plot(r - 1, s_np[r - 1], marker='o', color=c, markersize=5,
                    label=f"r={r}: {p:.1f}%")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Index des Singulärwerts")
        ax.set_ylabel("Singulärwerte" + (" (log)" if yscale == "log" else ""))
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(ncol=4, fontsize=8)

    # Layout: 3 Zeilen, 2 Spalten
    # Row 0: SVD (log) span beide Spalten
    # Row 1: SVD (linear) span beide Spalten
    # Row 2: Heatmaps U_r (links) & Vh_r (rechts) (nur wenn rank gesetzt)
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.2, 1.2, 1.0]
    )

    # Zeile 0: SVD log
    ax_log = fig.add_subplot(gs[0, :])
    plot_singular(ax_log, "log", f"SVD-Analyse (log) – {layer_name}")

    # Zeile 1: SVD linear
    ax_lin = fig.add_subplot(gs[1, :])
    plot_singular(ax_lin, None, "SVD-Analyse (linear)")

    # Zeile 2: Heatmaps, nur wenn rank vorhanden
    if rank is not None and rank >= 1:
        use_r = min(rank, n_sv)
        U_r = U[:, :use_r].numpy()
        Vh_r = Vh[:use_r, :].numpy()

        ax_u = fig.add_subplot(gs[2, 0])
        im_u = ax_u.imshow(U_r, cmap='viridis', aspect='auto')
        ax_u.set_title(f"Linke Vektoren (U_r), Shape: {U_r.shape}")
        ax_u.set_xlabel("Rang-Dimension")
        ax_u.set_ylabel("Original-Dimension")
        fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.02)

        ax_vh = fig.add_subplot(gs[2, 1])
        im_v = ax_vh.imshow(Vh_r, cmap='viridis', aspect='auto')
        ax_vh.set_title(f"Rechte Vektoren (Vh_r), Shape: {Vh_r.shape}")
        ax_vh.set_xlabel("Original-Dimension")
        ax_vh.set_ylabel("Rang-Dimension")
        fig.colorbar(im_v, ax=ax_vh, fraction=0.046, pad=0.02)

        kept_pct = float(np.sum(s2[:use_r]) / tot * 100.0)
        supt = (f"SVD-Komponenten & Coverage – {layer_name}\n"
                f"Top-{use_r} SVs erfassen {kept_pct:.2f}% der Energie – "
                f"alle RANKS markiert")
    else:
        supt = (f"SVD & Coverage – {layer_name} (ohne spezifischen Rank)\n"
                f"Max Rank Marker: r={max(cov_ranks) if cov_ranks else '-'} → "
                f"{(cov_pcts[cov_ranks.index(max(cov_ranks))] if cov_ranks else 0.0):.2f}%")

    fig.suptitle(supt, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Speichern
    out = os.path.join(
        output_path,
        layer_name.replace('.', '_') + "_svd_4charts_with_coverage.png"
    )
    plt.savefig(out, dpi=150)
    plt.close(fig)


    
def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-quantized residual connection models")
    
    # Input configuration
    parser.add_argument("--residual_models_dir", type=str, default="/home/nudel/Documents/peft/train_results_debugger/quantized_residuals", help="Directory containing quantized residual models")
    
    # Evaluation configuration
    parser.add_argument("--tasks", type=str, default="arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande", help="Evaluation tasks")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Evaluation batch size")
    
    # Filtering options
    parser.add_argument("--ranks", type=str, help="Comma-separated ranks to evaluate (e.g., '256,512')")
    parser.add_argument("--bits", type=str, help="Comma-separated bits to evaluate (e.g., '2,3,4')")
    
    # Output configuration  
    parser.add_argument("--output_dir", type=str, default="./residual_eval_results", help="Output directory for results")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
    parser.add_argument("--generate_latex", action="store_true", help="Generate LaTeX table")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover models
    print(f"🔍 Discovering models in: {args.residual_models_dir}")
    all_models = discover_residual_models(args.residual_models_dir)
    
    # Filter models if specified
    if args.ranks:
        target_ranks = [int(r.strip()) for r in args.ranks.split(",")]
        all_models = [m for m in all_models if m["rank"] in target_ranks]
    
    if args.bits:
        target_bits = [int(b.strip()) for b in args.bits.split(",")]
        all_models = [m for m in all_models if m["bits"] in target_bits]
    
    print(f"📊 Found {len(all_models)} models to evaluate:")
    for model in all_models:
        print(f"   - {os.path.basename(model['path'])}: r={model['rank']}, {model['bits']}-bit, gs={model.get('group_size', 'N/A')}")
    
    if not all_models:
        print("❌ No models found to evaluate")
        return
    
    # Run evaluations
    print(f"\n🚀 Starting evaluation of {len(all_models)} models...")
    all_results = []
    for i, model_info in enumerate(all_models, 1):
        print(f"\n{'='*60}")
        print(f"🏁 EVALUATING MODEL {i}/{len(all_models)}")
        print(f"{'='*60}")
        
        result = evaluate_residual_model(
            model_info=model_info,
            base_dir=args.residual_models_dir,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            per_device_eval_batch_size=args.per_device_eval_batch_size
        )
        
        all_results.append(result)
        
        if result["status"] == "success":
            print(f"✅ Successfully evaluated {os.path.basename(model_info['path'])}")
        else:
            print(f"❌ Failed to evaluate {os.path.basename(model_info['path'])}")
    
    # Prepare summary
    successful_results = [r for r in all_results if r["status"] == "success"]
    failed_results = [r for r in all_results if r["status"] == "failed"]
    
    print(f"\n📊 EVALUATION SUMMARY:")
    print(f"   ✅ Successful: {len(successful_results)}")
    print(f"   ❌ Failed: {len(failed_results)}")
    
    # Save results
    if args.save_results and successful_results:
        results_file = os.path.join(args.output_dir, "residual_evaluation_results.json")
        
        summary_data = {
            "summary": {
                "total_models": len(all_models),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results),
                "tasks_evaluated": args.tasks,
                "num_fewshot": args.num_fewshot
            },
            "results": all_results
        }
        
        with open(results_file, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
    
    # Generate LaTeX table
    if args.generate_latex and successful_results:
        print("\n📄 Generating LaTeX table...")
        create_residual_performance_table(successful_results, args.output_dir)
    
    print("\n✅ Residual model evaluation completed!")


if __name__ == "__main__":
    main()