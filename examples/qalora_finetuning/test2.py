import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm
from datasets import load_dataset

# =================================================================================
# 1. DATA LOADING UTILITY
# =================================================================================
def get_c4_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """
    Loads the C4 dataset from Hugging Face and prepares tokenized samples for calibration.
    """
    print(" Lade C4 Kalibrierungsdatensatz...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    samples = []
    for row in dataset:
        if len(samples) == n_samples:
            break
        text = row['text']
        tokens = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
        if tokens.input_ids.shape[1] == seq_len:
            samples.append(tokens)
    
    print(f"✅ {len(samples)} Kalibrierungssamples geladen.")
    return samples

# =================================================================================
# 2. HESSIAN VS. MAGNITUDE ANALYSIS CLASS (Unchanged)
# =================================================================================
class HessianMagnitudeAnalyser:
    def __init__(self, model, tokenizer, top_percentage=0.1, n_samples=128, seq_len=512, calibration_data=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.top_percentage = top_percentage
        
        if calibration_data is None:
            self.calibration_data = get_c4_calibration_data(tokenizer, n_samples, seq_len)
        else:
            self.calibration_data = calibration_data
        
        self.output_dir = f"hessian_magnitude_analysis_{self.top_percentage}pct"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f" Speichere Ergebnisse in '{self.output_dir}'")

    def _calculate_hessian_for_layer(self, layer_module):
        columns = layer_module.weight.shape[1]
        H = torch.zeros((columns, columns), device='cpu', dtype=torch.float32)
        nsamples = 0

        captured_inputs = []
        def hook(_, inp, __):
            captured_inputs.append(inp[0].detach().cpu())
        
        handle = layer_module.register_forward_hook(hook)

        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_data:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)
        
        handle.remove()

        for inp in captured_inputs:
            reshaped_inp = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
            batch_size = reshaped_inp.shape[0]
            
            H *= nsamples / (nsamples + batch_size)
            nsamples += batch_size
            H += (2 / nsamples) * (reshaped_inp.T @ reshaped_inp)
        
        return H.to(self.device)

    def run(self, max_layers=None):
        linear_layers = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
        
        if max_layers is not None:
            linear_layers = linear_layers[:max_layers]

        for name, module in linear_layers:
            print("\n" + "="*80)
            print(f" Analysiere Layer: {name} ({module.weight.shape})")
            print("="*80)

            print(" Berechne Hessian-Matrix...")
            H = self._calculate_hessian_for_layer(module)
            hessian_diag = torch.diag(H)
            weights_flat = module.weight.data.flatten()
            magnitudes = torch.abs(weights_flat)
            
            rows, cols = module.weight.shape
            hessian_diag_reshaped = hessian_diag.unsqueeze(0)
            hessian_importance_matrix = hessian_diag_reshaped.repeat(rows, 1)
            hessian_importance_flat = hessian_importance_matrix.flatten()

            if magnitudes.numel() != hessian_importance_flat.numel():
                raise RuntimeError(f"FATAL: Dimension Mismatch in layer {name}.")

            num_params = magnitudes.numel()
            k = max(1, int(num_params * self.top_percentage / 100.0))

            top_k_mag_indices = torch.topk(magnitudes, k=k).indices
            top_k_hess_indices = torch.topk(hessian_importance_flat, k=k).indices
            
            set_mag = set(top_k_mag_indices.tolist())
            set_hess = set(top_k_hess_indices.tolist())
            intersection_indices = torch.tensor(list(set_mag.intersection(set_hess)), device=self.device, dtype=torch.long)
            
            print(f"  - Top-{self.top_percentage}% ({k}) Magnitude-Parameter: {len(set_mag)}")
            print(f"  - Top-{self.top_percentage}% ({k}) Hessian-Parameter: {len(set_hess)}")
            print(f"  - Schnittmenge: {intersection_indices.numel()} Parameter")
            # Plots and JSON saving can be added back here if needed

# =================================================================================
# 3. NEW SVD IMPACT ANALYSIS BASED ON HESSIAN
# =================================================================================
def analyze_svd_impact_on_hessian_removal(model, tokenizer, calibration_data, percentages_to_remove=[0.1, 1.0]):
    """
    Analyzes the impact on a matrix's singular values when a percentage of its
    top HESSIAN-IMPORTANT weights are set to zero.
    """
    print("\n" + "="*80)
    print(f" SVD Impact Analysis: Comparing singular value changes after removing")
    print(f" top {percentages_to_remove}% HESSIAN-IMPORTANT weights.")
    print("="*80)

    results = {p: {} for p in percentages_to_remove}
    linear_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    # Helper function to calculate Hessian for a single layer
    def _calculate_hessian_for_layer(layer_module):
        columns = layer_module.weight.shape[1]
        H = torch.zeros((columns, columns), device='cpu', dtype=torch.float32)
        nsamples = 0
        captured_inputs = []
        def hook(_, inp, __):
            captured_inputs.append(inp[0].detach().cpu())
        handle = layer_module.register_forward_hook(hook)
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                model(**batch)
        handle.remove()
        for inp in captured_inputs:
            reshaped_inp = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
            batch_size = reshaped_inp.shape[0]
            H *= nsamples / (nsamples + batch_size)
            nsamples += batch_size
            H += (2 / nsamples) * (reshaped_inp.T @ reshaped_inp)
        return H.to(model.device)

    for name, module in tqdm(linear_layers, desc="Analyzing Layers for SVD (Hessian-based)"):
        W_orig = module.weight.data.clone().float()
        
        # Calculate Hessian importance for the current layer
        H = _calculate_hessian_for_layer(module)
        hessian_diag = torch.diag(H)
        rows, cols = W_orig.shape
        hessian_diag_reshaped = hessian_diag.unsqueeze(0)
        hessian_importance_matrix = hessian_diag_reshaped.repeat(rows, 1)
        hessian_importance_flat = hessian_importance_matrix.flatten()

        _, S_orig, _ = torch.linalg.svd(W_orig, full_matrices=False)

        for p in percentages_to_remove:
            k = int(W_orig.numel() * p / 100.0)
            if k == 0:
                results[p][name] = 0.0
                continue

            # Find top k indices based on HESSIAN, not magnitude
            _, top_k_indices = torch.topk(hessian_importance_flat, k=k)

            W_mod_flat = W_orig.flatten().clone()
            W_mod_flat[top_k_indices] = 0.0
            W_mod = W_mod_flat.reshape(W_orig.shape)

            _, S_mod, _ = torch.linalg.svd(W_mod, full_matrices=False)
            
            min_len = min(len(S_orig), len(S_mod))
            S_orig_trunc, S_mod_trunc = S_orig[:min_len], S_mod[:min_len]

            svd_error = torch.linalg.norm(S_orig_trunc - S_mod_trunc) / (torch.linalg.norm(S_orig_trunc) + 1e-9)
            results[p][name] = svd_error.item() * 100

    print("\n--- SVD Impact Summary (Hessian-based Removal) ---")
    header = f"{'Layer Name':<45} |"
    for p in percentages_to_remove:
        header += f" Change at {p}% (%) |"
    print(header)
    print("-" * len(header))

    last_p = percentages_to_remove[-1]
    sorted_layers = sorted(results[last_p].keys(), key=lambda name: results[last_p][name], reverse=True)

    for name in sorted_layers:
        row = f"{name:<45} |"
        for p in percentages_to_remove:
            error = results[p].get(name, 0.0)
            row += f" {error:>15.4f} |"
        print(row)
    print("-" * len(header))
    
    return results

# =================================================================================
# 4. MAIN EXECUTION BLOCK
# =================================================================================
if __name__ == '__main__':
    # --- Configuration ---
    model_name = "TinyLlama/TinyLlama_v1.1"
    num_calibration_samples = 32  # Reduced for faster SVD analysis
    max_layers_to_analyze_hessian = 0 # Set to 0 to skip the first detailed analysis if desired
    percentages_for_svd = [0.1, 1.0]

    # --- Load Model and Tokenizer ---
    print(f"Lade Modell und Tokenizer '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # --- Pre-load calibration data once ---
    calibration_data = get_c4_calibration_data(tokenizer, n_samples=num_calibration_samples)

    # # --- (Optional) Run the original Hessian vs. Magnitude Analysis ---
    # if max_layers_to_analyze_hessian > 0:
    #     for p in [0.1, 1.0]:
    #         print("\n" + "#"*80)
    #         print(f"# Running Hessian vs. Magnitude Analysis for Top {p}%")
    #         print("#"*80)
    #         analyser = HessianMagnitudeAnalyser(
    #             model, 
    #             tokenizer, 
    #             top_percentage=p, 
    #             calibration_data=calibration_data
    #         )
    #         analyser.run(max_layers=max_layers_to_analyze_hessian)
    #     print("\n🎉 Hessian vs. Magnitude Analyse abgeschlossen.")

    # --- Run the NEW SVD Impact Analysis based on Hessian importance ---
    analyze_svd_impact_on_hessian_removal(
        model, 
        tokenizer, 
        calibration_data, 
        percentages_to_remove=percentages_for_svd
    )
    print("\n🎉 Hessian-based SVD Impact Analyse abgeschlossen.")
