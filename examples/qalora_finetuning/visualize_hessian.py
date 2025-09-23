import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset

# =================================================================================
# 1. KONFIGURATION
# =================================================================================
MODEL_NAME = "TinyLlama/TinyLlama_v1.1"
OUTLIER_PERCENTAGE = 0.1  # Wie viel Prozent der Gewichte als Outlier gelten sollen
N_CALIBRATION_SAMPLES = 128 # Anzahl der Samples zur Berechnung des Hessians
SEQ_LEN = 512
MAX_LAYERS_TO_PLOT = 10 # Begrenzt die Anzahl der erstellten Plots, um Zeit zu sparen. 0 für alle.
POOLING_KERNEL_SIZE = 16 # Größe des Max-Pooling-Fensters für die Heatmap

# =================================================================================
# 2. HILFSFUNKTIONEN (Adaptiert aus Ihren Skripten)
# =================================================================================

def get_c4_calibration_data(tokenizer, n_samples, seq_len):
    """Lädt und tokenisiert Kalibrierungsdaten aus dem C4-Datensatz."""
    print(f"Lade {n_samples} Kalibrierungssamples aus 'allenai/c4'...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    samples = []
    pbar = tqdm(total=n_samples, desc="  -> Samples sammeln")
    for row in dataset:
        if len(samples) == n_samples:
            break
        text = row['text']
        tokens = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
        if tokens.input_ids.shape[1] == seq_len:
            samples.append(tokens)
            pbar.update(1)
    pbar.close()
    
    if not samples:
        raise ValueError("Konnte keine Kalibrierungsdaten mit der geforderten Sequenzlänge laden.")
    
    print(f"✅ {len(samples)} Kalibrierungssamples geladen.")
    return samples

def calculate_hessian_saliency_for_layer(model, layer_module, calibration_data):
    """Berechnet die Diagonale der Hessian-Matrix für einen einzelnen Layer."""
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
            try:
                model(**batch)
            except Exception as e:
                print(f"Warnung: Fehler bei einem Forward-Pass während der Kalibrierung: {e}")
                continue
    
    handle.remove()

    for inp in captured_inputs:
        reshaped_inp = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
        batch_size = reshaped_inp.shape[0]
        
        H *= nsamples / (nsamples + batch_size)
        nsamples += batch_size
        H += (2 / nsamples) * (reshaped_inp.T @ reshaped_inp)
    
    hessian_diag = torch.diag(H)
    rows, _ = layer_module.weight.shape
    hessian_saliency_matrix = hessian_diag.unsqueeze(0).expand(rows, -1)
    
    return hessian_saliency_matrix.to(model.device)


# =================================================================================
# 3. VISUALISIERUNGS-FUNKTIONEN
# =================================================================================

def visualize_matrix_heatmap(magnitudes, outlier_indices, layer_name, output_dir):
    """
    Visualisiert die Gewichtsmatrix als gepoolte Heatmap und hebt Outlier-Regionen hervor.
    """
    rows, cols = magnitudes.shape
    
    # Erstelle eine 2D-Maske, die die Positionen der Outlier markiert
    outlier_mask = torch.zeros_like(magnitudes, dtype=torch.float32)
    flat_mask = outlier_mask.flatten()
    flat_mask[outlier_indices] = 1.0
    outlier_mask_2d = flat_mask.reshape(rows, cols)

    # Bereite die Tensoren für das Pooling vor (N, C, H, W)
    magnitudes_4d = magnitudes.unsqueeze(0).unsqueeze(0).float()
    outlier_mask_4d = outlier_mask_2d.unsqueeze(0).unsqueeze(0)

    # Wende Max-Pooling an, um die Matrizen zu verkleinern
    pooled_magnitudes = F.max_pool2d(magnitudes_4d, kernel_size=POOLING_KERNEL_SIZE)
    # Pooling auf der Maske: Wenn ein Wert > 0 ist, war mindestens ein Outlier im Pool-Fenster
    pooled_outliers = F.max_pool2d(outlier_mask_4d, kernel_size=POOLING_KERNEL_SIZE)

    # Konvertiere zu NumPy-Arrays für Matplotlib
    pooled_magnitudes_np = pooled_magnitudes.squeeze().cpu().numpy()
    pooled_outliers_np = pooled_outliers.squeeze().cpu().numpy()

    # Erstelle den Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Zeige die gepoolten Magnituden als Hintergrund-Heatmap
    im = ax.imshow(pooled_magnitudes_np, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, label=f'Max Magnitude in {POOLING_KERNEL_SIZE}x{POOLING_KERNEL_SIZE} Block')

    # Erstelle eine rote Maske für die Outlier-Regionen
    # Wo pooled_outliers_np == 0 ist, wird die Zelle transparent
    outlier_heatmap = np.ma.masked_where(pooled_outliers_np == 0, pooled_outliers_np)
    ax.imshow(outlier_heatmap, cmap='Reds', aspect='auto', interpolation='nearest', alpha=0.7)

    ax.set_title(f"Heatmap of Weight Matrix for Layer: {layer_name}\n(Red blocks contain top {OUTLIER_PERCENTAGE}% Hessian outliers)")
    ax.set_xlabel(f"Input Features (pooled by {POOLING_KERNEL_SIZE})")
    ax.set_ylabel(f"Output Features (pooled by {POOLING_KERNEL_SIZE})")

    # Speichere die Figur
    plot_filename = os.path.join(output_dir, f"{layer_name.replace('.', '_')}_heatmap.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_layer_outliers(model, tokenizer, calibration_data, outlier_percentage):
    """
    Analysiert lineare Layer, berechnet Hessian-Wichtigkeit und visualisiert die Ergebnisse.
    """
    output_dir = f"hessian_visualization_{outlier_percentage}pct"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 Starte Analyse. Plots werden in '{output_dir}' gespeichert.")

    linear_layers = [
        (name, module) for name, module in model.named_modules() 
        if isinstance(module, nn.Linear) and "lm_head" not in name
    ]
    
    if MAX_LAYERS_TO_PLOT > 0:
        print(f"ℹ️  Begrenze Analyse auf die ersten {MAX_LAYERS_TO_PLOT} Layer.")
        linear_layers = linear_layers[:MAX_LAYERS_TO_PLOT]

    for name, module in tqdm(linear_layers, desc="Analysiere Layer"):
        
        hessian_saliency = calculate_hessian_saliency_for_layer(model, module, calibration_data)
        magnitudes = torch.abs(module.weight.data)
        
        hessian_flat = hessian_saliency.flatten().cpu().numpy()
        magnitudes_flat = magnitudes.flatten().cpu().to(torch.float32).numpy()
        
        num_params = len(hessian_flat)        
        k = max(1, int(num_params * outlier_percentage / 100.0))
        
        outlier_indices = np.argpartition(hessian_flat, -k)[-k:]
        
        # --- PLOT 1: Scatter Plot (wie bisher) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(magnitudes_flat, hessian_flat, alpha=0.2, s=5, label=f"Andere Gewichte ({100-outlier_percentage}%)", color='cornflowerblue')
        ax.scatter(magnitudes_flat[outlier_indices], hessian_flat[outlier_indices], alpha=0.8, s=15, label=f"Top {outlier_percentage}% Hessian Outlier", color='red')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Absolute Weight Magnitude (log scale)")
        ax.set_ylabel("Hessian Saliency (log scale)")
        ax.set_title(f"Hessian Outlier vs. Magnitude in Layer: {name}")
        ax.legend()
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        plot_filename = os.path.join(output_dir, f"{name.replace('.', '_')}_scatter.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # --- PLOT 2: Matrix Heatmap (NEU) ---
        visualize_matrix_heatmap(magnitudes, torch.from_numpy(outlier_indices).to(magnitudes.device), name, output_dir)

    print(f"\n🎉 Analyse abgeschlossen. {len(linear_layers)*2} Plots wurden in '{output_dir}' gespeichert.")


# =================================================================================
# 4. HAUPT-AUSFÜHRUNG
# =================================================================================

if __name__ == '__main__':
    print(f"Lade Modell und Tokenizer für '{MODEL_NAME}'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    calibration_data = get_c4_calibration_data(
        tokenizer, 
        n_samples=N_CALIBRATION_SAMPLES, 
        seq_len=SEQ_LEN
    )

    visualize_layer_outliers(
        model, 
        tokenizer, 
        calibration_data, 
        outlier_percentage=OUTLIER_PERCENTAGE
    )