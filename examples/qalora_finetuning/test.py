import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm
from datasets import load_dataset

# Die Funktion get_c4_calibration_data bleibt unverändert.
def get_c4_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """
    Lädt den C4-Datensatz und bereitet Kalibrierungsdaten vor.
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


class HessianMagnitudeAnalyser:
    # NEU: 'std_threshold' wurde durch 'top_k' ersetzt.
    def __init__(self, model, tokenizer, top_k=100, n_samples=128, seq_len=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.top_k = top_k
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.calibration_data = get_c4_calibration_data(tokenizer, n_samples, seq_len)
        
        self.output_dir = "hessian_magnitude_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f" Speichere Ergebnisse in '{self.output_dir}'")

    def _calculate_hessian(self, layer_module):
        """
        Berechnet die Hessian-Approximation (H = 2 * X^T * X) für einen gegebenen Layer.
        (Diese Funktion bleibt unverändert)
        """
        columns = layer_module.weight.shape[1]
        H = torch.zeros((columns, columns), device='cpu', dtype=torch.float32)
        nsamples = 0

        captured_inputs = []
        def hook(_, inp, __):
            captured_inputs.append(inp[0].detach().cpu())
        
        handle = layer_module.register_forward_hook(hook)

        print(" Sammle Layer-Inputs über Kalibrierungsdaten...")
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.calibration_data, desc="Kalibrierung"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)
        
        handle.remove()

        print(" Berechne Hessian-Matrix...")
        for inp in tqdm(captured_inputs, desc="Hessian-Berechnung"):
            reshaped_inp = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
            batch_size = reshaped_inp.shape[0]
            
            H *= nsamples / (nsamples + batch_size)
            nsamples += batch_size
            H += (2 / nsamples) * (reshaped_inp.T @ reshaped_inp)
        
        return H.to(self.device)

    # NEU: Die Plot-Funktion wurde angepasst, um mit den Top-K Indizes zu arbeiten
    def _create_plots(self, layer_name, weights_flat, hessian_diag, top_k_mag_indices, top_k_hess_indices, intersection_indices):
        """Erstellt und speichert die 4-Panel-Visualisierung basierend auf Top-K."""
        fig, axs = plt.subplots(2, 2, figsize=(20, 18))
        fig.suptitle(f"Hessian vs. Magnitude Analyse (Top-{self.top_k}) für Layer: {layer_name}", fontsize=20)
        
        magnitudes = torch.abs(weights_flat).float()
        
        # Schwellenwerte basierend auf dem k-ten Element bestimmen
        mag_threshold = magnitudes[top_k_mag_indices[-1]]
        hess_threshold = hessian_diag[top_k_hess_indices[-1]]
        
        # --- Plot 1: Hessian Verteilung ---
        ax = axs[0, 0]
        hessian_log = torch.log10(hessian_diag.float() + 1e-9).cpu()
        hess_threshold_log = torch.log10(hess_threshold.float() + 1e-9).cpu()
        ax.hist(hessian_log, bins=100, color='gray', label="Alle Hessian-Werte")
        ax.axvline(hess_threshold_log, color='blue', linestyle='--', label=f'Top-{self.top_k} Schwelle')
        ax.set_title("1. Verteilung der Hessian-Diagonalwerte (log10)", fontsize=14)
        ax.set_xlabel("log10(Hessian-Wert)", fontsize=12)
        ax.set_ylabel("Häufigkeit", fontsize=12)
        ax.legend()

        # --- Plot 2: Magnitude Verteilung ---
        ax = axs[0, 1]
        ax.hist(magnitudes.cpu().numpy(), bins=100, color='gray', label="Alle Magnituden")
        ax.axvline(mag_threshold.cpu().numpy(), color='red', linestyle='--', label=f'Top-{self.top_k} Schwelle')
        ax.set_title("2. Verteilung der Gewichts-Magnituden", fontsize=14)
        ax.set_xlabel("Absoluter Gewichtswert", fontsize=12)
        ax.legend()
        
        # --- Plot 3: Scatter Plot ---
        ax = axs[1, 0]
        magnitudes_cpu = magnitudes.cpu()
        hessian_diag_cpu = hessian_diag.float().cpu()
        colors = np.full(magnitudes_cpu.shape, '#cccccc', dtype=object) # hellgrau
        colors[top_k_mag_indices.cpu().numpy()] = 'red'
        colors[top_k_hess_indices.cpu().numpy()] = 'blue'
        colors[intersection_indices.cpu().numpy()] = 'magenta'
        
        ax.scatter(magnitudes_cpu, hessian_diag_cpu, c=colors, alpha=0.6, s=10)
        ax.scatter([], [], c='red', label=f'Top-{self.top_k} Magnitude ({top_k_mag_indices.numel()})')
        ax.scatter([], [], c='blue', label=f'Top-{self.top_k} Hessian ({top_k_hess_indices.numel()})')
        ax.scatter([], [], c='magenta', label=f'Schnittmenge ({intersection_indices.numel()})')
        
        ax.set_title("3. Hessian vs. Magnitude", fontsize=14)
        ax.set_xlabel("Gewichts-Magnitude", fontsize=12)
        ax.set_ylabel("Hessian-Diagonalwert", fontsize=12)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)

        # --- Plot 4: Heatmap ---
        ax = axs[1, 1]
        heatmap_data = torch.abs(self.model.get_submodule(layer_name).weight.data).float().cpu()
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        ax.set_title("4. Heatmap der Gewichts-Magnituden", fontsize=14)
        fig.colorbar(im, ax=ax, label="Absoluter Gewichtswert")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(self.output_dir, f"{layer_name.replace('.', '_')}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f" Plot gespeichert unter: {plot_path}")

    def _save_top_parameters(self, layer_name, weights_flat, hessian_diag, intersection_indices):
        """Speichert die Top-Parameterlisten als JSON."""
        
        # Die Logik hier ist bereits "Top-K", also passt sie perfekt.
        top_k_hess_indices = torch.topk(hessian_diag, k=min(self.top_k, hessian_diag.numel())).indices
        
        intersection_hessian_vals = hessian_diag[intersection_indices]
        sorted_intersect_indices = intersection_indices[torch.argsort(intersection_hessian_vals, descending=True)]
        
        def get_param_info(indices):
            info = []
            rows, cols = self.model.get_submodule(layer_name).weight.shape
            for idx in indices:
                row = idx // cols
                col = idx % cols
                info.append({
                    "layer": layer_name,
                    "row": row.item(),
                    "col": col.item(),
                    "weight": weights_flat[idx].item(),
                    "hessian_diag": hessian_diag[idx].item()
                })
            return info

        results = {
            f"info_a": f"Die {self.top_k} Parameter mit den absolut höchsten Hessian-Werten.",
            f"top_{self.top_k}_hessian": get_param_info(top_k_hess_indices),
            f"info_b": f"Die Top Parameter aus der Schnittmenge von Top-{self.top_k} Magnitude und Top-{self.top_k} Hessian, sortiert nach Hessian-Wert.",
            "important_outliers_intersection": get_param_info(sorted_intersect_indices)
        }
        
        json_path = os.path.join(self.output_dir, f"{layer_name.replace('.', '_')}_params.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f" Parameterlisten gespeichert unter: {json_path}")

    def run(self, max_layers=None):
        """Führt die Analyse für alle linearen Layer im Modell durch."""
        linear_layers = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]
        
        if max_layers is not None:
            linear_layers = linear_layers[:max_layers]

        for name, module in linear_layers:
            print("\n" + "="*80)
            print(f" Analysiere Layer: {name} ({module.weight.shape})")
            print("="*80)

            H = self._calculate_hessian(module)
            hessian_diag = torch.diag(H)
            weights_flat = module.weight.data.flatten()
            magnitudes = torch.abs(weights_flat)
            
            # --- KORREKTUR HIER ---
            # Wir ersetzen die .expand() Methode durch eine robustere Variante mit .repeat()
            rows, cols = module.weight.shape
            
            # 1. Füge eine neue Dimension hinzu: (in_features) -> (1, in_features)
            hessian_diag_reshaped = hessian_diag.unsqueeze(0)
            # 2. Wiederhole diesen Vektor 'rows' mal: (1, in_features) -> (out_features, in_features)
            hessian_importance_matrix = hessian_diag_reshaped.repeat(rows, 1)
            
            # 3. Flache diese Matrix, damit sie die gleiche Dimension wie 'magnitudes' hat.
            hessian_importance_flat = hessian_importance_matrix.flatten()

            # Diese Überprüfung sollte nun sicherstellen, dass alles passt.
            if magnitudes.numel() != hessian_importance_flat.numel():
                raise RuntimeError(f"FATAL: Dimension Mismatch in layer {name}. Magnitudes: {magnitudes.numel()}, Hessian: {hessian_importance_flat.numel()}")
            
            # Finde die Top-K Indizes auf den korrekt dimensionierten Tensoren
            top_k_mag_indices = torch.topk(magnitudes, k=min(self.top_k, len(magnitudes))).indices
            top_k_hess_indices = torch.topk(hessian_importance_flat, k=min(self.top_k, len(hessian_importance_flat))).indices
            
            # Finde die Schnittmenge der beiden Top-K Listen
            set_mag = set(top_k_mag_indices.tolist())
            set_hess = set(top_k_hess_indices.tolist())
            intersection_indices = torch.tensor(list(set_mag.intersection(set_hess)), device=self.device, dtype=torch.long)
            
            print(f"  - Top-{self.top_k} Magnitude-Parameter: {len(set_mag)}")
            print(f"  - Top-{self.top_k} Hessian-Parameter: {len(set_hess)}")
            print(f"  - Schnittmenge: {intersection_indices.numel()} Parameter")
            
            # Übergebe den korrekt geformten Hessian-Tensor an die Plot- und Speicher-Funktionen
            self._create_plots(name, weights_flat, hessian_importance_flat, top_k_mag_indices, top_k_hess_indices, intersection_indices)
            self._save_top_parameters(name, weights_flat, hessian_importance_flat, intersection_indices)


if __name__ == '__main__':
    model_name = "TinyLlama/TinyLlama_v1.1"
    
    print(f"Lade Modell und Tokenizer '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # NEU: Initialisiere die Klasse mit top_k=100 anstelle von std_threshold
    analyser = HessianMagnitudeAnalyser(model, tokenizer, top_k=1000, n_samples=64)
    analyser.run(max_layers=10)

    print("\n🎉 Analyse abgeschlossen.")