import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset
import imageio # You might need to install this: pip install imageio

# =================================================================================
# 1. DATA LOADING UTILITY (from your code)
# =================================================================================
def get_c4_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """
    Loads the C4 dataset from Hugging Face and prepares tokenized samples for calibration.
    """
    print(" Lade C4 Kalibrierungsdatensatz...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    samples = []
    # Using 'iter' and 'next' is a bit cleaner for streaming datasets
    dataset_iterator = iter(dataset)
    
    with tqdm(total=n_samples, desc="Lade Daten") as pbar:
        while len(samples) < n_samples:
            try:
                row = next(dataset_iterator)
                text = row['text']
                tokens = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
                if tokens.input_ids.shape[1] == seq_len:
                    samples.append(tokens)
                    pbar.update(1)
            except StopIteration:
                print("Warnung: Ende des Datasets erreicht, bevor n_samples gefüllt wurden.")
                break

    print(f"✅ {len(samples)} Kalibrierungssamples geladen.")
    return samples

# =================================================================================
# 2. HESSIAN VISUALIZATION SIMULATOR
# =================================================================================
def simulate_hessian_approximation(model, tokenizer, layer_name, n_samples=50):
    """
    Simulates the Hessian approximation step-by-step for a single layer
    and generates GIFs of its evolution.
    """
    
    # --- Setup ---
    print(f"\nStarte Simulation für Layer: {layer_name}")
    try:
        # Get the specific layer module from the model
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Linear):
            print(f"Fehler: Layer {layer_name} ist kein nn.Linear Modul.")
            return
    except KeyError:
        print(f"Fehler: Layer {layer_name} nicht im Modell gefunden.")
        return

    device = module.weight.device
    columns = module.weight.shape[1]

    # Create directories for saving plot frames
    frames_dir_heatmap = "hessian_frames_heatmap"
    frames_dir_3d = "hessian_frames_3d"
    os.makedirs(frames_dir_heatmap, exist_ok=True)
    os.makedirs(frames_dir_3d, exist_ok=True)
    
    # --- Data Loading ---
    # We only need the inputs to the specified layer, so we use a forward hook
    captured_inputs = []
    def hook(_, inp, __):
        # We only need one batch of input per sample, so we detach and move to CPU
        captured_inputs.append(inp[0].detach().cpu())
    
    handle = module.register_forward_hook(hook)
    
    # Load data
    calibration_data = get_c4_calibration_data(tokenizer, n_samples=n_samples, seq_len=512)

    print("Sammle Layer-Inputs durch einen Forward-Pass...")
    model.eval()
    with torch.no_grad():
        for sample in tqdm(calibration_data, desc="Sammle Inputs"):
            sample = {k: v.to(device) for k, v in sample.items()}
            model(**sample)
    handle.remove()
    
    # --- Step-by-Step Simulation ---
    H = torch.zeros((columns, columns), device='cpu', dtype=torch.float32)
    nsamples_processed = 0
    
    heatmap_frames = []
    plot3d_frames = []

    # Define two fixed, orthogonal directions in the parameter space for the 3D plot
    d1 = torch.randn(columns)
    d1 /= torch.linalg.norm(d1)
    d2 = torch.randn(columns)
    d2 -= (d2 @ d1) * d1 # Make d2 orthogonal to d1
    d2 /= torch.linalg.norm(d2)

    print("\nBeginne schrittweise Hessian-Approximation und Visualisierung...")
    for i, inp in enumerate(tqdm(captured_inputs, desc="Approximiere Hessian")):
        # Reshape input tensor and treat each token's activation as a sample
        reshaped_inp = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
        batch_size = reshaped_inp.shape[0]

        # Update Hessian using the running average formula
        H *= nsamples_processed / (nsamples_processed + batch_size)
        nsamples_processed += batch_size
        H += (2 / nsamples_processed) * (reshaped_inp.T @ reshaped_inp)
        
        # --- Visualization for the current step ---
        # To avoid generating too many frames, we can plot every N steps
        if i % 2 == 0 or i == n_samples - 1:
            # 1. Heatmap of the Hessian Matrix
            fig_h, ax_h = plt.subplots(figsize=(8, 6))
            im = ax_h.imshow(H.numpy(), cmap='viridis')
            fig_h.colorbar(im, ax=ax_h)
            ax_h.set_title(f"Hessian Approximation nach Sample {i+1}")
            ax_h.set_xlabel("Parameter-Index")
            ax_h.set_ylabel("Parameter-Index")
            heatmap_frame_path = os.path.join(frames_dir_heatmap, f"frame_{i+1:03d}.png")
            fig_h.savefig(heatmap_frame_path)
            plt.close(fig_h)
            heatmap_frames.append(heatmap_frame_path)
            
            # 2. 3D Plot of the approximated loss surface
            # We plot f(a,b) = 0.5 * (a*d1 + b*d2)^T * H * (a*d1 + b*d2)
            alpha = np.linspace(-1.0, 1.0, 30)
            beta = np.linspace(-1.0, 1.0, 30)
            alpha_grid, beta_grid = np.meshgrid(alpha, beta)
            
            term1 = (d1.T @ H @ d1).item()
            term2 = (d2.T @ H @ d2).item()
            term3 = (d1.T @ H @ d2).item()
            
            Z = 0.5 * (term1 * alpha_grid**2 + term2 * beta_grid**2 + 2 * term3 * alpha_grid * beta_grid)
            
            fig_3d = plt.figure(figsize=(9, 7))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot_surface(alpha_grid, beta_grid, Z, cmap=cm.viridis, rstride=1, cstride=1)
            ax_3d.set_title(f"Approximierte Loss-Krümmung nach Sample {i+1}")
            ax_3d.set_xlabel("Abweichung in Richtung d1")
            ax_3d.set_ylabel("Abweichung in Richtung d2")
            ax_3d.set_zlabel("Δ Loss")
            ax_3d.view_init(elev=30, azim=45) # Fix the viewing angle
            plot3d_frame_path = os.path.join(frames_dir_3d, f"frame_{i+1:03d}.png")
            fig_3d.savefig(plot3d_frame_path)
            plt.close(fig_3d)
            plot3d_frames.append(plot3d_frame_path)

    # --- Compile GIFs ---
    print("\nErstelle Animationen aus den Frames...")
    imageio.mimsave('hessian_evolution_heatmap.gif', [imageio.imread(f) for f in heatmap_frames], duration=0.2)
    imageio.mimsave('loss_approximation_3d.gif', [imageio.imread(f) for f in plot3d_frames], duration=0.2)

    print("\n🎉 Simulation abgeschlossen!")
    print(" GIFs gespeichert als 'hessian_evolution_heatmap.gif' und 'loss_approximation_3d.gif'")

# =================================================================================
# 3. MAIN EXECUTION BLOCK
# =================================================================================
if __name__ == '__main__':
    # --- Configuration ---
    model_name = "TinyLlama/TinyLlama_v1.1"
    
    # Let's pick one specific linear layer to analyze
    # Good choices are often in the MLP blocks, e.g., 'model.layers.0.mlp.gate_proj'
    layer_to_analyze = "model.layers.7.self_attn.p_proj"
    num_simulation_samples = 500 # Number of data samples to process one-by-one

    # --- Load Model and Tokenizer ---
    print(f"Lade Modell und Tokenizer '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- Run Simulation ---
    simulate_hessian_approximation(
        model=model,
        tokenizer=tokenizer,
        layer_name=layer_to_analyze,
        n_samples=num_simulation_samples
    )