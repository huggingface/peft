import pandas as pd
import json
import os
import re
from pathlib import Path

# Hauptverzeichnis, in dem die Ergebnisse Ihrer Experimente liegen
BASE_RESULTS_DIR = Path("./train_results_group_exp")
OUTPUT_CSV_PATH = Path("./master_results_final.csv")

# Konfigurationen für bekannte Modelle (können bei Bedarf erweitert/angepasst werden)
# Werte sind grobe Referenzen aus HF-Configs; bitte verifizieren bei neuen Modellen.
MODEL_CONFIGS = {
    "SmolLM2-1.7B": {
        "total_params": 1_687_261_184,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 32,
    },
    # TinyLlama Varianten
    "TinyLlama-1.1B-Chat-v1.0": {
        "total_params": 1_100_835_840,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
    },
    "TinyLlama_v1.1": {  # Falls der Kurzname so erzeugt wird
        "total_params": 1_100_835_840,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
    },
    # Optionale weitere Modelle (Werte ggf. anpassen/prüfen):
    "Llama-3.2-1B": {
        "total_params": 1_048_576_000,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
    },
    "phi-2": {
        "total_params": 2_780_000_000,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 32,
    },
}


def parse_experiment_name(name: str):
    """Extrahiert Parameter aus dem Experimentennamen mit Regex."""
    params = {}
    pattern = re.compile(
        r"(?P<model_name>[\w.-]+)_"
        r"(?P<training_mode>[\w-]+)_"
        r"r(?P<rank>\d+)_"
        r"b(?P<bits>\d+)_"
        r"d(?P<dataset>[\w-]+)"
        r"(?:_group(?P<group_size>\d+))?"
    )
    match = pattern.match(name)
    if match:
        data = match.groupdict()
        # ints wo sinnvoll
        for k, v in data.items():
            if v is None:
                params[k] = None
            elif v.isdigit():
                params[k] = int(v)
            else:
                params[k] = v
        params['experiment_name'] = name
    else:
        # Fallback: wenigsten den Namen mitgeben
        params = {
            'experiment_name': name,
            'model_name': None,
            'training_mode': None,
            'rank': None,
            'bits': None,
            'dataset': None,
            'group_size': None,
        }
    return params


def calculate_bpw_metrics(params: dict):
    """Berechnet sowohl den 'effective_bpw_base' (Quant-Teil) als auch den 'total_bpw' inkl. Adapter-Overhead.

    Annahmen:
    - Quant-Metadaten (scale, zero-point) zusammen 32 Bits pro Gruppe.
    - Adapter in 16 Bit (BF16/FP16) gespeichert.
    - Vereinfachte Schätzung der Adapter-Parameter basierend auf Hidden/Intermediate-Größen.
    """
    bits = params.get('bits')
    group_size = params.get('group_size')
    rank = params.get('rank')
    model_name = params.get('model_name')
    mode = params.get('training_mode')

    metrics = {'effective_bpw_base': None, 'total_bpw': None}

    if not bits or not group_size:
        return metrics

    # 16-bit scale + 16-bit zero-point
    metadata_bits = 32
    metrics['effective_bpw_base'] = bits + (metadata_bits / group_size)

    # Für total_bpw brauchen wir Modelldaten + Rank
    if not rank or not model_name or model_name not in MODEL_CONFIGS:
        return metrics

    cfg = MODEL_CONFIGS[model_name]
    total_params = cfg['total_params']
    hidden = cfg['hidden_size']
    inter = cfg['intermediate_size']
    n_layers = cfg['num_hidden_layers']

    # Grobe Schätzung der LoRA-Parameter pro Layer für gängige Projektionen
    # Q, K, V, O (4x hidden->hidden)
    lora_params_qkvo = 4 * (rank * (hidden + hidden))  # A: hidden x r, B: r x hidden => r*(hidden+hidden)
    # MLP: gate (hidden->inter), up (hidden->inter), down (inter->hidden)
    lora_params_mlp = (rank * (hidden + inter)) + (rank * (hidden + inter)) + (rank * (inter + hidden))
    params_per_layer = lora_params_qkvo + lora_params_mlp
    total_adapter_params = n_layers * params_per_layer

    # Adapter in 16 Bit
    total_adapter_bits = total_adapter_params * 16

    # Basisbits: quantisierte Gewichte (ggf. ohne Adapter-Gewichte)
    # Für PISSA-only-Adapter könnte man annehmen, dass Basisgewichte unverändert quantisiert bleiben
    # und Adapter additiv sind. Für andere Modi ebenso.
    total_bits = (total_params * metrics['effective_bpw_base']) + total_adapter_bits

    metrics['total_bpw'] = round(total_bits / total_params, 6)
    return metrics


def main():
    all_results = []

    print(f"Suche nach Ergebnissen in: {BASE_RESULTS_DIR}")

    if not BASE_RESULTS_DIR.exists():
        print("Keine Ergebnisse gefunden. Beende.")
        return

    for exp_dir in BASE_RESULTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue

        params = parse_experiment_name(exp_dir.name)
        if not params.get('model_name'):
            print(f"Warnung: Konnte Parameter für '{exp_dir.name}' nicht parsen. Überspringe.")
            continue

        evaluation_dir = exp_dir / "evaluation"
        # LM Harness Ergebnisse
        harness_path = evaluation_dir / "lm_harness_results.json"
        if harness_path.exists():
            try:
                with open(harness_path, 'r') as f:
                    results_data = json.load(f).get("results", {})
                # Beispiel: tinyMMLU
                params['mmlu_acc'] = results_data.get("tinyMMLU", {}).get("acc,none")
                # Optional weitere Tasks hier extrahieren
            except Exception as e:
                print(f"Warnung: Konnte Harness-Resultate in {harness_path} nicht lesen: {e}")

        # Trainings-/Evaluationsmetriken
        metrics_path = evaluation_dir / "training_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    params.update(json.load(f))
            except Exception as e:
                print(f"Warnung: Konnte Trainingsmetriken in {metrics_path} nicht lesen: {e}")

        # BPW Metriken
        params.update(calculate_bpw_metrics(params))

        all_results.append(params)

    if not all_results:
        print("Keine Ergebnisse gefunden. Beende.")
        return

    df = pd.DataFrame(all_results)

    ordered_columns = [
        'experiment_name', 'model_name', 'training_mode', 'rank', 'bits',
        'group_size', 'dataset', 'effective_bpw_base', 'total_bpw',
        'mmlu_acc', 'peak_vram_gb', 'training_time_min', 'final_loss'
    ]
    final_columns = [col for col in ordered_columns if col in df.columns]
    df = df[final_columns]

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Erfolgreich {len(df)} Experimente aggregiert und in '{OUTPUT_CSV_PATH}' gespeichert.")
    print("\nVorschau der ersten 5 Zeilen:")
    print(df.head())


if __name__ == "__main__":
    main()