import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from alpaca_eval import evaluate

# ==============================================================================
# TEIL 1: FUNKTION ZUR SICHEREN EVALUATION
# ==============================================================================
def erstelle_sicheres_leaderboard(
    all_model_outputs_path: str,
    reference_outputs_path: str,
    leaderboard_output_path: str,
    annotators_config: str,
    sort_by_column: str = "win_rate"
):
    """
    Erstellt ein Leaderboard, indem es jedes Modell einzeln evaluiert.
    Diese Methode ist robuster gegen Index-Duplikate in alpaca_eval.
    """
    print(f"Lade Modellausgaben von: {all_model_outputs_path}")
    try:
        all_model_outputs = pd.read_json(all_model_outputs_path)
    except Exception as e:
        print(f"Fehler beim Laden der Modellausgaben: {e}")
        return

    alle_ergebnisse = []

    for model_name in all_model_outputs['generator'].unique():
        print(f"\n--- Evaluiere Modell: {model_name} ---")
        model_outputs_df = all_model_outputs[all_model_outputs['generator'] == model_name].copy()
        
        # WICHTIGER FIX: Duplikate entfernen und Index zurücksetzen, um Fehler zu vermeiden
        model_outputs_df = model_outputs_df.drop_duplicates(subset=['instruction']).reset_index(drop=True)
        print(f"Anzahl einzigartiger Instruktionen für '{model_name}': {len(model_outputs_df)}")

        leaderboard_teil, _ = evaluate(
            model_outputs=model_outputs_df,
            reference_outputs=reference_outputs_path,
            annotators_config=annotators_config,
            name=model_name,
            is_return_instead_of_print=True,
        )

        if leaderboard_teil is not None and not leaderboard_teil.empty:
            # Stelle sicher, dass der Modellname als Spalte und nicht als Index gespeichert wird
            leaderboard_teil['model_name'] = leaderboard_teil.index
            alle_ergebnisse.append(leaderboard_teil)
        else:
            print(f"WARNUNG: Keine Ergebnisse für das Modell '{model_name}' erhalten.")

    if not alle_ergebnisse:
        print("Keine Modelle konnten evaluiert werden.")
        return

    finales_leaderboard = pd.concat(alle_ergebnisse).reset_index(drop=True)
    finales_leaderboard = finales_leaderboard.sort_values(by=sort_by_column, ascending=False)
    
    print(f"\nSpeichere finales Leaderboard unter: {leaderboard_output_path}")
    finales_leaderboard.to_csv(leaderboard_output_path, index=False)
    print("--- Evaluation abgeschlossen ---")


# ==============================================================================
# TEIL 2: FUNKTION ZUR VISUALISIERUNG
# ==============================================================================
def parse_fields(model_name: str):
    """
    Extrahiert method, bits, rank, training aus dem model_name.
    Erwartete Patterns wie:
      - mode_qalora_bits_3_rank_256_group_32
      - mode_gptq_lora_bits_2_rank_128_group_32
      - mode_post_quantization_bits_3_rank_16_group_32_training_skip_None
      - initial_quantization_2bit
    Alles andere wird als baseline/unknown klassifiziert.
    """
    method = "BASELINE"
    bits = None
    rank = None
    training = "Trained"

    # Methode
    m = re.search(r"mode_(\w+)", model_name)
    if m:
        method = m.group(1).replace("_", " ").upper()
    elif model_name.startswith("initial_quantization"):
        method = "INITIAL QUANTIZATION"
    else:
        method = "BASELINE"

    # Bits
    b = re.search(r"bits_(\d+)", model_name)
    if b:
        bits = int(b.group(1))
    else:
        b2 = re.search(r"(\d)bit", model_name)
        if b2:
            bits = int(b2.group(1))

    # Rank
    r = re.search(r"rank_(\d+)", model_name)
    if r:
        rank = int(r.group(1))

    # Training-Status
    if "training_skip_True" in model_name:
        training = "Untrained"

    return method, bits, rank, training


def plotte_leaderboard_heatmap(csv_path: str, metric_col: str = "win_rate", save_basename: str = "leaderboard_heatmap"):
    """
    Erstellt eine Heatmap-Tabelle:
      - Facets: Bits (2, 3)
      - Zeilen: Methode
      - Spalten: Rank
      - Zellen: Win-Rate (mit ± Standardfehler als Annotation)
    """
    print(f"\nLese Leaderboard-Daten von: {csv_path}")
    df = pd.read_csv(csv_path)

    # Felder extrahieren
    parsed = df["model_name"].apply(parse_fields)
    df_parsed = pd.DataFrame(parsed.tolist(), columns=["method", "bits", "rank", "training"])
    df = pd.concat([df, df_parsed], axis=1)

    # Nur Zeilen mit erkannten bits und rank verwenden
    df = df.dropna(subset=["bits", "rank"])
    df["bits"] = df["bits"].astype(int)
    df["rank"] = df["rank"].astype(int)

    # Optional: Nur Trained betrachten (wenn gewünscht). Hier zeigen wir beide gemischt an.
    # df = df[df["training"] == "Trained"]

    # Reihenfolge definieren
    rank_order = sorted(df["rank"].unique())  # z.B. [16, 128, 256]
    method_order = ["QALORA", "GPTQ LORA", "POST QUANTIZATION", "INITIAL QUANTIZATION", "BASELINE"]
    # Nicht vorhandene Methoden filtern
    method_order = [m for m in method_order if m in set(df["method"])]

    # Für Annotation: ± standard_error (falls vorhanden)
    has_se = "standard_error" in df.columns

    # Facets pro Bits vorbereiten (2 und 3, falls vorhanden)
    bits_list = sorted(df["bits"].unique())

    if len(bits_list) == 0:
        print("Keine gültigen Einträge mit bits/rank gefunden.")
        return

    # Figure vorbereiten
    ncols = len(bits_list)
    fig, axes = plt.subplots(
        1, ncols, figsize=(6.0 * ncols, 6.5), squeeze=False, dpi=140
    )
    axes = axes[0]

    # gemeinsame Farbskala 0–100 (Win-Rate in %)
    vmin, vmax = 0, 100
    cmap = sns.color_palette("light:teal", as_cmap=True)

    for i, bits in enumerate(bits_list):
        ax = axes[i]
        df_b = df[df["bits"] == bits].copy()

        # Aggregation falls mehrere Einträge je (method, rank) existieren
        agg = {
            metric_col: "mean",
        }
        if has_se:
            # Wenn mehrere SEs vorliegen, mitteln wir sie grob (robust wäre pooling).
            agg["standard_error"] = "mean"

        pivot_vals = df_b.groupby(["method", "rank"], as_index=False).agg(agg)

        # Pivot: Zeilen=Methoden, Spalten=Ranks
        table = pivot_vals.pivot(index="method", columns="rank", values=metric_col)

        # Reihenfolge der Achsen anwenden
        # fehlende Methoden/Ranks auffüllen (NaN) für konsistente Tabelle
        table = table.reindex(index=method_order, columns=rank_order)

        # Heatmap zeichnen
        sns.heatmap(
            table,
            ax=ax,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar=(i == ncols - 1),  # Colorbar nur einmal ganz rechts
            linewidths=0.6,
            linecolor="white",
            square=True,
            annot=False,  # Annotation separat, um Std-Error dazuzuschreiben
            fmt="",
        )

        # Annotationen (Wert ± SE)
        if has_se:
            # Merge, um die SEs passend zuzuordnen
            se_tab = pivot_vals.pivot(index="method", columns="rank", values="standard_error")
            se_tab = se_tab.reindex(index=method_order, columns=rank_order)
        else:
            se_tab = None

        for r_i, method in enumerate(table.index):
            for c_i, rank in enumerate(table.columns):
                val = table.loc[method, rank]
                if pd.isna(val):
                    text = "—"
                else:
                    if se_tab is not None and not pd.isna(se_tab.loc[method, rank]):
                        text = f"{val:.1f}\n±{se_tab.loc[method, rank]:.1f}"
                    else:
                        text = f"{val:.1f}"
                ax.text(
                    c_i + 0.5, r_i + 0.5, text,
                    ha="center", va="center",
                    fontsize=11, color="black"
                )

        ax.set_title(f"{bits}-bit", fontsize=14, weight="bold", pad=10)
        ax.set_xlabel("LoRA Rank", fontsize=12)
        ax.set_ylabel("Methode", fontsize=12)
        ax.set_xticklabels([str(x) for x in rank_order], rotation=0)
        ax.set_yticklabels([str(x) for x in table.index], rotation=0)

    # Gesamttitel und Layout
    fig.suptitle("AlpacaEval Win-Rates nach Bits (Spalten: Ranks, Zeilen: Methoden)", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.96, 0.95])

    # Dateien speichern
    base = os.path.splitext(os.path.basename(csv_path))[0]
    png_path = f"{save_basename}_{base}.png"
    pdf_path = f"{save_basename}_{base}.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Plots gespeichert unter:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)


# ==============================================================================
# TEIL 3: HAUPTSKRIPT ZUR AUSFÜHRUNG
# ==============================================================================
if __name__ == '__main__':
    # --- BITTE HIER ANPASSEN ---
    ALL_MODELS_FILE = '/home/nudel/Documents/peft/train_results_group_exp_merged/alpaca_super_collection_2_bit_and_3_bit.json'
    REFERENCE_FILE = '/home/nudel/Documents/peft/train_results_group_exp/fullprecision_smollm/alpaca_eval_results.json'
    LEADERBOARD_FILE = 'mein_eigenes_leaderboard_2bit_und_3bit.csv'
    ANNOTATOR_CONFIG = 'deepseek_v3_eval'
    
    # Schritt 1: Evaluation (wenn CSV noch nicht existiert)
    if not os.path.exists(LEADERBOARD_FILE):
        print(">>> SCHRITT 1: Starte die Evaluation der Modelle...")
        erstelle_sicheres_leaderboard(
            all_model_outputs_path=ALL_MODELS_FILE,
            reference_outputs_path=REFERENCE_FILE,
            leaderboard_output_path=LEADERBOARD_FILE,
            annotators_config=ANNOTATOR_CONFIG
        )

    # Schritt 2: Heatmap erstellen
    print("\n>>> SCHRITT 2: Erstelle die Heatmap-Tabelle...")
    if os.path.exists(LEADERBOARD_FILE):
        plotte_leaderboard_heatmap(csv_path=LEADERBOARD_FILE, metric_col="win_rate")
    else:
        print(f"FEHLER: Die Leaderboard-Datei '{LEADERBOARD_FILE}' wurde nicht gefunden.")