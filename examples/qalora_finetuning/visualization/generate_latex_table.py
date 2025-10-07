import os
import json
import re
import pandas as pd

# ------------------- KONFIGURATION -------------------
PARENT_PATH = '/home/tonthat/Documents/peft/train_results_group_exp'

# Mapping von Tasknamen (auch tiny-Varianten) zu Tabellenspalten
BENCHMARK_MAPPING = {
    'arc_challenge': 'ARC-c', 'tinyArc': 'ARC-c',
    'arc_easy': 'ARC-e',
    'boolq': 'BoolQ',
    'hellaswag': 'HellaSwag', 'tinyHellaswag': 'HellaSwag',
    'openbookqa': 'OpBoQA',
    'piqa': 'PIQA',
    'winogrande': 'Winogrande',
    'wikitext': 'WikiText',
    # können extrahiert werden, werden aber aktuell nicht in der Tabelle gelistet
    'tinyGSM8k': 'GSM8k',
    'tinyMMLU': 'MMLU',
}

# LaTeX-Spalten in der gewünschten (alten) Reihenfolge
LATEX_COLUMN_ORDER = [
    'ARC-c', 'ARC-e', 'BoolQ', 'HellaSwag', 'OpBoQA', 'PIQA', 'Winogrande', 'WikiText', 'Size'
]
# ----------------------------------------------------

def normalize_dataset_name(dataset_in_folder: str) -> str:
    """Nur führendes 'd' vor bekannten Namen entfernen (dc4→c4, dalpaca-cleaned→alpaca-cleaned)."""
    if dataset_in_folder == 'dc4':
        return 'c4'
    if dataset_in_folder == 'dalpaca-cleaned':
        return 'alpaca-cleaned'
    # falls ohne führendes d bereits korrekt ist
    if dataset_in_folder in ('c4', 'alpaca-cleaned'):
        return dataset_in_folder
    # generischer Fallback: nur erstes führendes 'd' entfernen
    return dataset_in_folder[1:] if dataset_in_folder.startswith('d') else dataset_in_folder


def parse_folder_name(folder_name: str):
    """Extrahiert Metadaten aus dem Ordnernamen. Group size ist optional."""
    pattern = re.compile(
        r'^([^_]+)_'                # 1. Model (z.B. "SmolLM2-1.7B")
        r'(.+?)_'                   # 2. Mode (z.B. "pissa_rank_analysis" oder "qalora")
        r'r(\d+)_'                  # 3. Rank (z.B. "4")
        r'b(\d+)_'                  # 4. Bits (z.B. "3")
        r'(d?c4|d?alpaca-cleaned)'   # 5. Dataset (z.B. "dalpaca-cleaned")
        r'(?:_group(\d+))?$'        # 6. Group Size (optional, z.B. "32")
    )
    match = pattern.match(folder_name)
    if not match:
        return None
    groups = match.groups()
    return {
        'model': groups[0],
        'mode': groups[1],
        'rank': int(groups[2]),
        'bits': int(groups[3]),
        'dataset_in_folder': groups[4],  # z.B. 'dc4' oder 'dalpaca-cleaned'
        'group_size': int(groups[5]) if groups[5] is not None else None,
    }


def build_eval_filename(params: dict) -> str:
    """Erzeugt den erwarteten eval_results-Dateinamen für ein Experiment."""
    group_size = params.get('group_size') if params.get('group_size') is not None else 32
    dataset_for_filename = normalize_dataset_name(params['dataset_in_folder'])
    filename = (
        f"mode_{params['mode']}_"
        f"bits_{params['bits']}_"
        f"rank_{params['rank']}_"
        f"group_{group_size}_"
        f"training_skip_None_"
        f"calibration_dataset_{dataset_for_filename}.json"
    )
    return filename


def build_inventory(parent_path: str):
    """Scannt alle Experimente und baut eine sortierte Inventarliste inkl. Pfad zur Eval-JSON."""
    inventory = []
    print(f"Durchsuche Hauptordner: {parent_path}...")
    for folder_name in os.listdir(parent_path):
        abs_folder = os.path.join(parent_path, folder_name)
        if not os.path.isdir(abs_folder):
            continue
        params = parse_folder_name(folder_name)
        if not params:
            continue
        eval_dir = os.path.join(abs_folder, 'eval_results')
        if not os.path.isdir(eval_dir):
            continue
        expected_file = build_eval_filename(params)
        eval_path = os.path.join(eval_dir, expected_file)
        if not os.path.exists(eval_path):
            continue
        inventory.append({
            'model': params['model'],
            'mode': params['mode'],
            'rank': params['rank'],
            'bits': params['bits'],
            'dataset_in_folder': params['dataset_in_folder'],
            'calibration_dataset': normalize_dataset_name(params['dataset_in_folder']),
            'group_size': params['group_size'] if params['group_size'] is not None else 32,
            'eval_results_path': eval_path,
        })

    # Sortierung: model -> bits -> calibration_dataset -> rank -> mode -> group_size
    inventory.sort(key=lambda x: (
        x['model'], x['bits'], x['calibration_dataset'], x['rank'], x['mode'], x['group_size']
    ))

    with open('experiments_inventory.json', 'w') as f:
        json.dump(inventory, f, indent=2)
    print("Erfolgreich! Inventar in 'experiments_inventory.json' gespeichert.")

    return inventory


def extract_benchmark_results(json_data: dict) -> dict:
    """Extrahiert gewünschte Benchmark-Werte je Task."""
    results = {}
    raw_results = json_data.get('results', {})
    for task_name, task_result in raw_results.items():
        table_key = BENCHMARK_MAPPING.get(task_name)
        if not table_key:
            continue
        value, stderr = None, None
        if task_name == 'wikitext':
            value = task_result.get('word_perplexity,none')
            stderr = task_result.get('word_perplexity_stderr,none')
        elif task_name in ('tinyGSM8k', 'gsm8k', 'GSM8k'):
            value = task_result.get('exact_match,flexible-extract', task_result.get('exact_match,strict-match'))
            stderr = task_result.get('exact_match_stderr,flexible-extract', task_result.get('exact_match_stderr,strict-match'))
        else:
            value = task_result.get('acc_norm,none', task_result.get('acc,none'))
            stderr = task_result.get('acc_norm_stderr,none', task_result.get('acc_stderr,none'))
        if stderr == 'N/A':
            stderr = None
        results[table_key] = {'value': value, 'stderr': stderr}
    return results


def format_value_for_latex(key: str, data: dict) -> str:
    """Formatiert Messwerte; nutzt ± anstatt \\pm."""
    if data is None or data.get('value') is None:
        return ''
    value, stderr = data['value'], data.get('stderr')
    if key == 'WikiText':  # Perplexity nicht in Prozent
        return f"{value:.1f}"
    value *= 100
    if stderr is not None:
        stderr *= 100
        return f"{value:.1f} ± {stderr:.1f}"
    return f"{value:.1f}"


def render_latex(df: pd.DataFrame, out_path: str = 'benchmark_table.tex'):
    with open(out_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n\\tiny\n\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{Performance evaluation by varying adaptation rank ($r$) and the quantization level of the residual matrix ($\\boldsymbol{W}_{res}$)}\n")
        f.write("\\label{tab:pissa_rank_quant_tradeoff}\n")
        col_def = 'lll' + 'c' * len(LATEX_COLUMN_ORDER)
        f.write(f"\\begin{{tabular}}{{{col_def}}}\n\\toprule\n")
        headers = ["\\textbf{Model}", "\\textbf{Rank ($r$)}", "\\textbf{Bits ($\\boldsymbol{W}_{res}$)}"] + \
                  ["\\textbf{" + c.replace('_', ' ') + "}" for c in LATEX_COLUMN_ORDER]
        f.write(" & ".join(headers) + " \\\\n\\midrule\n\n")
        num_columns = 3 + len(LATEX_COLUMN_ORDER)
        for model_name, model_group in df.groupby('model', sort=False):
            model_str = f"\\multirow{{{len(model_group)}}}{{*}}{{\\shortstack{{{model_name.replace('-', ' \\\\ ')}}}}}"
            is_first_row_in_model = True
            for rank, rank_group in model_group.groupby('rank', sort=False):
                if not is_first_row_in_model:
                    f.write(f"\\cmidrule{{2-{num_columns}}}\n")
                is_first_row_in_rank = True
                for _, row in rank_group.iterrows():
                    model_cell = model_str if is_first_row_in_model else ''
                    rank_cell = f"\\multirow{{{len(rank_group)}}}{{*}}{{{row['rank']}}}" if is_first_row_in_rank else ''
                    mode_str = str(row['mode']).replace('_', ' ').title()
                    bits_cell = f"{row['bits']}-bit ({mode_str}, {row['dataset_clean']})"
                    gs = row.get('group_size')
                    if pd.notna(gs):
                        bits_cell += f" (Gs {int(gs)})"
                    f.write(f"{model_cell} & {rank_cell} & {bits_cell}")
                    for col in LATEX_COLUMN_ORDER:
                        cell_data = row.get(col)
                        formatted = format_value_for_latex(col, cell_data) if pd.notna(cell_data) else ''
                        f.write(f" & {formatted}")
                    f.write(" \\\\ \n")
                    is_first_row_in_model = False
                    is_first_row_in_rank = False
            f.write("\n")
        f.write("\\midrule\n\\end{tabular}\n\\end{table}\n")


def main():
    # 1) Inventar aufbauen und speichern
    inventory = build_inventory(PARENT_PATH)
    if not inventory:
        print('Keine Ergebnisse gefunden. Bitte Pfad/Namensschema prüfen.')
        return

    # 2) Metriken aggregieren und speichern
    all_results = []
    for item in inventory:
        with open(item['eval_results_path'], 'r') as f:
            json_data = json.load(f)
        bench = extract_benchmark_results(json_data)
        row = {
            'model': item['model'],
            'mode': item['mode'],
            'rank': item['rank'],
            'bits': item['bits'],
            'dataset_in_folder': item['dataset_in_folder'],
            'calibration_dataset': item['calibration_dataset'],
            'group_size': item['group_size'],
        }
        row.update(bench)
        all_results.append(row)

    with open('aggregated_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Erfolgreich! Aggregierte Ergebnisse in 'aggregated_results.json' gespeichert.")

    # 3) DataFrame vorbereiten und LaTeX rendern
    df = pd.DataFrame(all_results)
    df['dataset_clean'] = df['calibration_dataset']
    df = df.sort_values(by=['model', 'bits', 'dataset_clean', 'rank', 'mode', 'group_size']).reset_index(drop=True)
    render_latex(df, out_path='benchmark_table.tex')
    print("Erfolgreich! LaTeX-Tabelle in 'benchmark_table.tex' gespeichert.")


if __name__ == '__main__':
    main()