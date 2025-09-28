import os
import json
import argparse
import sys

def merge_and_rename_alpaca_results_recursive(base_path, output_file):
    """
    Durchsucht rekursiv alle Unterverzeichnisse eines Basispfads nach 'alpaca_eval_results.json',
    benennt das 'generator'-Feld in den Namen des unmittelbaren Eltern-Ordners um und führt
    alle Ergebnisse in einer einzigen Ausgabedatei zusammen.
    """
    # Das Tilde-Zeichen '~' muss zum Home-Verzeichnis des Benutzers erweitert werden.
    expanded_base_path = os.path.expanduser(base_path)

    if not os.path.isdir(expanded_base_path):
        print(f"Fehler: Der angegebene Pfad '{expanded_base_path}' ist kein gültiges Verzeichnis.")
        sys.exit(1)

    merged_data = []
    print(f"Durchsuche rekursiv alle Ordner in: {expanded_base_path}")

    # os.walk durchläuft den gesamten Verzeichnisbaum von oben nach unten.
    # dirpath: Der Pfad des aktuellen Ordners
    # dirnames: Eine Liste der Unterordner im aktuellen Ordner
    # filenames: Eine Liste der Dateien im aktuellen Ordner
    for dirpath, dirnames, filenames in os.walk(expanded_base_path):
        # Prüfe, ob die gesuchte Datei im aktuellen Ordner vorhanden ist
        if 'alpaca_eval_results.json' in filenames:
            json_file_path = os.path.join(dirpath, 'alpaca_eval_results.json')
            
            print(f"-> Verarbeite gefundene Datei: {json_file_path}")
            
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Der neue Generator-Name ist der Name des Ordners, der die Datei enthält.
                # os.path.basename(dirpath) extrahiert den letzten Teil des Pfades.
                new_generator_name = os.path.basename(dirpath)

                # Iteriere durch jeden Eintrag und aktualisiere den Generator
                for entry in data:
                    entry['generator'] = new_generator_name

                # Füge die modifizierten Daten zur Hauptliste hinzu
                merged_data.extend(data)
                print(f"   - {len(data)} Einträge hinzugefügt mit generator='{new_generator_name}'")

            except json.JSONDecodeError:
                print(f"   - WARNUNG: Konnte JSON in {json_file_path} nicht parsen. Datei wird übersprungen.")
            except Exception as e:
                print(f"   - FEHLER: Ein unerwarteter Fehler ist aufgetreten bei der Verarbeitung von {json_file_path}: {e}")

    if not merged_data:
        print("\nKeine Daten zum Zusammenführen gefunden. Die Ausgabedatei wird nicht erstellt.")
        return

    # Schreibe die zusammengeführten Daten in die Ausgabedatei
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=4 sorgt für eine gut lesbare JSON-Datei
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        print(f"\nErfolgreich! {len(merged_data)} Einträge wurden in '{output_file}' zusammengeführt.")
    except IOError as e:
        print(f"\nFEHLER: Konnte die Ausgabedatei nicht schreiben: {e}")


if __name__ == '__main__':
    # Richte den Argumenten-Parser ein
    parser = argparse.ArgumentParser(
        description="Führt mehrere 'alpaca_eval_results.json' Dateien aus rekursiv durchsuchten "
                    "Unterverzeichnissen zusammen und benennt den 'generator' entsprechend dem Eltern-Ordner um."
    )
    parser.add_argument(
        "base_path",
        type=str,
        help="Der Pfad zum Hauptverzeichnis, das die Experiment-Ordner enthält."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="merged_alpaca_eval_results_recursive.json",
        help="Name der Ausgabedatei (Standard: merged_alpaca_eval_results_recursive.json)"
    )

    args = parser.parse_args()
    merge_and_rename_alpaca_results_recursive(args.base_path, args.output)