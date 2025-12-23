"""
Script pour remplacer toutes les références HAR par DCL dans les fichiers JSON.
"""

import json
from pathlib import Path

def fix_json_file(filepath):
    """Remplace har -> dcl dans un fichier JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    modified = False

    # Modifier loss_function
    if data.get("loss_function") == "har":
        data["loss_function"] = "dcl"
        print(f"  > {filepath.name}: loss_function har -> dcl")
        modified = True

    # Modifier les clés dans config
    config = data.get("config", {})
    if "har_alpha" in config:
        config["dcl_alpha"] = config.pop("har_alpha")
        print(f"  > {filepath.name}: har_alpha -> dcl_alpha")
        modified = True

    if "har_temperature" in config:
        config["dcl_temperature"] = config.pop("har_temperature")
        print(f"  > {filepath.name}: har_temperature -> dcl_temperature")
        modified = True

    # Sauvegarder seulement si modifié
    if modified:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    return modified

def main():
    results_dir = Path("results")

    # Trouver tous les fichiers JSON
    json_files = list(results_dir.glob("*.json"))

    fixed_count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Vérifier si ce fichier contient "har" n'importe où
            config = data.get("config", {})
            needs_fix = (data.get("loss_function") == "har" or
                        "har_alpha" in config or
                        "har_temperature" in config)

            if needs_fix:
                print(f"\nFixing {json_file.name}...")
                if fix_json_file(json_file):
                    fixed_count += 1
        except Exception as e:
            print(f"WARNING: Erreur avec {json_file.name}: {e}")

    print(f"\n{'='*60}")
    print(f"DONE: {fixed_count} fichiers corriges (HAR -> DCL)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
