"""
Script pour lancer toutes les configurations de loss functions avec WandB logging.

Usage:
    python run_all_experiments.py

Ce script lance automatiquement 4 configurations:
1. BCE (Baseline)
2. Focal Loss
3. DCL Loss
4. Hybrid (Focal + DCL)
"""

import subprocess
import sys
import time
from datetime import datetime

# Paramètres communs pour tous les entraînements
COMMON_PARAMS = [
    "--use_memory",
    "--n_epoch", "50",
    "--patience", "10",
    "--lr", "1e-4",
    "--node_dim", "200",
    "--time_dim", "200",
    "--memory_dim", "200",
    "--message_dim", "200",
    "--n_runs", "6",
    "--use_wandb"
]

# Configurations à tester
CONFIGURATIONS = [
    {
        "name": "BCE Baseline",
        "prefix": "tgn-bce",
        "params": []
    },
    {
        "name": "Focal Loss",
        "prefix": "tgn-focal",
        "params": [
            "--use_focal_loss",
            "--focal_alpha", "0.25",
            "--focal_gamma", "2.0"
        ]
    },
    {
        "name": "DCL Loss",
        "prefix": "tgn-dcl",
        "params": [
            "--use_dcl_loss",
            "--dcl_alpha", "0.5",
            "--dcl_temperature", "0.07"
        ]
    },
    {
        "name": "Hybrid (Focal + DCL)",
        "prefix": "tgn-hybrid",
        "params": [
            "--use_focal_loss",
            "--use_dcl_loss",
            "--focal_alpha", "0.25",
            "--focal_gamma", "2.0",
            "--dcl_alpha", "0.5",
            "--dcl_temperature", "0.07"
        ]
    }
]


def run_experiment(config, index, total):
    """
    Lance une expérience avec la configuration donnée.

    Args:
        config (dict): Configuration de l'expérience
        index (int): Numéro de la configuration
        total (int): Nombre total de configurations

    Returns:
        bool: True si succès, False si échec
    """
    print("\n" + "="*80)
    print(f"Configuration {index}/{total}: {config['name']}")
    print("="*80)

    # Construire la commande
    cmd = ["python", "train_self_supervised.py"]
    cmd.extend(["--prefix", config["prefix"]])
    cmd.extend(COMMON_PARAMS)
    cmd.extend(config["params"])

    # Afficher la commande
    print(f"\nCommande: {' '.join(cmd)}\n")

    # Lancer l'entraînement
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        elapsed_time = time.time() - start_time

        print(f"\n✓ {config['name']} terminé en {elapsed_time/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time

        print(f"\n❌ {config['name']} a échoué après {elapsed_time/60:.1f} minutes")
        print(f"   Code de sortie: {e.returncode}")
        return False

    except KeyboardInterrupt:
        print(f"\n⚠️  Expérience interrompue par l'utilisateur")
        return False


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("LANCEMENT DE TOUTES LES EXPÉRIENCES")
    print("="*80)
    print(f"\nDate de début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Nombre de configurations: {len(CONFIGURATIONS)}")
    print(f"Runs par configuration: 6")
    print(f"Epochs par run: 50")
    print("\n⚠️  Estimation du temps total: ~4-8 heures (dépend de ton GPU)")

    input("\nAppuie sur Entrée pour commencer les expériences...")

    total_start_time = time.time()
    results = []

    # Lancer chaque configuration
    for i, config in enumerate(CONFIGURATIONS, 1):
        success = run_experiment(config, i, len(CONFIGURATIONS))
        results.append({
            "name": config["name"],
            "success": success
        })

        # Petite pause entre les expériences
        if i < len(CONFIGURATIONS):
            print("\n" + "-"*80)
            print("Pause de 5 secondes avant la prochaine expérience...")
            print("-"*80)
            time.sleep(5)

    # Résumé final
    total_elapsed = time.time() - total_start_time

    print("\n" + "="*80)
    print("RÉSUMÉ DES EXPÉRIENCES")
    print("="*80)
    print(f"\nDurée totale: {total_elapsed/3600:.2f} heures")
    print("\nRésultats:")

    for result in results:
        status = "✓ Succès" if result["success"] else "❌ Échec"
        print(f"  {status:12} - {result['name']}")

    successes = sum(1 for r in results if r["success"])
    print(f"\nTotal: {successes}/{len(results)} expériences réussies")

    if successes > 0:
        print("\n" + "="*80)
        print("Pour visualiser les résultats:")
        print("  python plot_loss_comparison.py")
        print("="*80)

    print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrompu par l'utilisateur\n")
        sys.exit(1)
