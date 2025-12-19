"""
Script pour comparer les courbes de loss entre différentes loss functions.

Usage:
    python plot_loss_comparison.py

Ce script lit automatiquement tous les fichiers JSON dans results/ et génère:
1. Courbes de training loss par epoch
2. Comparaison des métriques de test finales
3. Courbes de validation metrics (MRR, Recall@K)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List

def load_results(results_dir: str = "results") -> Dict[str, dict]:
    """
    Charge tous les résultats JSON du dossier results/.

    Returns:
        dict: {loss_name: results_dict}
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Le dossier {results_dir}/ n'existe pas")
        return {}

    all_results = {}

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            loss_name = data.get("loss_function", json_file.stem)
            all_results[loss_name] = data
            print(f"✓ Loaded {json_file.name}")
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement de {json_file.name}: {e}")

    return all_results


def plot_training_loss_comparison(results: Dict[str, dict], save_path: str = "loss_comparison_training.png"):
    """
    Compare les courbes de training loss entre différentes loss functions.
    """
    plt.figure(figsize=(12, 7))

    # Couleurs et styles pour chaque loss
    colors = {
        "bce": "#3498db",      # Bleu
        "focal": "#e74c3c",    # Rouge
        "har": "#2ecc71",      # Vert
        "hybrid": "#9b59b6"    # Violet
    }

    markers = {
        "bce": "o",
        "focal": "s",
        "har": "^",
        "hybrid": "D"
    }

    for loss_name, data in results.items():
        train_losses = data.get("training", {}).get("losses", [])

        if not train_losses:
            print(f"⚠️  Pas de training losses pour {loss_name}")
            continue

        epochs = list(range(1, len(train_losses) + 1))
        color = colors.get(loss_name, "#95a5a6")
        marker = markers.get(loss_name, "o")

        # Plot avec marqueurs moins fréquents pour la lisibilité
        plt.plot(epochs, train_losses,
                label=loss_name.upper(),
                color=color,
                linewidth=2.5,
                marker=marker,
                markersize=6,
                markevery=max(1, len(epochs) // 10),  # Marker tous les 10%
                alpha=0.8)

    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=13, fontweight='bold')
    plt.title('Training Loss Comparison: BCE vs Focal vs HAR vs Hybrid', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training loss plot saved: {save_path}")
    plt.close()


def plot_test_metrics_comparison(results: Dict[str, dict], save_path: str = "loss_comparison_test_metrics.png"):
    """
    Compare les métriques de test finales entre les loss functions.
    """
    # Extraire les métriques de test
    loss_names = []
    aurocs = []
    aps = []
    mrrs = []
    recall_10s = []
    recall_50s = []

    for loss_name, data in results.items():
        test_data = data.get("test", {})

        if not test_data:
            continue

        loss_names.append(loss_name.upper())
        aurocs.append(test_data.get("auc", 0))
        aps.append(test_data.get("ap", 0))
        mrrs.append(test_data.get("mrr", 0))
        recall_10s.append(test_data.get("recall_10", 0))
        recall_50s.append(test_data.get("recall_50", 0))

    if not loss_names:
        print("⚠️  Pas de données de test disponibles")
        return

    # Créer le bar plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Test Metrics Comparison Across Loss Functions', fontsize=16, fontweight='bold')

    metrics = [
        ("AUROC", aurocs, axes[0, 0]),
        ("Average Precision (AP)", aps, axes[0, 1]),
        ("MRR", mrrs, axes[0, 2]),
        ("Recall@10", recall_10s, axes[1, 0]),
        ("Recall@50", recall_50s, axes[1, 1])
    ]

    colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for metric_name, values, ax in metrics:
        x_pos = np.arange(len(loss_names))
        bars = ax.bar(x_pos, values, color=colors_list[:len(loss_names)],
                     alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Loss Function', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_names, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Ajouter les valeurs sur les barres
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Supprimer le dernier subplot vide
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Test metrics plot saved: {save_path}")
    plt.close()


def plot_validation_metrics_over_epochs(results: Dict[str, dict], save_path: str = "loss_comparison_val_metrics.png"):
    """
    Compare l'évolution des métriques de validation au cours des epochs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Validation Metrics Over Epochs', fontsize=16, fontweight='bold')

    colors = {
        "bce": "#3498db",
        "focal": "#e74c3c",
        "har": "#2ecc71",
        "hybrid": "#9b59b6"
    }

    for loss_name, data in results.items():
        val_data = data.get("validation", {})

        if not val_data:
            continue

        color = colors.get(loss_name, "#95a5a6")

        # MRR
        mrr = val_data.get("mrr", [])
        if mrr:
            epochs = list(range(1, len(mrr) + 1))
            axes[0, 0].plot(epochs, mrr, label=loss_name.upper(),
                          color=color, linewidth=2.5, alpha=0.8)

        # AP
        ap = val_data.get("ap", [])
        if ap:
            epochs = list(range(1, len(ap) + 1))
            axes[0, 1].plot(epochs, ap, label=loss_name.upper(),
                          color=color, linewidth=2.5, alpha=0.8)

        # Recall@10
        recall_10 = val_data.get("recall_10", [])
        if recall_10:
            epochs = list(range(1, len(recall_10) + 1))
            axes[1, 0].plot(epochs, recall_10, label=loss_name.upper(),
                          color=color, linewidth=2.5, alpha=0.8)

        # Recall@50
        recall_50 = val_data.get("recall_50", [])
        if recall_50:
            epochs = list(range(1, len(recall_50) + 1))
            axes[1, 1].plot(epochs, recall_50, label=loss_name.upper(),
                          color=color, linewidth=2.5, alpha=0.8)

    # Configuration des subplots
    metrics_config = [
        (axes[0, 0], "Validation MRR"),
        (axes[0, 1], "Validation AP"),
        (axes[1, 0], "Validation Recall@10"),
        (axes[1, 1], "Validation Recall@50")
    ]

    for ax, title in metrics_config:
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Validation metrics plot saved: {save_path}")
    plt.close()


def create_summary_table(results: Dict[str, dict], save_path: str = "loss_comparison_summary.csv"):
    """
    Crée un tableau résumé avec toutes les métriques finales.
    """
    rows = []

    for loss_name, data in results.items():
        test_data = data.get("test", {})
        new_nodes_data = data.get("new_nodes_test", {})
        config = data.get("config", {})

        row = {
            "Loss Function": loss_name.upper(),
            "Focal Alpha": config.get("focal_alpha"),
            "Focal Gamma": config.get("focal_gamma"),
            "HAR Alpha": config.get("har_alpha"),
            "HAR Temperature": config.get("har_temperature"),
            "Test AUROC": test_data.get("auc"),
            "Test AP": test_data.get("ap"),
            "Test MRR": test_data.get("mrr"),
            "Test Recall@10": test_data.get("recall_10"),
            "Test Recall@50": test_data.get("recall_50"),
            "New Nodes AUROC": new_nodes_data.get("auc"),
            "New Nodes AP": new_nodes_data.get("ap"),
            "New Nodes MRR": new_nodes_data.get("mrr"),
            "New Nodes Recall@10": new_nodes_data.get("recall_10"),
            "New Nodes Recall@50": new_nodes_data.get("recall_50"),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"✓ Summary table saved: {save_path}")

    # Afficher aussi dans le terminal
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("LOSS FUNCTION COMPARISON ANALYSIS")
    print("="*80 + "\n")

    # Charger les résultats
    results = load_results("results")

    if not results:
        print("\n❌ Aucun fichier de résultats trouvé dans results/")
        print("   Entraîne d'abord ton modèle avec différentes loss functions:")
        print("   - python train_self_supervised.py --data crunchbase")
        print("   - python train_self_supervised.py --data crunchbase --use_focal_loss")
        print("   - python train_self_supervised.py --data crunchbase --use_har_loss")
        print("   - python train_self_supervised.py --data crunchbase --use_focal_loss --use_har_loss")
        return

    print(f"\n✓ Chargé {len(results)} configurations de loss\n")

    # Créer le dossier de sortie
    output_dir = Path("loss_comparison_plots")
    output_dir.mkdir(exist_ok=True)

    # Générer les plots
    print("Génération des visualisations...")
    plot_training_loss_comparison(results, output_dir / "training_loss_comparison.png")
    plot_test_metrics_comparison(results, output_dir / "test_metrics_comparison.png")
    plot_validation_metrics_over_epochs(results, output_dir / "validation_metrics_over_epochs.png")
    create_summary_table(results, output_dir / "summary_table.csv")

    print(f"\n{'='*80}")
    print(f"✅ Tous les plots ont été générés dans: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
