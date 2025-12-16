"""
Script pour comparer les résultats de tous les modèles entraînés
"""

import re
import os
from pathlib import Path

def parse_temporal_validation_log(log_file):
    """Parse un fichier de log de temporal validation"""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    results = {}

    # Parse probabilités globales
    match = re.search(r'Médiane:\s+([\d.]+)', content)
    if match:
        results['prob_median_global'] = float(match.group(1))

    match = re.search(r'Moyenne:\s+([\d.]+)', content)
    if match:
        results['prob_mean_global'] = float(match.group(1))

    # Parse probabilités vrais liens
    section = re.search(r'VRAIS LIENS FUTURS.*?Médiane:\s+([\d.]+)', content, re.DOTALL)
    if section:
        results['prob_median_true'] = float(section.group(1))

    section = re.search(r'VRAIS LIENS FUTURS.*?Moyenne:\s+([\d.]+)', content, re.DOTALL)
    if section:
        results['prob_mean_true'] = float(section.group(1))

    # Parse rangs
    match = re.search(r'Rang médian:\s+([\d,]+)', content)
    if match:
        results['rank_median'] = int(match.group(1).replace(',', ''))

    match = re.search(r'Rang moyen:\s+([\d,]+)', content)
    if match:
        results['rank_mean'] = int(match.group(1).replace(',', ''))

    # Parse Precision/Recall@1000
    match = re.search(r'Precision@\s*1,000:\s+([\d.]+)\s+\((\d+)/(\d+)\)', content)
    if match:
        results['precision_1000'] = float(match.group(1))
        results['precision_1000_hits'] = int(match.group(2))
        results['precision_1000_total'] = int(match.group(3))

    match = re.search(r'Recall@\s*1,000:\s+([\d.]+)\s+\((\d+)/(\d+)\)', content)
    if match:
        results['recall_1000'] = float(match.group(1))
        results['recall_1000_hits'] = int(match.group(2))
        results['recall_1000_total'] = int(match.group(3))

    # Parse Precision@5000
    match = re.search(r'Precision@\s*5,000:\s+([\d.]+)\s+\((\d+)/(\d+)\)', content)
    if match:
        results['precision_5000'] = float(match.group(1))
        results['recall_5000_hits'] = int(match.group(2))

    # Parse amélioration vs random
    match = re.search(r'Amélioration:\s+([\d.]+)x', content)
    if match:
        results['vs_random'] = float(match.group(1))

    return results


def print_comparison_table(models):
    """Affiche un tableau de comparaison"""
    print("\n" + "="*100)
    print("COMPARAISON DES MODÈLES - TEMPORAL VALIDATION")
    print("="*100)

    # Header
    header = f"{'Métrique':<30}"
    for model_name in models.keys():
        header += f"{model_name:>18}"
    print(header)
    print("-"*100)

    # Rows
    metrics = [
        ('Prob Médiane (global)', 'prob_median_global', '{:.3f}'),
        ('Prob Médiane (vrais)', 'prob_median_true', '{:.3f}'),
        ('Gap (vrais - global)', None, '{:.3f}'),  # Calculé
        ('Rang Médian', 'rank_median', '{:,}'),
        ('Precision@1000', 'precision_1000', '{:.4f}'),
        ('Recall@1000', 'recall_1000', '{:.4f}'),
        ('Precision@5000', 'precision_5000', '{:.4f}'),
        ('vs Random (x)', 'vs_random', '{:.2f}x'),
    ]

    for metric_name, metric_key, fmt in metrics:
        row = f"{metric_name:<30}"

        for model_name, results in models.items():
            if results is None:
                row += f"{'N/A':>18}"
            elif metric_key is None:  # Gap calculé
                if 'prob_median_true' in results and 'prob_median_global' in results:
                    gap = results['prob_median_true'] - results['prob_median_global']
                    row += f"{fmt.format(gap):>18}"
                else:
                    row += f"{'N/A':>18}"
            elif metric_key in results:
                value = results[metric_key]
                if 'x' in fmt:
                    row += f"{fmt.format(value):>18}"
                else:
                    row += f"{fmt.format(value):>18}"
            else:
                row += f"{'N/A':>18}"

        print(row)

    print("="*100)

    # Recommendations
    print("\n📊 ANALYSE:")

    # Find best model for each metric
    best_models = {}

    for metric_name, metric_key, _ in metrics:
        if metric_key is None:
            continue

        best_value = None
        best_model = None

        for model_name, results in models.items():
            if results is None or metric_key not in results:
                continue

            value = results[metric_key]

            # For rank, lower is better; for others, higher is better
            if metric_key in ['rank_median', 'rank_mean']:
                if best_value is None or value < best_value:
                    best_value = value
                    best_model = model_name
            else:
                if best_value is None or value > best_value:
                    best_value = value
                    best_model = model_name

        if best_model:
            best_models[metric_name] = best_model

    print("\n🏆 Meilleur modèle par métrique:")
    for metric_name, best_model in best_models.items():
        print(f"   {metric_name:<30}: {best_model}")

    # Overall recommendation
    print("\n💡 RECOMMANDATION:")

    # Count wins
    model_wins = {}
    for model_name in models.keys():
        model_wins[model_name] = sum(1 for m in best_models.values() if m == model_name)

    if model_wins:
        winner = max(model_wins.items(), key=lambda x: x[1])
        print(f"   Le modèle '{winner[0]}' performe le mieux ({winner[1]} métriques gagnées)")

        # Specific insights
        for model_name, results in models.items():
            if results and 'precision_1000' in results:
                p1000 = results['precision_1000']
                if p1000 >= 0.01:  # 1% or higher
                    print(f"   ✅ '{model_name}' atteint {p1000:.2%} de Precision@1000 (bon pour TechRank)")
                elif p1000 >= 0.005:  # 0.5% or higher
                    print(f"   ⚠️  '{model_name}' atteint {p1000:.2%} de Precision@1000 (acceptable)")
                else:
                    print(f"   ❌ '{model_name}' atteint seulement {p1000:.2%} de Precision@1000 (faible)")


def main():
    """Main function"""
    results_dir = Path("results")

    # Models to compare
    model_names = ['baseline', 'focal', 'hardneg', 'focal-hardneg']

    models = {}
    for model_name in model_names:
        log_file = results_dir / f"{model_name}_temporal_validation.log"
        results = parse_temporal_validation_log(log_file)
        models[model_name] = results

        if results:
            print(f"✅ Loaded results for {model_name}")
        else:
            print(f"❌ No results found for {model_name} (file: {log_file})")

    # Print comparison
    if any(r is not None for r in models.values()):
        print_comparison_table(models)
    else:
        print("\n❌ No results found. Please run training and evaluation first.")
        print("\nRun this command to train and evaluate all models:")
        print("   bash compare_all_models.sh")
        print("\nOr run individual training commands from HARD_NEGATIVE_MINING_README.md")


if __name__ == "__main__":
    main()
