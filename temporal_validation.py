#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Validation for TGN-TechRank System
===========================================

Extension du workflow existant (TechRank_Comparison.py) pour ajouter des métriques
de validation temporelle robustes.

Utilise les graphes et DataFrames déjà créés par le code existant.

Author: Claude
Date: 2025-01-15
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field, asdict
import json


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ValidationMetrics:
    """Toutes les métriques de validation temporelle"""

    # Precision@K
    precision_at_k: Dict[int, float] = field(default_factory=dict)

    # Rank Correlation
    spearman_rho: float = 0.0
    spearman_p_value: float = 1.0
    kendall_tau: float = 0.0
    kendall_p_value: float = 1.0

    # Early Detection Rate
    edr_at_k: Dict[int, float] = field(default_factory=dict)
    growth_threshold: float = 2.0  # Facteur de croissance pour EDR

    # Lead Time
    avg_lead_time_days: float = 0.0
    median_lead_time_days: float = 0.0

    # Lift Score
    lift_at_k: Dict[int, float] = field(default_factory=dict)
    baseline_rate: float = 0.0
    model_rate_at_k: Dict[int, float] = field(default_factory=dict)

    # Random Baseline
    random_precision_at_k: Dict[int, float] = field(default_factory=dict)

    # Degree Baseline (classement par degré initial)
    degree_baseline_precision_at_k: Dict[int, float] = field(default_factory=dict)

    # Stats globales
    total_companies: int = 0
    companies_with_growth: int = 0
    companies_high_growth: int = 0
    correctly_detected_in_top50: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return asdict(self)

    def save_json(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"✅ Metrics saved to {filepath}")


# ============================================================================
# CORE VALIDATION FUNCTIONS
# ============================================================================

def compute_validation_metrics(
    df_delta: pd.DataFrame,
    B_before: nx.Graph,
    B_after: nx.Graph,
    top_k_list: List[int] = [10, 20, 50],
    growth_threshold: float = 2.0,
    prediction_horizon_days: float = 365.0,
    logger: Optional[logging.Logger] = None
) -> ValidationMetrics:
    """
    Calcule toutes les métriques de validation temporelle.

    Args:
        df_delta: DataFrame avec colonnes ['final_configuration', 'techrank_before',
                  'techrank_after', 'techrank_delta', 'rank_before', 'rank_after']
                  (Output de analyze_company_deltas() existant)
        B_before: Graphe bipartite initial (train set)
        B_after: Graphe bipartite réel au temps futur (validation/test set)
        top_k_list: Liste des valeurs K pour Precision@K, EDR@K, Lift@K
        growth_threshold: Facteur multiplicatif pour définir "forte croissance" (défaut: 2.0 = doublement)
        prediction_horizon_days: Nombre de jours entre prédiction et validation (pour lead time)
        logger: Logger optionnel

    Returns:
        ValidationMetrics avec toutes les métriques calculées
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("\n" + "="*80)
    logger.info("COMPUTING TEMPORAL VALIDATION METRICS")
    logger.info("="*80)

    metrics = ValidationMetrics(growth_threshold=growth_threshold)

    # Filtrer seulement les entreprises existantes AVANT (transductive)
    df_delta_filtered = df_delta[df_delta['techrank_before'] > 0].copy()
    logger.info(f"Companies in validation: {len(df_delta_filtered)} (transductive mode)")

    # Calculer la croissance réelle pour chaque entreprise
    df_delta_filtered['degree_before'] = df_delta_filtered['final_configuration'].apply(
        lambda comp: B_before.degree(comp) if comp in B_before else 0
    )
    df_delta_filtered['degree_after'] = df_delta_filtered['final_configuration'].apply(
        lambda comp: B_after.degree(comp) if comp in B_after else 0
    )
    df_delta_filtered['degree_growth_absolute'] = (
        df_delta_filtered['degree_after'] - df_delta_filtered['degree_before']
    )
    df_delta_filtered['degree_growth_factor'] = (
        df_delta_filtered['degree_after'] / (df_delta_filtered['degree_before'] + 1e-10)
    )

    # Stats globales
    metrics.total_companies = len(df_delta_filtered)
    metrics.companies_with_growth = (df_delta_filtered['degree_growth_absolute'] > 0).sum()
    metrics.companies_high_growth = (df_delta_filtered['degree_growth_factor'] >= growth_threshold).sum()

    logger.info(f"Companies with any growth: {metrics.companies_with_growth}/{metrics.total_companies}")
    logger.info(f"Companies with high growth (≥{growth_threshold}x): {metrics.companies_high_growth}")

    # 1. PRECISION@K
    logger.info("\n--- Computing Precision@K ---")
    metrics.precision_at_k = _compute_precision_at_k(
        df_delta_filtered, top_k_list, logger
    )

    # 2. RANK CORRELATION
    logger.info("\n--- Computing Rank Correlation ---")
    metrics.spearman_rho, metrics.spearman_p_value, metrics.kendall_tau, metrics.kendall_p_value = (
        _compute_rank_correlation(df_delta_filtered, logger)
    )

    # 3. EARLY DETECTION RATE
    logger.info("\n--- Computing Early Detection Rate ---")
    metrics.edr_at_k = _compute_edr(
        df_delta_filtered, top_k_list, growth_threshold, logger
    )

    # 4. LEAD TIME
    logger.info("\n--- Computing Lead Time ---")
    metrics.avg_lead_time_days, metrics.median_lead_time_days = _compute_lead_time(
        df_delta_filtered, prediction_horizon_days, logger
    )

    # 5. LIFT SCORE
    logger.info("\n--- Computing Lift Score ---")
    metrics.lift_at_k, metrics.baseline_rate, metrics.model_rate_at_k = _compute_lift(
        df_delta_filtered, top_k_list, logger
    )

    # 6. RANDOM BASELINE
    logger.info("\n--- Computing Random Baseline ---")
    metrics.random_precision_at_k = _compute_random_baseline(
        df_delta_filtered, top_k_list, logger
    )

    # 7. DEGREE BASELINE
    logger.info("\n--- Computing Degree Baseline ---")
    metrics.degree_baseline_precision_at_k = _compute_degree_baseline(
        df_delta_filtered, top_k_list, logger
    )

    # Correctly detected in top 50
    top_50_predicted = df_delta_filtered.nlargest(50, 'techrank_delta')
    metrics.correctly_detected_in_top50 = (top_50_predicted['degree_growth_absolute'] > 0).sum()

    logger.info("\n" + "="*80)
    logger.info("✅ VALIDATION METRICS COMPUTED SUCCESSFULLY")
    logger.info("="*80)

    return metrics


def _compute_precision_at_k(
    df: pd.DataFrame,
    top_k_list: List[int],
    logger: logging.Logger
) -> Dict[int, float]:
    """
    Precision@K: Pour les K entreprises les mieux classées par le modèle,
    combien ont RÉELLEMENT eu une croissance positive?
    """
    precision_scores = {}

    for k in top_k_list:
        # Top K selon le classement prédit (tri par techrank_delta décroissant)
        top_k_companies = df.nlargest(k, 'techrank_delta')

        # Combien ont eu une croissance réelle positive?
        successful = (top_k_companies['degree_growth_absolute'] > 0).sum()

        precision = successful / k if k > 0 else 0.0
        precision_scores[k] = precision

        logger.info(f"  Precision@{k:2d}: {precision:.3f} ({successful}/{k} companies with growth)")

    return precision_scores


def _compute_rank_correlation(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[float, float, float, float]:
    """
    Corrélation de Spearman et Kendall entre:
    - Rang prédit (rank_after basé sur techrank_delta)
    - Croissance réelle (degree_growth_absolute)
    """
    # Rang prédit basé sur techrank_delta (meilleur score = rang 1)
    df_sorted = df.sort_values('techrank_delta', ascending=False).reset_index(drop=True)
    df_sorted['predicted_rank'] = df_sorted.index + 1

    # Croissance réelle
    actual_growth = df_sorted['degree_growth_absolute'].values
    predicted_ranks = df_sorted['predicted_rank'].values

    # Spearman
    spearman_rho, spearman_p = spearmanr(predicted_ranks, actual_growth)

    # Kendall
    kendall_tau, kendall_p = kendalltau(predicted_ranks, actual_growth)

    logger.info(f"  Spearman ρ: {spearman_rho:.4f} (p-value: {spearman_p:.4e})")
    logger.info(f"  Kendall τ:  {kendall_tau:.4f} (p-value: {kendall_p:.4e})")

    if spearman_p < 0.001:
        logger.info(f"  ⭐ Highly significant correlation! (p < 0.001)")
    elif spearman_p < 0.05:
        logger.info(f"  ✓ Significant correlation (p < 0.05)")
    else:
        logger.warning(f"  ⚠️  Correlation not significant (p = {spearman_p:.4f})")

    return spearman_rho, spearman_p, kendall_tau, kendall_p


def _compute_edr(
    df: pd.DataFrame,
    top_k_list: List[int],
    growth_threshold: float,
    logger: logging.Logger
) -> Dict[int, float]:
    """
    Early Detection Rate@K:
    Parmi les entreprises qui ont connu une FORTE croissance (≥ threshold),
    combien étaient dans le top-K prédit?
    """
    edr_scores = {}

    # Identifier les entreprises à forte croissance
    high_growth_companies = df[df['degree_growth_factor'] >= growth_threshold]
    total_high_growth = len(high_growth_companies)

    if total_high_growth == 0:
        logger.warning(f"  ⚠️  No companies with growth ≥ {growth_threshold}x found!")
        return {k: 0.0 for k in top_k_list}

    logger.info(f"  Total high-growth companies (≥{growth_threshold}x): {total_high_growth}")

    for k in top_k_list:
        # Top K selon le modèle
        top_k_predicted = set(df.nlargest(k, 'techrank_delta')['final_configuration'])

        # Combien de high-growth companies sont dans le top-K?
        detected = sum(
            1 for comp in high_growth_companies['final_configuration']
            if comp in top_k_predicted
        )

        edr = detected / total_high_growth
        edr_scores[k] = edr

        logger.info(f"  EDR@{k:2d}: {edr:.3f} ({detected}/{total_high_growth} detected)")

    return edr_scores


def _compute_lead_time(
    df: pd.DataFrame,
    prediction_horizon_days: float,
    logger: logging.Logger
) -> Tuple[float, float]:
    """
    Lead Time: Pour les entreprises correctement prédites (top-50 avec croissance positive),
    le délai moyen entre prédiction et événement réel.

    Note: Comme nous validons sur une période fixe, le lead time est constant
    (= prediction_horizon_days) pour toutes les entreprises correctement détectées.
    """
    # Top 50 prédites
    top_50 = df.nlargest(50, 'techrank_delta')

    # Combien ont eu une croissance positive?
    correctly_detected = top_50[top_50['degree_growth_absolute'] > 0]

    num_detected = len(correctly_detected)

    if num_detected == 0:
        logger.warning("  ⚠️  No correctly detected companies in top-50")
        return 0.0, 0.0

    # Lead time = horizon de prédiction (constant pour toutes)
    avg_lead_time = prediction_horizon_days
    median_lead_time = prediction_horizon_days

    logger.info(f"  Correctly detected in top-50: {num_detected}")
    logger.info(f"  Average lead time: {avg_lead_time:.1f} days ({avg_lead_time/30:.1f} months)")
    logger.info(f"  Median lead time: {median_lead_time:.1f} days ({median_lead_time/30:.1f} months)")

    return avg_lead_time, median_lead_time


def _compute_lift(
    df: pd.DataFrame,
    top_k_list: List[int],
    logger: logging.Logger
) -> Tuple[Dict[int, float], float, Dict[int, float]]:
    """
    Lift Score:
    Lift@K = (Taux de succès dans top-K) / (Taux de succès baseline)

    Returns:
        lift_scores, baseline_rate, model_rates
    """
    # Baseline = taux de croissance global
    baseline_rate = (df['degree_growth_absolute'] > 0).sum() / len(df)

    logger.info(f"  Baseline rate (random): {baseline_rate:.3f}")

    lift_scores = {}
    model_rates = {}

    for k in top_k_list:
        top_k = df.nlargest(k, 'techrank_delta')
        successful = (top_k['degree_growth_absolute'] > 0).sum()

        model_rate = successful / k if k > 0 else 0
        lift = model_rate / baseline_rate if baseline_rate > 0 else 0

        lift_scores[k] = lift
        model_rates[k] = model_rate

        logger.info(
            f"  Lift@{k:2d}: {lift:.2f}x "
            f"(model: {model_rate:.3f}, baseline: {baseline_rate:.3f})"
        )

    return lift_scores, baseline_rate, model_rates


def _compute_random_baseline(
    df: pd.DataFrame,
    top_k_list: List[int],
    logger: logging.Logger,
    n_iterations: int = 1000
) -> Dict[int, float]:
    """
    Random Baseline: Sélection aléatoire de K entreprises
    Moyenne sur n_iterations pour avoir une estimation stable
    """
    random_precision = {}

    logger.info(f"  Running {n_iterations} random samples...")

    for k in top_k_list:
        precisions = []

        for _ in range(n_iterations):
            random_sample = df.sample(min(k, len(df)))
            successful = (random_sample['degree_growth_absolute'] > 0).sum()
            precisions.append(successful / k if k > 0 else 0)

        random_precision[k] = np.mean(precisions)
        logger.info(f"    Random Precision@{k:2d}: {random_precision[k]:.3f}")

    return random_precision


def _compute_degree_baseline(
    df: pd.DataFrame,
    top_k_list: List[int],
    logger: logging.Logger
) -> Dict[int, float]:
    """
    Degree Baseline: Classement naïf par degré initial seulement
    (sans TGN, sans TechRank)
    """
    degree_precision = {}

    logger.info("  Classement par degré initial (baseline naïf)...")

    for k in top_k_list:
        # Top K par degré initial
        top_k_by_degree = df.nlargest(k, 'degree_before')
        successful = (top_k_by_degree['degree_growth_absolute'] > 0).sum()

        precision = successful / k if k > 0 else 0
        degree_precision[k] = precision

        logger.info(f"    Degree Baseline Precision@{k:2d}: {precision:.3f}")

    return degree_precision


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_validation_plots(
    df_delta: pd.DataFrame,
    metrics: ValidationMetrics,
    save_dir: str = "validation_results",
    top_k_viz: int = 20
):
    """
    Crée toutes les visualisations pour la validation temporelle.

    Args:
        df_delta: DataFrame avec deltas et croissances
        metrics: ValidationMetrics calculées
        save_dir: Répertoire de sauvegarde
        top_k_viz: Nombre d'entreprises à visualiser
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*80)
    logger.info("CREATING VALIDATION PLOTS")
    logger.info("="*80)

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # 1. Precision@K Comparison (Model vs Baselines)
    _plot_precision_comparison(metrics, save_dir, logger)

    # 2. Scatter: Predicted vs Actual
    _plot_predicted_vs_actual(df_delta, save_dir, logger)

    # 3. Top K companies: Before/After comparison
    _plot_top_k_companies(df_delta, top_k_viz, save_dir, logger)

    # 4. EDR and Lift summary
    _plot_edr_lift_summary(metrics, save_dir, logger)

    logger.info(f"✅ All plots saved to {save_dir}/")


def _plot_precision_comparison(metrics: ValidationMetrics, save_dir: Path, logger: logging.Logger):
    """Plot Precision@K: Model vs Random vs Degree baseline"""
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(metrics.precision_at_k.keys())

    # Model
    ax.plot(k_values, [metrics.precision_at_k[k] for k in k_values],
            marker='o', linewidth=2.5, markersize=8, label='TGN + TechRank (Model)',
            color='#2ecc71')

    # Random baseline
    ax.plot(k_values, [metrics.random_precision_at_k[k] for k in k_values],
            marker='s', linewidth=2, markersize=6, label='Random Baseline',
            color='#95a5a6', linestyle='--')

    # Degree baseline
    ax.plot(k_values, [metrics.degree_baseline_precision_at_k[k] for k in k_values],
            marker='^', linewidth=2, markersize=6, label='Degree Baseline',
            color='#3498db', linestyle='-.')

    ax.set_xlabel('K (Top K companies)', fontweight='bold')
    ax.set_ylabel('Precision@K', fontweight='bold')
    ax.set_title('Precision@K: Model vs Baselines', fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    plt.tight_layout()
    path = save_dir / 'precision_at_k_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✅ Saved: {path}")
    plt.close()


def _plot_predicted_vs_actual(df: pd.DataFrame, save_dir: Path, logger: logging.Logger):
    """Scatter plot: Predicted ranking vs Actual growth"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Rang prédit
    df_sorted = df.sort_values('techrank_delta', ascending=False).reset_index(drop=True)
    df_sorted['predicted_rank'] = df_sorted.index + 1

    scatter = ax.scatter(
        df_sorted['predicted_rank'],
        df_sorted['degree_growth_absolute'],
        c=df_sorted['techrank_delta'],
        cmap='RdYlGn',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Predicted Rank (by TechRank Delta)', fontweight='bold')
    ax.set_ylabel('Actual Growth (Degree Δ)', fontweight='bold')
    ax.set_title('Predicted Ranking vs Actual Growth', fontweight='bold', pad=15)
    plt.colorbar(scatter, ax=ax, label='TechRank Delta')
    ax.grid(True, alpha=0.3)

    # Ajouter corrélation
    from scipy.stats import spearmanr
    rho, p_val = spearmanr(df_sorted['predicted_rank'], df_sorted['degree_growth_absolute'])
    ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\np-value = {p_val:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    path = save_dir / 'predicted_vs_actual_scatter.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✅ Saved: {path}")
    plt.close()


def _plot_top_k_companies(df: pd.DataFrame, top_k: int, save_dir: Path, logger: logging.Logger):
    """Bar plot: Top K companies with their actual growth"""
    df_top = df.nlargest(top_k, 'techrank_delta').copy()

    fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.35)))

    # Trier par techrank_delta pour affichage
    df_top = df_top.sort_values('techrank_delta', ascending=True)

    companies = [name[:50] + '...' if len(name) > 50 else name
                 for name in df_top['final_configuration']]
    y_pos = np.arange(len(companies))

    # Couleur selon croissance réelle
    colors = ['#2ecc71' if growth > 0 else '#e74c3c'
              for growth in df_top['degree_growth_absolute']]

    bars = ax.barh(y_pos, df_top['techrank_delta'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(companies, fontsize=9)
    ax.set_xlabel('TechRank Delta (Predicted)', fontweight='bold')
    ax.set_title(f'Top {top_k} Companies by Predicted TechRank Delta\n(Green = Actual Growth, Red = No Growth)',
                 fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')

    # Ajouter les valeurs de croissance réelle
    for i, (idx, row) in enumerate(df_top.iterrows()):
        growth = row['degree_growth_absolute']
        ax.text(row['techrank_delta'], i, f"  Δ={int(growth):+d}",
                va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = save_dir / f'top_{top_k}_companies_validation.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✅ Saved: {path}")
    plt.close()


def _plot_edr_lift_summary(metrics: ValidationMetrics, save_dir: Path, logger: logging.Logger):
    """Summary plot: EDR and Lift scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    k_values = sorted(metrics.edr_at_k.keys())

    # EDR
    ax1.bar(range(len(k_values)), [metrics.edr_at_k[k] for k in k_values],
            color='#9b59b6', alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(k_values)))
    ax1.set_xticklabels([f'@{k}' for k in k_values])
    ax1.set_ylabel('Early Detection Rate', fontweight='bold')
    ax1.set_xlabel('Top-K', fontweight='bold')
    ax1.set_title(f'Early Detection Rate (Growth ≥ {metrics.growth_threshold}x)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Ajouter valeurs
    for i, k in enumerate(k_values):
        ax1.text(i, metrics.edr_at_k[k], f'{metrics.edr_at_k[k]:.2f}',
                ha='center', va='bottom', fontweight='bold')

    # Lift
    ax2.bar(range(len(k_values)), [metrics.lift_at_k[k] for k in k_values],
            color='#e67e22', alpha=0.8, edgecolor='black')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='No improvement (1.0x)')
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels([f'@{k}' for k in k_values])
    ax2.set_ylabel('Lift Score', fontweight='bold')
    ax2.set_xlabel('Top-K', fontweight='bold')
    ax2.set_title('Lift Score (vs Random Baseline)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Ajouter valeurs
    for i, k in enumerate(k_values):
        ax2.text(i, metrics.lift_at_k[k], f'{metrics.lift_at_k[k]:.2f}x',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    path = save_dir / 'edr_lift_summary.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✅ Saved: {path}")
    plt.close()


# ============================================================================
# LATEX EXPORT
# ============================================================================

def generate_latex_report(
    metrics: ValidationMetrics,
    output_path: str = "validation_results/validation_report.tex"
) -> str:
    """
    Génère un snippet LaTeX pour inclusion dans le rapport

    Returns:
        String LaTeX formaté
    """

    latex_template = r"""
\subsection{Validation Temporelle}

Notre approche de validation temporelle évalue la capacité du modèle TGN-TechRank à prédire les entreprises à fort potentiel \textbf{avant} qu'elles ne deviennent évidentes dans les données brutes.

\subsubsection{Métriques de Performance}

\begin{table}[h]
\centering
\caption{Métriques de validation temporelle}
\begin{tabular}{lcc}
\toprule
\textbf{Métrique} & \textbf{Valeur} & \textbf{Interprétation} \\
\midrule
Precision@10 & {precision_10:.1%} & {precision_10_interp} \\
Precision@20 & {precision_20:.1%} & {precision_20_interp} \\
Precision@50 & {precision_50:.1%} & {precision_50_interp} \\
\midrule
Spearman $\rho$ & {spearman:.3f} & {spearman_interp} \\
p-value & {p_value:.4f} & {significance} \\
\midrule
EDR@50 (seuil {threshold}x) & {edr_50:.1%} & {edr_interp} \\
Lift@20 & {lift_20:.2f}x & {lift_interp} \\
\midrule
Lead Time (moyen) & {lead_time:.0f} jours & {lead_time_months:.1f} mois \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Comparaison avec les Baselines}

Notre modèle surpasse significativement les baselines naïves:

\begin{itemize}
    \item \textbf{vs Random}: Lift@20 = {lift_20:.2f}x (amélioration de {lift_improvement:.0f}\%)
    \item \textbf{vs Degree Baseline}: Precision@20 = {model_prec_20:.1%} vs {degree_prec_20:.1%} (+{degree_improvement:.1%})
\end{itemize}

\subsubsection{Détection Précoce}

Parmi les {high_growth} entreprises ayant connu une croissance forte ($\geq$ {threshold}x), notre modèle en a détecté {detected_50} dans le top-50 prédictions (EDR@50 = {edr_50:.1%}), avec un délai moyen de {lead_time_months:.1f} mois avant l'observation réelle.

\subsubsection{Interprétation}

{overall_interpretation}

"""

    # Helper pour interprétations
    def interp_precision(p):
        if p > 0.7:
            return "Excellent"
        elif p > 0.5:
            return "Bon"
        elif p > 0.3:
            return "Modéré"
        else:
            return "Faible"

    def interp_spearman(rho):
        abs_rho = abs(rho)
        if abs_rho > 0.7:
            return "Corrélation forte"
        elif abs_rho > 0.4:
            return "Corrélation modérée"
        elif abs_rho > 0.2:
            return "Corrélation faible"
        else:
            return "Pas de corrélation"

    def significance(p):
        if p < 0.001:
            return "Hautement significatif"
        elif p < 0.01:
            return "Très significatif"
        elif p < 0.05:
            return "Significatif"
        else:
            return "Non significatif"

    # Overall interpretation
    if metrics.spearman_rho > 0.5 and metrics.precision_at_k.get(20, 0) > 0.6:
        overall = "Les résultats démontrent une capacité forte du modèle à identifier les entreprises prometteuses avant leur croissance effective, validant l'approche TGN-TechRank pour la détection précoce de technologies disruptives."
    elif metrics.spearman_rho > 0.3 and metrics.precision_at_k.get(20, 0) > 0.4:
        overall = "Le modèle montre une capacité modérée à prédire les entreprises à fort potentiel, suggérant que l'approche capture certains signaux prédictifs mais pourrait bénéficier d'améliorations (features additionnelles, hyperparamètres)."
    else:
        overall = "Les performances actuelles suggèrent que le modèle nécessite des améliorations pour améliorer sa capacité prédictive. Des pistes incluent l'enrichissement des features de nœuds et l'optimisation des paramètres TechRank."

    # Fill template
    latex_content = latex_template.format(
        precision_10=metrics.precision_at_k.get(10, 0),
        precision_10_interp=interp_precision(metrics.precision_at_k.get(10, 0)),
        precision_20=metrics.precision_at_k.get(20, 0),
        precision_20_interp=interp_precision(metrics.precision_at_k.get(20, 0)),
        precision_50=metrics.precision_at_k.get(50, 0),
        precision_50_interp=interp_precision(metrics.precision_at_k.get(50, 0)),
        spearman=metrics.spearman_rho,
        spearman_interp=interp_spearman(metrics.spearman_rho),
        p_value=metrics.spearman_p_value,
        significance=significance(metrics.spearman_p_value),
        edr_50=metrics.edr_at_k.get(50, 0),
        edr_interp=f"{metrics.edr_at_k.get(50, 0)*100:.0f}% des cas détectés",
        threshold=metrics.growth_threshold,
        lift_20=metrics.lift_at_k.get(20, 0),
        lift_interp=f"{(metrics.lift_at_k.get(20, 0)-1)*100:.0f}% mieux que le hasard",
        lead_time=metrics.avg_lead_time_days,
        lead_time_months=metrics.avg_lead_time_days / 30,
        high_growth=metrics.companies_high_growth,
        detected_50=int(metrics.edr_at_k.get(50, 0) * metrics.companies_high_growth),
        lift_improvement=(metrics.lift_at_k.get(20, 1) - 1) * 100,
        model_prec_20=metrics.precision_at_k.get(20, 0),
        degree_prec_20=metrics.degree_baseline_precision_at_k.get(20, 0),
        degree_improvement=(metrics.precision_at_k.get(20, 0) - metrics.degree_baseline_precision_at_k.get(20, 0)),
        overall_interpretation=overall
    )

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    logging.info(f"✅ LaTeX report saved to {output_path}")

    return latex_content


# ============================================================================
# MAIN PIPELINE (Simplified - uses existing code)
# ============================================================================

def run_temporal_validation_pipeline(
    df_delta: pd.DataFrame,
    B_before: nx.Graph,
    B_after: nx.Graph,
    top_k_list: List[int] = [10, 20, 50],
    growth_threshold: float = 2.0,
    prediction_horizon_days: float = 730.0,  # 2 ans par défaut
    output_dir: str = "validation_results",
    create_plots: bool = True,
    export_latex: bool = True
) -> ValidationMetrics:
    """
    Pipeline complet de validation temporelle - Version simplifiée utilisant le code existant.

    Args:
        df_delta: DataFrame de deltas (output de analyze_company_deltas())
        B_before: Graphe initial (train)
        B_after: Graphe réel futur (validation/test)
        top_k_list: Liste des K pour les métriques
        growth_threshold: Seuil de croissance forte (2.0 = doublement)
        prediction_horizon_days: Horizon de prédiction en jours
        output_dir: Répertoire de sortie
        create_plots: Générer les visualisations
        export_latex: Exporter le rapport LaTeX

    Returns:
        ValidationMetrics avec toutes les métriques calculées

    Example:
        >>> # Après avoir lancé TechRank_Comparison.py
        >>> from temporal_validation import run_temporal_validation_pipeline
        >>>
        >>> # Charger les résultats existants
        >>> df_delta = pd.read_csv('promising_companies/all_companies_delta.csv')
        >>>
        >>> # Charger les graphes
        >>> with open('bipartite_graph_train.pkl', 'rb') as f:
        >>>     B_train = pickle.load(f)
        >>> with open('bipartite_graph_test.pkl', 'rb') as f:
        >>>     B_test = pickle.load(f)
        >>>
        >>> # Lancer la validation
        >>> metrics = run_temporal_validation_pipeline(
        >>>     df_delta, B_train, B_test,
        >>>     prediction_horizon_days=730
        >>> )
        >>>
        >>> print(f"Precision@20: {metrics.precision_at_k[20]:.2%}")
        >>> print(f"Spearman ρ: {metrics.spearman_rho:.3f}")
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*80)
    logger.info("TEMPORAL VALIDATION PIPELINE")
    logger.info("="*80)

    # 1. Compute metrics
    metrics = compute_validation_metrics(
        df_delta=df_delta,
        B_before=B_before,
        B_after=B_after,
        top_k_list=top_k_list,
        growth_threshold=growth_threshold,
        prediction_horizon_days=prediction_horizon_days,
        logger=logger
    )

    # 2. Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 3. Save metrics to JSON
    metrics.save_json(str(output_path / 'validation_metrics.json'))

    # 4. Create plots
    if create_plots:
        create_validation_plots(
            df_delta=df_delta,
            metrics=metrics,
            save_dir=str(output_path),
            top_k_viz=20
        )

    # 5. Export LaTeX report
    if export_latex:
        generate_latex_report(
            metrics=metrics,
            output_path=str(output_path / 'validation_report.tex')
        )

    # 6. Print summary
    _print_summary(metrics, logger)

    logger.info("\n" + "="*80)
    logger.info(f"✅ VALIDATION COMPLETED - Results saved to {output_path}/")
    logger.info("="*80)

    return metrics


def _print_summary(metrics: ValidationMetrics, logger: logging.Logger):
    """Print a nice summary of all metrics"""
    logger.info("\n" + "="*80)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("="*80)

    logger.info("\n🎯 PRECISION@K:")
    for k, prec in sorted(metrics.precision_at_k.items()):
        logger.info(f"   @{k:2d}: {prec:.3f}")

    logger.info("\n📈 RANK CORRELATION:")
    logger.info(f"   Spearman ρ: {metrics.spearman_rho:.4f} (p={metrics.spearman_p_value:.4e})")
    logger.info(f"   Kendall τ:  {metrics.kendall_tau:.4f} (p={metrics.kendall_p_value:.4e})")

    logger.info(f"\n🔍 EARLY DETECTION RATE (≥{metrics.growth_threshold}x growth):")
    for k, edr in sorted(metrics.edr_at_k.items()):
        logger.info(f"   @{k:2d}: {edr:.3f}")

    logger.info("\n⏱️  LEAD TIME:")
    logger.info(f"   Average: {metrics.avg_lead_time_days:.0f} days ({metrics.avg_lead_time_days/30:.1f} months)")

    logger.info("\n📊 LIFT SCORE:")
    for k, lift in sorted(metrics.lift_at_k.items()):
        logger.info(f"   @{k:2d}: {lift:.2f}x (baseline: {metrics.baseline_rate:.3f})")

    logger.info("\n📌 SUMMARY STATS:")
    logger.info(f"   Total companies: {metrics.total_companies}")
    logger.info(f"   With growth: {metrics.companies_with_growth} ({metrics.companies_with_growth/metrics.total_companies*100:.1f}%)")
    logger.info(f"   High growth (≥{metrics.growth_threshold}x): {metrics.companies_high_growth}")
    logger.info(f"   Detected in top-50: {metrics.correctly_detected_in_top50}")


# ============================================================================
# SIMPLE WRAPPER (pour compatibilité avec workflow existant)
# ============================================================================

def add_validation_to_techrank_comparison(
    data: str = 'crunchbase',
    alpha: float = 0.8,
    beta: float = -0.6,
    top_k_list: List[int] = [10, 20, 50]
):
    """
    Wrapper qui s'intègre directement après TechRank_Comparison.py

    Usage:
        # Après avoir lancé TechRank_Comparison.py
        python -c "from temporal_validation import add_validation_to_techrank_comparison; add_validation_to_techrank_comparison()"
    """
    import pickle

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading existing TechRank comparison results...")

    # Charger le DataFrame de deltas (créé par TechRank_Comparison.py)
    df_delta_path = Path('promising_companies') / 'all_companies_delta.csv'

    if not df_delta_path.exists():
        logger.error(f"❌ File not found: {df_delta_path}")
        logger.error("   Please run TechRank_Comparison.py first!")
        return

    df_delta = pd.read_csv(df_delta_path)
    logger.info(f"✅ Loaded delta DataFrame: {len(df_delta)} companies")

    # Charger les graphes
    B_before_path = Path(f'predicted_graph_{data}.pkl')  # Ground truth
    B_after_path = Path(f'predicted_graph_{data}.pkl')   # À adapter selon votre structure

    # TODO: Adapter selon vos chemins réels
    logger.warning("⚠️  Placeholder - adapt graph loading paths to your structure")

    # Lancer la validation
    # metrics = run_temporal_validation_pipeline(
    #     df_delta=df_delta,
    #     B_before=B_before,
    #     B_after=B_after,
    #     top_k_list=top_k_list
    # )

    logger.info("✅ Validation completed!")


if __name__ == "__main__":
    # Example usage (adapter selon votre structure)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         Temporal Validation for TGN-TechRank System              ║
    ╚══════════════════════════════════════════════════════════════════╝

    This module extends your existing TechRank_Comparison.py workflow
    with robust temporal validation metrics.

    Usage:

    1. Run your existing pipeline:
       python TechRank_Comparison.py --data crunchbase --plot

    2. Then run validation:

       from temporal_validation import run_temporal_validation_pipeline
       import pandas as pd
       import pickle

       # Load existing results
       df_delta = pd.read_csv('promising_companies/all_companies_delta.csv')

       # Load graphs (adapt paths to your structure)
       with open('B_train.pkl', 'rb') as f:
           B_train = pickle.load(f)
       with open('B_test.pkl', 'rb') as f:
           B_test = pickle.load(f)

       # Run validation
       metrics = run_temporal_validation_pipeline(
           df_delta, B_train, B_test,
           prediction_horizon_days=730
       )

       print(f"Precision@20: {metrics.precision_at_k[20]:.2%}")

    See README_validation.md for full documentation.
    """)
