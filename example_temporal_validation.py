#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Complete Temporal Validation Workflow
===============================================

Ce script montre comment utiliser le module temporal_validation.py
avec vos données Crunchbase et votre pipeline TGN-TechRank existant.

Usage:
    python example_temporal_validation.py

Author: Claude
Date: 2025-01-15
"""

import logging
import pickle
from pathlib import Path
import pandas as pd
import networkx as nx

from temporal_validation import (
    run_temporal_validation_pipeline,
    compute_validation_metrics,
    create_validation_plots,
    generate_latex_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_existing_results(data_name='crunchbase'):
    """
    Charge les résultats déjà créés par TechRank_Comparison.py

    Returns:
        df_delta, B_train, B_test
    """
    logger.info("="*80)
    logger.info("LOADING EXISTING RESULTS")
    logger.info("="*80)

    # 1. DataFrame de deltas (créé par TechRank_Comparison.py)
    df_delta_path = Path('techrank_comparison/company_techrank_deltas.csv')

    if not df_delta_path.exists():
        logger.error(f"❌ File not found: {df_delta_path}")
        logger.error("   Please run TechRank_Comparison.py first!")
        logger.error("   Command: python TechRank_Comparison.py --data crunchbase --alpha 0.8 --beta -0.6 --plot")
        return None, None, None

    df_delta = pd.read_csv(df_delta_path)
    logger.info(f"✅ Loaded delta DataFrame: {len(df_delta)} companies")

    # 2. Charger les graphes (adapter selon votre structure)
    # Option A: Si vous avez sauvegardé les splits temporels
    split_dir = Path('savings/bipartite_invest_comp/networks/split_50000')

    train_graph_path = split_dir / 'bipartite_graph_train.gpickle'
    test_graph_path = split_dir / 'bipartite_graph_test.gpickle'

    if not train_graph_path.exists() or not test_graph_path.exists():
        logger.error(f"❌ Graph files not found in {split_dir}")
        logger.error("   Please run data preparation with temporal split first!")
        logger.error("   Command: python -m data.bipartite_investor_comp")
        return None, None, None

    with open(train_graph_path, 'rb') as f:
        B_train = pickle.load(f)

    with open(test_graph_path, 'rb') as f:
        B_test = pickle.load(f)

    logger.info(f"✅ Loaded train graph: {B_train.number_of_nodes()} nodes, {B_train.number_of_edges()} edges")
    logger.info(f"✅ Loaded test graph:  {B_test.number_of_nodes()} nodes, {B_test.number_of_edges()} edges")

    return df_delta, B_train, B_test


def example_basic_usage(df_delta, B_train, B_test):
    """
    Exemple 1: Utilisation basique - Calculer seulement les métriques
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: BASIC USAGE")
    logger.info("="*80)

    metrics = compute_validation_metrics(
        df_delta=df_delta,
        B_before=B_train,
        B_after=B_test,
        top_k_list=[10, 20, 50],
        growth_threshold=2.0,
        prediction_horizon_days=730  # 2 ans
    )

    # Afficher les résultats principaux
    logger.info("\n📊 KEY RESULTS:")
    logger.info(f"   Precision@20: {metrics.precision_at_k[20]:.2%}")
    logger.info(f"   Spearman ρ:   {metrics.spearman_rho:.4f} (p={metrics.spearman_p_value:.4e})")
    logger.info(f"   EDR@50:       {metrics.edr_at_k[50]:.2%}")
    logger.info(f"   Lift@20:      {metrics.lift_at_k[20]:.2f}x")

    return metrics


def example_full_pipeline(df_delta, B_train, B_test):
    """
    Exemple 2: Pipeline complet avec visualisations et export LaTeX
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: FULL PIPELINE")
    logger.info("="*80)

    metrics = run_temporal_validation_pipeline(
        df_delta=df_delta,
        B_before=B_train,
        B_after=B_test,
        top_k_list=[10, 20, 50],
        growth_threshold=2.0,
        prediction_horizon_days=730,
        output_dir='validation_results',
        create_plots=True,
        export_latex=True
    )

    logger.info("\n✅ Pipeline completed!")
    logger.info(f"   Results saved to: validation_results/")
    logger.info(f"   Files generated:")
    logger.info(f"     - validation_metrics.json")
    logger.info(f"     - validation_report.tex")
    logger.info(f"     - precision_at_k_comparison.png")
    logger.info(f"     - predicted_vs_actual_scatter.png")
    logger.info(f"     - top_20_companies_validation.png")
    logger.info(f"     - edr_lift_summary.png")

    return metrics


def example_sensitivity_analysis(df_delta, B_train, B_test):
    """
    Exemple 3: Analyse de sensibilité - Tester différents seuils de croissance
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: SENSITIVITY ANALYSIS")
    logger.info("="*80)

    thresholds = [1.5, 2.0, 2.5, 3.0]
    results = {}

    for threshold in thresholds:
        logger.info(f"\nTesting growth_threshold = {threshold}x...")

        metrics = compute_validation_metrics(
            df_delta=df_delta,
            B_before=B_train,
            B_after=B_test,
            top_k_list=[20, 50],
            growth_threshold=threshold,
            prediction_horizon_days=730
        )

        results[threshold] = {
            'precision_20': metrics.precision_at_k[20],
            'edr_50': metrics.edr_at_k[50],
            'num_high_growth': metrics.companies_high_growth,
            'lift_20': metrics.lift_at_k[20]
        }

        logger.info(f"   Precision@20: {metrics.precision_at_k[20]:.3f}")
        logger.info(f"   EDR@50:       {metrics.edr_at_k[50]:.3f}")
        logger.info(f"   High-growth companies: {metrics.companies_high_growth}")
        logger.info(f"   Lift@20:      {metrics.lift_at_k[20]:.2f}x")

    # Résumé
    logger.info("\n" + "="*80)
    logger.info("SENSITIVITY ANALYSIS SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{'Threshold':<12} {'Precision@20':<15} {'EDR@50':<10} {'Lift@20':<10} {'High-Growth':<15}")
    logger.info("-"*70)
    for threshold, res in results.items():
        logger.info(
            f"{threshold:<12.1f} "
            f"{res['precision_20']:<15.3f} "
            f"{res['edr_50']:<10.3f} "
            f"{res['lift_20']:<10.2f} "
            f"{res['num_high_growth']:<15d}"
        )

    return results


def example_error_analysis(df_delta, B_train, B_test):
    """
    Exemple 4: Analyse des erreurs - Identifier les faux positifs et négatifs
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 4: ERROR ANALYSIS")
    logger.info("="*80)

    # Ajouter les croissances réelles
    df_delta['degree_before'] = df_delta['final_configuration'].apply(
        lambda comp: B_train.degree(comp) if comp in B_train else 0
    )
    df_delta['degree_after'] = df_delta['final_configuration'].apply(
        lambda comp: B_test.degree(comp) if comp in B_test else 0
    )
    df_delta['degree_growth'] = df_delta['degree_after'] - df_delta['degree_before']

    # Top 20 prédites
    top_20_predicted = df_delta.nlargest(20, 'techrank_delta')

    # FAUX POSITIFS: bien classées mais pas de croissance
    false_positives = top_20_predicted[top_20_predicted['degree_growth'] <= 0]

    logger.info("\n❌ FALSE POSITIVES (Top-20 without real growth):")
    logger.info(f"   Count: {len(false_positives)}/20")

    if len(false_positives) > 0:
        logger.info(f"\n   {'Company':<50} {'TechRank Δ':<12} {'Degree Before':<15} {'Degree After':<15}")
        logger.info("   " + "-"*95)
        for idx, row in false_positives.iterrows():
            logger.info(
                f"   {row['final_configuration'][:48]:<50} "
                f"{row['techrank_delta']:>11.4f} "
                f"{int(row['degree_before']):>14d} "
                f"{int(row['degree_after']):>14d}"
            )
    else:
        logger.info("   None! All top-20 had positive growth ✓")

    # VRAIS POSITIFS
    true_positives = top_20_predicted[top_20_predicted['degree_growth'] > 0]
    logger.info(f"\n✅ TRUE POSITIVES: {len(true_positives)}/20 ({len(true_positives)/20*100:.0f}%)")

    # FAUX NÉGATIFS: forte croissance mais mal classées
    high_growth = df_delta[df_delta['degree_growth'] >= 5]  # Au moins +5 investisseurs
    top_50_predicted_set = set(df_delta.nlargest(50, 'techrank_delta')['final_configuration'])

    missed = high_growth[~high_growth['final_configuration'].isin(top_50_predicted_set)]
    missed = missed.sort_values('degree_growth', ascending=False)

    logger.info(f"\n⚠️  FALSE NEGATIVES (High growth but missed from top-50):")
    logger.info(f"   Count: {len(missed)}")

    if len(missed) > 0:
        logger.info(f"\n   {'Company':<50} {'Degree Δ':<12} {'TechRank Δ':<12}")
        logger.info("   " + "-"*75)
        for idx, row in missed.head(10).iterrows():
            logger.info(
                f"   {row['final_configuration'][:48]:<50} "
                f"{int(row['degree_growth']):>11d} "
                f"{row['techrank_delta']:>11.4f}"
            )

    # Statistiques d'erreur
    total_growth = (df_delta['degree_growth'] > 0).sum()
    precision_20 = len(true_positives) / 20
    recall_50 = len(high_growth[high_growth['final_configuration'].isin(top_50_predicted_set)]) / len(high_growth) if len(high_growth) > 0 else 0

    logger.info(f"\n📊 ERROR STATISTICS:")
    logger.info(f"   Total companies with growth: {total_growth}/{len(df_delta)}")
    logger.info(f"   Precision@20: {precision_20:.3f}")
    logger.info(f"   Recall@50 (high-growth): {recall_50:.3f}")


def main():
    """
    Main function - Lance tous les exemples
    """
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "TEMPORAL VALIDATION - COMPLETE EXAMPLE" + " "*20 + "║")
    logger.info("╚" + "="*78 + "╝")

    # 1. Charger les données
    df_delta, B_train, B_test = load_existing_results()

    if df_delta is None:
        logger.error("\n❌ Cannot proceed without data. Please run the preparation steps first.")
        return

    # 2. Exemple 1: Utilisation basique
    metrics_basic = example_basic_usage(df_delta, B_train, B_test)

    # 3. Exemple 2: Pipeline complet
    metrics_full = example_full_pipeline(df_delta, B_train, B_test)

    # 4. Exemple 3: Analyse de sensibilité
    sensitivity_results = example_sensitivity_analysis(df_delta, B_train, B_test)

    # 5. Exemple 4: Analyse des erreurs
    example_error_analysis(df_delta, B_train, B_test)

    # Résumé final
    logger.info("\n" + "="*80)
    logger.info("✅ ALL EXAMPLES COMPLETED!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Check validation_results/ for plots and reports")
    logger.info("  2. Include validation_report.tex in your LaTeX document")
    logger.info("  3. Analyze false positives/negatives to improve the model")
    logger.info("  4. Try different TechRank parameters (α, β)")
    logger.info("  5. Optimize TGN hyperparameters based on validation metrics")
    logger.info("\nFor more details, see:")
    logger.info("  - README_validation.md (documentation)")
    logger.info("  - temporal_validation_demo.ipynb (interactive notebook)")


if __name__ == "__main__":
    main()
