"""
Métriques alternatives pour identifier les entreprises prometteuses
Au-delà du simple delta absolu TechRank
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_comparison_data():
    """Charger les données de comparaison"""
    df = pd.read_csv('techrank_comparison/company_techrank_merged_filtered.csv')
    return df

def compute_alternative_metrics(df):
    """Calculer des metriques alternatives au delta absolu"""

    # Calculer les colonnes manquantes
    df['techrank_delta'] = df['techrank_after'] - df['techrank_before']

    # Utiliser initial_position ou ground_truth_rank selon ce qui est disponible
    if 'initial_position_before' in df.columns:
        df['rank_before'] = df['initial_position_before']
    elif 'ground_truth_rank_before' in df.columns:
        df['rank_before'] = df['ground_truth_rank_before']
    else:
        df['rank_before'] = 0

    if 'initial_position_after' in df.columns:
        df['rank_after'] = df['initial_position_after']
    elif 'ground_truth_rank_after' in df.columns:
        df['rank_after'] = df['ground_truth_rank_after']
    else:
        df['rank_after'] = 0

    df['rank_change'] = df['rank_before'] - df['rank_after']  # Positif = amelioration

    print("="*70)
    print("METRIQUES ALTERNATIVES POUR IDENTIFIER LES ENTREPRISES PROMETTEUSES")
    print("="*70)

    # ========================================
    # 1. RATIO MULTIPLICATEUR (Before → After)
    # ========================================
    print("\n[*] 1. RATIO MULTIPLICATEUR (After / Before)")
    print("-"*70)

    # Éviter division par zéro
    df['ratio_multiplier'] = np.where(
        df['techrank_before'] > 0,
        df['techrank_after'] / df['techrank_before'],
        np.nan
    )

    # Top par ratio (croissance relative)
    df_ratio = df[df['ratio_multiplier'].notna()].sort_values('ratio_multiplier', ascending=False)

    print("\nTop 10 par MULTIPLICATION du score:")
    print(f"{'Rank':<6} {'Company':<45} {'Before':<12} {'After':<12} {'Ratio':<10}")
    print("-"*95)

    for idx, (_, row) in enumerate(df_ratio.head(10).iterrows(), 1):
        display_name = row['display_name'] if 'display_name' in row else row['final_configuration']
        display_name = display_name.replace('COMPANY_', '').replace('INVESTOR_', '')
        print(f"{idx:<6} {display_name[:43]:<45} {row['techrank_before']:<12.6f} "
              f"{row['techrank_after']:<12.6f} {row['ratio_multiplier']:<10.2f}x")

    # ========================================
    # 2. NOUVELLES APPARITIONS (0 → positive)
    # ========================================
    print("\n\n[*] 2. NOUVELLES APPARITIONS (Score passe de 0 a positif)")
    print("-"*70)

    df_new = df[(df['techrank_before'] == 0) & (df['techrank_after'] > 0)].sort_values('techrank_after', ascending=False)

    print(f"\nTotal nouvelles entrées: {len(df_new)}")
    print(f"\nTop 10 nouvelles companies:")
    print(f"{'Rank':<6} {'Company':<45} {'Score After':<12} {'Rank After':<10}")
    print("-"*85)

    for idx, (_, row) in enumerate(df_new.head(10).iterrows(), 1):
        display_name = row['display_name'] if 'display_name' in row else row['final_configuration']
        display_name = display_name.replace('COMPANY_', '').replace('INVESTOR_', '')
        print(f"{idx:<6} {display_name[:43]:<45} {row['techrank_after']:<12.6f} {int(row['rank_after']):<10}")

    # ========================================
    # 3. CHANGEMENT DE RANG ABSOLU
    # ========================================
    print("\n\n[*] 3. MEILLEURE PRESERVATION DU RANG (moins de perte)")
    print("-"*70)
    print("Note: Rangs plus eleves = position plus basse. Graphe apres a plus d'entreprises.")

    # Trier par rank_change croissant (moins négatif = meilleure préservation)
    df_rank_preservation = df.sort_values('rank_change', ascending=False)

    print(f"\nTop 10 par preservation du rang (perte minimale):")
    print(f"{'Company':<45} {'Rank Before':<12} {'Rank After':<12} {'Change':<10}")
    print("-"*85)

    for _, row in df_rank_preservation.head(10).iterrows():
        display_name = row['display_name'] if 'display_name' in row else row['final_configuration']
        display_name = display_name.replace('COMPANY_', '').replace('INVESTOR_', '')
        change_str = f"{int(row['rank_change']):+d}"
        print(f"{display_name[:43]:<45} {int(row['rank_before']):<12} "
              f"{int(row['rank_after']):<12} {change_str:<10}")

    # ========================================
    # 4. PERCENTILE ANALYSIS
    # ========================================
    print("\n\n[*] 4. ANALYSE PAR PERCENTILE")
    print("-"*70)

    # Calculer les percentiles
    df['percentile_before'] = df['techrank_before'].rank(pct=True) * 100
    df['percentile_after'] = df['techrank_after'].rank(pct=True) * 100
    df['percentile_change'] = df['percentile_after'] - df['percentile_before']

    df_percentile = df.sort_values('percentile_change', ascending=False)

    print(f"\nTop 10 par changement de percentile:")
    print(f"{'Company':<45} {'%ile Before':<12} {'%ile After':<12} {'Delta %ile':<10}")
    print("-"*85)

    for _, row in df_percentile.head(10).iterrows():
        display_name = row['display_name'] if 'display_name' in row else row['final_configuration']
        display_name = display_name.replace('COMPANY_', '').replace('INVESTOR_', '')
        print(f"{display_name[:43]:<45} {row['percentile_before']:<12.1f} "
              f"{row['percentile_after']:<12.1f} {row['percentile_change']:+<10.1f}")

    # ========================================
    # 5. Z-SCORE (Distance de la moyenne)
    # ========================================
    print("\n\n[*] 5. Z-SCORE ANALYSIS (Ecarts-types de la moyenne)")
    print("-"*70)

    # Calculer les z-scores
    mean_delta = df['techrank_delta'].mean()
    std_delta = df['techrank_delta'].std()

    df['z_score'] = (df['techrank_delta'] - mean_delta) / std_delta

    df_zscore = df.sort_values('z_score', ascending=False)

    print(f"\nMoyenne delta: {mean_delta:.6f}")
    print(f"Écart-type: {std_delta:.6f}")
    print(f"\nTop 10 par Z-score (outliers positifs):")
    print(f"{'Company':<45} {'Delta':<12} {'Z-score':<10} {'Interpretation':<15}")
    print("-"*95)

    for _, row in df_zscore.head(10).iterrows():
        display_name = row['display_name'] if 'display_name' in row else row['final_configuration']
        display_name = display_name.replace('COMPANY_', '').replace('INVESTOR_', '')

        # Interprétation du Z-score
        z = row['z_score']
        if z > 3:
            interp = "Exceptionnel"
        elif z > 2:
            interp = "Très fort"
        elif z > 1:
            interp = "Au-dessus moy"
        else:
            interp = "Normal"

        print(f"{display_name[:43]:<45} {row['techrank_delta']:<12.6f} "
              f"{row['z_score']:<10.2f} {interp:<15}")

    # ========================================
    # 6. MÉTHODE COMBINÉE (Score composite)
    # ========================================
    print("\n\n[*] 6. SCORE COMPOSITE (Combine plusieurs metriques)")
    print("-"*70)

    # Normaliser chaque métrique entre 0 et 1
    df['norm_delta'] = (df['techrank_delta'] - df['techrank_delta'].min()) / (df['techrank_delta'].max() - df['techrank_delta'].min())
    df['norm_ratio'] = (df['ratio_multiplier'] - df['ratio_multiplier'].min()) / (df['ratio_multiplier'].max() - df['ratio_multiplier'].min())
    df['norm_rank_change'] = (df['rank_change'] - df['rank_change'].min()) / (df['rank_change'].max() - df['rank_change'].min())
    df['norm_percentile'] = (df['percentile_change'] - df['percentile_change'].min()) / (df['percentile_change'].max() - df['percentile_change'].min())

    # Remplacer NaN par 0
    df['norm_delta'] = df['norm_delta'].fillna(0)
    df['norm_ratio'] = df['norm_ratio'].fillna(0)
    df['norm_rank_change'] = df['norm_rank_change'].fillna(0)
    df['norm_percentile'] = df['norm_percentile'].fillna(0)

    # Score composite (moyenne pondérée)
    df['composite_score'] = (
        0.3 * df['norm_delta'] +           # 30% delta absolu
        0.3 * df['norm_ratio'] +            # 30% ratio multiplicateur
        0.2 * df['norm_rank_change'] +      # 20% amélioration rang
        0.2 * df['norm_percentile']         # 20% changement percentile
    )

    df_composite = df.sort_values('composite_score', ascending=False)

    print(f"\nTop 10 par SCORE COMPOSITE (recommandé!):")
    print(f"{'Rank':<6} {'Company':<40} {'Score':<10} {'Delta':<12} {'Ratio':<8}")
    print("-"*85)

    for idx, (_, row) in enumerate(df_composite.head(10).iterrows(), 1):
        display_name = row['display_name'] if 'display_name' in row else row['final_configuration']
        display_name = display_name.replace('COMPANY_', '').replace('INVESTOR_', '')
        ratio = row['ratio_multiplier'] if pd.notna(row['ratio_multiplier']) else 0
        print(f"{idx:<6} {display_name[:38]:<40} {row['composite_score']:<10.3f} "
              f"{row['techrank_delta']:<12.6f} {ratio:<8.2f}x")

    # ========================================
    # SAUVEGARDER LES RÉSULTATS
    # ========================================

    # Sauvegarder le DataFrame enrichi
    output_path = Path('techrank_comparison/promising_companies_all_metrics.csv')
    df_composite.to_csv(output_path, index=False)
    print(f"\n[SAVE] Resultats sauvegardes: {output_path}")

    # Sauvegarder les tops par métrique
    tops = {
        'by_ratio': df_ratio.head(20),
        'new_entries': df_new.head(20),
        'by_rank_preservation': df_rank_preservation.head(20),
        'by_percentile': df_percentile.head(20),
        'by_zscore': df_zscore.head(20),
        'by_composite': df_composite.head(20)
    }

    for name, data in tops.items():
        path = Path(f'techrank_comparison/promising_{name}.csv')
        data.to_csv(path, index=False)
        print(f"   • {path}")

    return df_composite

def print_summary_recommendations(df):
    """Afficher un resume avec recommandations"""
    print("\n" + "="*70)
    print("[RECOMMANDATIONS FINALES]")
    print("="*70)

    print("\nSelon l'objectif d'investissement:")

    print("\n1. CROISSANCE EXPLOSIVE (nouvelles entrées avec score élevé):")
    df_new = df[(df['techrank_before'] == 0) & (df['techrank_after'] > 0)].sort_values('techrank_after', ascending=False)
    for idx, (_, row) in enumerate(df_new.head(3).iterrows(), 1):
        name = row['display_name'].replace('COMPANY_', '').replace('INVESTOR_', '')
        print(f"   {idx}. {name} (Score: {row['techrank_after']:.6f})")

    print("\n2. MOMENTUM FORT (meilleur ratio de croissance):")
    df_ratio = df[df['ratio_multiplier'].notna()].sort_values('ratio_multiplier', ascending=False)
    for idx, (_, row) in enumerate(df_ratio.head(3).iterrows(), 1):
        name = row['display_name'].replace('COMPANY_', '').replace('INVESTOR_', '')
        print(f"   {idx}. {name} (×{row['ratio_multiplier']:.1f})")

    print("\n3. MONTÉE EN PUISSANCE (meilleur score composite):")
    df_composite = df.sort_values('composite_score', ascending=False)
    for idx, (_, row) in enumerate(df_composite.head(3).iterrows(), 1):
        name = row['display_name'].replace('COMPANY_', '').replace('INVESTOR_', '')
        print(f"   {idx}. {name} (Score composite: {row['composite_score']:.3f})")

    print("\n" + "="*70)

if __name__ == "__main__":
    print("Chargement des données...")
    df = load_comparison_data()

    print(f"Total companies: {len(df)}\n")

    # Créer display_name si pas déjà présent
    if 'display_name' not in df.columns:
        df['display_name'] = df['final_configuration'].str.replace('COMPANY_', '', regex=False)
        df['display_name'] = df['display_name'].str.replace('INVESTOR_', '', regex=False)

    # Calculer toutes les métriques
    df_enriched = compute_alternative_metrics(df)

    # Recommandations finales
    print_summary_recommendations(df_enriched)

    print("\n[OK] Analyse terminee!")
