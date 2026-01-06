
"""
TechRank Delta Analysis : Identifier les entreprises prometteuses
Analyse le delta TechRank (Après TGN - Avant TGN) pour chaque entreprise
"""

import argparse
import logging
import pickle
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def parse_args():
    parser = argparse.ArgumentParser('TechRank Delta Analysis for Companies')
    parser.add_argument('--data', type=str, default='crunchbase', help='Dataset name')
    parser.add_argument('--mapping_dir', type=str, default='data/mappings')
    parser.add_argument('--alpha', type=float, default=0.8, help='TechRank alpha parameter')
    parser.add_argument('--beta', type=float, default=-0.6, help='TechRank beta parameter')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum relative delta to consider (e.g., 0.01 = 1%% improvement, 0.5 = 50%% improvement)')
    parser.add_argument('--top_k', type=int, default=50, 
                        help='Number of top promising companies to export')
    parser.add_argument('--top_k_viz', type=int, default=20, 
                        help='Number of companies to show in before/after visualization')
    parser.add_argument('--save_dir', type=str, default='promising_companies', 
                        help='Directory to save results')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    return parser.parse_args()

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger()

def load_mappings(mapping_dir, dataset_name, logger):
    """Charge les mappings depuis les CSV"""
    mapping_dir = Path(mapping_dir)
    
    csv_company = mapping_dir / f"{dataset_name}_filtered_company_map_verification.csv"
    csv_investor = mapping_dir / f"{dataset_name}_filtered_investor_map_verification.csv"
    
    if csv_company.exists() and csv_investor.exists():
        df_comp = pd.read_csv(csv_company)
        df_inv = pd.read_csv(csv_investor)
        
        id_to_company = {int(row['Company_ID_TGN']): str(row['Company_Name']) 
                        for _, row in df_comp.iterrows()}
        id_to_investor = {int(row['Investor_ID_TGN']): str(row['Investor_Name']) 
                         for _, row in df_inv.iterrows()}
        
        logger.info(f"[OK] Mappings loaded: {len(id_to_company)} companies, {len(id_to_investor)} investors")
        return id_to_company, id_to_investor
    
    logger.error("[ERROR] CSV mapping files not found!")
    return {}, {}

def build_ground_truth_graph(dataset_name, id_to_company, id_to_investor, logger):
    """
    Construit le graphe bipartite AVANT TGN (données brutes de test)
    Convention : companies=0, investors=1
    """
    logger.info("\n" + "="*70)
    logger.info("CONSTRUCTION DU GRAPHE AVANT TGN (Ground Truth)")
    logger.info("="*70)
    
    # Charger les données de test
    test_csv = f'./data/data_split/{dataset_name}_filtered_test.csv'
    if not Path(test_csv).exists():
        logger.error(f"[ERROR] Test file not found: {test_csv}")
        return None, None, None
    
    df_test = pd.read_csv(test_csv)
    logger.info(f"Loaded test data: {len(df_test)} interactions")
    
    # Filtrer seulement les arêtes positives (label=1)
    df_positive = df_test[df_test['label'] == 1].copy()
    logger.info(f"Positive edges (label=1): {len(df_positive)}")
    
    # Créer le graphe bipartite
    B = nx.Graph()
    
    dict_companies = {}
    dict_investors = {}
    edge_funding_info = {}
    
    COMPANY_BIPARTITE = 0
    INVESTOR_BIPARTITE = 1
    
    logger.info("Building bipartite graph from raw test data...")
    
    for idx, row in df_positive.iterrows():
        company_id = int(row['u'])
        investor_id = int(row['i'])
        timestamp = float(row['ts'])

        # Récupérer les noms de base
        company_base_name = id_to_company.get(company_id, f"company_{company_id}")
        investor_base_name = id_to_investor.get(investor_id, f"investor_{investor_id}")

        # [WARNING] IMPORTANT: Utiliser les mêmes préfixes que dans TGN_eval.py
        # pour que les noms correspondent lors du merge des dataframes!
        company_name = f"COMPANY_{company_base_name}"
        investor_name = f"INVESTOR_{investor_base_name}"
        
        if company_name not in B:
            B.add_node(company_name, bipartite=COMPANY_BIPARTITE)
        
        if investor_name not in B:
            B.add_node(investor_name, bipartite=INVESTOR_BIPARTITE)
        
        if company_name not in dict_companies:
            dict_companies[company_name] = {
                'id': company_id,
                'name': company_name,
                'base_name': company_base_name,  # Pour affichage
                'technologies': [],
                'total_funding': 0.0,
                'num_funding_rounds': 0
            }

        if investor_name not in dict_investors:
            dict_investors[investor_name] = {
                'investor_id': investor_id,
                'name': investor_name,
                'base_name': investor_base_name,  # Pour affichage
                'num_investments': 0,
                'total_invested': 0.0
            }
        
        edge_key = (company_name, investor_name)
        if edge_key not in edge_funding_info:
            edge_funding_info[edge_key] = {
                'funding_rounds': [],
                'total_raised_amount_usd': 0.0,
                'num_funding_rounds': 0,
                'weight': 0.0
            }
        
        weight = 1.0  # Poids unitaire pour ground truth
        
        edge_funding_info[edge_key]['funding_rounds'].append({
            'timestamp': timestamp,
            'weight': weight
        })
        edge_funding_info[edge_key]['total_raised_amount_usd'] += weight
        edge_funding_info[edge_key]['num_funding_rounds'] += 1
        edge_funding_info[edge_key]['weight'] += weight
        
        dict_companies[company_name]['total_funding'] += weight
        dict_companies[company_name]['num_funding_rounds'] += 1
        dict_investors[investor_name]['num_investments'] += 1
        dict_investors[investor_name]['total_invested'] += weight
    
    for (comp_name, inv_name), funding_info in edge_funding_info.items():
        B.add_edge(
            comp_name,
            inv_name,
            weight=funding_info['weight'],
            funding_rounds=funding_info['funding_rounds'],
            total_raised_amount_usd=funding_info['total_raised_amount_usd'],
            num_funding_rounds=funding_info['num_funding_rounds']
        )
    
    # Ajouter tous les nœuds mappés avec préfixes
    for nid, base_name in id_to_company.items():
        company_name = f"COMPANY_{base_name}"
        if company_name not in B:
            B.add_node(company_name, bipartite=COMPANY_BIPARTITE)
            if company_name not in dict_companies:
                dict_companies[company_name] = {
                    'id': int(nid),
                    'name': company_name,
                    'base_name': base_name,  # Pour affichage
                    'technologies': [],
                    'total_funding': 0.0,
                    'num_funding_rounds': 0
                }

    for nid, base_name in id_to_investor.items():
        investor_name = f"INVESTOR_{base_name}"
        if investor_name not in B:
            B.add_node(investor_name, bipartite=INVESTOR_BIPARTITE)
            if investor_name not in dict_investors:
                dict_investors[investor_name] = {
                    'investor_id': int(nid),
                    'name': investor_name,
                    'base_name': base_name,  # Pour affichage
                    'num_investments': 0,
                    'total_invested': 0.0
                }
    
    logger.info(f"[OK] Ground truth graph created:")
    logger.info(f"   Nodes: {B.number_of_nodes()}")
    logger.info(f"   Edges: {B.number_of_edges()}")
    logger.info(f"   Companies: {len(dict_companies)}")
    logger.info(f"   Investors: {len(dict_investors)}")

    # Debug: vérifier les premiers noms
    logger.info(f"\nDEBUG - Premiers noms de companies dans dict_companies:")
    for i, name in enumerate(list(dict_companies.keys())[:5]):
        logger.info(f"   {i+1}. {name}")

    logger.info(f"\nDEBUG - Premiers noms d'investors dans dict_investors:")
    for i, name in enumerate(list(dict_investors.keys())[:5]):
        logger.info(f"   {i+1}. {name}")
    
    return B, dict_companies, dict_investors

def load_predicted_graph(dataset_name, logger):
    """Charge le graphe APRÈS TGN (prédictions)"""
    logger.info("\n" + "="*70)
    logger.info("CHARGEMENT DU GRAPHE APRÈS TGN (Predictions)")
    logger.info("="*70)
    
    graph_path = Path(f'predicted_graph_{dataset_name}.pkl')
    dict_comp_path = Path(f'dict_companies_{dataset_name}.pickle')
    dict_inv_path = Path(f'dict_investors_{dataset_name}.pickle')
    
    if not graph_path.exists():
        logger.error(f"[ERROR] Predicted graph not found: {graph_path}")
        logger.error("   Run: python tgn_evaluation_fixed.py --run_techrank first!")
        return None, None, None
    
    with open(graph_path, 'rb') as f:
        B_pred = pickle.load(f)
    
    with open(dict_comp_path, 'rb') as f:
        dict_companies_pred = pickle.load(f)
    
    with open(dict_inv_path, 'rb') as f:
        dict_investors_pred = pickle.load(f)
    
    logger.info(f"[OK] Predicted graph loaded:")
    logger.info(f"   Nodes: {B_pred.number_of_nodes()}")
    logger.info(f"   Edges: {B_pred.number_of_edges()}")
    logger.info(f"   Companies: {len(dict_companies_pred)}")
    logger.info(f"   Investors: {len(dict_investors_pred)}")

    # Debug: vérifier les premiers noms
    logger.info(f"\nDEBUG - Premiers noms dans dict_companies_pred:")
    for i, name in enumerate(list(dict_companies_pred.keys())[:5]):
        logger.info(f"   {i+1}. {name}")

    logger.info(f"\nDEBUG - Premiers noms dans dict_investors_pred:")
    for i, name in enumerate(list(dict_investors_pred.keys())[:5]):
        logger.info(f"   {i+1}. {name}")

    return B_pred, dict_companies_pred, dict_investors_pred

def run_techrank_on_graph(B, dict_companies, dict_investors, alpha, beta, label, logger):
    """Lance TechRank sur un graphe"""
    logger.info(f"\n{'='*70}")
    logger.info(f"CALCUL TECHRANK - {label}")
    logger.info(f"{'='*70}")
    logger.info(f"Alpha: {alpha}, Beta: {beta}")
    
    try:
        from code.TechRank import run_techrank
        
        num_nodes = B.number_of_nodes()
        
        logger.info(f"Running TechRank...")
        # [WARNING] IMPORTANT: Ordre corrigé pour correspondre à TGN_eval.py
        # run_techrank retourne (df_companies, df_investors) dans cet ordre
        df_companies_rank, df_investors_rank, _, _ = run_techrank(
            num_comp=num_nodes,
            num_tech=num_nodes,
            flag_cybersecurity=False,
            alpha=alpha,
            beta=beta,
            do_plot=False,
            dict_investors=dict_investors,
            dict_comp=dict_companies,
            B=B
        )
        
        logger.info(f"[OK] TechRank completed for {label}")

        if df_companies_rank is not None:
            non_zero = (df_companies_rank['techrank'] > 0).sum()
            max_score = df_companies_rank['techrank'].max()
            logger.info(f"   Companies with score > 0: {non_zero}/{len(df_companies_rank)}")
            logger.info(f"   Max TechRank score: {max_score:.6f}")

        # [WARNING] IMPORTANT: Retourner dans l'ordre (companies, investors) pour cohérence
        return df_companies_rank, df_investors_rank
        
    except Exception as e:
        logger.error(f"[ERROR] Error running TechRank: {e}", exc_info=True)
        return None, None

def analyze_company_deltas(df_before, df_after, threshold, top_k, save_dir, logger, plot=False, top_k_viz=20):
    """
    Analyse des deltas TechRank RELATIFS par entreprise.
    Delta relatif = (after - before) / (before + epsilon)
    Le delta n'est calculé QUE si techrank_before != 0
    """
    logger.info("\n" + "="*70)
    logger.info("ANALYSE DES DELTAS TECHRANK RELATIFS PAR ENTREPRISE")
    logger.info("="*70)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if df_before is None or df_after is None:
        logger.error("[ERROR] Missing TechRank results")
        return None, None

    # Merge BEFORE / AFTER
    df_delta = df_before.merge(
        df_after,
        on='final_configuration',
        how='outer',
        suffixes=('_before', '_after')
    )

    # Remplacer NaN par 0 UNIQUEMENT pour les scores
    df_delta['techrank_before'] = df_delta['techrank_before'].fillna(0)
    df_delta['techrank_after'] = df_delta['techrank_after'].fillna(0)

    # Créer une colonne d'affichage sans préfixe COMPANY_ ou INVESTOR_
    df_delta['display_name'] = df_delta['final_configuration'].str.replace('COMPANY_', '', regex=False)
    df_delta['display_name'] = df_delta['display_name'].str.replace('INVESTOR_', '', regex=False)

    # 🔒 Filtrer : on garde uniquement les entreprises avec un TechRank initial significatif
    # Threshold minimum pour éviter les deltas relatifs artificiellement élevés
    min_techrank_threshold = 1e-6

    logger.info(f"\nFiltrage des entreprises:")
    logger.info(f"   Avant filtrage: {len(df_delta)} entreprises")
    logger.info(f"   Threshold minimum: techrank_before > {min_techrank_threshold}")

    # Compter combien seront filtrées AVANT de filtrer
    num_filtered = (df_delta['techrank_before'] <= min_techrank_threshold).sum()

    df_delta = df_delta[df_delta['techrank_before'] > min_techrank_threshold].copy()

    logger.info(f"   Après filtrage: {len(df_delta)} entreprises")
    logger.info(f"   Entreprises filtrées (techrank_before ≤ {min_techrank_threshold}): {num_filtered}")

    # Sauvegarde intermédiaire
    df_delta.to_csv('techrank_comparison/company_techrank_merged_filtered.csv', index=False)

    # ============================
    # Calcul des deltas RELATIFS
    # ============================
    # Delta relatif: (after - before) / (before + epsilon)
    # Normalise par la valeur initiale pour comparer les améliorations relatives
    epsilon = 1e-8
    df_delta['techrank_delta'] = (
        (df_delta['techrank_after'] - df_delta['techrank_before']) /
        (df_delta['techrank_before'] + epsilon)
    )

    # Delta en pourcentage (×100 du delta relatif)
    df_delta['techrank_delta_pct'] = df_delta['techrank_delta'] * 100

    df_delta.to_csv('techrank_comparison/company_techrank_deltas.csv', index=False)

    # ============================
    # Calcul des rangs
    # ============================
    df_delta['rank_before'] = df_delta['techrank_before'].rank(
        ascending=False, method='dense'
    )
    df_delta['rank_after'] = df_delta['techrank_after'].rank(
        ascending=False, method='dense'
    )
    df_delta['rank_change'] = df_delta['rank_before'] - df_delta['rank_after']

    # ============================
    # Spearman Rank Correlation
    # ============================
    if len(df_delta) > 1:
        spearman_corr, p_value = spearmanr(
            df_delta['techrank_before'],
            df_delta['techrank_after']
        )

        logger.info(f"\nSPEARMAN RANK CORRELATION ANALYSIS:")
        logger.info(f"   Correlation: {spearman_corr:.4f}")
        logger.info(f"   P-value: {p_value:.6f}")
        logger.info(f"   Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        logger.info(f"   Sample size: {len(df_delta)} companies")

        # Interprétation
        if spearman_corr > 0.9:
            interpretation = "Very strong positive correlation (rankings are very similar)"
        elif spearman_corr > 0.7:
            interpretation = "Strong positive correlation (rankings are similar)"
        elif spearman_corr > 0.5:
            interpretation = "Moderate positive correlation (some similarity in rankings)"
        elif spearman_corr > 0.3:
            interpretation = "Weak positive correlation (rankings differ significantly)"
        elif spearman_corr > -0.3:
            interpretation = "No correlation (rankings are independent)"
        else:
            interpretation = "Negative correlation (rankings are inverted)"

        logger.info(f"   Interpretation: {interpretation}")

    # ============================
    # Stats globales
    # ============================
    logger.info(f"\nSTATISTIQUES GLOBALES:")
    logger.info(f"   Total companies analysed: {len(df_delta)}")
    logger.info(f"   Positive delta: {(df_delta['techrank_delta'] > 0).sum()}")
    logger.info(f"   Negative delta: {(df_delta['techrank_delta'] < 0).sum()}")
    logger.info(f"   Unchanged: {(df_delta['techrank_delta'] == 0).sum()}")

    # ============================
    # Entreprises prometteuses
    # ============================
    df_promising = df_delta[df_delta['techrank_delta'] > threshold].copy()
    df_promising = df_promising.sort_values('techrank_delta', ascending=False)

    logger.info(f"\nENTREPRISES PROMETTEUSES (delta > {threshold}): {len(df_promising)}")

    if not df_promising.empty:
        logger.info(
            f"\n{'Rank':<6} {'Company':<50} {'Before':<12} {'After':<12} "
            f"{'Δ Relative':<12} {'Δ %':<10} {'Rank Δ':<8}"
        )
        logger.info("-"*120)

        for _, row in df_promising.head(top_k).iterrows():
            # Utiliser display_name au lieu de final_configuration
            display = row['display_name'][:48] if 'display_name' in row else row['final_configuration'][:48]
            logger.info(
                f"{int(row['rank_after']):<6} "
                f"{display:<50} "
                f"{row['techrank_before']:<12.6f} "
                f"{row['techrank_after']:<12.6f} "
                f"{row['techrank_delta']:<12.6f} "
                f"{row['techrank_delta_pct']:<10.2f} "
                f"{int(row['rank_change']):+<8d}"
            )

        df_promising.head(top_k).to_csv(
            save_dir / f'promising_companies_top{top_k}.csv', index=False
        )

    # ============================
    # Entreprises en déclin
    # ============================
    df_declining = df_delta[df_delta['techrank_delta'] < -threshold]
    df_declining = df_declining.sort_values('techrank_delta')

    logger.info(f"\nENTREPRISES EN DÉCLIN (delta < -{threshold}): {len(df_declining)}")

    # ============================
    # Visualisations
    # ============================
    if plot:
        create_visualizations(
            df_delta, df_promising, threshold, save_dir, logger, top_k_viz
        )

    return df_delta, df_promising


def create_cross_methodology_bump_chart(save_dir, logger):
    """
    Crée un bump chart comparant les classements TGN avec les métriques Crunchbase.

    Compare 4 métriques pour le top 4 des entreprises TGN:
    - TGN Ranking (basé sur TechRank après prédiction)
    - Heat Score Ranking (activité Crunchbase)
    - Growth Score Ranking (croissance Crunchbase)
    - Crunchbase Ranking (rang global CB)

    Les données proviennent du tableau du rapport (extraction manuelle Crunchbase).
    """
    logger.info(f"\nGenerating cross-methodology bump chart...")

    # Données du tableau du rapport (top 4 TGN)
    data = {
        'Company': ['Quantistry', 'Phaseshift Tech.', 'QSIM Plus', 'Global Telecom'],
        'TGN_Rank': [1, 2, 3, 4],
        'Growth_Score': [92, 86, 90, 94],
        'CB_Rank': [24664, 36647, 115969, 18185],
        'Heat_Score': [78, 64, 55, 92]
    }

    df = pd.DataFrame(data)

    # Calculer les rangs relatifs entre les 4 entreprises
    df['Heat_Score_Rank'] = df['Heat_Score'].rank(ascending=False, method='dense').astype(int)
    df['Growth_Score_Rank'] = df['Growth_Score'].rank(ascending=False, method='dense').astype(int)
    df['CB_Rank_Relative'] = df['CB_Rank'].rank(ascending=True, method='dense').astype(int)

    logger.info(f"\nRanking comparison:")
    for _, row in df.iterrows():
        logger.info(f"  {row['Company']}: TGN={row['TGN_Rank']}, "
                   f"Heat={row['Heat_Score_Rank']}, "
                   f"Growth={row['Growth_Score_Rank']}, "
                   f"CB={row['CB_Rank_Relative']}")

    # Créer le bump chart
    fig, ax = plt.subplots(figsize=(14, 10))

    # Positions sur l'axe X
    positions = [0, 1, 2, 3]
    labels = ['TGN\nRanking', 'Heat Score\nRanking', 'Growth Score\nRanking', 'Crunchbase\nRanking']

    # Couleurs distinctes pour chaque entreprise
    colors = {
        'Quantistry': '#e74c3c',          # Rouge
        'Phaseshift Tech.': '#3498db',    # Bleu
        'QSIM Plus': '#2ecc71',           # Vert
        'Global Telecom': '#9b59b6'       # Violet
    }

    # Tracer les lignes pour chaque entreprise
    for idx, row in df.iterrows():
        company = row['Company']
        ranks = [
            row['TGN_Rank'],
            row['Heat_Score_Rank'],
            row['Growth_Score_Rank'],
            row['CB_Rank_Relative']
        ]

        color = colors[company]

        # Tracer la ligne
        ax.plot(positions, ranks,
               color=color, alpha=0.8, linewidth=3.5,
               marker='o', markersize=14,
               markeredgecolor='black', markeredgewidth=1.5,
               label=company, zorder=3)

        # Ajouter le nom de l'entreprise à droite
        ax.text(3.08, ranks[3], company,
               va='center', fontsize=13, fontweight='bold', color=color)

        # Ajouter les rangs sur chaque point
        for pos, rank in zip(positions, ranks):
            ax.text(pos, rank - 0.15, str(rank),
                   ha='center', va='bottom', fontsize=10,
                   fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=color, alpha=0.8))

    # Configuration des axes
    ax.set_xlim(-0.3, 3.7)
    ax.set_ylim(4.6, 0.4)  # Inverser l'axe Y (1 = meilleur en haut)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank (1 = Best)', fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['1st', '2nd', '3rd', '4th'], fontsize=13, fontweight='bold')

    ax.set_title('Cross-Methodology Ranking Comparison: TGN vs Crunchbase Metrics\n'
                 'Top 4 Companies from TGN Predictions',
                 fontsize=15, fontweight='bold', pad=20)

    # Grille horizontale
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Légende
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95,
             title='Companies', title_fontsize=13, ncol=1)

    plt.tight_layout()

    # Sauvegarder
    png_path = save_dir / 'bump_chart_cross_methodology.png'
    pdf_path = save_dir / 'bump_chart_cross_methodology.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    logger.info(f"   [OK] Cross-methodology bump chart saved:")
    logger.info(f"      PNG: {png_path}")
    logger.info(f"      PDF: {pdf_path}")

    plt.close()

    # Sauvegarder les données en CSV
    csv_path = save_dir / 'cross_methodology_rankings.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"      CSV: {csv_path}")

    # Analyse des divergences
    logger.info(f"\nDivergence Analysis:")
    for idx, row in df.iterrows():
        company = row['Company']
        tgn_rank = row['TGN_Rank']
        heat_rank = row['Heat_Score_Rank']
        growth_rank = row['Growth_Score_Rank']

        heat_div = tgn_rank - heat_rank
        growth_div = tgn_rank - growth_rank

        logger.info(f"\n  {company}:")
        logger.info(f"    TGN vs Heat Score:   {heat_div:+d}")
        logger.info(f"    TGN vs Growth Score: {growth_div:+d}")


def create_bump_chart(df_delta, save_dir, logger, top_n=20):
    """
    Crée un bump chart pour visualiser les changements de ranking.

    Un bump chart montre comment le rang de chaque entreprise évolue
    entre "Before TGN" et "After TGN" avec des lignes connectant les positions.

    Args:
        df_delta: DataFrame avec rank_before et rank_after
        save_dir: Dossier où sauvegarder le graphique
        top_n: Nombre d'entreprises à afficher (par défaut 20)
    """
    logger.info(f"\nGenerating bump chart for top {top_n} rank changes...")

    # Sélectionner les top N entreprises avec le plus grand changement de rang absolu
    df_top = df_delta.nlargest(top_n, 'rank_change').copy()

    # Trier par rank_after pour avoir un affichage cohérent
    df_top = df_top.sort_values('rank_after')

    fig, ax = plt.subplots(figsize=(14, max(10, top_n * 0.5)))

    # Utiliser display_name si disponible
    name_col = 'display_name' if 'display_name' in df_top.columns else 'final_configuration'

    # Créer les lignes du bump chart
    for idx, row in df_top.iterrows():
        company_name = row[name_col]
        rank_before = row['rank_before']
        rank_after = row['rank_after']
        rank_change = row['rank_change']

        # Couleur: vert si amélioration (rank_change > 0), rouge si dégradation
        if rank_change > 0:
            color = '#2ecc71'  # Vert pour amélioration
            alpha = 0.8
            linewidth = 2.5
        elif rank_change < 0:
            color = '#e74c3c'  # Rouge pour dégradation
            alpha = 0.6
            linewidth = 1.5
        else:
            color = '#95a5a6'  # Gris pour inchangé
            alpha = 0.4
            linewidth = 1.0

        # Tracer la ligne entre before et after
        ax.plot([0, 1], [rank_before, rank_after],
               color=color, alpha=alpha, linewidth=linewidth,
               marker='o', markersize=8, markeredgecolor='black', markeredgewidth=0.5)

        # Ajouter le nom de l'entreprise à droite (position after)
        # Tronquer le nom si trop long
        display_name = company_name[:35] + '...' if len(company_name) > 35 else company_name
        ax.text(1.02, rank_after, display_name,
               va='center', fontsize=10, fontweight='bold')

        # Ajouter le changement de rang à gauche (position before)
        change_text = f"+{int(rank_change)}" if rank_change > 0 else f"{int(rank_change)}"
        ax.text(-0.02, rank_before, change_text,
               va='center', ha='right', fontsize=9,
               color=color, fontweight='bold')

    # Configuration des axes
    ax.set_xlim(-0.15, 1.4)
    ax.set_ylim(df_top['rank_after'].max() + 5, df_top['rank_before'].min() - 5)  # Inverser l'axe Y

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before TGN', 'After TGN'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank (1 = Best)', fontsize=13, fontweight='bold')
    ax.set_title(f'Ranking Shifts: Top {top_n} Companies by Absolute Rank Change\n'
                f'(Before vs After TGN Predictions)',
                fontsize=14, fontweight='bold', pad=20)

    # Grille horizontale pour faciliter la lecture
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Légende
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=2.5, label='Rank Improved (↑)'),
        Line2D([0], [0], color='#e74c3c', linewidth=1.5, label='Rank Declined (↓)'),
        Line2D([0], [0], color='#95a5a6', linewidth=1.0, label='No Change (=)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    bump_chart_path = save_dir / f'bump_chart_top{top_n}_rank_changes.png'
    plt.savefig(bump_chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"   [OK] Saved bump chart: {bump_chart_path}")
    plt.close()


def create_visualizations(df_delta, df_promising, threshold, save_dir, logger, top_k_viz=20):
    """Crée des visualisations pour l'analyse"""
    logger.info(f"\nGenerating visualizations...")

    save_dir = Path(save_dir)
    
    # Style
    sns.set_style("whitegrid")
    # ============================
    # GLOBAL FONT CONFIGURATION
    # ============================
    plt.rcParams.update({
        'font.size': 18,          # taille par défaut
        'axes.titlesize': 20,     # titres des plots
        'axes.labelsize': 18,     # labels des axes
        'xtick.labelsize': 16,    # ticks axe X
        'ytick.labelsize': 16,    # ticks axe Y
        'legend.fontsize': 16,    # légende
        'figure.titlesize': 22    # titre figure
    })

    # 0. BUMP CHART - Visualisation des changements de ranking
    create_bump_chart(df_delta, save_dir, logger, top_n=top_k_viz)

    # 0b. CROSS-METHODOLOGY BUMP CHART - Comparaison TGN vs Crunchbase
    create_cross_methodology_bump_chart(save_dir, logger)

    # 1. NOUVEAU: Graphique Before/After pour top K entreprises
    if len(df_promising) > 0:
        fig, ax = plt.subplots(figsize=(14, max(10, top_k_viz * 0.4)))
        
        topK = df_promising.head(top_k_viz).copy()
        topK = topK.sort_values('techrank_delta', ascending=True)  # Pour avoir le plus grand en haut

        # Utiliser display_name (sans préfixe COMPANY_)
        name_col = 'display_name' if 'display_name' in topK.columns else 'final_configuration'
        companies = [name[:45] + '...' if len(name) > 45 else name
                    for name in topK[name_col]]
        y_pos = np.arange(len(companies))
        
        bar_height = 0.35
        
        # Barres AVANT TGN (en bleu)
        bars_before = ax.barh(y_pos - bar_height/2, topK['techrank_before'], 
                              bar_height, label='Before TGN', 
                              color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.2)
        
        # Barres APRÈS TGN (en vert)
        bars_after = ax.barh(y_pos + bar_height/2, topK['techrank_after'], 
                            bar_height, label='After TGN', 
                            color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.2)
        
        # Ajouter les valeurs sur les barres
        for i, (before, after) in enumerate(zip(topK['techrank_before'], topK['techrank_after'])):
            # Valeur before
            ax.text(before, i - bar_height/2, f'{before:.4f}', 
                   va='center', ha='left', fontsize=10, fontweight='bold')
            # Valeur after
            ax.text(after, i + bar_height/2, f'{after:.4f}', 
                   va='center', ha='left', fontsize=10, fontweight='bold')
        
        # ax.set_yticks(y_pos)
        # ax.set_yticklabels(companies, fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            companies,
            fontsize=16,
            fontweight='bold'
        )
        ax.set_xlabel('TechRank Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_k_viz} Most Promising Companies: TechRank Before vs After TGN', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Ajouter une ligne verticale à 0
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        before_after_path = save_dir / f'top{top_k_viz}_before_after_comparison.png'
        plt.savefig(before_after_path, dpi=300, bbox_inches='tight')
        logger.info(f"   [OK] Saved plot: {before_after_path}")
        plt.close()
    
    # 2. Distribution des deltas
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram des deltas relatifs
    ax1 = axes[0, 0]
    ax1.hist(df_delta['techrank_delta'], bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('TechRank Relative Delta: (After - Before) / Before', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Companies', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of TechRank Relative Delta', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot: Before vs After
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df_delta['techrank_before'], 
        df_delta['techrank_after'],
        c=df_delta['techrank_delta'],
        cmap='RdYlGn',
        alpha=0.6,
        s=30,
        edgecolors='black',
        linewidth=0.5
    )
    max_val = max(df_delta['techrank_before'].max(), df_delta['techrank_after'].max())
    ax2.plot([0, max_val], [0, max_val], 
             'k--', alpha=0.5, linewidth=2, label='No change line')
    ax2.set_xlabel('TechRank Before TGN', fontsize=11, fontweight='bold')
    ax2.set_ylabel('TechRank After TGN', fontsize=11, fontweight='bold')
    ax2.set_title('TechRank: Before vs After', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Relative Delta')
    
    # Top 20 promising companies - Relative Delta only
    ax3 = axes[1, 0]
    if len(df_promising) > 0:
        top20 = df_promising.head(20).copy()
        y_pos = np.arange(len(top20))
        bars = ax3.barh(y_pos, top20['techrank_delta'], color='#2ecc71', alpha=0.8, edgecolor='black')
        ax3.set_yticks(y_pos)
        # Utiliser display_name (sans préfixe COMPANY_)
        name_col = 'display_name' if 'display_name' in top20.columns else 'final_configuration'
        ax3.set_yticklabels([name[:35] for name in top20[name_col]], fontsize=12)
        ax3.set_xlabel('TechRank Relative Delta', fontsize=11, fontweight='bold')
        ax3.set_title('Top 20 Promising Companies (by Relative Delta)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()

        # Ajouter les valeurs sur les barres (format avec % pour clarté)
        for i, (idx, row) in enumerate(top20.iterrows()):
            ax3.text(row['techrank_delta'], i, f" +{row['techrank_delta']:.3f} ({row['techrank_delta_pct']:.1f}%)",
                    va='center', fontsize=7, fontweight='bold')
    
    # Cumulative distribution
    ax4 = axes[1, 1]
    sorted_deltas = np.sort(df_delta['techrank_delta'].values)
    cumsum = np.arange(1, len(sorted_deltas) + 1) / len(sorted_deltas) * 100
    ax4.plot(sorted_deltas, cumsum, linewidth=2, color='#3498db')
    ax4.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('TechRank Relative Delta', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative % of Companies', fontsize=11, fontweight='bold')
    ax4.set_title('Cumulative Distribution of TechRank Relative Delta', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / 'techrank_delta_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"   [OK] Saved plot: {plot_path}")
    plt.close()
    
    # 3. Rank changes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_top_movers = df_delta.nlargest(30, 'rank_change')
    if len(df_top_movers) > 0:
        y_pos = np.arange(len(df_top_movers))
        bars = ax.barh(y_pos, df_top_movers['rank_change'], color='#9b59b6', alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        # Utiliser display_name (sans préfixe COMPANY_)
        name_col = 'display_name' if 'display_name' in df_top_movers.columns else 'final_configuration'
        ax.set_yticklabels([name[:40] for name in df_top_movers[name_col]], fontsize=9)
        ax.set_xlabel('Rank Improvement (positive = moved up)', fontsize=11, fontweight='bold')
        ax.set_title('Top 30 Companies by Rank Change', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Ajouter les valeurs
        for i, (idx, row) in enumerate(df_top_movers.iterrows()):
            rank_val = int(row['rank_change'])
            sign = "+" if rank_val >= 0 else ""
            ax.text(row['rank_change'], i, f" {sign}{rank_val}",
                   va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        rank_plot_path = save_dir / 'rank_changes.png'
        plt.savefig(rank_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"   [OK] Saved plot: {rank_plot_path}")
        plt.close()

def main():
    args = parse_args()
    logger = setup_logger()
    
    logger.info("="*70)
    logger.info("TECHRANK DELTA ANALYSIS: IDENTIFIER LES ENTREPRISES PROMETTEUSES")
    logger.info("="*70)
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Top K to export: {args.top_k}")
    
    # Charger les mappings
    id_to_company, id_to_investor = load_mappings(args.mapping_dir, args.data, logger)
    
    if not id_to_company or not id_to_investor:
        logger.error("[ERROR] Failed to load mappings. Exiting.")
        return
    
    # 1. Construire le graphe AVANT TGN (ground truth)
    B_before, dict_comp_before, dict_inv_before = build_ground_truth_graph(
        args.data, id_to_company, id_to_investor, logger
    )
    
    if B_before is None:
        logger.error("[ERROR] Failed to build ground truth graph. Exiting.")
        return
    
    # 2. Charger le graphe APRÈS TGN (prédictions)
    B_after, dict_comp_after, dict_inv_after = load_predicted_graph(args.data, logger)
    
    if B_after is None:
        logger.error("[ERROR] Failed to load predicted graph.")
        logger.error("   Run: python tgn_evaluation_fixed.py --run_techrank first!")
        return
    
    # 3. Calculer TechRank AVANT TGN
    # [WARNING] Maintenant run_techrank_on_graph retourne (companies, investors)
    df_comp_before, _ = run_techrank_on_graph(
        B_before, dict_comp_before, dict_inv_before,
        args.alpha, args.beta, "AVANT TGN (Ground Truth)", logger
    )

    # Debug: vérifier les premiers noms
    if df_comp_before is not None and len(df_comp_before) > 0:
        logger.info(f"\nDEBUG - Premiers noms dans df_comp_before:")
        for name in df_comp_before['final_configuration'].head(5):
            logger.info(f"   {name}")

    df_comp_before = df_comp_before[df_comp_before['techrank'] != 0.0]
    df_comp_before.to_csv("techrank_comparison/before_tgn.csv", index=False)


    # 4. Calculer TechRank APRÈS TGN
    # [WARNING] Maintenant run_techrank_on_graph retourne (companies, investors)
    df_comp_after, _ = run_techrank_on_graph(
        B_after, dict_comp_after, dict_inv_after,
        args.alpha, args.beta, "APRÈS TGN (Predictions)", logger
    )
    df_comp_after = df_comp_after[df_comp_after['techrank'] != 0.0]
    df_comp_after.to_csv("techrank_comparison/after_tgn.csv", index=False)

    # 5. Analyser les deltas et identifier les entreprises prometteuses
    df_delta, df_promising = analyze_company_deltas(
        df_comp_before, df_comp_after, 
        args.threshold, args.top_k, 
        args.save_dir, logger, plot=args.plot, top_k_viz=args.top_k_viz
    )
    
    # 6. Résumé final
    logger.info("\n" + "="*70)
    logger.info("[OK] ANALYSE TERMINÉE!")
    logger.info("="*70)
    
    if df_promising is not None and len(df_promising) > 0:
        logger.info(f"\nRÉSULTATS CLÉS:")
        logger.info(f"   - {len(df_promising)} entreprises prometteuses identifiées (delta > {args.threshold})")
        logger.info(f"   - Top {args.top_k} sauvegardées dans: {args.save_dir}/")
        logger.info(f"   - Fichiers générés:")
        logger.info(f"     • promising_companies_top{args.top_k}.csv")
        logger.info(f"     • all_companies_delta.csv")
        if args.plot:
            logger.info(f"     • techrank_delta_analysis.png")
            logger.info(f"     • rank_changes.png")
        
        logger.info(f"\nUTILISATION:")
        logger.info(f"   Les entreprises dans promising_companies_top{args.top_k}.csv sont celles")
        logger.info(f"   que le modèle TGN considère comme sous-évaluées par les données brutes.")
        logger.info(f"   Un delta élevé suggère un potentiel de croissance non capturé dans les")
        logger.info(f"   interactions historiques mais détecté par le modèle.")
    else:
        logger.warning(f"\n[WARNING]  Aucune entreprise avec delta > {args.threshold} trouvée")
        logger.info(f"   Essayez de réduire le threshold avec --threshold")

if __name__ == "__main__":
    main()