import pandas as pd
import pickle
import os
from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix
import json
from tqdm import tqdm


# ===================================================================
# CONFIGURATION
# ===================================================================
NUM_COMP = 500
NUM_TECH = 500

FLAG_CYBERSECURITY = False

# SAVE_DIR_CLASSES = "savings/bipartite_tech_comp/classes"
# SAVE_DIR_NETWORKS = "savings/bipartite_tech_comp/networks"
# SAVE_DIR_M = "savings/bipartite_tech_comp/M"
SAVE_DIR_CLASSES = "savings/bipartite_invest_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_invest_comp/networks"
SAVE_DIR_M = "savings/bipartite_invest_comp/M"
SAVE_DIR_RESULTS = "savings/csv_results"
SAVE_DIR_PLOTS = "plots/rank_evolution"

OPTIMAL_ALPHA_COMP = 0.8
OPTIMAL_BETA_COMP = -0.6

# ===================================================================
# UTILS
# ===================================================================

def create_directories():
    for directory in [SAVE_DIR_CLASSES, SAVE_DIR_NETWORKS, SAVE_DIR_M, SAVE_DIR_RESULTS, SAVE_DIR_PLOTS]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def extract_nodes(B, bipartite_value):
    return [node for node, data in B.nodes(data=True) if data.get('bipartite') == bipartite_value]

def preferences_to_string(preferences):
    if preferences is None:
        return "default"
    return "_".join([f"{k}_{v}" for k,v in preferences.items()])

# ===================================================================
# LOADING FUNCTIONS
# ===================================================================

def load_saved_data(num_comp, num_tech, flag_cybersecurity):
    prefix = "cybersecurity_" if flag_cybersecurity else ""
    
    # name_file_com = f'{SAVE_DIR_CLASSES}/{prefix}dict_companies_ranked_cybersecurity_{num_comp}.pickle'
    # name_file_tech = f'{SAVE_DIR_CLASSES}/{prefix}dict_tech_ranked_cybersecurity_{num_tech}.pickle'
    # name_file_graph = f'{SAVE_DIR_NETWORKS}/{prefix}cybersecurity_bipartite_graph_{num_comp}.gpickle'
    
    name_file_com = f'{SAVE_DIR_CLASSES}/{prefix}dict_companies_{num_comp}.pickle'
    name_file_tech = f'{SAVE_DIR_CLASSES}/{prefix}dict_investors_{num_tech}.pickle'
    name_file_graph = f'{SAVE_DIR_NETWORKS}/{prefix}bipartite_graph_{num_comp}.gpickle'
    
    with open(name_file_com, 'rb') as f:
        dict_companies = pickle.load(f)
    
    with open(name_file_tech, 'rb') as f:
        dict_tech = pickle.load(f)
    
    with open(name_file_graph, 'rb') as f:
        B = pickle.load(f)
    
    print(f" Données chargées: {len(dict_companies)} entreprises, {len(dict_tech)} technologies, Graphe: {B.number_of_nodes()} noeuds, {B.number_of_edges()} arêtes")
    return dict_companies, dict_tech, B

# ===================================================================
# MATRIX OPERATIONS
# ===================================================================

def create_adjacency_matrix(B):
    set0 = extract_nodes(B, 0)
    set1 = extract_nodes(B, 1)
    
    print(f"Création matrice d'adjacence: Investors={len(set0)}, Companies={len(set1)}")
    adj_matrix = bipartite.biadjacency_matrix(B, set0, set1, format='csr')
    print(f" Matrice créée: Shape={adj_matrix.shape}, Arêtes={adj_matrix.sum()}")
    return adj_matrix, set0, set1

def save_matrix(M, num_comp, num_tech, flag_cybersecurity, dict_companies, dict_tech):
    if flag_cybersecurity == False: # all fields
        name_file_M = f'{SAVE_DIR_M}/comp_{len(dict_companies)}_tech_{len(dict_tech)}.npy'                                     
    else: # only companies in cybersecurity
        name_file_M = f'{SAVE_DIR_M}/cybersecurity_comp_{len(dict_companies)}_tech_{len(dict_tech)}.npy'
    
    np.save(name_file_M, M)
    print(f" Matrice sauvegardée: {name_file_M}")

# ===================================================================
# TECHRANK FUNCTIONS - CORRIGÉES
# ===================================================================

def zero_order_score(M):
    k_c = M.sum(axis=1)
    k_t = M.sum(axis=0)
    print(f" Scores ordre zéro: deg_investors={np.mean(k_c):.2f}, deg_comp={np.mean(k_t):.2f}")
    return k_c, k_t

def make_G_hat(M, alpha=1, beta=1, eps=1e-12):
    k_c = M.sum(axis=1).astype(float)
    k_t = M.sum(axis=0).astype(float)
    
    # Éviter division par zéro
    k_c_safe = np.where(k_c > eps, k_c, eps)
    k_t_safe = np.where(k_t > eps, k_t, eps)
    
    # Calcul G_ct
    G_ct = M * (k_c_safe ** (-beta))[:, np.newaxis]
    
    # Normalisation
    col_sums = G_ct.sum(axis=0)
    col_sums_safe = np.where(col_sums > eps, col_sums, eps)
    G_ct = G_ct / col_sums_safe
    
    # Même chose pour G_tc
    G_tc = M.T * (k_t_safe ** (-alpha))[:, np.newaxis]
    col_sums_tc = G_tc.sum(axis=0)
    col_sums_tc_safe = np.where(col_sums_tc > eps, col_sums_tc, eps)
    G_tc = G_tc / col_sums_tc_safe
    
    return {'G_ct': G_ct, 'G_tc': G_tc}

def next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev):
    """Version pour numpy arrays"""
    fitness_next = G_ct.dot(ubiquity_prev)
    ubiquity_next = G_tc.dot(fitness_prev)
    return fitness_next, ubiquity_next

def generator_order_w(M, alpha, beta, normalize=True):
    """Générateur adapté pour numpy arrays"""
    G_hat = make_G_hat(M, alpha, beta)
    G_ct, G_tc = G_hat['G_ct'], G_hat['G_tc']
    fitness_0, ubiquity_0 = zero_order_score(M)
    
    # Normalisation initiale
    if normalize:
        # fitness_0 = fitness_0 / (np.linalg.norm(fitness_0) + 1e-12)
        # ubiquity_0 = ubiquity_0 / (np.linalg.norm(ubiquity_0) + 1e-12)
        fitness_0 = fitness_0 / (np.linalg.norm(fitness_0))
        ubiquity_0 = ubiquity_0 / (np.linalg.norm(ubiquity_0))
    
    fitness_next, ubiquity_next = fitness_0, ubiquity_0
    i = 0
    
    while True:
        i += 1
        fitness_prev, ubiquity_prev = fitness_next, ubiquity_next
        fitness_next, ubiquity_next = next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev)
        
        # NORMALISATION CRITIQUE pour éviter l'explosion/implosion
        if normalize:
            fitness_norm = np.linalg.norm(fitness_next)
            ubiquity_norm = np.linalg.norm(ubiquity_next)
            
            if fitness_norm > 1e-12:
                fitness_next = fitness_next / fitness_norm
            if ubiquity_norm > 1e-12:
                ubiquity_next = ubiquity_next / ubiquity_norm
        
        yield {'iteration': i, 'fitness': fitness_next, 'ubiquity': ubiquity_next}

def find_convergence_debug(M, alpha, beta, fit_or_ubiq, do_plot=False, flag_cybersecurity=False, preferences=''):
    if fit_or_ubiq=='fitness':
        name='Investors'; M_shape=M.shape[0]
    else:
        name='Companies'; M_shape=M.shape[1]
    
    rankings=[]; scores=[]
    prev_rankdata = np.zeros(M_shape)
    stops_flag=0
    weights = generator_order_w(M, alpha, beta, normalize=True)
    max_iterations = 1000
    
    print(f"=== TECHRANK {name} ===")
    print(f"Matrice: {M.shape}, Alpha={alpha}, Beta={beta}")
    
    for stream_data in weights:
        iteration = stream_data['iteration']
        data = stream_data[fit_or_ubiq]
        data = np.nan_to_num(data, nan=0.0)
        rankdata = data.argsort().argsort()
        
        # Debug détaillé
        if iteration <= 10 or iteration % 100 == 0:
            unique_vals = len(np.unique(np.round(data, 6)))
            print(f"Iter {iteration:3d}: min={np.min(data):8.6f}, max={np.max(data):8.6f}, "
                  f"unique={unique_vals:5d}, changes={np.sum(rankdata != prev_rankdata) if iteration > 1 else M_shape}")
        
        if iteration==1:
            initial_conf = rankdata
            
        # Critère de convergence basé sur les changements de rang
        rank_changes = np.sum(rankdata != prev_rankdata) if iteration > 1 else M_shape
        rank_stability = (rank_changes / M_shape) < 0.01  # Moins de 1% de changements
        
        if rank_stability:
            stops_flag += 1
            if stops_flag >= 20:  # 20 itérations stables
                convergence_iteration = iteration
                print(f" CONVERGENCE à l'itération {iteration} (stabilité des rangs)")
                break
        else:
            stops_flag = 0
            
        rankings.append(rankdata)
        scores.append(data)
        prev_rankdata = rankdata
        
        if iteration >= max_iterations:
            convergence_iteration = iteration
            print(f" Maximum d'itérations atteint: {iteration}")
            break
    
    final_data = scores[-1]
    print(f"\n=== RÉSULTATS FINAUX {name} ===")
    print(f"Itérations: {convergence_iteration}")
    print(f"Scores: min={np.min(final_data):.6f}, max={np.max(final_data):.6f}")
    print(f"Valeurs uniques: {len(np.unique(np.round(final_data, 6)))}")
    print(f"Écart-type: {np.std(final_data):.6f}")
    
    return {fit_or_ubiq: final_data, 'iteration': convergence_iteration, 
            'initial_conf': initial_conf, 'final_conf': rankdata}

def rank_df_class_corrected(convergence, dict_class):
    
    """Version CORRIGÉE de rank_df_class avec ground truth"""
    fit_or_ubiq = 'fitness' if 'fitness' in convergence else 'ubiquity'
    list_names = list(dict_class.keys())
    n = len(list_names)
    
    print(f"Debug rank_df_class: {n} éléments, scores range: [{np.min(convergence[fit_or_ubiq]):.6f}, {np.max(convergence[fit_or_ubiq]):.6f}]")
    
    # Vérifier que nous avons assez de données
    if len(convergence[fit_or_ubiq]) < n:
        print(f"ATTENTION: {len(convergence[fit_or_ubiq])} scores pour {n} éléments")
        n = len(convergence[fit_or_ubiq])
        list_names = list_names[:n]
    
    # Créer le DataFrame avec les bonnes données
    df_final = pd.DataFrame({
        'final_configuration': list_names,
        'initial_position': convergence['initial_conf'][:n],
        'techrank': convergence[fit_or_ubiq][:n]
    })
    
    # Trier par techrank (ordre décroissant - meilleur score en premier)
    df_final = df_final.sort_values('techrank', ascending=False).reset_index(drop=True)
    
    # Ajouter le rang explicite
    df_final['TeckRank_int'] = df_final.index + 1
    
    # AJOUT: Récupérer le ground truth depuis les objets si disponible
    ground_truth_ranks = []
    ground_truth_scores = []
    
    for name in list_names:
        # Essayer différentes sources de ground truth
        ground_truth = None
        if hasattr(dict_class[name], 'rank_CB'):
            ground_truth = getattr(dict_class[name], 'rank_CB')
        elif hasattr(dict_class[name], 'rank_analytic'):
            ground_truth = getattr(dict_class[name], 'rank_analytic')
        elif hasattr(dict_class[name], 'true_rank'):
            ground_truth = getattr(dict_class[name], 'true_rank')
        
        ground_truth_ranks.append(ground_truth)
        
        # Essayer de récupérer un score ground truth aussi
        ground_score = None
        if hasattr(dict_class[name], 'true_score'):
            ground_score = getattr(dict_class[name], 'true_score')
        elif hasattr(dict_class[name], 'score_CB'):
            ground_score = getattr(dict_class[name], 'score_CB')
        
        ground_truth_scores.append(ground_score)
    
    # Ajouter les colonnes ground truth au DataFrame
    df_final['ground_truth_rank'] = ground_truth_ranks
    df_final['ground_truth_score'] = ground_truth_scores
    
    # Calculer la performance de l'algorithme
    if any(rank is not None for rank in ground_truth_ranks):
        valid_indices = [i for i, rank in enumerate(ground_truth_ranks) if rank is not None]
        if len(valid_indices) > 10:  # Au moins 10 valeurs pour calculer une corrélation
            techranks = df_final.iloc[valid_indices]['TeckRank_int'].values
            ground_truths = [ground_truth_ranks[i] for i in valid_indices]
            
            # Calculer la corrélation de Spearman
            from scipy.stats import spearmanr
            correlation, p_value = spearmanr(techranks, ground_truths)
            print(f" Corrélation avec ground truth: {correlation:.3f} (p-value: {p_value:.3f})")
            
            # Ajouter la corrélation comme attribut
            df_final.attrs['spearman_correlation'] = correlation
            df_final.attrs['correlation_p_value'] = p_value
    
    print(f"Debug - DataFrame créé: {len(df_final)} lignes")
    print(f"Debug - Scores dans DataFrame: min={df_final['techrank'].min():.6f}, max={df_final['techrank'].max():.6f}")
    
    # Afficher les statistiques sur le ground truth
    valid_ground_truth = [r for r in ground_truth_ranks if r is not None]
    if valid_ground_truth:
        print(f" Ground truth disponible pour {len(valid_ground_truth)} éléments")
    
    return df_final, dict_class

def save_corrected_results(df_investors, df_companies, num_invest, num_comp, flag_cybersecurity, preferences_comp, preferences_tech):
    pref_comp = preferences_to_string(preferences_comp)
    pref_tech = preferences_to_string(preferences_tech)
    
    # Réorganiser les colonnes - AJOUT des colonnes ground truth
    investors_columns = ['TeckRank_int', 'final_configuration', 'techrank', 'ground_truth_rank', 'ground_truth_score', 'initial_position']
    invest_columns = ['TeckRank_int', 'final_configuration', 'techrank', 'ground_truth_rank', 'ground_truth_score', 'initial_position']
    
    # Fichier entreprises
    investors_filename = f'{SAVE_DIR_RESULTS}/companies_rank_{num_invest}_{pref_comp}.csv'
    # Garder seulement les colonnes qui existent
    available_invest_cols = [col for col in investors_columns if col in df_investors.columns]
    df_investors[available_invest_cols].to_csv(investors_filename, index=False)
    print(f" Fichier entreprises sauvegardé: {investors_filename}")
    
    # Fichier technologies
    comp_filename = f'{SAVE_DIR_RESULTS}/technologies_rank_{num_comp}_{pref_tech}.csv'
    available_tech_cols = [col for col in invest_columns if col in df_companies.columns]
    df_companies[available_tech_cols].to_csv(comp_filename, index=False)
    print(f" Fichier technologies sauvegardé: {comp_filename}")
    
    # Afficher les corrélations si disponibles
    if hasattr(df_investors, 'attrs') and 'spearman_correlation' in df_investors.attrs:
        print(f" Corrélation entreprises: {df_investors.attrs['spearman_correlation']:.3f}")
    
    if hasattr(df_companies, 'attrs') and 'spearman_correlation' in df_companies.attrs:
        print(f" Corrélation technologies: {df_companies.attrs['spearman_correlation']:.3f}")


# ===================================================================
# MAIN FUNCTION CORRIGÉE
# ===================================================================

def run_techrank(num_comp=NUM_COMP, num_tech=NUM_TECH, flag_cybersecurity=FLAG_CYBERSECURITY,
                 preferences_comp=None, preferences_tech=None,
                 alpha=OPTIMAL_ALPHA_COMP, beta=OPTIMAL_BETA_COMP, do_plot=False 
                 ,dict_investors=None, dict_comp=None, B=None
                ):
    
    create_directories()
    # dict_investors, dict_comp, B = load_saved_data(num_comp, num_tech, flag_cybersecurity)
    
    # ✅ MODIFICATION: Charger depuis les fichiers OU utiliser les paramètres
    if dict_investors is None or dict_comp is None or B is None:
        # Mode normal: charger depuis les fichiers
        dict_investors, dict_comp, B = load_saved_data(num_comp, num_tech, flag_cybersecurity)
    else:
        # Mode avec données fournies: utiliser directement
        print(f"✓ Utilisation des données fournies: {len(dict_investors)} investors, {len(dict_comp)} companies")
    
    # Vérifier que B n'est pas None
    if B is None:
        raise ValueError("Le graphe B est None! Vérifiez que les données sont correctement chargées.")
    # Création de la matrice
    set0 = extract_nodes(B, 0)
    set1 = extract_nodes(B, 1)
    adj_matrix = bipartite.biadjacency_matrix(B, set0, set1)
    adj_matrix_dense = adj_matrix.todense()
    M = np.squeeze(np.asarray(adj_matrix_dense))
    
    print(f" Matrice M créée: {M.shape}")
    
    # Ranking Companies
    print("\n" + "="*60)
    print("CLASSEMENT DES ENTREPRISES")
    print("="*60)
    start_time = time.time()
    convergence_comp = find_convergence_debug(M, alpha, beta, 'fitness', do_plot=do_plot)
    time_conv_comp = time.time() - start_time
    
    df_final_companies, dict_investors = rank_df_class_corrected(convergence_comp, dict_investors)
    
    # Ranking Technologies  
    print("\n" + "="*60)
    print("CLASSEMENT DES TECHNOLOGIES")
    print("="*60)
    start_time = time.time()
    convergence_tech = find_convergence_debug(M, alpha, beta, 'ubiquity', do_plot=do_plot)
    time_conv_tech = time.time() - start_time
    
    df_final_tech, dict_comp = rank_df_class_corrected(convergence_tech, dict_comp)
    
    # Sauvegarde des résultats
    save_corrected_results(df_final_companies, df_final_tech, num_comp, num_tech, flag_cybersecurity, preferences_comp, preferences_tech)
    
    # Affichage des résultats - CORRECTION ICI : utiliser df_final_tech au lieu de df_tech
    print("\n" + "="*60)
    print("RÉSULTATS FINAUX - TOP 10")
    print("="*60)
    
    print("\n TOP 10 ENTREPRISES (par influence technologique):")
    top_comp = df_final_companies[['TeckRank_int', 'final_configuration', 'techrank']].head(10)
    for _, row in top_comp.iterrows():
        print(f"#{int(row['TeckRank_int']):2d} {row['final_configuration']:30} → Score: {row['techrank']:.6f}")
    
    print("\n TOP 10 TECHNOLOGIES (par adoption stratégique):")
    top_tech = df_final_tech[['TeckRank_int', 'final_configuration', 'techrank']].head(10)  # CORRECTION : df_final_tech
    for _, row in top_tech.iterrows():
        print(f"#{int(row['TeckRank_int']):2d} {row['final_configuration']:30} → Score: {row['techrank']:.6f}")
    
    print(f"\n Temps d'execution:")
    print(f"   Entreprises: {time_conv_comp:.1f}s")
    print(f"   Technologies: {time_conv_tech:.1f}s")
    print(f"   Total: {time_conv_comp + time_conv_tech:.1f}s")
    
    return df_final_companies, df_final_tech, dict_investors, dict_comp

# ===================================================================
# EXECUTION
# ===================================================================

# if __name__ == "__main__":
#     df_investors, df_comp, dict_investors, dict_invest = run_techrank(do_plot=False)
#     # print(df_investors.head(), df_comp.head())