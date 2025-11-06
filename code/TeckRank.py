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
import classes
import json
from tqdm import tqdm

# ===================================================================
# CONFIGURATION
# ===================================================================

# Paramètres du réseau
NUM_COMP = 500
NUM_TECH = 500

# Préférences pour le ranking
PREFERENCES_COMP = {
    "previous_investments": 100,
    "crunchbase_rank": 0
}

PREFERENCES_TECH = {
    "previous_investments": 100
}

# Flags
FLAG_CYBERSECURITY = False

# Chemins de sauvegarde
SAVE_DIR_CLASSES = "savings/bipartite_tech_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_tech_comp/networks"
SAVE_DIR_M = "savings/bipartite_tech_comp/M"
SAVE_DIR_RESULTS = "savings/csv_results"
SAVE_DIR_PLOTS = "plots/rank_evolution"

# Paramètres de l'algorithme
OPTIMAL_ALPHA_COMP = 0.5
OPTIMAL_BETA_COMP = 0.5


# ===================================================================
# UTILS
# ===================================================================

def create_directories():
    """Crée tous les répertoires nécessaires"""
    for directory in [SAVE_DIR_CLASSES, SAVE_DIR_NETWORKS, SAVE_DIR_M, SAVE_DIR_RESULTS, SAVE_DIR_PLOTS]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def extract_nodes(B, bipartite_value):
    """Extrait les noeuds d'un graphe bipartite selon leur valeur bipartite
    
    Args:
        B: graphe bipartite
        bipartite_value: 0 pour companies, 1 pour technologies
    
    Return:
        liste des noeuds
    """
    return [node for node, data in B.nodes(data=True) 
            if data.get('bipartite') == bipartite_value]


def preferences_to_string(preferences):
    """Convertit le dictionnaire de préférences en string pour les noms de fichiers"""
    return str(preferences)


# ===================================================================
# LOADING FUNCTIONS
# ===================================================================

def load_saved_data(num_comp, num_tech, flag_cybersecurity):
    """Charge les données sauvegardées (dictionnaires et graphe)
    
    Args:
        num_comp: nombre de companies
        num_tech: nombre de technologies
        flag_cybersecurity: flag pour filtrage cybersecurity
    
    Return:
        dict_companies, dict_tech, B
    """
    prefix = "cybersecurity_" if flag_cybersecurity else ""
    
    # Chemins des fichiers
    name_file_com = f'{SAVE_DIR_CLASSES}/{prefix}dict_companies_ranked_cybersecurity_{num_comp}.pickle'
    name_file_tech = f'{SAVE_DIR_CLASSES}/{prefix}dict_tech_ranked_cybersecurity_{num_tech}.pickle'
    name_file_graph = f'{SAVE_DIR_NETWORKS}/{prefix}cybersecurity_bipartite_graph_{num_comp}.gpickle'
    
    # Charger les dictionnaires
    with open(name_file_com, 'rb') as f:
        dict_companies = pickle.load(f)
    
    with open(name_file_tech, 'rb') as f:
        dict_tech = pickle.load(f)
    
    # Charger le graphe
    # B = nx.read_gpickle(name_file_graph)
    with open(name_file_graph, 'rb') as f:
        B = pickle.load(f)
    
    print(f"✓ Données chargées:")
    print(f"  - {len(dict_companies)} entreprises")
    print(f"  - {len(dict_tech)} technologies")
    print(f"  - Graphe: {B.number_of_nodes()} noeuds, {B.number_of_edges()} arêtes")
    
    return dict_companies, dict_tech, B


# ===================================================================
# MATRIX OPERATIONS
# ===================================================================

def create_adjacency_matrix(B, flag_cybersecurity):
    """Crée la matrice d'adjacence à partir du graphe bipartite"""
    
    set0 = extract_nodes(B, 0)  # companies
    set1 = extract_nodes(B, 1)  # technologies
    
    print(f"🔧 Création matrice d'adjacence")
    print(f"  - Companies: {len(set0)}")
    print(f"  - Technologies: {len(set1)}")
    
    # Utiliser la fonction de NetworkX
    adj_matrix = bipartite.biadjacency_matrix(B, set0, set1)
    adj_matrix_dense = adj_matrix.todense()
    
    # Convertir en array numpy
    M = np.squeeze(np.asarray(adj_matrix_dense))
    
    print(f"✓ Matrice créée:")
    print(f"  - Shape: {M.shape}")
    print(f"  - Arêtes: {np.sum(M)}")
    
    return M, set0, set1


def save_matrix(M, num_comp, num_tech, flag_cybersecurity):
    """Sauvegarde la matrice M"""
    if flag_cybersecurity:
        name_file_M = f'{SAVE_DIR_M}/cybersecurity_comp_{num_comp}_tech_{num_tech}.npy'
    else:
        name_file_M = f'{SAVE_DIR_M}/comp_{num_comp}_tech_{num_tech}.npy'
    
    np.save(name_file_M, M)
    print(f"✓ Matrice sauvegardée: {name_file_M}")


def M_test_triangular(M, flag_cybersecurity=False):
    """Test the triangularity of M matrix"""

    user_edits_sum = M.sum(axis=1).flatten()
    article_edits_sum = M.sum(axis=0).flatten()

    user_edits_order = user_edits_sum.argsort()
    article_edits_order = article_edits_sum.argsort()

    M_sorted = M[user_edits_order, :]

    if len(M_sorted.shape) > 2:
        M_sorted = M_sorted[0]

    M_sorted_transpose = M_sorted.transpose()
    M_sorted_transpose = M_sorted_transpose[article_edits_order, :]

    if len(M_sorted_transpose.shape) > 2:
        M_sorted_transpose = M_sorted_transpose[0]

    M_sorted_sorted = M_sorted_transpose

    params = {
        'axes.labelsize': 18,
        'axes.titlesize': 28, 
        'legend.fontsize': 22, 
        'xtick.labelsize': 16, 
        'ytick.labelsize': 16}

    plt.figure(figsize=(10, 10))
    plt.rcParams.update(params)
    plt.imshow(M_sorted_sorted, cmap=plt.cm.bone, interpolation='nearest')
    plt.xlabel("Companies")
    plt.ylabel("Technologies")

    # if flag_cybersecurity:
    #     name_plot_M = f'plots/M_triangulize/cybersec_matrix_{str(M_sorted.shape)}'
    # else:
    #     name_plot_M = f'plots/M_triangulize/matrix_{str(M_sorted.shape)}'

    # plt.savefig(f'{name_plot_M}.pdf')
    # plt.savefig(f'{name_plot_M}.png')
    # plt.show()

    return


# ===================================================================
# TECHRANK ALGORITHM FUNCTIONS
# ===================================================================

def zero_order_score(M):
    """Calcule le score d'ordre zéro (degré)"""
    k_c = M.sum(axis=1)  # degree companies
    k_t = M.sum(axis=0)  # degree technologies
    
    print(f"✓ Scores d'ordre zéro calculés")
    print(f"  - Degree moyen companies: {np.mean(k_c):.2f}")
    print(f"  - Degree moyen technologies: {np.mean(k_t):.2f}")
    
    return k_c, k_t


def Gct_beta(M, c, t, k_c, beta):
    """Calcule la probabilité de transition de company c vers technologie t"""
    num = (M[c, t]) * (k_c[c] ** (-beta))

    # sum over the technologies
    M_t = M[:, t].flatten()
    k_c_beta = [x ** (-1 * beta) for x in k_c]

    den = float(np.dot(M_t, k_c_beta))
    return num / den


def Gtc_alpha(M, c, t, k_t, alpha):
    """Calcule la probabilité de transition de technologie t vers company c"""
    num = (M.T[t, c]) * (k_t[t] ** (-alpha))
    
    # sum over the companies
    M_c = M[c, :].flatten()
    k_t_alpha = [x ** (-1 * alpha) for x in k_t]
    
    den = float(np.dot(M_c, k_t_alpha))
    
    return num / den


def make_G_hat(M, alpha=1, beta=1):
    """G hat is Markov chain of length 2"""
    
    # zero order score
    k_c, k_t = zero_order_score(M)
    
    # allocate space
    G_tc = np.zeros(shape=M.T.shape)
    G_ct = np.zeros(shape=M.shape)
    
    # Gct_beta
    for [c, t], val in np.ndenumerate(M):
        G_ct[c, t] = Gct_beta(M, c, t, k_c, beta)
    
    # Gtc_alpha
    for [t, c], val in np.ndenumerate(M.T):
        G_tc[t, c] = Gtc_alpha(M, c, t, k_t, alpha)
    
    return {'G_ct': G_ct, "G_tc": G_tc}


def next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev):
    """Generates w^(n+1) from w^n"""
    fitness_next = np.sum(G_ct * ubiquity_prev, axis=1)
    ubiquity_next = np.sum(G_tc * fitness_prev, axis=1)
    
    return fitness_next, ubiquity_next


def generator_order_w(M, alpha, beta):
    """Generates w_t^{n+1} and w_c^{n+1}"""
    
    # transition probabilities
    G_hat = make_G_hat(M, alpha, beta)
    G_ct = G_hat['G_ct']
    G_tc = G_hat['G_tc']
    
    # starting point
    fitness_0, ubiquity_0 = zero_order_score(M)
    
    fitness_next = fitness_0
    ubiquity_next = ubiquity_0
    i = 0
    
    while True:
        fitness_prev = fitness_next
        ubiquity_prev = ubiquity_next
        i += 1
        
        fitness_next, ubiquity_next = next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev)
        
        yield {'iteration': i, 'fitness': fitness_next, 'ubiquity': ubiquity_next}


def find_convergence(M, 
                    alpha, 
                    beta, 
                    fit_or_ubiq, 
                    do_plot=False, 
                    flag_cybersecurity=False,
                    preferences=''):
    """TechRank evolution: finds the convergence point"""
    
    if fit_or_ubiq == 'fitness':
        M_shape = M.shape[0]
        name = 'Companies'
    elif fit_or_ubiq == 'ubiquity':
        name = 'Technologies'
        M_shape = M.shape[1]

    rankings = list()
    scores = list()
    
    prev_rankdata = np.zeros(M_shape)
    iteration = 0
    stops_flag = 0

    weights = generator_order_w(M, alpha, beta)
    max_iterations = 500
    pbar = tqdm(total=max_iterations, desc=f"TechRank {name}", unit="it")

    for stream_data in weights:
        
        iteration = stream_data['iteration']
        data = stream_data[fit_or_ubiq]
        
        # Gérer les NaN
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
        
        rankdata = data.argsort().argsort()
        
        if iteration == 1:
            initial_conf = rankdata

        # Test de convergence
        if stops_flag == 10:
            print(f"Converge at {iteration}")
            convergence_iteration = iteration
            for i in range(90):
                rankings.append(rankdata)
                scores.append(data)
            break

        elif np.equal(rankdata, prev_rankdata).all():
            if stops_flag == 0:
                convergence_iteration = iteration
            stops_flag += 1
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata = rankdata

        elif iteration == 5000:
            print("We break because we reach a too high number of iterations")
            convergence_iteration = iteration
            break

        else:
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata = rankdata
            stops_flag = 0
        pbar.update(1)

    pbar.update(1) 

    final_conf = rankdata
    
    # Plot
    if do_plot and iteration > 2:
        params = {
            'axes.labelsize': 26,
            'axes.titlesize': 28, 
            'legend.fontsize': 22, 
            'xtick.labelsize': 16, 
            'ytick.labelsize': 16}

        plt.figure(figsize=(10, 10))
        plt.rcParams.update(params)
        plt.xlabel('Iterations')
        plt.ylabel('Rank, higher is better')
        plt.title(f'{name} rank evolution')
        plt.semilogx(range(1, len(rankings) + 1), rankings, '-,', alpha=0.5)

        if flag_cybersecurity:
            name_plot = f'{SAVE_DIR_PLOTS}/techrank_cybersecurity_{name}_{M_shape}_{preferences}'
        else:
            name_plot = f'{SAVE_DIR_PLOTS}/techrank_{name}_{M_shape}_{preferences}'
        
        plt.savefig(f'{name_plot}.pdf')
        plt.savefig(f'{name_plot}.png')
        plt.show()

    return {fit_or_ubiq: scores[-1], 
            'iteration': convergence_iteration, 
            'initial_conf': initial_conf, 
            'final_conf': final_conf}


def rank_df_class(convergence, dict_class):
    """Creates a DataFrame and updates the class with the rank"""
    
    if 'fitness' in convergence.keys():
        fit_or_ubiq = 'fitness'
    elif 'ubiquity' in convergence.keys():
        fit_or_ubiq = 'ubiquity'
    
    list_names = [*dict_class]
    n = len(list_names)

    # Vérifier si rank_CB existe
    if hasattr(dict_class[list_names[0]], 'rank_CB'):
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank', 'rank_CB']
    else:
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank']

    df_final = pd.DataFrame(columns=columns_final, index=range(n))

    if n > len(convergence['initial_conf']):
        n = n - 1
    
    for i in range(n):
        name = list_names[i]
        
        ini_pos = convergence['initial_conf'][i]
        final_pos = convergence['final_conf'][i]
        rank = round(convergence[fit_or_ubiq][i], 3)
        degree = dict_class[name].degree
        
        df_final.loc[final_pos, 'final_configuration'] = name
        df_final.loc[final_pos, 'degree'] = degree
        df_final.loc[final_pos, 'initial_position'] = ini_pos
        df_final.loc[final_pos, 'techrank'] = rank

        if hasattr(dict_class[name], 'rank_CB'):
            rank_CB = dict_class[name].rank_CB
            df_final.loc[final_pos, 'rank_CB'] = rank_CB
        
        # Update class's instances with rank
        dict_class[name].rank_algo = rank
    
    return df_final, dict_class


# ===================================================================
# SAVING RESULTS
# ===================================================================

def save_results(df_companies, df_tech, dict_companies, dict_tech, 
                 num_comp, num_tech, flag_cybersecurity, preferences_comp, preferences_tech):
    """Sauvegarde les résultats"""
    
    # Sauvegarder les DataFrames
    name_csv_comp = f'{SAVE_DIR_RESULTS}/complete_companies_{num_comp}_{preferences_comp}.csv'
    name_csv_tech = f'{SAVE_DIR_RESULTS}/complete_tech_{num_tech}_{preferences_tech}.csv'
    
    df_companies.to_csv(name_csv_comp, index=False, header=True)
    df_tech.to_csv(name_csv_tech, index=False, header=True)
    
    print(f"\n✓ Résultats sauvegardés:")
    print(f"  - {name_csv_comp}")
    print(f"  - {name_csv_tech}")


# ===================================================================
# MAIN FUNCTION
# ===================================================================

def run_techrank(num_comp=NUM_COMP, 
                 num_tech=NUM_TECH, 
                 flag_cybersecurity=FLAG_CYBERSECURITY,
                 preferences_comp=PREFERENCES_COMP,
                 preferences_tech=PREFERENCES_TECH,
                 alpha=OPTIMAL_ALPHA_COMP,
                 beta=OPTIMAL_BETA_COMP,
                 do_plot=True):
    """
    Fonction principale pour exécuter TechRank
    """
    create_directories()
    
    print("\n" + "="*70)
    print("TECHRANK ALGORITHM")
    print("="*70)
    
    # 1. Charger les données
    dict_companies, dict_tech, B = load_saved_data(num_comp, num_tech, flag_cybersecurity)
    
    # 2. Créer la matrice d'adjacence
    M, set0, set1 = create_adjacency_matrix(B, flag_cybersecurity)
    save_matrix(M, num_comp, num_tech, flag_cybersecurity)
    
    # 3. Test de triangularité
    M_test_triangular(M, flag_cybersecurity)
    
    # 4. Scores d'ordre zéro
    k_c, k_t = zero_order_score(M)
    
    # 5. Ranking Companies
    print("\n" + "="*70)
    print("RANKING COMPANIES")
    print("="*70)
    
    start_time = time.time()
    convergence_comp = find_convergence(
        M,
        alpha=alpha,
        beta=beta,
        fit_or_ubiq='fitness',
        do_plot=do_plot,
        flag_cybersecurity=flag_cybersecurity,
        preferences=preferences_comp
    )
    end_time = time.time()
    time_conv_comp = end_time - start_time
    
    df_final_companies, dict_companies = rank_df_class(convergence_comp, dict_companies)
    
    # Normalisation pour Spearman
    df_final_companies['techrank_normlized'] = df_final_companies['techrank'] / np.max(list(df_final_companies['techrank'])) * 10
    if 'rank_CB' in df_final_companies.columns:
        n = np.max(df_final_companies['rank_CB']) + 1
        df_final_companies['rank_CB_normlized'] = n - df_final_companies['rank_CB']
    df_final_companies['TeckRank_int'] = df_final_companies.index + 1.0
    
    print(f"✓ Ranking companies terminé en {time_conv_comp:.2f}s")
    
    # 6. Ranking Technologies
    print("\n" + "="*70)
    print("RANKING TECHNOLOGIES")
    print("="*70)
    
    start_time = time.time()
    convergence_tech = find_convergence(
        M,
        alpha=alpha,
        beta=beta,
        fit_or_ubiq='ubiquity',
        do_plot=do_plot,
        flag_cybersecurity=flag_cybersecurity,
        preferences=preferences_tech
    )
    end_time = time.time()
    time_conv_tech = end_time - start_time
    
    df_final_tech, dict_tech = rank_df_class(convergence_tech, dict_tech)
    df_final_tech['TeckRank_int'] = df_final_tech.index + 1.0
    
    print(f"✓ Ranking technologies terminé en {time_conv_tech:.2f}s")
    
    # 7. Sauvegarder les résultats
    save_results(df_final_companies, df_final_tech, dict_companies, dict_tech,
                 num_comp, num_tech, flag_cybersecurity, preferences_comp, preferences_tech)
    
    # 8. Afficher les tops
    print("\n" + "="*70)
    print("TOP 10 COMPANIES")
    print("="*70)
    print(df_final_companies.head(10))
    
    print("\n" + "="*70)
    print("TOP 10 TECHNOLOGIES")
    print("="*70)
    print(df_final_tech.head(10))
    
    return df_final_companies, df_final_tech, dict_companies, dict_tech


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    # Exécuter TechRank
    df_companies, df_tech, dict_companies, dict_tech = run_techrank(
        num_comp=NUM_COMP,
        num_tech=NUM_TECH,
        flag_cybersecurity=FLAG_CYBERSECURITY,
        preferences_comp=PREFERENCES_COMP,
        preferences_tech=PREFERENCES_TECH,
        alpha=0.5,
        beta=0.5,
        do_plot=True
    )