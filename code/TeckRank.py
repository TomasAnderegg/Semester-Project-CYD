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

# Importer les fonctions externes si disponibles
try:
    # from functions.fun_external_factors import rank_comparison, calibrate_analytic, create_exogenous_rank
    EXTERNAL_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("⚠ Fonctions externes non disponibles - certaines fonctionnalités seront limitées")
    EXTERNAL_FUNCTIONS_AVAILABLE = False


# ===================================================================
# CONFIGURATION
# ===================================================================

# Paramètres du réseau
NUM_COMP = 500
NUM_TECH = 500

# Préférences pour le ranking
PREFERENCES_COMP = {
    "previous_investments": 0,
    "geo_position": 100
}

PREFERENCES_TECH = {
    "previous_investments": 100
}

# Flags
FLAG_CYBERSECURITY = True

# Chemins de sauvegarde
SAVE_DIR_CLASSES = "savings/bipartite_tech_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_tech_comp/networks"
SAVE_DIR_M = "savings/bipartite_tech_comp/M"
SAVE_DIR_RESULTS = "savings/bipartite_tech_comp/results"
SAVE_DIR_PLOTS = "plots/rank_evolution"

# Paramètres de l'algorithme
OPTIMAL_ALPHA_COMP = 0.5  # À ajuster selon calibration
OPTIMAL_BETA_COMP = 0.5   # À ajuster selon calibration


# ===================================================================
# UTILS
# ===================================================================

def create_directories():
    """Crée tous les répertoires nécessaires"""
    for directory in [SAVE_DIR_CLASSES, SAVE_DIR_NETWORKS, SAVE_DIR_M, SAVE_DIR_RESULTS, SAVE_DIR_PLOTS]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_paths(num_comp, num_tech, flag_cybersecurity):
    """Génère les chemins de fichiers selon les paramètres"""
    prefix = "cybersecurity_" if flag_cybersecurity else ""
    
    paths = {
        'companies': f'savings/bipartite_tech_comp/classes/{prefix}dict_companies_ranked_{num_comp}.pickle',
        'tech': f'savings/bipartite_tech_comp/classes/{prefix}dict_tech_ranked_{num_tech}.pickle',
        'graph': f'savings/bipartite_tech_comp/networks/{prefix}bipartite_graph_{num_comp}.gpickle',
        'matrix': f'savings/bipartite_tech_comp/M/{prefix}comp_{num_comp}_tech_{num_tech}.npy'
    }
    return paths


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
    # Créer une version simplifiée pour le nom de fichier
    parts = []
    for key, value in preferences.items():
        # Raccourcir les noms de clés
        short_key = key[:4]  # Prendre les 4 premiers caractères
        parts.append(f"{short_key}{value}")
    return "_".join(parts)


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
    paths = get_file_paths(num_comp, num_tech, flag_cybersecurity)
    
    # Charger les dictionnaires
    with open(paths['companies'], 'rb') as f:
        dict_companies = pickle.load(f)
    
    with open(paths['tech'], 'rb') as f:
        dict_tech = pickle.load(f)
    
    # Charger le graphe
    with open(paths['graph'], 'rb') as f:
        B = pickle.load(f)
    
    print(f"✓ Données chargées:")
    print(f"  - {len(dict_companies)} entreprises")
    print(f"  - {len(dict_tech)} technologies")
    print(f"  - Graphe: {B.number_of_nodes()} noeuds, {B.number_of_edges()} arêtes")
    
    return dict_companies, dict_tech, B


# ===================================================================
# MATRIX OPERATIONS
# ===================================================================

def create_adjacency_matrix_simple(B):
    """Version simple et robuste qui garantit une matrice valide"""
    
    set0 = extract_nodes(B, 0)
    set1 = extract_nodes(B, 1)
    
    print("🔧 CRÉATION MATRICE SIMPLE")
    print(f"  - Companies: {len(set0)}")
    print(f"  - Technologies: {len(set1)}")
    print(f"  - Arêtes graphe: {B.number_of_edges()}")
    
    # Créer la matrice manuellement pour éviter tout bug NetworkX
    if len(set0) == 0 or len(set1) == 0:
        print("❌ Aucun nœud dans une des partitions")
        return np.array([]), [], []
    
    M = np.zeros((len(set0), len(set1)))
    company_to_idx = {company: i for i, company in enumerate(set0)}
    tech_to_idx = {tech: i for i, tech in enumerate(set1)}
    
    edge_count = 0
    for company in set0:
        i = company_to_idx[company]
        for tech in B.neighbors(company):
            if tech in tech_to_idx:
                j = tech_to_idx[tech]
                M[i, j] = 1
                edge_count += 1
    
    print(f"✅ Matrice créée manuellement:")
    print(f"  - Shape: {M.shape}")
    print(f"  - Arêtes comptées: {edge_count}")
    print(f"  - Somme matrice: {np.sum(M)}")
    
    # Vérification cohérence
    if edge_count != B.number_of_edges():
        print(f"⚠️  Attention: {edge_count} arêtes comptées vs {B.number_of_edges()} dans le graphe")
    
    return M, set0, set1
   


def Gct_beta(M, c, t, k_c, beta):
    """Calcule la probabilité de transition de company c vers technologie t"""
    num = (M[c,t]) * (k_c[c] ** (- beta))

    # sum over the technologies
    M_t = M[:,t].flatten()
    k_c_beta = [x ** (-1 * beta) for x in k_c]

    den = float(np.dot(M_t, k_c_beta))
    return num / den


def Gtc_alpha(M, c, t, k_t, alpha):
    """Calcule la probabilité de transition de technologie t vers company c"""
    # Gérer les cas de division par zéro
    num = (M.T[t,c]) * (k_t[t] ** (- alpha))
    
    # sum over the companies
    M_c = M[c,:].flatten()
    k_t_alpha = [x ** (-1 * alpha) for x in k_t]
    
    type(M_c)
    type(k_t_alpha)
    
    den = float(np.dot(M_c, k_t_alpha))
    
    return num / den


def next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev):
    """Generates w^(n+1) from w^n"""
    fitness_next = np.sum(G_ct * ubiquity_prev, axis=1)
    ubiquity_next = np.sum(G_tc * fitness_prev, axis=1)
    
    return fitness_next, ubiquity_next


def make_G_hat(M, alpha=1, beta=1):
    """G hat is Markov chain of length 2
    Gct is a matrix to go from companies to technologies and  
    Gtc is a matrix to go from technologies to companies"""
    
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


def save_matrix(M, num_comp, num_tech, flag_cybersecurity):
    """Sauvegarde la matrice M"""
    paths = get_file_paths(num_comp, num_tech, flag_cybersecurity)
    np.save(paths['matrix'], M)
    print(f"✓ Matrice sauvegardée: {paths['matrix']}")


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
    plt.show()

    return


def generator_order_w(M, alpha, beta):
    """Generates w_t^{n+1} and w_c^{n+1}
    
    fitness_next = w_t next
    ubliq_next = w_c next
    """
    
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
# ===================================================================
# VALIDITY FUNCTIONS
# ===================================================================
def validate_graph_and_matrix(B, M, companies=None, technologies=None, do_plot=True):
    """
    Vérifie la cohérence entre le graphe bipartite B et la matrice d'adjacence M.
    
    Args:
        B: graphe bipartite (NetworkX)
        M: matrice d'adjacence (numpy array)
        companies: liste des noeuds "companies" (bipartite=0)
        technologies: liste des noeuds "technologies" (bipartite=1)
        do_plot: affiche ou non les visualisations
        
    Returns:
        dict contenant les résultats de validation
    """

    results = {}

    # ----------------------------
    # 1. Vérifications de base
    # ----------------------------
    if companies is None or technologies is None:
        companies = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
        technologies = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]

    n_comp, n_tech = len(companies), len(technologies)
    results["num_companies"] = n_comp
    results["num_technologies"] = n_tech

    if M.shape != (n_comp, n_tech):
        print(f"❌ Shape mismatch: M.shape={M.shape} vs ({n_comp}, {n_tech}) attendus")
        results["shape_match"] = False
    else:
        results["shape_match"] = True
        print(f"✓ Shape cohérente: {M.shape}")

    # ----------------------------
    # 2. Vérification du nombre d'arêtes
    # ----------------------------
    num_edges_graph = B.number_of_edges()
    num_edges_matrix = int(np.sum(M))
    results["edges_graph"] = num_edges_graph
    results["edges_matrix"] = num_edges_matrix

    if num_edges_graph != num_edges_matrix:
        print(f"⚠️  Différence dans le nombre d'arêtes: Graphe={num_edges_graph}, Matrice={num_edges_matrix}")
        results["edges_match"] = False
    else:
        print(f"✓ Nombre d'arêtes cohérent: {num_edges_graph}")
        results["edges_match"] = True

    # ----------------------------
    # 3. Vérification de la bipartition
    # ----------------------------
    if nx.is_bipartite(B):
        print("✓ Graphe confirmé bipartite")
        results["is_bipartite"] = True
    else:
        print("❌ Graphe non bipartite")
        results["is_bipartite"] = False

    # ----------------------------
    # 4. Vérification de la correspondance des arêtes
    # ----------------------------
    mismatches = 0
    for i, c in enumerate(companies):
        for j, t in enumerate(technologies):
            if M[i, j] == 1 and not B.has_edge(c, t):
                mismatches += 1
    if mismatches == 0:
        print("✓ Toutes les arêtes de M existent dans B")
        results["edges_consistent"] = True
    else:
        print(f"⚠️  {mismatches} arêtes présentes dans M mais absentes dans B")
        results["edges_consistent"] = False

    # ----------------------------
    # 5. Visualisations
    # ----------------------------
    if do_plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Heatmap de la matrice
        axs[0].imshow(M, cmap="Greys", interpolation="nearest", aspect="auto")
        axs[0].set_title("Matrice d'adjacence")
        axs[0].set_xlabel("Technologies")
        axs[0].set_ylabel("Companies")

        # Distribution des degrés
        deg_comp = [B.degree(n) for n in companies]
        deg_tech = [B.degree(n) for n in technologies]
        axs[1].hist(deg_comp, bins=20, alpha=0.7, label="Companies")
        axs[1].hist(deg_tech, bins=20, alpha=0.7, label="Technologies")
        axs[1].set_title("Distribution des degrés")
        axs[1].legend()
        axs[1].grid(alpha=0.3)

        # Graphe simplifié (petit sous-échantillon)
        sub_B = B.copy()
        if sub_B.number_of_nodes() > 100:
            nodes_sample = list(np.random.choice(list(B.nodes), size=100, replace=False))
            sub_B = B.subgraph(nodes_sample)
        pos = nx.spring_layout(sub_B, seed=42)
        nx.draw(sub_B, pos, node_size=30, alpha=0.6, ax=axs[2])
        axs[2].set_title("Visualisation du graphe")

        plt.tight_layout()
        plt.show()

    return results

# ===================================================================
# TECHRANK ALGORITHM FUNCTIONS
# ===================================================================

def zero_order_score(M):
    """Calcule le score d'ordre zéro (degré)
    
    Args:
        M: matrice d'adjacence
    
    Return:
        k_c: degrés des companies (somme par ligne)
        k_t: degrés des technologies (somme par colonne)
    """
    k_c = np.sum(M, axis=1)  # degree companies
    k_t = np.sum(M, axis=0)  # degree technologies
    
    print(f"✓ Scores d'ordre zéro calculés")
    print(f"  - Degree moyen companies: {np.mean(k_c):.2f}")
    print(f"  - Degree moyen technologies: {np.mean(k_t):.2f}")
    
    return k_c, k_t


from tqdm import tqdm

def find_convergence(M, 
                    alpha, 
                    beta, 
                    fit_or_ubiq, 
                    do_plot=False, 
                    flag_cybersecurity=False,
                    preferences=''):
    
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

    # Barre de progression
    max_iterations = 5000  # même valeur que ton break
    pbar = tqdm(total=max_iterations, desc=f"TechRank {name}", unit="it")

    for stream_data in weights:
        iteration = stream_data['iteration']
        data = stream_data[fit_or_ubiq]
        
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
        
        rankdata = data.argsort().argsort()
        if iteration == 1:
            initial_conf = rankdata

        # test de convergence
        if stops_flag == 10:
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
        elif iteration == max_iterations:
            convergence_iteration = iteration
            break
        else:
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata = rankdata
            stops_flag = 0

        pbar.update(1)  # mise à jour de la barre

    pbar.close()  # fermer la barre quand terminé

    final_conf = rankdata

    # plot
    if do_plot and iteration > 2:
        # ton code existant pour plot
        ...

    return {fit_or_ubiq: scores[-1], 
            'iteration': convergence_iteration, 
            'initial_conf': initial_conf, 
            'final_conf': final_conf}



def plot_convergence_results(scores, fit_or_ubiq, flag_cybersecurity):
    """Affiche les résultats de convergence"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sorted(scores, reverse=True), 'b-', linewidth=2)
    plt.xlabel('Rank')
    plt.ylabel(f'{fit_or_ubiq.capitalize()} Score')
    plt.title(f'Distribution des scores - {fit_or_ubiq}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel(f'{fit_or_ubiq.capitalize()} Score')
    plt.ylabel('Fréquence')
    plt.title(f'Histogramme des scores')
    plt.grid(True, alpha=0.3)
    
    status = "Cybersecurity" if flag_cybersecurity else "All Fields"
    plt.suptitle(f'TechRank Results - {status}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def rank_df_class_robust(convergence, dict_class, active_names=None):
    """Version ultra-robuste avec gestion explicite des noms actifs"""
    
    if 'fitness' in convergence.keys():
        fit_or_ubiq = 'fitness'
    elif 'ubiquity' in convergence.keys():
        fit_or_ubiq = 'ubiquity'
    else:
        print("❌ Aucune clé 'fitness' ou 'ubiquity' trouvée dans convergence")
        return pd.DataFrame(), dict_class
    
    # Utiliser les noms actifs si fournis, sinon tous les noms du dictionnaire
    if active_names is None:
        active_names = list(dict_class.keys())
    
    n_convergence = len(convergence[fit_or_ubiq])
    n_active = len(active_names)
    
    print(f"🔍 TAILLES - Convergence: {n_convergence}, Actives: {n_active}")
    
    # Prendre le minimum pour éviter les dépassements
    n = min(n_convergence, n_active)
    
    if n_convergence != n_active:
        print(f"⚠️  Ajustement: utilisation de {n} éléments sur {n_convergence} convergence et {n_active} actives")
    
    # Créer le DataFrame
    columns_final = ['initial_position', 'final_configuration', 'techrank']
    if hasattr(dict_class[active_names[0]], 'rank_CB'):
        columns_final.append('rank_CB')
    
    df_final = pd.DataFrame(columns=columns_final, index=range(n))
    
    for i in range(n):
        name = active_names[i]
        
        try:
            ini_pos = convergence['initial_conf'][i]
            final_pos = convergence['final_conf'][i]
            rank = round(convergence[fit_or_ubiq][i], 6)
            
            df_final.loc[final_pos, 'final_configuration'] = name
            df_final.loc[final_pos, 'initial_position'] = ini_pos
            df_final.loc[final_pos, 'techrank'] = rank
            
            if hasattr(dict_class[name], 'rank_CB'):
                rank_CB = dict_class[name].rank_CB
                df_final.loc[final_pos, 'rank_CB'] = rank_CB
            
            # update class's instances with rank
            dict_class[name].rank_algo = rank
            
        except Exception as e:
            print(f"❌ Erreur pour {name} (index {i}): {e}")
            continue
    
    return df_final, dict_class

# ===================================================================
# CALIBRATION (si fonctions externes disponibles)
# ===================================================================

# def run_calibration_companies(M, dict_companies, preferences_comp):
#     """Calibration pour les companies"""
#     if not EXTERNAL_FUNCTIONS_AVAILABLE:
#         print("⚠ Calibration non disponible - fonctions externes manquantes")
#         return OPTIMAL_ALPHA_COMP, OPTIMAL_BETA_COMP
    
#     print("\n" + "="*70)
#     print("CALIBRATION COMPANIES")
#     print("="*70)
    
#     start_time = time.time()
    
#     # best_par = calibrate_analytic(
#     #     M=M,
#     #     ua='Companies',
#     #     dict_class=dict_companies,
#     #     # exogenous_rank=create_exogenous_rank('Companies', dict_companies, preferences_comp),
#     #     index_function=lambda x: (x - 50) / 25,
#     #     title='Correlation for Companies',
#     #     do_plot=True,
#     #     preferences=preferences_comp
#     # )
    
#     end_time = time.time()
    
#     print(f"✓ Calibration terminée en {end_time - start_time:.2f}s")
#     print(f"  Meilleurs paramètres: {best_par}")
    
#     return best_par


# def run_calibration_technologies(M, dict_tech, preferences_tech):
#     """Calibration pour les technologies"""
#     if not EXTERNAL_FUNCTIONS_AVAILABLE:
#         print("⚠ Calibration non disponible - fonctions externes manquantes")
#         return OPTIMAL_ALPHA_COMP, OPTIMAL_BETA_COMP
    
#     print("\n" + "="*70)
#     print("CALIBRATION TECHNOLOGIES")
#     print("="*70)
    
#     start_time = time.time()
    
#     best_par = calibrate_analytic(
#         M=M,
#         ua='Technologies',
#         dict_class=dict_tech,
#         exogenous_rank=create_exogenous_rank('Technologies', dict_tech, preferences_tech),
#         index_function=lambda x: (x - 50) / 25,
#         title='Correlation for Technologies',
#         do_plot=True,
#         preferences=preferences_tech
#     )
    
#     end_time = time.time()
    
#     print(f"✓ Calibration terminée en {end_time - start_time:.2f}s")
#     print(f"  Meilleurs paramètres: {best_par}")
    
#     return best_par


# ===================================================================
# SAVING RESULTS
# ===================================================================

def save_results(df_companies, df_tech, dict_companies, dict_tech, 
                 num_comp, num_tech, flag_cybersecurity):
    """Sauvegarde les résultats"""
    prefix = "cybersecurity_" if flag_cybersecurity else ""
    
    # Sauvegarder les DataFrames
    df_companies.to_csv(f'{SAVE_DIR_RESULTS}/{prefix}companies_ranked_{num_comp}.csv', index=False)
    df_tech.to_csv(f'{SAVE_DIR_RESULTS}/{prefix}tech_ranked_{num_tech}.csv', index=False)
    
    # Sauvegarder les dictionnaires mis à jour
    with open(f'{SAVE_DIR_CLASSES}/dict_companies_ranked_{prefix}{num_comp}.pickle', 'wb') as f:
        pickle.dump(dict_companies, f)
    
    with open(f'{SAVE_DIR_CLASSES}/dict_tech_ranked_{prefix}{num_tech}.pickle', 'wb') as f:
        pickle.dump(dict_tech, f)
    
    print(f"\n✓ Résultats sauvegardés dans {SAVE_DIR_RESULTS}/")


# ===================================================================
# MAIN FUNCTION
# ===================================================================

def run_techrank(num_comp=NUM_COMP, num_tech=NUM_TECH, 
                 flag_cybersecurity=FLAG_CYBERSECURITY,
                 preferences_comp=PREFERENCES_COMP,
                 preferences_tech=PREFERENCES_TECH,
                 do_calibration=False,
                 alpha=OPTIMAL_ALPHA_COMP,
                 beta=OPTIMAL_BETA_COMP):
    """
    Fonction principale CORRIGÉE avec gestion des tailles
    """
    create_directories()
    
    print("\n" + "="*70)
    print("TECHRANK ALGORITHM - VERSION SYNCHRONISÉE")
    print("="*70)
    
    # Initialiser les variables de retour
    df_companies = pd.DataFrame()
    df_tech = pd.DataFrame()
    
    # 1. Charger les données
    dict_companies, dict_tech, B = load_saved_data(num_comp, num_tech, flag_cybersecurity)
    
    # 2. Créer la matrice d'adjacence
    print("\n=== CRÉATION MATRICE ADJACENCE ===")
    try:
        M, active_companies, active_techs = create_adjacency_matrix_simple(B)

        validation_results = validate_graph_and_matrix(B, M, active_companies, active_techs)
        print(json.dumps(validation_results, indent=4))
    except Exception as e:
        print(f"❌ Erreur création matrice: {e}")
        return df_companies, df_tech, dict_companies, dict_tech
    
    # # VÉRIFICATION CRITIQUE
    # if M.size == 0 or np.sum(M) == 0:
    #     print("❌ Matrice vide - arrêt de l'algorithme")
    #     return df_companies, df_tech, dict_companies, dict_tech
    
    # FILTRAGE ET SYNCHRONISATION
    # print("\n=== SYNCHRONISATION DES DONNÉES ===")
    
    # # Filtrer les dictionnaires pour garder seulement les actifs
    # dict_companies_active = {name: dict_companies[name] for name in active_companies if name in dict_companies}
    # dict_tech_active = {name: dict_tech[name] for name in active_techs if name in dict_tech}
    
    # print(f"📊 SYNCHRONISATION:")
    # print(f"  - Companies: {len(active_companies)} actives vs {len(dict_companies_active)} dans dict")
    # print(f"  - Technologies: {len(active_techs)} actives vs {len(dict_tech_active)} dans dict")
    
    # # Vérification cohérence
    # if len(active_companies) != len(dict_companies_active):
    #     print("⚠️  Ajustement des companies actives...")
    #     active_companies = [name for name in active_companies if name in dict_companies_active]
    
    # if len(active_techs) != len(dict_tech_active):
    #     print("⚠️  Ajustement des technologies actives...")
    #     active_techs = [name for name in active_techs if name in dict_tech_active]
    
    # # Filtrer les entreprises sans connexion dans la matrice
    # row_sums = np.sum(M, axis=1)
    # zero_degree_mask = row_sums == 0
    # if np.sum(zero_degree_mask) > 0:
    #     print(f"🔄 Filtrage des {np.sum(zero_degree_mask)} companies sans connexion...")
    #     M = M[~zero_degree_mask, :]
    #     # Mettre à jour la liste des companies actives
    #     active_companies = [company for i, company in enumerate(active_companies) if not zero_degree_mask[i]]
    #     # Mettre à jour le dictionnaire
    #     dict_companies_active = {name: dict_companies_active[name] for name in active_companies if name in dict_companies_active}
    #     print(f"  - Nouvelle shape: {M.shape}")
    #     print(f"  - Companies actives après filtrage: {len(active_companies)}")
    
    # if np.sum(M) == 0:
    #     print("❌ Matrice vide après filtrage - arrêt")
    #     return df_companies, df_tech, dict_companies, dict_tech
    
    save_matrix(M, num_comp, num_tech, flag_cybersecurity)
    
    # 3. Calcul des scores d'ordre zéro
    k_c, k_t = zero_order_score(M)
    
    # 4. Algorithme TechRank pour companies
    print("\n" + "="*70)
    print("RANKING COMPANIES")
    print("="*70)
    
    try:
        convergence_comp = find_convergence(
            M,
            alpha=alpha,
            beta=beta,
            fit_or_ubiq='fitness',
            do_plot=True,
            flag_cybersecurity=flag_cybersecurity,
            preferences=preferences_comp
        )
        
        # ⚠️ CORRECTION : Utiliser les companies actives et le dictionnaire filtré
        df_companies, dict_companies_active = rank_df_class_robust(
            convergence_comp, 
            dict_companies_active,
            active_names=active_companies
        )
        print(f"✓ Ranking companies terminé: {len(df_companies)} entreprises rankées")
        
    except Exception as e:
        print(f"❌ Erreur lors du ranking companies: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Algorithme TechRank pour technologies
    print("\n" + "="*70)
    print("RANKING TECHNOLOGIES")
    print("="*70)
    
    try:
        # Transposer M pour les technologies
        M_T = M.T
        
        convergence_tech = find_convergence(
            M_T,
            alpha=alpha,
            beta=beta,
            fit_or_ubiq='ubiquity',
            do_plot=True,
            flag_cybersecurity=flag_cybersecurity,
            preferences=preferences_tech
        )
        
        # ⚠️ CORRECTION : Utiliser les technologies actives et le dictionnaire filtré
        df_tech, dict_tech_active = rank_df_class_robust(
            convergence_tech, 
            dict_tech_active,
            active_names=active_techs
        )
        print(f"✓ Ranking technologies terminé: {len(df_tech)} technologies rankées")
        
    except Exception as e:
        print(f"❌ Erreur lors du ranking technologies: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Sauvegarder les résultats
    if not df_companies.empty and not df_tech.empty:
        save_results(df_companies, df_tech, dict_companies_active, dict_tech_active, 
                     num_comp, num_tech, flag_cybersecurity)
        
        # Afficher le top 10
        print("\n" + "="*70)
        print("TOP 10 COMPANIES")
        print("="*70)
        print(df_companies.head(10))
        
        print("\n" + "="*70)
        print("TOP 10 TECHNOLOGIES")
        print("="*70)
        print(df_tech.head(10))
    else:
        print("❌ Aucun résultat à sauvegarder")
    
    return df_companies, df_tech, dict_companies_active, dict_tech_active

# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    # Exécuter TechRank avec les paramètres par défaut
    df_companies, df_tech, dict_companies, dict_tech = run_techrank(
        num_comp=NUM_COMP,
        num_tech=NUM_TECH,
        flag_cybersecurity=FLAG_CYBERSECURITY,
        preferences_comp=PREFERENCES_COMP,
        preferences_tech=PREFERENCES_TECH,
        do_calibration=False,  # Mettre True pour faire la calibration
        alpha=0.5,
        beta=0.5
    )