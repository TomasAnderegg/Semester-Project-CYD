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

# Importer les fonctions externes si disponibles
try:
    from functions.fun_external_factors import rank_comparison, calibrate_analytic, create_exogenous_rank
    EXTERNAL_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("⚠ Fonctions externes non disponibles - certaines fonctionnalités seront limitées")
    EXTERNAL_FUNCTIONS_AVAILABLE = False


# ===================================================================
# CONFIGURATION
# ===================================================================

# Paramètres du réseau
NUM_COMP = 10000
NUM_TECH = 10000

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

# Paramètres de l'algorithme
OPTIMAL_ALPHA_COMP = 0.5  # À ajuster selon calibration
OPTIMAL_BETA_COMP = 0.5   # À ajuster selon calibration


# ===================================================================
# UTILS
# ===================================================================

def create_directories():
    """Crée tous les répertoires nécessaires"""
    for directory in [SAVE_DIR_CLASSES, SAVE_DIR_NETWORKS, SAVE_DIR_M, SAVE_DIR_RESULTS]:
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
    # B = nx.read_gpickle(paths['graph'])
    
    print(f"✓ Données chargées:")
    print(f"  - {len(dict_companies)} entreprises")
    print(f"  - {len(dict_tech)} technologies")
    print(f"  - Graphe: {B.number_of_nodes()} noeuds, {B.number_of_edges()} arêtes")
    
    return dict_companies, dict_tech, B


# ===================================================================
# MATRIX OPERATIONS
# ===================================================================

def create_adjacency_matrix(B):
    """Crée la matrice d'adjacence du graphe bipartite
    
    Args:
        B: graphe bipartite
    
    Return:
        M: matrice d'adjacence dense (numpy array)
        set0: liste des companies
        set1: liste des technologies
    """
    set0 = extract_nodes(B, 0) #toutes les companies
    set1 = extract_nodes(B, 1) #toutes les technologies
    
    # Matrice d'adjacence bipartite (sparse)
    adj_matrix = bipartite.biadjacency_matrix(B, set0, set1) #on cree la matrice adjacente, ou les lignes sont les companies et les colonnes les techs
    
    # Convertir en matrice dense
    adj_matrix_dense = adj_matrix.todense() #transformer en matrice dense (numpy.matrix). Chaque element d'une matrice dense est stockee explictement en memoire
    
    # Convertir en numpy array
    M = np.squeeze(np.asarray(adj_matrix_dense)) #np.asarray pour transformer en array, np.squeeze pour enlever les dimensions inutiles
    
    print(f"✓ Matrice créée: shape {M.shape}")
    
    return M, set0, set1


def save_matrix(M, num_comp, num_tech, flag_cybersecurity):
    """Sauvegarde la matrice M"""
    paths = get_file_paths(num_comp, num_tech, flag_cybersecurity)
    np.save(paths['matrix'], M)
    print(f"✓ Matrice sauvegardée: {paths['matrix']}")


def M_test_triangular(M, flag_cybersecurity):
    """Teste si la matrice est triangulaire (fonction de validation)"""
    is_upper = np.allclose(M, np.triu(M))
    is_lower = np.allclose(M, np.tril(M))
    
    status = "cybersecurity" if flag_cybersecurity else "all fields"
    
    if is_upper or is_lower:
        print(f"⚠ Matrice {status} est triangulaire")
    else:
        print(f"✓ Matrice {status} n'est pas triangulaire (OK)")
    
    return is_upper or is_lower


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


def find_convergence(M, alpha=0.5, beta=0.5, fit_or_ubiq='fitness', 
                     max_iter=100, tol=1e-6, do_plot=True, 
                     flag_cybersecurity=False, preferences=None):
    """Algorithme itératif pour trouver le ranking (TechRank)
    
    Args:
        M: matrice d'adjacence
        alpha: paramètre alpha
        beta: paramètre beta
        fit_or_ubiq: 'fitness' ou 'ubiquity'
        max_iter: nombre max d'itérations
        tol: tolérance pour convergence
        do_plot: afficher les graphiques
        flag_cybersecurity: flag cyber
        preferences: dictionnaire de préférences
    
    Return:
        convergence: dictionnaire avec résultats
    """
    n_comp, n_tech = M.shape
    
    # Initialisation
    if fit_or_ubiq == 'fitness':
        F = np.ones(n_comp)  # fitness des companies
        Q = np.ones(n_tech)  # quality des technologies
    else:
        F = np.ones(n_comp)  # ubiquity
        Q = np.ones(n_tech)
    
    initial_conf = np.arange(n_comp)
    
    print(f"\n{'='*70}")
    print(f"ALGORITHME TECHRANK - {fit_or_ubiq.upper()}")
    print(f"Alpha: {alpha}, Beta: {beta}")
    print(f"{'='*70}")
    
    # Itérations
    for iteration in range(max_iter):
        F_old = F.copy()
        Q_old = Q.copy()
        
        # Update quality of technologies
        Q = np.dot(M.T, F ** alpha)
        Q = Q / np.sum(Q)  # normalisation
        
        # Update fitness of companies
        F = np.dot(M, Q ** beta)
        F = F / np.sum(F)  # normalisation
        
        # Vérifier la convergence
        diff_F = np.max(np.abs(F - F_old))
        diff_Q = np.max(np.abs(Q - Q_old))
        
        if (iteration + 1) % 10 == 0:
            print(f"Itération {iteration + 1}: diff_F = {diff_F:.6f}, diff_Q = {diff_Q:.6f}")
        
        if diff_F < tol and diff_Q < tol:
            print(f"✓ Convergence atteinte après {iteration + 1} itérations")
            break
    
    # Trier les résultats
    sorted_indices = np.argsort(-F)  # ordre décroissant
    final_conf = sorted_indices
    
    convergence = {
        'initial_conf': initial_conf,
        'final_conf': final_conf,
        fit_or_ubiq: F,
        'iterations': iteration + 1
    }
    
    if do_plot:
        plot_convergence_results(F, fit_or_ubiq, flag_cybersecurity)
    
    return convergence


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


def rank_df_class(convergence, dict_class):
    """Crée un DataFrame avec les résultats et met à jour les classes
    
    Args:
        convergence: résultats de l'algorithme
        dict_class: dictionnaire des classes (companies ou tech)
    
    Return:
        df_final: DataFrame avec les résultats
        dict_class: dictionnaire mis à jour
    """
    if 'fitness' in convergence.keys():
        fit_or_ubiq = 'fitness'
    elif 'ubiquity' in convergence.keys():
        fit_or_ubiq = 'ubiquity'
    
    list_names = list(dict_class.keys())
    n = len(list_names)
    
    # Vérifier si rank_CB existe
    has_rank_cb = hasattr(list(dict_class.values())[0], 'rank_CB') if n > 0 else False
    
    if has_rank_cb:
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank', 'rank_CB']
    else:
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank']
    
    df_final = pd.DataFrame(columns=columns_final, index=range(n))
    
    # Ajuster n si nécessaire
    if n > len(convergence['initial_conf']):
        n = len(convergence['initial_conf'])
    
    for i in range(n):
        name = list_names[i]
        
        ini_pos = convergence['initial_conf'][i]
        final_pos = convergence['final_conf'][i]
        rank = round(convergence[fit_or_ubiq][i], 3)
        degree = getattr(dict_class[name], 'degree', 0)
        
        df_final.loc[final_pos, 'final_configuration'] = name
        df_final.loc[final_pos, 'degree'] = degree
        df_final.loc[final_pos, 'initial_position'] = ini_pos
        df_final.loc[final_pos, 'techrank'] = rank
        
        if has_rank_cb:
            rank_CB = dict_class[name].rank_CB
            df_final.loc[final_pos, 'rank_CB'] = rank_CB
        
        # Mettre à jour la classe
        dict_class[name].rank_algo = rank
    
    return df_final, dict_class


# ===================================================================
# CALIBRATION (si fonctions externes disponibles)
# ===================================================================

def run_calibration_companies(M, dict_companies, preferences_comp):
    """Calibration pour les companies"""
    if not EXTERNAL_FUNCTIONS_AVAILABLE:
        print("⚠ Calibration non disponible - fonctions externes manquantes")
        return OPTIMAL_ALPHA_COMP, OPTIMAL_BETA_COMP
    
    print("\n" + "="*70)
    print("CALIBRATION COMPANIES")
    print("="*70)
    
    start_time = time.time()
    
    best_par = calibrate_analytic(
        M=M,
        ua='Companies',
        dict_class=dict_companies,
        exogenous_rank=create_exogenous_rank('Companies', dict_companies, preferences_comp),
        index_function=lambda x: (x - 50) / 25,
        title='Correlation for Companies',
        do_plot=True,
        preferences=preferences_comp
    )
    
    end_time = time.time()
    
    print(f"✓ Calibration terminée en {end_time - start_time:.2f}s")
    print(f"  Meilleurs paramètres: {best_par}")
    
    return best_par


def run_calibration_technologies(M, dict_tech, preferences_tech):
    """Calibration pour les technologies"""
    if not EXTERNAL_FUNCTIONS_AVAILABLE:
        print("⚠ Calibration non disponible - fonctions externes manquantes")
        return OPTIMAL_ALPHA_COMP, OPTIMAL_BETA_COMP
    
    print("\n" + "="*70)
    print("CALIBRATION TECHNOLOGIES")
    print("="*70)
    
    start_time = time.time()
    
    best_par = calibrate_analytic(
        M=M,
        ua='Technologies',
        dict_class=dict_tech,
        exogenous_rank=create_exogenous_rank('Technologies', dict_tech, preferences_tech),
        index_function=lambda x: (x - 50) / 25,
        title='Correlation for Technologies',
        do_plot=True,
        preferences=preferences_tech
    )
    
    end_time = time.time()
    
    print(f"✓ Calibration terminée en {end_time - start_time:.2f}s")
    print(f"  Meilleurs paramètres: {best_par}")
    
    return best_par


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
    Fonction principale pour exécuter l'algorithme TechRank
    
    Args:
        num_comp: nombre de companies
        num_tech: nombre de technologies
        flag_cybersecurity: flag pour cybersecurity
        preferences_comp: préférences pour companies (pas utilisee)
        preferences_tech: préférences pour technologies (pas utilisee)
        do_calibration: effectuer la calibration
        alpha: paramètre alpha (si pas de calibration)
        beta: paramètre beta (si pas de calibration)
    
    Return:
        df_companies: DataFrame des companies rankées
        df_tech: DataFrame des technologies rankées
        dict_companies: dictionnaire mis à jour
        dict_tech: dictionnaire mis à jour
    """
    create_directories()
    
    print("\n" + "="*70)
    print("TECHRANK ALGORITHM")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Companies: {num_comp}")
    print(f"  - Technologies: {num_tech}")
    print(f"  - Cybersecurity filter: {flag_cybersecurity}")
    print("="*70 + "\n")
    
    # 1. Charger les données
    dict_companies, dict_tech, B = load_saved_data(num_comp, num_tech, flag_cybersecurity)
    
    # 2. Créer la matrice d'adjacence
    M, set0, set1 = create_adjacency_matrix(B)
    save_matrix(M, num_comp, num_tech, flag_cybersecurity)
    
    # 3. Test de la matrice
    '''
    TeckRank se base sur les sommations de connexions et les iterations enttre entreprises et technologies. Si la matrice est triangulaire, cela signifie qu'il y a une hiérarchie stricte
    certaines technologies ou entreprises n'ont jamais de connexions avec certaines autres. Le ranking deviendrait biaisé, car certaines entreprises ou technologies seraient systématiquement avantagées ou désavantagées.
    Par exemple, si une entreprise n'a jamais investi dans une technologie spécifique (représentée par un zéro dans la matrice triangulaire), cette entreprise ne pourra jamais être classée favorablement par rapport à d'autres qui ont investi dans cette technologie.
    En résumé, une matrice triangulaire indiquerait une structure de données inadéquate pour l'algorithme TechRank, compromettant ainsi la validité des résultats de classement obtenus..
    '''
    M_test_triangular(M, flag_cybersecurity) #fonction qui sert a verifier que la matrice n'est pas triangulaire, car si elle l'est, l'algorithme TechRank ne fonctionnera pas correctement.
 
    # 4. Calcul des scores d'ordre zéro
    k_c, k_t = zero_order_score(M)
    
    # 5. Calibration (optionnel)
    ''''
    Calibration des paramètres alpha et beta en fonction des préférences données.
    Cette étape ajuste les paramètres de l'algorithme TechRank pour mieux refléter les préférences spécifiques des utilisateurs ou des analystes.
    Si do_calibration est True et que les fonctions externes sont disponibles, la calibration est effectuée.
    '''
    if do_calibration and EXTERNAL_FUNCTIONS_AVAILABLE:
        alpha, beta = run_calibration_companies(M, dict_companies, preferences_comp)
        run_calibration_technologies(M, dict_tech, preferences_tech)
    
    # 6. Algorithme TechRank pour companies
    print("\n" + "="*70)
    print("RANKING COMPANIES")
    print("="*70)
    start_time = time.time()
    
    convergence_comp = find_convergence(
        M,
        alpha=alpha,
        beta=beta,
        fit_or_ubiq='fitness',
        do_plot=True,
        flag_cybersecurity=flag_cybersecurity,
        preferences=preferences_comp
    )
    
    time_comp = time.time() - start_time
    print(f"✓ Temps de calcul companies: {time_comp:.2f}s")
    
    df_companies, dict_companies = rank_df_class(convergence_comp, dict_companies)
    
    # 7. Algorithme TechRank pour technologies
    print("\n" + "="*70)
    print("RANKING TECHNOLOGIES")
    print("="*70)
    start_time = time.time()
    
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
    
    time_tech = time.time() - start_time
    print(f"✓ Temps de calcul technologies: {time_tech:.2f}s")
    
    df_tech, dict_tech = rank_df_class(convergence_tech, dict_tech)
    
    # 8. Sauvegarder les résultats
    save_results(df_companies, df_tech, dict_companies, dict_tech, 
                 num_comp, num_tech, flag_cybersecurity)
    
    # 9. Afficher le top 10
    print("\n" + "="*70)
    print("TOP 10 COMPANIES")
    print("="*70)
    print(df_companies.head(10))
    
    print("\n" + "="*70)
    print("TOP 10 TECHNOLOGIES")
    print("="*70)
    print(df_tech.head(10))
    
    return df_companies, df_tech, dict_companies, dict_tech


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
        do_calibration=True,  # Mettre True pour faire la calibration
        alpha=0.5,
        beta=0.5
    )