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

def Gct_beta(M, c, t, k_c, beta):

    num = (M[c,t]) * (k_c[c] ** (- beta))

    # sum over the technologies
    M_t = M[:,t].flatten()
    k_c_beta = [x ** (-1 * beta) for x in k_c]

    den = float(np.dot(M_t, k_c_beta))
    
    return num/den


def Gtc_alpha(M, c, t, k_t, alpha):
    
    num = (M.T[t,c]) * (k_t[t] ** (- alpha))
    
    # sum over the companies
    M_c = M[c,:].flatten()
    k_t_alpha = [x ** (-1 * alpha) for x in k_t]
    
    type(M_c)
    type(k_t_alpha)
    
    den = float(np.dot(M_c, k_t_alpha))
    
    return num/den

def next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev):
    '''Generates w^(n+1) from w^n
    '''
    
    fitness_next = np.sum( G_ct * ubiquity_prev, axis=1 )
    ubiquity_next = np.sum( G_tc * fitness_prev, axis=1 )
    
    return fitness_next, ubiquity_next


def make_G_hat(M, alpha=1, beta=1):
    '''G hat is Markov chain of length 2
    Gct is a matrix to go from  companies to technologies and  
    Gtc is a matrix to go from technologies to companies'''
    
    # zero order score
    k_c, k_t = zero_order_score(M)
    
    # allocate space
    G_tc = np.zeros(shape=M.T.shape)
    G_ct = np.zeros(shape=M.shape)
    
    # Gct_beta
    for [c, t], val in np.ndenumerate(M):
        G_ct[c,t] = Gct_beta(M, c, t, k_c, beta)
    
    # Gtc_alpha
    for [t, c], val in np.ndenumerate(M.T):
        G_tc[t,c] = Gtc_alpha(M, c, t, k_t, alpha)
    
    return {'G_ct': G_ct, "G_tc" : G_tc}


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

    M_sorted = M[user_edits_order,:]

    if len(M_sorted.shape)>2: # the matrix in inside the first
        M_sorted = M_sorted[0] # so it becomes of size 2

    M_sorted_transpose = M_sorted.transpose()

    M_sorted_transpose = M_sorted_transpose[article_edits_order,:]

    if len(M_sorted_transpose.shape)>2: # the matrix in inside the first
        M_sorted_transpose = M_sorted_transpose[0] # so it becomes of size 2

    M_sorted_sorted = M_sorted_transpose#.transpose()

    params = {
        'axes.labelsize': 18,
        'axes.titlesize':28, 
        'legend.fontsize': 22, 
        'xtick.labelsize': 16, 
        'ytick.labelsize': 16}

    plt.figure(figsize=(10, 10))
    plt.rcParams.update(params)
    plt.imshow(M_sorted_sorted, cmap=plt.cm.bone, interpolation='nearest')
    plt.xlabel("Companies")
    plt.ylabel("Technologies")

    # if flag_cybersecurity==False: # all fields
    #     name_plot_M = f'plots/M_triangulize/matrix_{str(M_sorted.shape)}'
    # else: # only companies in cybersecurity
    #     name_plot_M = f'plots/M_triangulize/cybersec_matrix_{str(M_sorted.shape)}'

    # plt.savefig(f'{name_plot_M}.pdf')
    # plt.savefig(f'{name_plot_M}.png')
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
    
    # strating point
    fitness_0, ubiquity_0  = zero_order_score(M)
    
    fitness_next = fitness_0
    ubiquity_next = ubiquity_0
    i = 0
    
    while True:
        
        fitness_prev = fitness_next
        ubiquity_prev = ubiquity_next
        i += 1
        
        fitness_next, ubiquity_next = next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev)
        
        yield {'iteration':i, 'fitness': fitness_next, 'ubiquity': ubiquity_next}
   

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


def find_convergence(M, 
                    alpha, 
                    beta, 
                    fit_or_ubiq, 
                    do_plot=False, 
                    flag_cybersecurity=False,
                    preferences = ''):
    '''TechRank evolution: finds the convergence point (or gives up after 1000 iterations)
    
    Args:
        - 
        
    Return: 
        - 
        
    '''
    
    # technologies or company
    if fit_or_ubiq == 'fitness':
        M_shape = M.shape[0]
        name = 'Companies'
    elif fit_or_ubiq == 'ubiquity':
        name = 'Technologies'
        M_shape = M.shape[1]

    #print(name)
    
    rankings = list()
    scores = list()
    
    prev_rankdata = np.zeros(M_shape)
    iteration = 0
    

    weights = generator_order_w(M, alpha, beta)


    stops_flag = 0

    for stream_data in weights:
        
        iteration = stream_data['iteration']
        
        data = stream_data[fit_or_ubiq] # weights
        
        rankdata = data.argsort().argsort()

        # print(f"iteration : {iteration}")
        
        if iteration==1:
            # print(iteration, rankdata)
            initial_conf = rankdata

        # print(f"Iteration: {iteration} stops flag: {stops_flag}")

        # stops in case algorithm does not change for some iterations
        if stops_flag==10:
            print(f"Converge at {iteration}")
            convergence_iteration = iteration
            for i in range(90):
                rankings.append(rankdata)
                scores.append(data)
            break

        # test for convergence, in case break
        elif np.equal(rankdata,prev_rankdata).all(): # no changes
            if stops_flag==0:
                convergence_iteration = iteration
            stops_flag += 1

            # reappend two times to make plot flat
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata = rankdata

        # max limit
        elif iteration == 5000: 
            print("We break becuase we reach a too high number of iterations")
            convergence_iteration = iteration
            break

        # go ahead
        else: 
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata = rankdata
            stops_flag = 0

            
    # print(iteration, rankdata)
    final_conf = rankdata
    
    # plot:
    if do_plot and iteration>2:

        params = {
            'axes.labelsize': 26,
            'axes.titlesize':28, 
            'legend.fontsize': 22, 
            'xtick.labelsize': 16, 
            'ytick.labelsize': 16}

        plt.figure(figsize=(10, 10))
        plt.rcParams.update(params)
        plt.xlabel('Iterations')
        plt.ylabel('Rank, higher is better')
        plt.title(f'{name} rank evolution')
        plt.semilogx(range(1,len(rankings)+1), rankings, '-,', alpha=0.5)

        # save figure 
        if flag_cybersecurity==False:
            name_plot = f'code/plots/rank_evolution/techrank_{name}_{str(M_shape)}_{str(preferences)}'
        else:
            name_plot = f'code/plots/rank_evolution/techrank_cybersecurity_{name}_{str(M_shape)}_{str(preferences)}'
        plt.savefig(f'{name_plot}.pdf')
        plt.savefig(f'{name_plot}.png')

        
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


def rank_df_class(convergence, dict_class):
    """Creates a Dataframe to have a representation of the evolution with the iterations and the rank and 
    update the class with the rank found with the find_convergence algorithm
    """
    
    if 'fitness' in convergence.keys():
        fit_or_ubiq = 'fitness'
    elif 'ubiquity' in convergence.keys():
        fit_or_ubiq = 'ubiquity'
    
    list_names = [*dict_class]
    
    n = len(list_names)

    if hasattr(list_names[0], 'rank_CB'):
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank', 'rank_CB']
    else:
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank']

    df_final = pd.DataFrame(columns=columns_final, index=range(n))

    # print(f"----{n}----{len(convergence['initial_conf'])}----")
    if n>len(convergence['initial_conf']):
        n = n-1
    
    for i in range(n):
        # print(i)
        
        name = list_names[i]
        
        ini_pos = convergence['initial_conf'][i] # initial position
        final_pos = convergence['final_conf'][i] # final position
        rank = round(convergence[fit_or_ubiq][i], 3) # final rank rounded
        degree = dict_class[name].degree
        
        df_final.loc[final_pos, 'final_configuration'] = name
        df_final.loc[final_pos, 'degree'] = degree
        df_final.loc[final_pos, 'initial_position'] = ini_pos
        df_final.loc[final_pos, 'techrank'] = rank


        if hasattr(dict_class[name], 'rank_CB'):
            rank_CB = dict_class[name].rank_CB
            df_final.loc[final_pos, 'rank_CB'] = rank_CB
        
        
        # update class's instances with rank
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