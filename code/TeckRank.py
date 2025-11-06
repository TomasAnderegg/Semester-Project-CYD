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

SAVE_DIR_CLASSES = "savings/bipartite_tech_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_tech_comp/networks"
SAVE_DIR_M = "savings/bipartite_tech_comp/M"
SAVE_DIR_RESULTS = "savings/csv_results"
SAVE_DIR_PLOTS = "plots/rank_evolution"

OPTIMAL_ALPHA_COMP = 1
OPTIMAL_BETA_COMP = 1

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
    
    name_file_com = f'{SAVE_DIR_CLASSES}/{prefix}dict_companies_ranked_cybersecurity_{num_comp}.pickle'
    name_file_tech = f'{SAVE_DIR_CLASSES}/{prefix}dict_tech_ranked_cybersecurity_{num_tech}.pickle'
    name_file_graph = f'{SAVE_DIR_NETWORKS}/{prefix}cybersecurity_bipartite_graph_{num_comp}.gpickle'
    
    with open(name_file_com, 'rb') as f:
        dict_companies = pickle.load(f)
    
    with open(name_file_tech, 'rb') as f:
        dict_tech = pickle.load(f)
    
    # B = nx.read_gpickle(name_file_graph)
    with open(name_file_graph, 'rb') as f:
        B = pickle.load(f)
    
    print(f"✓ Données chargées: {len(dict_companies)} entreprises, {len(dict_tech)} technologies, Graphe: {B.number_of_nodes()} noeuds, {B.number_of_edges()} arêtes")
    return dict_companies, dict_tech, B

# ===================================================================
# MATRIX OPERATIONS
# ===================================================================

def create_adjacency_matrix(B):
    set0 = extract_nodes(B, 0)
    set1 = extract_nodes(B, 1)
    
    print(f"Création matrice d'adjacence: Companies={len(set0)}, Technologies={len(set1)}")
    adj_matrix = bipartite.biadjacency_matrix(B, set0, set1, format='csr')
    print(f"✓ Matrice créée: Shape={adj_matrix.shape}, Arêtes={adj_matrix.sum()}")
    return adj_matrix, set0, set1

def save_matrix(M, num_comp, num_tech, flag_cybersecurity):
    name_file_M = f'{SAVE_DIR_M}/{"cybersecurity_" if flag_cybersecurity else ""}comp_{num_comp}_tech_{num_tech}.npy'
    np.save(name_file_M, M)
    print(f"✓ Matrice sauvegardée: {name_file_M}")

def M_test_triangular(M):
    if hasattr(M, "toarray"):
        M = M.toarray()
    user_edits_sum = M.sum(axis=1).flatten()
    article_edits_sum = M.sum(axis=0).flatten()
    user_order = user_edits_sum.argsort()
    article_order = article_edits_sum.argsort()
    M_sorted = M[user_order, :][:, article_order]
    plt.figure(figsize=(10,10))
    plt.imshow(M_sorted, cmap=plt.cm.bone, interpolation='nearest')
    plt.xlabel("Companies")
    plt.ylabel("Technologies")
    plt.show()

# ===================================================================
# TECHRANK FUNCTIONS
# ===================================================================

def zero_order_score(M):
    """Calcule le degré des companies et technologies"""
    if isinstance(M, csr_matrix):
        k_c = np.array(M.sum(axis=1)).flatten()
        k_t = np.array(M.sum(axis=0)).flatten()
    else:  # numpy array
        k_c = M.sum(axis=1)
        k_t = M.sum(axis=0)
    print(f"✓ Scores ordre zéro: deg_companies={np.mean(k_c):.2f}, deg_tech={np.mean(k_t):.2f}")
    return k_c, k_t

def make_G_hat(M, alpha=1, beta=1, eps=1e-12):
    k_c = np.array(M.sum(axis=1)).flatten().astype(float)
    k_t = np.array(M.sum(axis=0)).flatten().astype(float)
    k_c_safe = np.where(k_c>0, k_c, eps)
    k_t_safe = np.where(k_t>0, k_t, eps)
    
    G_ct = M.multiply((k_c_safe ** (-beta))[:, None])
    # Normalisation colonnes
    col_sums = np.array(G_ct.sum(axis=0)).flatten()
    col_sums_safe = np.where(col_sums>0, col_sums, eps)
    G_ct = G_ct / col_sums_safe
    
    G_tc = M.T.multiply((k_t_safe ** (-alpha))[:, None])
    col_sums_tc = np.array(G_tc.sum(axis=0)).flatten()
    col_sums_tc_safe = np.where(col_sums_tc>0, col_sums_tc, eps)
    G_tc = G_tc / col_sums_tc_safe
    
    return {'G_ct': G_ct, 'G_tc': G_tc}

def next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev):
    fitness_next = G_ct.dot(ubiquity_prev)
    ubiquity_next = G_tc.dot(fitness_prev)
    return fitness_next, ubiquity_next

def generator_order_w(M, alpha, beta):
    G_hat = make_G_hat(M, alpha, beta)
    G_ct, G_tc = G_hat['G_ct'], G_hat['G_tc']
    fitness_0, ubiquity_0 = zero_order_score(M)
    fitness_next, ubiquity_next = fitness_0, ubiquity_0
    i = 0
    while True:
        i += 1
        fitness_prev, ubiquity_prev = fitness_next, ubiquity_next
        fitness_next, ubiquity_next = next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev)
        yield {'iteration': i, 'fitness': fitness_next, 'ubiquity': ubiquity_next}

def find_convergence(M, alpha, beta, fit_or_ubiq, do_plot=False, flag_cybersecurity=False, preferences=''):
    if fit_or_ubiq=='fitness':
        name='Companies'; M_shape=M.shape[0]
    else:
        name='Technologies'; M_shape=M.shape[1]
    
    rankings=[]; scores=[]
    prev_rankdata = np.zeros(M_shape)
    stops_flag=0
    weights = generator_order_w(M, alpha, beta)
    max_iterations = 5000
    pbar = tqdm(total=max_iterations, desc=f"TechRank {name}", unit="it")
    convergence_iteration = 0
    
    for stream_data in weights:
        iteration = stream_data['iteration']
        data = stream_data[fit_or_ubiq]
        data = np.nan_to_num(data, nan=0.0)
        rankdata = data.argsort().argsort()
        if iteration==1:
            initial_conf = rankdata
        if stops_flag==10:
            convergence_iteration=iteration
            for _ in range(90):
                rankings.append(rankdata)
                scores.append(data)
            break
        elif np.equal(rankdata, prev_rankdata).all():
            if stops_flag==0: convergence_iteration=iteration
            stops_flag+=1
            rankings.append(rankdata)
            scores.append(data)
        else:
            stops_flag=0
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata=rankdata
        if iteration>=max_iterations: break
        pbar.update(1)
    pbar.close()
    
    if do_plot and iteration>2:
        plt.figure(figsize=(10,10))
        plt.semilogx(range(1,len(rankings)+1), rankings, '-,', alpha=0.5)
        plt.xlabel("Iterations"); plt.ylabel("Rank, higher is better")
        plt.title(f"{name} rank evolution")
        name_plot = f'{SAVE_DIR_PLOTS}/techrank_{"cybersecurity_" if flag_cybersecurity else ""}{name}_{M_shape}_{preferences}'
        plt.savefig(f'{name_plot}.pdf'); plt.savefig(f'{name_plot}.png')
        plt.show()
    
    return {fit_or_ubiq: scores[-1], 'iteration': convergence_iteration, 'initial_conf': initial_conf, 'final_conf': rankdata}

def rank_df_class(convergence, dict_class):
    fit_or_ubiq = 'fitness' if 'fitness' in convergence else 'ubiquity'
    list_names = list(dict_class.keys())
    n = len(list_names)
    df_final = pd.DataFrame(columns=['initial_position','final_configuration','degree','techrank','rank_CB','rank_analytic'], index=range(n))
    for i,name in enumerate(list_names):
        ini_pos = convergence['initial_conf'][i]
        final_pos = convergence['final_conf'][i]
        rank = round(convergence[fit_or_ubiq][i],3)
        degree = getattr(dict_class[name],'degree',0)
        df_final.loc[final_pos,'final_configuration']=name
        df_final.loc[final_pos,'initial_position']=ini_pos
        df_final.loc[final_pos,'techrank']=rank
        df_final.loc[final_pos,'degree']=degree
        df_final.loc[final_pos,'rank_CB']=getattr(dict_class[name],'rank_CB',None)
        df_final.loc[final_pos,'rank_analytic']=getattr(dict_class[name],'rank_analytic',None)
        dict_class[name].rank_algo=rank
    return df_final, dict_class

def save_results(df_companies, df_tech, num_comp, num_tech, flag_cybersecurity, preferences_comp, preferences_tech):
    pref_comp = preferences_to_string(preferences_comp)
    pref_tech = preferences_to_string(preferences_tech)
    df_companies.to_csv(f'{SAVE_DIR_RESULTS}/complete_companies_{num_comp}_{pref_comp}.csv', index=False)
    df_tech.to_csv(f'{SAVE_DIR_RESULTS}/complete_tech_{num_tech}_{pref_tech}.csv', index=False)
    
    # Global CSV pour suivre les expériences
    df_all = pd.DataFrame([{
        'num_comp': num_comp, 'num_tech': num_tech, 'flag_cybersecurity': flag_cybersecurity,
        'preferences_comp': pref_comp, 'preferences_tech': pref_tech
    }])
    df_all.to_csv(f'{SAVE_DIR_RESULTS}/df_rank_evolu.csv', index=False)
    print("✓ Résultats sauvegardés")

# ===================================================================
# MAIN FUNCTION
# ===================================================================

def run_techrank(num_comp=NUM_COMP, num_tech=NUM_TECH, flag_cybersecurity=FLAG_CYBERSECURITY,
                 preferences_comp=None, preferences_tech=None,
                 alpha=OPTIMAL_ALPHA_COMP, beta=OPTIMAL_BETA_COMP, do_plot=False):
    
    create_directories()
    dict_companies, dict_tech, B = load_saved_data(num_comp, num_tech, flag_cybersecurity)
    M, _, _ = create_adjacency_matrix(B)
    save_matrix(M, num_comp, num_tech, flag_cybersecurity)
    
    # Triangularité optionnelle
    # M_test_triangular(M)
    
    # Ranking Companies
    convergence_comp = find_convergence(M, alpha, beta, 'fitness', do_plot, flag_cybersecurity, preferences_to_string(preferences_comp))
    df_companies, dict_companies = rank_df_class(convergence_comp, dict_companies)
    
    # Ranking Technologies
    convergence_tech = find_convergence(M, alpha, beta, 'ubiquity', do_plot, flag_cybersecurity, preferences_to_string(preferences_tech))
    df_tech, dict_tech = rank_df_class(convergence_tech, dict_tech)
    
    # Sauvegarde
    save_results(df_companies, df_tech, num_comp, num_tech, flag_cybersecurity, preferences_comp, preferences_tech)
    
    print("TOP 10 Companies:\n", df_companies.head(10))
    print("TOP 10 Technologies:\n", df_tech.head(10))
    
    return df_companies, df_tech, dict_companies, dict_tech

# ===================================================================
# EXECUTION
# ===================================================================

if __name__ == "__main__":
    df_companies, df_tech, dict_companies, dict_tech = run_techrank(do_plot=False)
