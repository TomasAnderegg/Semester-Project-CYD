import math
import arrow
import ipynb 
import os.path
import json
import pickle
import sys
import time
import random
import operator

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
from networkx.algorithms import bipartite
from importlib import reload
from typing import List

# import functions from py file 

import functions.fun
reload(functions.fun)
from functions.fun import CB_data_cleaning, df_from_api_CB, extract_nodes, extract_data_from_column, field_extraction
from functions.fun import nx_dip_graph_from_pandas, plot_bipartite_graph, filter_dict, check_desc
from functions.fun import extract_classes_company_tech, degree_bip, insert_data_classes

# import functions from py file 

import functions.fun_meth_reflections
reload(functions.fun_meth_reflections)
from functions.fun_meth_reflections import zero_order_score, Gct_beta, Gtc_alpha, make_G_hat, next_order_score, generator_order_w
from functions.fun_meth_reflections import M_test_triangular, w_stream, find_convergence, rank_df_class, w_star_analytic

# import functions from py file 

import functions.fun_external_factors
reload(functions.fun_external_factors)
from functions.fun_external_factors import rank_comparison, calibrate_analytic, create_exogenous_rank

# import classes 

import classes
reload(classes)

# cybersecurity
size_comp = [10, 100, 499,997, 1494, 1990, 2429]
size_tech = [26, 134, 306, 372, 431, 456, 477]
flag_cybersecurity = True

size_comp = [100]
size_tech = [134]

# medicine
# size_comp = [10, 100, 999, 4996, 9974, 14954, 19938, 25203]
# size_tech = [32, 95, 254, 437, 507, 549, 586, 604]
# flag_cybersecurity = False

preferences_comp = {"previous_investments":100,
                    "geo_position":0}
preferences_tech = {"previous_investments":100}

print(preferences_comp)

for i in range(len(size_comp)): 

    num_comp = size_comp[i]
    num_tech = size_tech[i]

    # upload data
    print(f'\n\n num comp:{num_comp}, num tech: {num_tech}\n')

    if flag_cybersecurity==False: # all fields
        name_file_com = f'savings/classes/medicine/dict_companies_{num_comp}.pickle'
        name_file_tech = f'savings/classes/medicine/dict_tech_{num_tech}.pickle'
        name_file_graph = f'savings/networks/comp_{num_comp}_tech_{num_tech}.gpickle'
    else: # only companies in cybersecurity
        name_file_com = f'savings/classes/dict_companies_cybersecurity_{num_comp}.pickle'
        name_file_tech = f'savings/classes/dict_tech_cybersecurity_{num_tech}.pickle'
        name_file_graph = f'savings/networks/cybersecurity_comp_{num_comp}_tech_{num_tech}.gpickle'

    with open(name_file_com, 'rb') as f:
        dict_companies = pickle.load(f)

    with open(name_file_tech, 'rb') as f:
        dict_tech = pickle.load(f)

    B = nx.read_gpickle(name_file_graph)

    set0 = extract_nodes(B, 0)
    set1 = extract_nodes(B, 1)

    # adjacency matrix of bipartite graph
    adj_matrix = biadjacency_matrix(B, set0, set1)
    adj_matrix_dense = adj_matrix.todense()

    a = np.squeeze(np.asarray(adj_matrix_dense))
    M = np.squeeze(np.asarray(adj_matrix_dense))

    if flag_cybersecurity==False: # all fields
        name_file_M = 'savings/M/comp_' + str(len(dict_companies)) + '_tech_' + str(len(dict_tech)) + '.npy'                                     
    else: # only companies in cybersecurity
        name_file_M = 'savings/M/cybersecurity_comp_'+ str(len(dict_companies)) + '_tech_' + str(len(dict_tech)) + '.npy'

    np.save(name_file_M, M)

    M_test_triangular(adj_matrix_dense, flag_cybersecurity)

    #par opt
    start_time = time.time()
    best_par = calibrate_analytic(M=M,
                       ua='Companies',
                       dict_class=dict_companies, 
                       exogenous_rank=create_exogenous_rank('Companies', dict_companies, preferences_comp), 
                       index_function=lambda x: (x-50)/25,
                       title='Correlation for Companies',
                       do_plot=True,
                       preferences = preferences_comp,
                       flag_cybersecurity = flag_cybersecurity)
    end_time = time.time()
    time_optimal_par_comp = end_time - start_time
    optimal_alpha_comp = best_par['alpha']
    optimal_beta_comp = best_par['beta']

    start_time = time.time()
    best_par = calibrate_analytic(M=M,
                       ua='Technologies',
                       dict_class=dict_tech, 
                       exogenous_rank=create_exogenous_rank('Technologies', dict_tech, preferences_tech), 
                       index_function=lambda x: (x-50)/25,
                       title='Correlation for Technologies',
                       do_plot=True,
                       preferences = preferences_tech,
                       flag_cybersecurity = flag_cybersecurity)
    end_time = time.time()
    optimal_alpha_tech = best_par['alpha']
    optimal_beta_tech = best_par['beta']
    time_optimal_par_tech = end_time - start_time

    k_c, k_t = zero_order_score(M)

    start_time = time.time()
    convergence_comp = find_convergence(M,
                                        alpha=optimal_alpha_comp, 
                                        beta=optimal_beta_comp, 
                                        fit_or_ubiq='fitness', 
                                        do_plot=True, 
                                        flag_cybersecurity=flag_cybersecurity,
                                        preferences = preferences_comp)
    end_time = time.time()
    time_conv_comp = end_time - start_time

    df_final_companies, dict_companies = rank_df_class(convergence_comp, dict_companies)
    df_final_companies['techrank_normlized'] = df_final_companies['techrank']/np.max(list(df_final_companies['techrank']))*10
    n = np.max(df_final_companies['rank_CB']) + 1
    df_final_companies['rank_CB_normlized'] = n - df_final_companies['rank_CB']
    df_final_companies['TeckRank_int'] = df_final_companies.index + 1.0
    df_spearman = df_final_companies[["TeckRank_int", "rank_CB_normlized"]]
    df_spearman = df_spearman.astype(float)
    df_spearman["name"] = df_final_companies['final_configuration']
    df_spearman.set_index("name")


    if flag_cybersecurity==False:
        name = "savings/csv_results/cybersecurity/complete_companies_" + str(len(dict_companies)) + "_" + str(preferences_comp) + ".csv"
    else:
        name = "savings/csv_results/complete_companies_" + str(len(dict_companies)) + "_" + str(preferences_comp) + ".csv"
   
    df_final_companies.to_csv(name, index = False, header=True)

    spear_corr = df_spearman.corr(method='spearman')

    start_time = time.time()
    convergence_tech = find_convergence(M,
                                        alpha=optimal_alpha_tech, 
                                        beta=optimal_beta_tech, 
                                        fit_or_ubiq='ubiquity', 
                                        do_plot=True, 
                                        flag_cybersecurity=flag_cybersecurity,
                                        preferences=preferences_tech)
    end_time = time.time()
    time_conv_tech = end_time - start_time
    df_final_tech, dict_tech = rank_df_class(convergence_tech, dict_tech)
    df_final_tech['TeckRank_int'] = df_final_tech.index + 1.0

    if flag_cybersecurity==False:
        name = "savings/csv_results/cybersecurity/complete_tech_" + str(len(dict_tech)) + "_" + str(preferences_tech) + ".csv"
    else:
        name = "savings/csv_results/complete_tech_" + str(len(dict_tech)) + "_" + str(preferences_tech) + ".csv"

    df_final_tech.to_csv(name, index = False, header=True)

    if flag_cybersecurity==True:
        name_csv = 'savings/useful_datasets/cybersecurity_df_rank_evolu.csv'
    else:
        name_csv = 'savings/useful_datasets/df_rank_evolu.csv'
    
    df_rank_evolu = pd.read_csv(name_csv)
    df_rank_evolu = df_rank_evolu.drop(['Unnamed: 0'], axis=1, errors='ignore')
    # check if that specific case is already in the csv
    if ((df_rank_evolu['num_comp'] == num_comp) &
        (df_rank_evolu['num_tech'] == num_tech) &
        (df_rank_evolu['preferences_comp'] == str(preferences_comp)) &
        (df_rank_evolu['preferences_tech'] == str(preferences_tech)) &
        (df_rank_evolu['number_iterations_comp'] == convergence_comp['iteration'])&
        (df_rank_evolu['number_iterations_tech'] == convergence_tech['iteration'])
        ).any(): # present

        print("Already analysed")

    else:
        new_row = {'num_comp': num_comp,
                   'num_tech': num_tech,
                    'preferences_comp': str(preferences_comp),
                    'preferences_tech': str(preferences_tech),
                    'optimal_alpha_comp': optimal_alpha_comp,
                    'optimal_beta_comp': optimal_beta_comp,
                    'optimal_alpha_tech': optimal_alpha_tech,
                    'optimal_beta_tech': optimal_beta_tech,
                    'number_iterations_comp': convergence_comp['iteration'],
                    'number_iterations_tech': convergence_tech['iteration'],
                    'time_optimal_par_comp': time_optimal_par_comp,
                    'time_optimal_par_tech': time_optimal_par_tech,
                    'time_conv_comp': time_conv_comp,
                    'time_conv_tech': time_conv_tech,
                    'time_conv_total' : time_conv_comp + time_conv_tech,
                    'spearman_corr_with_cb': spear_corr['rank_CB_normlized']['TeckRank_int']
                }
        df_rank_evolu = df_rank_evolu.append(new_row, ignore_index=True)

    df_rank_evolu = df_rank_evolu.drop(['Unnamed: 0'], axis=1, errors='ignore')
    df_rank_evolu.to_csv(name_csv)
    
 

