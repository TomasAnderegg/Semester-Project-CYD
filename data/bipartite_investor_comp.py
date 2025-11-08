import pandas as pd
import pickle
import os
from pathlib import Path
import duckdb
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import math
import classes
import matplotlib
matplotlib.use('Qt5Agg')  # ou 'TkAgg' selon ton installation


# ===================================================================
# CONFIGURATION
# ===================================================================

USE_DUCKDB = True  # True = utiliser DuckDB, False = CSV

DATA_PATH_DUCKDB = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
DATA_PATH_INVESTMENTS_CSV = r"data/data_cb/investments.csv"
DATA_PATH_FUNDING_CSV = r"data/data_cb/funding_rounds.csv"

TABLE_NAME_INVESTMENTS = "investments"
TABLE_NAME_FUNDING = "funding_rounds"
TABLE_NAME_ORGA = "organizations"

SAVE_DIR_NETWORKS = "savings/bipartite_invest/networks"

SAVE_DIR_CLASSES = "savings/bipartite_invest_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_invest_comp/networks"
SAVE_DIR_CSV = "savings/bipartite_invest_comp/csv_exports"  # ✅ NOUVEAU: dossier pour les CSV

FLAG_FILTER = False  # Mettre True si tu veux filtrer
FILTER_KEYWORDS = ['quantum computing']  # Keywords pour filtrage optionnel
LIMITS = [500]  # Nombre d'entrées à traiter

# ===================================================================
# UTILS
# ===================================================================

def create_directories():
    Path(SAVE_DIR_NETWORKS).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_CSV).mkdir(parents=True, exist_ok=True)  # ✅ NOUVEAU


def load_data_from_duckdb(filepath, table_name):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier DuckDB introuvable: {filepath}")
    conn = duckdb.connect(filepath, read_only=True)
    df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
    conn.close()
    return df


def load_data_from_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier CSV introuvable: {filepath}")
    return pd.read_csv(filepath)


def load_data(use_duckdb=True, table_name=""):
    if use_duckdb:
        return load_data_from_duckdb(DATA_PATH_DUCKDB, table_name)
    else:
        if table_name == TABLE_NAME_INVESTMENTS:
            return load_data_from_csv(DATA_PATH_INVESTMENTS_CSV)
        elif table_name == TABLE_NAME_FUNDING:
            return load_data_from_csv(DATA_PATH_FUNDING_CSV)
        else:
            raise ValueError(f"Table name {table_name} non reconnue")
        
def save_graph_and_dicts(B, df_companies, dict_companies, dict_tech, limit, flag_cybersecurity):
    """Sauvegarde le graphe et les dictionnaires associés."""
    prefix = "cybersecurity_" if flag_cybersecurity else ""

    os.makedirs(SAVE_DIR_CLASSES, exist_ok=True)
    os.makedirs(SAVE_DIR_NETWORKS, exist_ok=True)

    # Sauvegarder les dictionnaires
    with open(f'{SAVE_DIR_CLASSES}/{prefix}dict_companies_ranked_{limit}.pickle', 'wb') as f:
        pickle.dump(dict_companies, f)

    with open(f'{SAVE_DIR_CLASSES}/{prefix}dict_tech_ranked_{limit}.pickle', 'wb') as f:
        pickle.dump(dict_tech, f)

    # ✅ Sauvegarder le graphe avec pickle directement (évite tout bug NetworkX)
    with open(f"{SAVE_DIR_NETWORKS}/{prefix}bipartite_graph_{limit}.gpickle", "wb") as f:
        pickle.dump(B, f)

    # Sauvegarder le DataFrame
    df_companies.to_csv(f'{SAVE_DIR_CLASSES}/{prefix}companies_ranked_{limit}.csv', index=False)

    print(f"\n✓ Résultats sauvegardés dans {SAVE_DIR_CLASSES}/ et {SAVE_DIR_NETWORKS}/")


# ===================================================================
# DATA CLEANING
# ===================================================================

def CB_data_cleaning(
    df: pd.DataFrame, 
    to_drop: List[str], 
    to_rename: Dict[str, str], 
    to_check_double: Dict[str, str],
    drop_if_nan: List[str], 
    sort_by: str = ""):
    """Performs the Data Cleaning part of the CB dataset

    Args:
        - df: dataset to clean
        - to_drop: columns to drop
        - to_rename: columns to rename and new name
        - to_check_double: columns to check. If they bring additional value and,
                           in case they don't, drop them
        - drop_if_nan: columns where NaN values should be dropped
        - sort_by: column by which sort values

    Return:
        - df: cleaned dataset
    """
    df = df.drop(to_drop, axis=1, errors='ignore')
    df = df.rename(columns=to_rename)

    for key, item in to_check_double.items():
        # item does not bring new info:
        if key in df.columns and item in df.columns:
            if (df[key] == df[item]).all() == True: 
                df = df.drop([item], axis=1)

    # drop if nan
    if len(drop_if_nan) > 0:
        for to_drop_col in drop_if_nan:
            if to_drop_col in df.columns:
                df = df.dropna(subset=[to_drop_col])

    if len(sort_by) > 0 and sort_by in df.columns:
        df = df.sort_values(sort_by)

    return df


def clean_investments_data(df):
    """Clean investments dataset"""
    to_drop = [
        'uuid',
        'permalink',
        'funding_round_name',
        'cb_url',
        'created_at',
        'updated_at',   
        'rank',
    ]
    to_rename = {}
    drop_if_nan = []
    to_check_double = {}
    sort_by = ""
    
    return CB_data_cleaning(df, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)


def clean_funding_data(df):
    """Clean funding rounds dataset"""
    to_drop = [
        'type',
        'permalink',
        'cb_url',   
        'rank',
        'funding_round_name',
        'created_at',
        'updated_at',
        'investor_type',
        'raised_amount',
        'is_lead_investor',
        'post_money_valuation_usd',                                      
        'post_money_valuation',                                    
        'post_money_valuation_currency_code',
    ]
    to_rename = {
        'category_list': 'category_groups',
        'uuid': 'funding_round_uuid'
    }
    drop_if_nan = ['name']
    to_check_double = {}
    sort_by = ""
    
    return CB_data_cleaning(df, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)


def merge_and_clean_final(df_funding, df_investments):
    """Merge funding and investments data, then clean"""
    # ✅ MERGE COMME TU LE VEUX
    df_merged = pd.merge(df_funding, df_investments, on='funding_round_uuid')
    
    # ✅ SAUVEGARDER LE RÉSULTAT DU MERGE COMPLET
    csv_path = f"{SAVE_DIR_CSV}/merged_funding_investments_full.csv"
    df_merged.to_csv(csv_path, index=False)
    print(f"✓ CSV du merge complet sauvegardé : {csv_path}")
    print(f"  Colonnes: {list(df_merged.columns)}")
    print(f"  Shape: {df_merged.shape}")
    
    # to_drop = [ ]
    # to_rename = {}
    # drop_if_nan = []
    # to_check_double = {}
    # sort_by = ""
    
    # df_clean = CB_data_cleaning(df_merged, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)
    
    # # ✅ Garder TOUTES les colonnes nécessaires pour le graphe
    # required_cols = ["org_name", "investor_name", "org_uuid", "investor_uuid", 
    #                  "raised_amount_usd", "num_investments"]
    
    # # Vérifier quelles colonnes existent
    # existing_cols = [col for col in required_cols if col in df_clean.columns]
    
    # if 'org_name' in df_clean.columns and 'investor_name' in df_clean.columns:
    #     df_graph = df_clean[existing_cols].copy()
    #     df_graph = df_graph.dropna(subset=['org_name', 'investor_name'])
        
    #     # ✅ SAUVEGARDER AUSSI LE RÉSULTAT FILTRÉ
    #     csv_path_graph = f"{SAVE_DIR_CSV}/merged_for_graph.csv"
    #     df_graph.to_csv(csv_path_graph, index=False)
    #     print(f"✓ CSV pour le graphe sauvegardé : {csv_path_graph}")
    #     print(f"  Colonnes gardées: {list(df_graph.columns)}")
        
    #     print("----------en tete de df_graph--------:")
    #     print(df_graph.head())
    return df_merged
    # else:
    #     raise ValueError("Colonnes 'org_name' ou 'investor_name' manquantes")


# ===================================================================
# BIPARTITE GRAPH CREATION
# ===================================================================

def nx_dip_graph_from_pandas(df):
    """Creates the bipartite graph from the dataset

    bipartite = 0 => investor (investor_name)
    bipartite = 1 => company (org_name)

    Args:
        - df: DataFrame with necessary columns

    Return:
        - B: bipartite graph 
    """
    dict_companies = {}
    dict_invest = {}

    B = nx.Graph()
    
    for index, row in df.iterrows():
        # Investor informations
        invest_name = row['investor_name']
        
        # Créer l'investisseur selon la structure de classes.Investor
        i = classes.Investor(
            investor_id=row['funding_round_uuid'],
            name=invest_name,
            raised_amount_usd=row['raised_amount_usd'],
            num_investors=row['investor_count']
        )
        
        dict_invest[invest_name] = i
        B.add_node(invest_name, bipartite=0)

        # Company informations
        comp_name = row['org_name']
        
        # Créer la company selon la structure de classes.Company
        c = classes.Company(
            id=row.get('org_uuid', f'org_{comp_name}'),
            name=comp_name,
            technologies=[]  # Liste vide par défaut
        )
        
        dict_companies[comp_name] = c
        B.add_node(comp_name, bipartite=1)

        B.add_edge(comp_name, invest_name)

    return B, dict_companies, dict_invest


def create_bipartite_from_dataframe(df, df_comp_clean):
    """Create bipartite graph from org_name and investor_name columns"""
    B = nx.Graph()
    
    for _, row in df.iterrows():
        org = row['org_name']
        investor = row['investor_name']
        
        if pd.notna(org) and pd.notna(investor):
            B.add_node(org, bipartite=0)
            B.add_node(investor, bipartite=1)
            B.add_edge(org, investor)
    
    return B


# ===================================================================
# VISUALIZATION
# ===================================================================

def plot_bipartite_graph(G, circular=False):
    """Plots the bipartite network"""
    print("\n========================== PLOTTING BIPARTITE GRAPH ==========================")

    set1 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 0]
    set2 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 1]

    if circular:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)

    if len(set1) >= 20:
        plt.figure(figsize=(25, 15))
    else:
        plt.figure(figsize=(19, 13))

    plt.ion()
    plt.axis('on')

    companies = set1
    investors = set2

    # calculate degree centrality
    companyDegree = nx.degree(G, companies) 
    investorDegree = nx.degree(G, investors)

    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=companies,
                           node_color='r',
                           node_size=[v * 100 for v in dict(companyDegree).values()],
                           alpha=0.6,
                           label='Companies')

    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=investors,
                           node_color='g',
                           node_size=[v * 200 for v in dict(investorDegree).values()],
                           alpha=0.6,
                           label='Investors')

    nx.draw_networkx_labels(G, pos, {n: n for n in companies}, font_size=10)
    nx.draw_networkx_labels(G, pos, {n: n for n in investors}, font_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4)

    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
    return pos


# ===================================================================
# SAVING
# ===================================================================

def save_graph(B, limit):
    file_graph = f"{SAVE_DIR_NETWORKS}/investment_graph_{limit}.gpickle"
    with open(file_graph, 'wb') as f:
        pickle.dump(B, f)
    print(f"✓ Graphe sauvegardé : {file_graph}")


def save_graph_and_dicts(B, dict_companies, dict_investors, limit):
    """Sauvegarde le graphe et les dictionnaires associés"""
    os.makedirs(SAVE_DIR_CLASSES, exist_ok=True)
    os.makedirs(SAVE_DIR_NETWORKS, exist_ok=True)

    # Sauvegarder les dictionnaires
    with open(f'{SAVE_DIR_CLASSES}/dict_companies_{limit}.pickle', 'wb') as f:
        pickle.dump(dict_companies, f)
    print(f"✓ Dictionnaire companies sauvegardé : {SAVE_DIR_CLASSES}/dict_companies_{limit}.pickle")

    with open(f'{SAVE_DIR_CLASSES}/dict_investors_{limit}.pickle', 'wb') as f:
        pickle.dump(dict_investors, f)
    print(f"✓ Dictionnaire investors sauvegardé : {SAVE_DIR_CLASSES}/dict_investors_{limit}.pickle")

    # Sauvegarder le graphe avec pickle directement
    with open(f"{SAVE_DIR_NETWORKS}/bipartite_graph_{limit}.gpickle", "wb") as f:
        pickle.dump(B, f)
    print(f"✓ Graphe sauvegardé : {SAVE_DIR_NETWORKS}/bipartite_graph_{limit}.gpickle")

    print(f"\n✓ Résultats sauvegardés dans {SAVE_DIR_CLASSES}/ et {SAVE_DIR_NETWORKS}/")


# ===================================================================
# MAIN
# ===================================================================

def main(max_companies_plot=20, max_investors_plot=20):
    create_directories()
    
    print("\n========================== LOADING DATA ==========================")
    df_investments = load_data(use_duckdb=USE_DUCKDB, table_name=TABLE_NAME_INVESTMENTS)
    df_funding = load_data(use_duckdb=USE_DUCKDB, table_name=TABLE_NAME_FUNDING)
    
    print("✓ Données chargées")
    
    print("\n========================== CLEANING DATA ==========================")
    df_investments_clean = clean_investments_data(df_investments)
    df_funding_clean = clean_funding_data(df_funding)
    
    print("✓ Données nettoyées")
    
    print("\n========================== MERGING DATA ==========================")
    df_graph_full = merge_and_clean_final(df_funding_clean, df_investments_clean)
    print(f"✓ Données mergées : {len(df_graph_full):,} lignes")
    
    for limit in LIMITS:
        print(f"\n{'='*70}")
        print(f"TRAITEMENT POUR LIMIT = {limit}")
        print(f"{'='*70}")
        
        # Limiter les données
        df_graph = df_graph_full.head(limit).copy()
        
        # Créer le graphe bipartite
        B, dict_companies, dict_investors = nx_dip_graph_from_pandas(df_graph)
        
        companies = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
        investors = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]
        
        print(f"✓ Graphe créé : {len(companies)} entreprises, {len(investors)} investisseurs")
        
        # Limiter pour le plotting
        top_companies = sorted(companies, key=lambda n: B.degree(n), reverse=True)[:max_companies_plot]
        top_investors = sorted(investors, key=lambda n: B.degree(n), reverse=True)[:max_investors_plot]
        
        nodes_to_keep = set(top_companies) | set(top_investors)
        B_sub = B.subgraph(nodes_to_keep).copy()
        
        print(f"✓ Sous-graphe créé : {len(B_sub.nodes())} noeuds, {len(B_sub.edges())} arêtes")
        
        # Plot
        pos = plot_bipartite_graph(B_sub)
        
        # Save - avec dictionnaires
        save_graph_and_dicts(B_sub, dict_companies, dict_investors, limit)


if __name__ == "__main__":
    # Plot seulement 15 entreprises et 15 investisseurs
    main(max_companies_plot=50, max_investors_plot=50)