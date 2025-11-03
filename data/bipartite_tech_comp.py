import pandas as pd
import pickle
import os
from pathlib import Path
import duckdb
import networkx as nx
import numpy as np
import classes
import matplotlib.pyplot as plt
from typing import List, Dict
import math
import matplotlib

matplotlib.use('Qt5Agg')  # ou 'TkAgg' selon ton installation


# ===================================================================
# CONFIGURATION
# ===================================================================

USE_DUCKDB = True  # True = utiliser DuckDB, False = CSV

DATA_PATH_DUCKDB = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
DATA_PATH_CSV = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Code\TechRank\5-TechRank-main\5-TechRank-main\data\sample CB data\organizations.csv"

ENTITY_NAME_1 = "organizations"

SAVE_DIR_CLASSES = "savings/bipartite_tech_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_tech_comp/networks"

FLAG_CYBERSECURITY = True
LIMITS = [10000]
CYBERSECURITY_KEYWORDS = ['quantum computing']

# ===================================================================
# UTILS
# ===================================================================

def create_directories():
    Path(SAVE_DIR_CLASSES).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_NETWORKS).mkdir(parents=True, exist_ok=True)


def convert_to_list(x):
    if isinstance(x, pd.Series):
        return x.apply(convert_to_list)
    if x is None or pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(item).strip() for item in x if pd.notna(item)]
    return [item.strip() for item in str(x).split(',') if item.strip()]


def load_data_from_duckdb(filepath, table_name):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier DuckDB introuvable: {filepath}")
    conn = duckdb.connect(filepath, read_only=True)
    df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
    conn.close()
    return df


def load_data_from_csv(filepath):
    return pd.read_csv(filepath)


def load_data(use_duckdb=True, entity_name=ENTITY_NAME_1):
    if use_duckdb:
        return load_data_from_duckdb(DATA_PATH_DUCKDB, entity_name)
    else:
        return load_data_from_csv(DATA_PATH_CSV)


# ===================================================================
# DATA CLEANING AND PROCESSING
# ===================================================================

def clean_data(df):
    df_clean = df.copy()
    columns_to_drop = [
        'type','permalink','cb_url','domain','address','state_code',
        'updated_at','legal_name','roles','postal_code','homepage_url','num_funding_rounds',
        'total_funding_currency_code','phone','email','num_exits','alias1','alias2','alias3',
        'logo_url','last_funding_on','twitter_url','facebook_url','linkedin_url','crunchbase_url',
        'overview','acquisitions','city','primary_role','region','founded_on','ipo','milestones',
        'news_articles','status','country_code','investment_type','post_money_valuation_usd',
        'pre_money_valuation_usd','closed_on'
    ]
    cols_to_drop = [c for c in columns_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    rename_mapping = {'category_list':'category_groups','category_groups_list':'category_groups'}
    actual_renames = {k:v for k,v in rename_mapping.items() if k in df_clean.columns}
    df_clean = df_clean.rename(columns=actual_renames)

    required_columns = ['category_groups','rank','short_description']
    missing_cols = [col for col in required_columns if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Colonnes requises manquantes: {missing_cols}")

    df_clean = df_clean.dropna(subset=required_columns)
    df_clean = df_clean.sort_values('rank').reset_index(drop=True)
    return df_clean


def process_category_groups(df):
    df_proc = df.copy()
    if "category_groups" not in df_proc.columns:
        raise ValueError(f"Colonne 'category_groups' introuvable")
    if df_proc.columns.duplicated().any():
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
    col_series = df_proc['category_groups']
    first_valid_idx = col_series.first_valid_index()
    if first_valid_idx is not None:
        first_valid = col_series.loc[first_valid_idx]
        if not isinstance(first_valid, list):
            df_proc['category_groups'] = col_series.apply(convert_to_list)
    df_proc = df_proc.reset_index(drop=True)
    return df_proc


def filter_cybersecurity(df, keywords):
    print("\n========================== FILTRAGE CYBER ==========================")
    df = df.reset_index(drop=True).copy()

    def check_in_categories(lst):
        if not isinstance(lst, list):
            return False
        flat_list = []
        for item in lst:
            if isinstance(item, list):
                flat_list.extend([str(x) for x in item])
            else:
                flat_list.append(str(item))
        return any(k.lower() in ' '.join(flat_list).lower() for k in keywords)
    
    mask_cat = df['category_groups'].apply(check_in_categories)
    mask_desc = df['short_description'].astype(str).str.contains('|'.join(keywords), case=False, na=False)
    mask_combined = mask_cat | mask_desc
    df_filtered = df[mask_combined].reset_index(drop=True)
    
    print(f"✓ Total d'entreprises cybersécurité: {len(df_filtered):,}")
    return df_filtered


# ===================================================================
# BIPARTITE CREATION FUNCTION
# ===================================================================

def extract_classes_company_tech(df):
    """Extracts the dictionaries of Companies and Technologies 
    from the dataset and create the network"""
    
    dict_companies = {}
    dict_tech = {}
    B = nx.Graph()

    for index, row in df.iterrows():
        comp_name = row['name']

        c = classes.Company(
            id=row['uuid'],
            name=comp_name,
            technologies=row['category_groups'],
        )

        if 'rank_company' in df.columns:
            c.rank_CB = row['rank_company']
        elif 'rank' in df.columns:
            c.rank_CB = row['rank']
        
        dict_companies[comp_name] = c
        B.add_node(comp_name, bipartite=0)
        
        if isinstance(row['category_groups'], list):
            for tech in row['category_groups']:
                if tech not in dict_tech:
                    dict_tech[tech] = classes.Technology(name=tech)
                    B.add_node(tech, bipartite=1)
                B.add_edge(comp_name, tech)
        else:
            tech = row['category_groups']
            if tech not in dict_tech:
                dict_tech[tech] = classes.Technology(name=tech)
                B.add_node(tech, bipartite=1)
            B.add_edge(comp_name, tech)

    return dict_companies, dict_tech, B


# ===================================================================
# VISUALIZATION FUNCTION
# ===================================================================

def filter_dict(G, percentage, set1, set2):
    degree_set2 = list(dict(nx.degree(G, set2)).values())
    threshold_companies = math.ceil(len(set2)/percentage)
    if threshold_companies > np.max(degree_set2):
        threshold_companies = np.mean(degree_set2)
    dict_nodes = nx.degree(G, set1) 
    to_delete = [key for key, value in dict(dict_nodes).items() if value <= threshold_companies]
    return to_delete


def plot_bipartite_graph(G, small_degree=True, percentage=10, circular=False):
    print("\n========================== PLOTTING BIPARTITE GRAPH ==========================")

    set1 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 0]
    set2 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 1]

    pos = nx.circular_layout(G) if circular else nx.spring_layout(G)
    plt.figure(figsize=(25,15) if len(set1) >= 20 else (19,13))
    plt.ion()
    plt.axis('on')

    company = set1
    value = set2

    companyDegree = nx.degree(G, company) 
    valueDegree = nx.degree(G, value)

    nx.draw_networkx_nodes(G, pos, nodelist=company, node_color='r',
                           node_size=[v * 100 for v in dict(companyDegree).values()],
                           alpha=0.6, label=company)

    nx.draw_networkx_nodes(G, pos, nodelist=value, node_color='b',
                           node_size=[v * 200 for v in dict(valueDegree).values()],
                           alpha=0.6, label=value)

    nx.draw_networkx_labels(G, pos, {n: n for n in company}, font_size=10)
    nx.draw_networkx_labels(G, pos, {n: n for n in value}, font_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4)

    plt.tight_layout()
    plt.show(block=True)
    return pos


# ===================================================================
# ✅ SAVING FUNCTION (corrigée)
# ===================================================================

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
# MAIN
# ===================================================================

def main(max_companies_plot=20, max_tech_plot=20):
    create_directories()
    df_comp = load_data(use_duckdb=USE_DUCKDB, entity_name=ENTITY_NAME_1)

    df_comp_clean = clean_data(df_comp)
    df_comp_proc = process_category_groups(df_comp_clean)

    if FLAG_CYBERSECURITY:
        df_comp_proc = filter_cybersecurity(df_comp_proc, CYBERSECURITY_KEYWORDS)
        if len(df_comp_proc) == 0:
            print("Aucune entreprise cybersécurité trouvée")
            return

    for limit in LIMITS:
        dict_companies, dict_tech, B = extract_classes_company_tech(df_comp_proc)

        companies = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
        techs = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]

        top_companies = sorted(companies, key=lambda n: B.degree(n), reverse=True)[:max_companies_plot]
        top_techs = sorted(techs, key=lambda n: B.degree(n), reverse=True)[:max_tech_plot]

        nodes_to_keep = set(top_companies) | set(top_techs)
        B_sub = B.subgraph(nodes_to_keep).copy()

        pos = plot_bipartite_graph(B_sub)

        # ✅ Appel corrigé
        save_graph_and_dicts(B, df_comp_proc, dict_companies, dict_tech, limit, FLAG_CYBERSECURITY)




if __name__ == "__main__":
    main(max_companies_plot=10, max_tech_plot=15)
