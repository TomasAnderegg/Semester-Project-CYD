"""
Script autonome de traitement des données CrunchBase pour TechRank
Auteur: Tomas (modifié avec visualisation tripartite)
Date: 2025
"""

import pandas as pd
import pickle
import os
from pathlib import Path
import duckdb
import networkx as nx
import numpy as np

# ===================================================================
# CONFIGURATION
# ===================================================================

USE_DUCKDB = True  # True = utiliser DuckDB, False = CSV

DATA_PATH_DUCKDB = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
DATA_PATH_CSV = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Code\TechRank\5-TechRank-main\5-TechRank-main\data\sample CB data\organizations.csv"

ENTITY_NAME_1 = "organizations"
ENTITY_NAME_2 = "investments"
ENTITY_NAME_3 = "funding_rounds"

SAVE_DIR_CLASSES = "savings/classes"
SAVE_DIR_NETWORKS = "savings/networks"

FLAG_CYBERSECURITY = True
LIMITS = [10000]
CYBERSECURITY_KEYWORDS = ['quantum computing']

# ===================================================================
# UTILITAIRES
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

# ===================================================================
# CHARGEMENT DES DONNÉES
# ===================================================================

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
# NETTOYAGE DES DONNÉES
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
# GRAPHE TRIPARTITE
# ===================================================================

def extract_tripartite_graph(companies_df, investments_df, funding_rounds_df):
    print("\n========================== CRÉATION DU GRAPHE ==========================")
    org_id_col = None
    for col in ['organization_id', 'uuid', 'id', 'org_id']:
        if col in companies_df.columns:
            org_id_col = col
            print(f"✓ Colonne ID organisation trouvée: {org_id_col}")
            break
    if org_id_col is None:
        raise ValueError("Aucune colonne ID organisation trouvée")

    B = nx.Graph()

    # --- Companies ---
    for _, row in companies_df.iterrows():
        company_id = row.get(org_id_col)
        if pd.notna(company_id):
            B.add_node(company_id, type='company', name=row.get('name'), categories=row.get('category_groups'))

    # --- Funding + Investments ---
    funding_info = funding_rounds_df[['uuid','raised_amount_usd','type','announced_on']].rename(
        columns={'uuid':'uuid_of_f_r', 'type':'funding_type'}
    )
    investments_full = investments_df.merge(
        funding_info, left_on='funding_round_uuid', right_on='uuid_of_f_r', how='left'
    )
    for _, row in investments_full.iterrows():
        funding_uuid = row.get('funding_round_uuid')
        if pd.notna(funding_uuid):
            B.add_node(funding_uuid, type='investment', money_raised_usd=row.get('raised_amount_usd'), announced_on=row.get('announced_on'))

    # --- Technologies ---
    tech_df = companies_df[[org_id_col,'category_groups']].copy().explode('category_groups')[['category_groups']].drop_duplicates()
    for _, row in tech_df.iterrows():
        tech_name = row.get('category_groups')
        if pd.notna(tech_name):
            B.add_node(tech_name, type='technology')

    # --- Arêtes Company ↔ Investment ---
    inv_org_col = None
    for col in ['org_uuid', 'organization_id', 'company_uuid', org_id_col]:
        if col in investments_full.columns:
            inv_org_col = col
            break
    if inv_org_col:
        for _, row in investments_full.iterrows():
            org_id = row.get(inv_org_col)
            funding_uuid = row.get('funding_round_uuid')
            if pd.notna(org_id) and pd.notna(funding_uuid):
                B.add_edge(org_id, funding_uuid, type='company_investment')

    # --- Arêtes Company ↔ Technology ---
    edges_CT = companies_df[[org_id_col,'category_groups']].copy().explode('category_groups')[[org_id_col,'category_groups']]
    for _, row in edges_CT.iterrows():
        company_id = row.get(org_id_col)
        tech_name = row.get('category_groups')
        if pd.notna(company_id) and pd.notna(tech_name):
            B.add_edge(company_id, tech_name, type='company_technology')

    # --- Arêtes Investment ↔ Technology ---
    if inv_org_col:
        edges_IT = edges_CT.merge(
            investments_full[['funding_round_uuid',inv_org_col]],
            on=inv_org_col, how='inner'
        )[['funding_round_uuid','category_groups']].drop_duplicates()
        for _, row in edges_IT.iterrows():
            funding_uuid = row.get('funding_round_uuid')
            tech_name = row.get('category_groups')
            if pd.notna(funding_uuid) and pd.notna(tech_name):
                B.add_edge(funding_uuid, tech_name, type='investment_technology')

    print(f"✓ Graphe créé : {B.number_of_nodes()} nœuds, {B.number_of_edges()} arêtes")
    return B

# ===================================================================
# VISUALISATION TRIPARTITE
# ===================================================================

def visualize_tripartite_graph(B, max_companies=15, max_investments=15, max_technologies=15):
    import matplotlib.pyplot as plt

    companies = [n for n, d in B.nodes(data=True) if d.get('type') == 'company']
    investments = [n for n, d in B.nodes(data=True) if d.get('type') == 'investment']
    technologies = [n for n, d in B.nodes(data=True) if d.get('type') == 'technology']

    top_companies = sorted(companies, key=lambda n: B.degree(n), reverse=True)[:max_companies]
    top_investments = sorted(investments, key=lambda n: B.degree(n), reverse=True)[:max_investments]
    top_tech = sorted(technologies, key=lambda n: B.degree(n), reverse=True)[:max_technologies]

    nodes_to_keep = set(top_companies) | set(top_investments) | set(top_tech)
    B_sub = B.subgraph(nodes_to_keep).copy()

    # Placement manuel par "couches"
    pos = {}
    y_offset = 0
    spacing = 1.5

    for i, n in enumerate(top_companies):
        pos[n] = (0, i * spacing)
    for i, n in enumerate(top_investments):
        pos[n] = (5, i * spacing)
    for i, n in enumerate(top_tech):
        pos[n] = (10, i * spacing)

    plt.figure(figsize=(18, 10))
    plt.axis('off')

    nx.draw_networkx_nodes(B_sub, pos, nodelist=top_companies, node_color='red', node_shape='s', label='Companies', alpha=0.6)
    nx.draw_networkx_nodes(B_sub, pos, nodelist=top_investments, node_color='green', node_shape='o', label='Investments', alpha=0.6)
    nx.draw_networkx_nodes(B_sub, pos, nodelist=top_tech, node_color='blue', node_shape='^', label='Technologies', alpha=0.6)

    nx.draw_networkx_edges(B_sub, pos, alpha=0.3)
    labels = {}
    for n in B_sub.nodes():
        if B_sub.nodes[n].get('type') == 'company':
            labels[n] = B_sub.nodes[n].get('name', n)
        elif B_sub.nodes[n].get('type') == 'technology':
            labels[n] = n
        elif B_sub.nodes[n].get('type') == 'investment':
            labels[n] = "INV"
    nx.draw_networkx_labels(B_sub, pos, labels, font_size=8)

    plt.title("Graph Tripartite: Companies ↔ Investments ↔ Technologies", fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# ===================================================================
# SAUVEGARDE
# ===================================================================

def save_graph_and_dicts(B, companies_df, limit, is_cybersecurity):
    suffix = "cybersecurity_" if is_cybersecurity else ""
    file_graph = f"{SAVE_DIR_NETWORKS}/{suffix}graph_{limit}.gpickle"
    with open(file_graph,'wb') as f:
        pickle.dump(B,f)
    print(f"✓ Graphe sauvegardé : {file_graph}")

# ===================================================================
# MAIN
# ===================================================================

def main():
    create_directories()
    df_comp = load_data(use_duckdb=USE_DUCKDB, entity_name=ENTITY_NAME_1)
    df_invest = load_data(use_duckdb=USE_DUCKDB, entity_name=ENTITY_NAME_2)
    df_fund = load_data(use_duckdb=USE_DUCKDB, entity_name=ENTITY_NAME_3)

    df_comp_clean = clean_data(df_comp)
    df_comp_proc = process_category_groups(df_comp_clean)

    if FLAG_CYBERSECURITY:
        df_comp_proc = filter_cybersecurity(df_comp_proc, CYBERSECURITY_KEYWORDS)
        if len(df_comp_proc)==0:
            print("⚠️ Aucune entreprise cybersécurité trouvée")
            return

    for limit in LIMITS:
        df_limited = df_comp_proc.iloc[:limit]
        company_uuids = df_limited['uuid'].unique()
        df_invest_filtered = df_invest[df_invest['funding_round_uuid'].isin(
            df_fund[df_fund['org_uuid'].isin(company_uuids)]['uuid']
        )]
        df_fund_filtered = df_fund[df_fund['org_uuid'].isin(company_uuids)]
        B = extract_tripartite_graph(df_limited, df_invest_filtered, df_fund_filtered)
        visualize_tripartite_graph(B, max_companies=10, max_investments=10, max_technologies=15)
        save_graph_and_dicts(B, df_limited, limit, FLAG_CYBERSECURITY)

if __name__=="__main__":
    main()
