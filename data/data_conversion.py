"""
Script autonome de traitement des données CrunchBase pour TechRank
Auteur: Tomas
Date: 2025

Fonctionnalités :
- Chargement CSV ou DuckDB
- Nettoyage et traitement des catégories
- Filtrage cybersécurité
- Création d'un graphe tripartite Companies ↔ Investments ↔ Technologies
- Sauvegarde des dictionnaires et du graphe
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
CYBERSECURITY_KEYWORDS = ['quantum computing']  # mots-clés pour cybersécurité

# ===================================================================
# UTILITAIRES
# ===================================================================

def create_directories():
    Path(SAVE_DIR_CLASSES).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_NETWORKS).mkdir(parents=True, exist_ok=True)

def convert_to_list(x):
    # Si x est une Series, appliquer récursivement
    if isinstance(x, pd.Series):
        return x.apply(convert_to_list)

    # Si NaN ou None, retourner liste vide
    if x is None or pd.isna(x):
        return []

    # Si c'est déjà une liste, nettoyer et convertir en string
    if isinstance(x, list):
        return [str(item).strip() for item in x if pd.notna(item)]

    # Sinon, convertir en string et split sur la virgule
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
        'type','permalink','cb_url','created_at','domain','address','state_code',
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
    
    # Vérifier que la colonne existe
    if "category_groups" not in df_proc.columns:
        raise ValueError(f"Colonne 'category_groups' introuvable")
    
    # Gérer les colonnes dupliquées
    if df_proc.columns.duplicated().any():
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
    
    # Obtenir la Series
    col_series = df_proc['category_groups']
    
    # Vérifier le type de la première valeur non-nulle
    first_valid_idx = col_series.first_valid_index()
    if first_valid_idx is not None:
        first_valid = col_series.loc[first_valid_idx]
        
        # Convertir en liste si ce n'est pas déjà une liste
        if not isinstance(first_valid, list):
            df_proc['category_groups'] = col_series.apply(convert_to_list)
    
    # IMPORTANT: Reset de l'index après le traitement
    df_proc = df_proc.reset_index(drop=True)
    
    return df_proc

def filter_cybersecurity(df, keywords):
    """Filtre les entreprises de cybersécurité - VERSION CORRIGÉE."""
    print(f"\n{'='*60}")
    print("FILTRAGE CYBERSÉCURITÉ")
    print(f"{'='*60}")
    
    # ÉTAPE 1: Forcer le reset d'index avant toute opération
    df = df.reset_index(drop=True).copy()
    
    # DIAGNOSTIC: Vérifier la structure de category_groups
    print(f"DEBUG: Longueur du DataFrame: {len(df)}")
    if len(df) > 0:
        sample = df['category_groups'].iloc[0]
        print(f"DEBUG: Type de category_groups[0]: {type(sample)}")
        print(f"DEBUG: Valeur: {sample}")
    
    # ÉTAPE 2: Fonction pour vérifier les catégories (gère les listes imbriquées)
    def check_in_categories(lst):
        if not isinstance(lst, list):
            return False
        # Aplatir les listes imbriquées
        flat_list = []
        for item in lst:
            if isinstance(item, list):
                flat_list.extend([str(x) for x in item])
            else:
                flat_list.append(str(item))
        # Vérifier si un keyword est présent
        return any(k.lower() in ' '.join(flat_list).lower() for k in keywords)
    
    # ÉTAPE 3: Créer les masques
    mask_cat = df['category_groups'].apply(check_in_categories)
    mask_desc = df['short_description'].astype(str).str.contains(
        '|'.join(keywords), case=False, na=False
    )
    
    # DIAGNOSTIC: Vérifier les shapes
    print(f"DEBUG: Shape mask_cat: {mask_cat.shape}")
    print(f"DEBUG: Shape mask_desc: {mask_desc.shape}")
    
    # ÉTAPE 4: Combinaison - GARDER en pandas Series pour éviter le problème de shape
    mask_combined = mask_cat | mask_desc
    
    # ÉTAPE 5: Appliquer le filtre
    df_filtered = df[mask_combined].reset_index(drop=True)
    
    # Affichage des résultats
    print(f"✓ Correspondances dans category_groups: {mask_cat.sum():,}")
    print(f"✓ Correspondances dans short_description: {mask_desc.sum():,}")
    print(f"✓ Total d'entreprises cybersécurité: {len(df_filtered):,}")
    
    if len(df_filtered) > 0:
        print(f"\n  Exemples d'entreprises filtrées:")
        for i, row in df_filtered.head(3).iterrows():
            print(f"  - {row.get('name', 'N/A')}: {row['category_groups']}")
    
    return df_filtered

# ===================================================================
# GRAPHE TRIPARTITE
# ===================================================================

def extract_tripartite_graph(companies_df, investments_df, funding_rounds_df):
    """
    Crée un graphe tripartite Companies ↔ Investments ↔ Technologies.
    """
    print(f"\n{'='*60}")
    print("CRÉATION DU GRAPHE TRIPARTITE")
    print(f"{'='*60}")
    
    # DIAGNOSTIC: Afficher les colonnes disponibles
    print(f"Colonnes companies_df: {companies_df.columns.tolist()}")
    print(f"Colonnes investments_df: {investments_df.columns.tolist()}")
    print(f"Colonnes funding_rounds_df: {funding_rounds_df.columns.tolist()}")
    
    # Identifier la colonne ID de l'organisation
    org_id_col = None
    for col in ['organization_id', 'uuid', 'id', 'org_id']:
        if col in companies_df.columns:
            org_id_col = col
            print(f"✓ Colonne ID organisation trouvée: {org_id_col}")
            break
    
    if org_id_col is None:
        raise ValueError(f"Aucune colonne ID trouvée. Colonnes disponibles: {companies_df.columns.tolist()}")
    
    B = nx.Graph()

    # --- Companies ---
    for idx, row in companies_df.iterrows():
        company_id = row.get(org_id_col)
        # Vérifier que l'ID n'est pas None
        if pd.notna(company_id) and company_id is not None:
            B.add_node(
                company_id,
                type='company',
                name=row.get('name'),
                categories=row.get('category_groups')
            )

    # --- Préparer les informations financières depuis funding_rounds_df ---
    # La colonne s'appelle 'type' et non 'funding_type'
    funding_info = funding_rounds_df[['uuid','raised_amount_usd','type','announced_on']].rename(
        columns={'uuid':'uuid_of_f_r', 'type':'funding_type'}
    )

    # Merge investments avec funding_rounds pour enrichir
    investments_full = investments_df.merge(
        funding_info,
        left_on='funding_round_uuid',
        right_on='uuid_of_f_r',
        how='left'
    )

    # --- Investments ---
    for idx, row in investments_full.iterrows():
        funding_uuid = row.get('funding_round_uuid')
        # Vérifier que l'UUID n'est pas None avant d'ajouter le nœud
        if pd.notna(funding_uuid) and funding_uuid is not None:
            B.add_node(
                funding_uuid,
                type='investment',
                money_raised_usd=row.get('raised_amount_usd'),
                announced_on=row.get('announced_on')
            )

    # --- Technologies ---
    tech_df = companies_df[[org_id_col,'category_groups']].copy()
    tech_df = tech_df.explode('category_groups')[['category_groups']].drop_duplicates()
    for idx, row in tech_df.iterrows():
        tech_name = row.get('category_groups')
        # Vérifier que le nom de la technologie n'est pas None
        if pd.notna(tech_name) and tech_name is not None:
            B.add_node(tech_name, type='technology')

    # --- Arêtes Company ↔ Investment ---
    # Identifier la colonne org_id dans investments_full
    # D'après le merge, il devrait y avoir 'org_uuid' depuis funding_rounds_df
    inv_org_col = None
    for col in ['org_uuid', 'organization_id', 'company_uuid', org_id_col]:
        if col in investments_full.columns:
            inv_org_col = col
            print(f"✓ Colonne org dans investments trouvée: {inv_org_col}")
            break
    
    if inv_org_col:
        for idx, row in investments_full.iterrows():
            org_id = row.get(inv_org_col)
            funding_uuid = row.get('funding_round_uuid')
            # Vérifier que les deux IDs ne sont pas None
            if pd.notna(org_id) and org_id is not None and pd.notna(funding_uuid) and funding_uuid is not None:
                B.add_edge(
                    org_id,
                    funding_uuid,
                    type='company_investment',
                    money_raised_usd=row.get('raised_amount_usd')
                )
        print(f"✓ Arêtes Company ↔ Investment créées")
    else:
        print(f"⚠️ Colonne organisation non trouvée dans investments. Colonnes: {investments_full.columns.tolist()}")

    # --- Arêtes Company ↔ Technology ---
    edges_CT = companies_df[[org_id_col,'category_groups']].copy()
    edges_CT = edges_CT.explode('category_groups')[[org_id_col,'category_groups']]
    for idx, row in edges_CT.iterrows():
        company_id = row.get(org_id_col)
        tech_name = row.get('category_groups')
        # Vérifier que les deux IDs ne sont pas None
        if pd.notna(company_id) and company_id is not None and pd.notna(tech_name) and tech_name is not None:
            B.add_edge(
                company_id,
                tech_name,
                type='company_technology'
            )

    # --- Arêtes Investment ↔ Technology ---
    if inv_org_col:
        edges_IT = edges_CT.merge(
            investments_full[['funding_round_uuid',inv_org_col]],
            on=inv_org_col
        )
        edges_IT = edges_IT[['funding_round_uuid','category_groups']].drop_duplicates()
        for idx, row in edges_IT.iterrows():
            funding_uuid = row.get('funding_round_uuid')
            tech_name = row.get('category_groups')
            # Vérifier que les deux IDs ne sont pas None
            if pd.notna(funding_uuid) and funding_uuid is not None and pd.notna(tech_name) and tech_name is not None:
                B.add_edge(
                    funding_uuid,
                    tech_name,
                    type='investment_technology'
                )
    
    print(f"✓ Graphe créé: {B.number_of_nodes()} nœuds, {B.number_of_edges()} arêtes")
    return B

# ===================================================================
# VISUALISATION
# ===================================================================

def visualize_graph(B, max_companies=20, max_technologies=30):
    import matplotlib.pyplot as plt
    companies = [n for n,d in B.nodes(data=True) if d.get('type')=='company']
    technologies = [n for n,d in B.nodes(data=True) if d.get('type')=='technology']

    top_companies = sorted(companies, key=lambda n:B.degree(n), reverse=True)[:max_companies]
    top_tech = sorted(technologies, key=lambda n:B.degree(n), reverse=True)[:max_technologies]
    nodes_to_keep = set(top_companies) | set(top_tech)
    B_sub = B.subgraph(nodes_to_keep).copy()

    pos = {}
    pos.update((n,(1,i)) for i,n in enumerate(top_companies))
    pos.update((n,(2,i)) for i,n in enumerate(top_tech))

    colors = ['lightblue' if B.nodes[n]['type']=='company' else 'lightgreen' for n in B_sub.nodes()]

    plt.figure(figsize=(12,8))
    nx.draw(B_sub, pos, with_labels=True, node_size=800, node_color=colors, edge_color='gray', alpha=0.7, font_size=8)
    plt.title(f"Top {len(top_companies)} Companies & Top {len(top_tech)} Technologies", fontsize=14)
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

    # Identifier la colonne ID
    org_id_col = None
    for col in ['organization_id', 'uuid', 'id', 'org_id']:
        if col in companies_df.columns:
            org_id_col = col
            break
    
    if org_id_col:
        dict_companies = {row[org_id_col]:row.to_dict() for idx,row in companies_df.iterrows()}
        file_companies = f"{SAVE_DIR_CLASSES}/dict_companies_{suffix}{limit}.pickle"
        with open(file_companies,'wb') as f:
            pickle.dump(dict_companies,f)
        print(f"✓ Dictionnaire entreprises sauvegardé : {file_companies}")
    else:
        print(f"⚠️ Impossible de sauvegarder le dictionnaire: colonne ID non trouvée")

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
        B = extract_tripartite_graph(df_limited, df_invest, df_fund)
        visualize_graph(B, max_companies=10, max_technologies=15)
        save_graph_and_dicts(B, df_limited, limit, FLAG_CYBERSECURITY)

if __name__=="__main__":
    main()