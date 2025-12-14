import pandas as pd
import pickle
import os
from pathlib import Path
import duckdb
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Set


import math
from . import classes
import matplotlib
from datetime import datetime 
from code.TechRank import run_techrank
matplotlib.use('Qt5Agg')  # ou 'TkAgg' selon ton installation
from sklearn.preprocessing import StandardScaler


# ===================================================================
# CONFIGURATION
# ===================================================================

USE_DUCKDB = True  # True = utiliser DuckDB, False = CSV

DATA_PATH_DUCKDB = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
DATA_PATH_INVESTMENTS_CSV = r"data/data_cb/investments.csv"
DATA_PATH_FUNDING_CSV = r"data/data_cb/funding_rounds.csv"
DATA_PATH_ORGA_CSV = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Code\savings\bipartite_tech_comp\classes\cybersecurity_companies_ranked_500.csv"

TABLE_NAME_INVESTMENTS = "investments"
TABLE_NAME_FUNDING = "funding_rounds"
TABLE_NAME_ORGA = "organizations"

SAVE_DIR_NETWORKS = "savings/bipartite_invest/networks"

SAVE_DIR_CLASSES = "savings/bipartite_invest_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_invest_comp/networks"
SAVE_DIR_CSV = "savings/bipartite_invest_comp/csv_exports"  # ✅ NOUVEAU: dossier pour les CSV

FLAG_FILTER = False  # Mettre True si tu veux filtrer
FILTER_KEYWORDS = ['Quantum Computing', 'Quantum Key Distribution']  # Keywords pour filtrage optionnel
LIMITS = [50000]  # Nombre d'entrées à traiter

# ===================================================================
# UTILS
# ===================================================================

def create_directories():
    Path(SAVE_DIR_NETWORKS).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_CSV).mkdir(parents=True, exist_ok=True) 


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
    
    # Détecter le séparateur (virgule ou point-virgule)
    df = pd.read_csv(
        filepath,
        sep=None,              # Détection automatique du séparateur
        engine='python',       # Nécessaire pour sep=None
        on_bad_lines='skip'
    )
    
    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    
    return df

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
        
# def save_graph_and_dicts(B, df_companies, dict_companies, dict_tech, limit, flag_cybersecurity):
#     """Sauvegarde le graphe et les dictionnaires associés."""
#     prefix = "cybersecurity_" if flag_cybersecurity else ""

#     os.makedirs(SAVE_DIR_CLASSES, exist_ok=True)
#     os.makedirs(SAVE_DIR_NETWORKS, exist_ok=True)

#     # Sauvegarder les dictionnaires
#     with open(f'{SAVE_DIR_CLASSES}/{prefix}dict_companies_ranked_{limit}.pickle', 'wb') as f:
#         pickle.dump(dict_companies, f)

#     with open(f'{SAVE_DIR_CLASSES}/{prefix}dict_tech_ranked_{limit}.pickle', 'wb') as f:
#         pickle.dump(dict_tech, f)

#     # Sauvegarder le graphe avec pickle directement (évite tout bug NetworkX)
#     with open(f"{SAVE_DIR_NETWORKS}/{prefix}bipartite_graph_{limit}.gpickle", "wb") as f:
#         pickle.dump(B, f)

#     # Sauvegarder le DataFrame
#     df_companies.to_csv(f'{SAVE_DIR_CLASSES}/{prefix}companies_ranked_{limit}.csv', index=False)

#     print(f"\n✓ Résultats sauvegardés dans {SAVE_DIR_CLASSES}/ et {SAVE_DIR_NETWORKS}/")

def run_techrank_on_bipartite(B, dict_companies, dict_investors, limit, 
                               alpha=0.8, beta=-0.6):
    """
    Exécute TechRank sur le graphe bipartite créé.
    
    Args:
        B: Graphe bipartite NetworkX
        dict_companies: Dictionnaire des entreprises
        dict_investors: Dictionnaire des investisseurs
        limit: Limite utilisée pour nommer les fichiers
        alpha: Paramètre alpha pour TechRank
        beta: Paramètre beta pour TechRank
    
    Returns:
        df_investors_rank: DataFrame avec le ranking des investisseurs
        df_companies_rank: DataFrame avec le ranking des entreprises
    """
    print("\n" + "="*70)
    print("EXÉCUTION DE TECHRANK")
    print("="*70)
    
    try:
        # Appeler run_techrank avec les données créées
        df_investors_rank, df_companies_rank, dict_inv_updated, dict_comp_updated = run_techrank(
            num_comp=limit,
            num_tech=limit,
            flag_cybersecurity=False,
            preferences_comp=None,
            preferences_tech=None,
            alpha=alpha,
            beta=beta,
            do_plot=False,
            # IMPORTANT: Passer les données directement
            dict_investors=dict_investors,
            dict_comp=dict_companies,
            B=B
        )
        
        print("\n✓ TechRank exécuté avec succès!")
        print(f"  - {len(df_investors_rank)} investisseurs classés")
        print(f"  - {len(df_companies_rank)} entreprises classées")
        
        return df_investors_rank, df_companies_rank
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution de TechRank: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def prepare_tgn_input(B, max_time=None, output_prefix="investment_bipartite"):
    """
    Prépare les fichiers TGN en ne gardant que les arêtes <= max_time si fourni.
    
    MODIFICATION AJOUTÉE: Export des mappings réels vers CSV pour vérification.
    """
    rows, feats = [], []
    user_map, item_map = {}, {}
    user_inverse, item_inverse = {}, {}

    for u, v, data in B.edges(data=True):
        # Pour NetworkX, u est le noeud du set 0 (Company), v est le noeud du set 1 (Investor)
        # Vérification du type (pour être sûr)
        u_bipartite = B.nodes[u].get('bipartite')
        v_bipartite = B.nodes[v].get('bipartite')

        # Si le graphe n'est pas strictement bipartite ou mal labellisé, on peut corriger ici
        if u_bipartite == 1 and v_bipartite == 0:
            # Si les rôles sont inversés dans l'itération, on les échange
            u, v = v, u 
            u_bipartite, v_bipartite = v_bipartite, u_bipartite # Nécessaire si on voulait réutiliser les variables

        # Si 'u' n'a pas bipartite=0 (Company) ou 'v' n'a pas bipartite=1 (Investor), il y a un problème
        if u_bipartite != 0 or v_bipartite != 1:
            print(f"Avertissement: Nœud {u} ({u_bipartite}) ou {v} ({v_bipartite}) n'a pas le label bipartite attendu (0/1). Ignoré.")
            continue


        # Extraire timestamp min/max des levées
        ts_list = []
        for fr in data.get('funding_rounds', []):
            announced_on = fr.get("announced_on")

            # Cas null / vide
            if announced_on is None or announced_on == "" or pd.isna(announced_on):
                continue

            # Cas : Pandas Timestamp
            if hasattr(announced_on, "timestamp"):
                ts_list.append(announced_on.timestamp())
                continue

            # Cas : string → tester plusieurs formats
            possible_formats = ["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"]
            parsed = False
            for fmt in possible_formats:
                try:
                    dt = datetime.strptime(announced_on, fmt)
                    ts_list.append(dt.timestamp())
                    parsed = True
                    break
                except Exception:
                    pass

            if not parsed:
                # si aucun format ne marche : on ignore
                continue
                
        if not ts_list:
            continue
        ts = min(ts_list) # Le timestamp de la première levée de fonds entre la paire

        if max_time is not None and ts > max_time:
            continue  # on ignore les arêtes futures

        # 1. Mapping du Nœud U (Company)
        # On garantit que u est dans item_map (Compagnie)
        if u not in item_map:
            item_map[u] = len(item_map)
            # item_inverse[item_map[u]] = u
            #    
        # 2. Mapping du Nœud V (Investor)
        # On garantit que v est dans user_map (Investisseur)
        if v not in user_map:
            user_map[v] = len(user_map)
            # user_inverse[user_map[v]] = v

        u_id, v_id = item_map[u], user_map[v]
        label = 1.0 # La transaction a eu lieu
        # Features de l'arête: Total levé, Nombre de rounds
        feat = np.array([np.log1p(data.get('total_raised_amount_usd', 0)), data.get('num_funding_rounds', 1)]) # l'argument de la fonctin get veut dire "Donne moi la valeur demandé ou si elle n'existe pas, donne moi 0 (ou 1 pour num_funding_rounds)"
        
        # PRINT EDGE FEATURES (une seule fois)
        if len(feats) == 0:
            print("\nEDGE FEATURES INFORMATION")
            print(f"  - Number of edge features: {len(feat)}")
            print(f"  - Edge feature names: ['log_total_raised_amount_usd', 'num_funding_rounds']")

        rows.append((u_id, v_id, ts, label)) # u → company, v → investor
        feats.append(feat)
    
    # print(f"liste du mapping company: {len(item_map)}")
    # print("------------------------------------------")
    # print(f"liste du mapping investor: {len(user_map)}")
    # Création des DataFrames TGN
    df = pd.DataFrame(rows, columns=['u', 'i', 'ts', 'label'])
    feats = np.array(feats)

    # Tri par timestamp et ajout idx
    df = df.sort_values('ts').reset_index(drop=True)
    df['idx'] = np.arange(len(df))

    # Sauvegarde des fichiers TGN
    Path("data").mkdir(exist_ok=True)
    df.to_csv(f"data/data_split/{output_prefix}.csv", index=False)
    np.save(f"data/data_split/{output_prefix}.npy", feats)
    
    # Création du fichier de features de nœuds (nécessaire par TGN, même vide)
    if not df.empty:
        max_node_id = max(df['u'].max(), df['i'].max())
    else:
        max_node_id = -1
        
    node_feats = np.zeros((max_node_id + 1, 200)) # 172 est la dimension par défaut/conventionnelle
    np.save(f"data/data_split/{output_prefix}_node.npy", node_feats)

    # Sauvegarde mappings (PICKLE - format nécessaire pour TGN)
    mapping_dir = Path("data/data_split")
    mapping_dir.mkdir(exist_ok=True)
    with open(mapping_dir / f"{output_prefix}_company_id_map.pickle", "wb") as f:
        pickle.dump(item_map, f)
    with open(mapping_dir / f"{output_prefix}_investor_id_map.pickle", "wb") as f:
        pickle.dump(user_map, f)
        
    print(f"\n✓ Fichiers TGN préparés pour '{output_prefix}'.")
    
    # =================================================================
    # VÉRIFICATION DES MAPPINGS EN CSV (AJOUT DEMANDÉ)
    # =================================================================

    # 1. Créer le DataFrame de mapping des ENTREPRISES (u = item)
    df_company_map = pd.DataFrame(
        item_map.items(), 
        columns=['Company_Name', 'Company_ID_TGN']
    ).sort_values('Company_ID_TGN')
    
    csv_company_map_path = mapping_dir / f"{output_prefix}_company_map_verification.csv"
    df_company_map.to_csv(csv_company_map_path, index=False)
    print(f"✓ Fichier de vérification Company ID sauvegardé: {csv_company_map_path}")

    # 2. Créer le DataFrame de mapping des INVESTISSEURS (v = user)
    df_investor_map = pd.DataFrame(
        user_map.items(), 
        columns=['Investor_Name', 'Investor_ID_TGN']
    ).sort_values('Investor_ID_TGN')
    
    csv_investor_map_path = mapping_dir / f"{output_prefix}_investor_map_verification.csv"
    df_investor_map.to_csv(csv_investor_map_path, index=False)
    print(f"✓ Fichier de vérification Investor ID sauvegardé: {csv_investor_map_path}")
    
    # =================================================================
    # FIN VÉRIFICATION
    # =================================================================

    return df, item_map, user_map #, item_inverse, user_inverse


# ===================================================================
# TEMPORAL SPLITTING FUNCTIONS
# ===================================================================

def temporal_split_graph(B, train_ratio=0.85, val_ratio=0.0):
    """
    Split temporel du graphe bipartite basé sur les timestamps des funding_rounds
    
    Returns:
        B_train, B_val, B_test: Graphes splittés
        max_train_time, max_val_time: Timestamps limites
    """
    print("\n" + "="*70)
    print("TEMPORAL SPLIT DU GRAPHE BIPARTITE")
    print("="*70)
    
    # Collecter tous les timestamps
    all_timestamps = []
    edge_timestamps = {}  # (u, v) -> min_timestamp
    
    for u, v, data in B.edges(data=True):
        ts_list = []
        for fr in data.get('funding_rounds', []):
            announced_on = fr.get("announced_on")
            if announced_on is None or announced_on == "" or pd.isna(announced_on):
                continue
                
            if hasattr(announced_on, "timestamp"):
                ts_list.append(announced_on.timestamp())
            else:
                possible_formats = ["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"]
                for fmt in possible_formats:
                    try:
                        dt = datetime.strptime(str(announced_on), fmt)
                        ts_list.append(dt.timestamp())
                        break
                    except:
                        pass
        
        if ts_list:
            min_ts = min(ts_list)
            edge_timestamps[(u, v)] = min_ts
            all_timestamps.append(min_ts)
    
    if not all_timestamps:
        print("❌ Aucun timestamp trouvé dans le graphe!")
        return B, B, B, None, None
    
    # Trier et calculer les seuils
    all_timestamps = sorted(all_timestamps)
    total_edges = len(all_timestamps)
    
    train_idx = int(total_edges * train_ratio)
    val_idx = int(total_edges * (train_ratio + val_ratio))
    
    max_train_time = all_timestamps[train_idx - 1] if train_idx > 0 else all_timestamps[0]
    max_val_time = all_timestamps[val_idx - 1] if val_idx < total_edges else all_timestamps[-1]
    
    print(f"\nStatistiques temporelles:")
    print(f"  Total edges: {total_edges:,}")
    print(f"  Train edges: {train_idx:,} ({train_ratio*100:.1f}%)")
    print(f"  Val edges: {val_idx - train_idx:,} ({val_ratio*100:.1f}%)")
    print(f"  Test edges: {total_edges - val_idx:,} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    print(f"  Max train timestamp: {datetime.fromtimestamp(max_train_time).strftime('%Y-%m-%d')}")
    print(f"  Max val timestamp: {datetime.fromtimestamp(max_val_time).strftime('%Y-%m-%d')}")
    
    # Créer les sous-graphes
    B_train = nx.Graph()
    B_val = nx.Graph()
    B_test = nx.Graph()
    
    # Ajouter tous les nœuds avec leurs attributs
    for node, data in B.nodes(data=True):
        B_train.add_node(node, **data)
        B_val.add_node(node, **data)
        B_test.add_node(node, **data)
    
    # Distribuer les arêtes selon leur timestamp
    for (u, v), ts in edge_timestamps.items():
        edge_data = B[u][v].copy()
        
        if ts <= max_train_time:
            B_train.add_edge(u, v, **edge_data)
        elif ts <= max_val_time:
            B_val.add_edge(u, v, **edge_data)
        else:
            B_test.add_edge(u, v, **edge_data)
    
    print(f"\nGraphes créés:")
    print(f"  Train: {B_train.number_of_nodes()} nœuds, {B_train.number_of_edges()} arêtes")
    print(f"  Val:   {B_val.number_of_nodes()} nœuds, {B_val.number_of_edges()} arêtes")
    print(f"  Test:  {B_test.number_of_nodes()} nœuds, {B_test.number_of_edges()} arêtes")
    
    return B_train, B_val, B_test, max_train_time, max_val_time


def split_dictionaries_by_graph(B_train, B_val, B_test, dict_companies, dict_investors):
    """
    Split les dictionnaires selon les nœuds présents dans chaque graphe
    """
    print("\n" + "="*70)
    print("SPLIT DES DICTIONNAIRES")
    print("="*70)
    
    # Extraire les nœuds de chaque type pour chaque split
    companies_train = {n for n, d in B_train.nodes(data=True) if d.get('bipartite') == 0}
    companies_val = {n for n, d in B_val.nodes(data=True) if d.get('bipartite') == 0}
    companies_test = {n for n, d in B_test.nodes(data=True) if d.get('bipartite') == 0}
    
    investors_train = {n for n, d in B_train.nodes(data=True) if d.get('bipartite') == 1}
    investors_val = {n for n, d in B_val.nodes(data=True) if d.get('bipartite') == 1}
    investors_test = {n for n, d in B_test.nodes(data=True) if d.get('bipartite') == 1}
    
    # Split dictionnaires companies
    dict_comp_train = {name: data for name, data in dict_companies.items() if name in companies_train}
    dict_comp_val = {name: data for name, data in dict_companies.items() if name in companies_val}
    dict_comp_test = {name: data for name, data in dict_companies.items() if name in companies_test}
    
    # Split dictionnaires investors
    dict_inv_train = {name: data for name, data in dict_investors.items() if name in investors_train}
    dict_inv_val = {name: data for name, data in dict_investors.items() if name in investors_val}
    dict_inv_test = {name: data for name, data in dict_investors.items() if name in investors_test}
    
    print(f"\nCompanies:")
    print(f"  Train: {len(dict_comp_train):,}")
    print(f"  Val:   {len(dict_comp_val):,}")
    print(f"  Test:  {len(dict_comp_test):,}")
    
    print(f"\nInvestors:")
    print(f"  Train: {len(dict_inv_train):,}")
    print(f"  Val:   {len(dict_inv_val):,}")
    print(f"  Test:  {len(dict_inv_test):,}")
    
    return {
        'companies': {
            'train': dict_comp_train,
            'val': dict_comp_val,
            'test': dict_comp_test
        },
        'investors': {
            'train': dict_inv_train,
            'val': dict_inv_val,
            'test': dict_inv_test
        }
    }


def save_split_data(B_train, B_val, B_test, dict_split, limit, output_dir=None):
    """
    Sauvegarde tous les splits (graphes + dictionnaires)
    """
    if output_dir is None:
        output_dir = SAVE_DIR_NETWORKS
    
    split_dir = Path(output_dir) / f"split_{limit}"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"SAUVEGARDE DES SPLITS DANS {split_dir}")
    print("="*70)
    
    # Sauvegarder les graphes
    for split_name, graph in [('train', B_train), ('val', B_val), ('test', B_test)]:
        graph_path = split_dir / f"bipartite_graph_{split_name}.gpickle"
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"✓ Graphe {split_name} sauvegardé: {graph_path}")
    
    # Sauvegarder les dictionnaires
    dict_dir = split_dir / "dictionaries"
    dict_dir.mkdir(exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        comp_path = dict_dir / f"dict_companies_{split_name}.pickle"
        inv_path = dict_dir / f"dict_investors_{split_name}.pickle"
        
        with open(comp_path, 'wb') as f:
            pickle.dump(dict_split['companies'][split_name], f)
        with open(inv_path, 'wb') as f:
            pickle.dump(dict_split['investors'][split_name], f)
        
        print(f"✓ Dictionnaires {split_name} sauvegardés")
    
    # Sauvegarder les métadonnées
    metadata = {
        'limit': limit,
        'train_nodes': B_train.number_of_nodes(),
        'train_edges': B_train.number_of_edges(),
        'val_nodes': B_val.number_of_nodes(),
        'val_edges': B_val.number_of_edges(),
        'test_nodes': B_test.number_of_nodes(),
        'test_edges': B_test.number_of_edges(),
        'companies_train': len(dict_split['companies']['train']),
        'companies_val': len(dict_split['companies']['val']),
        'companies_test': len(dict_split['companies']['test']),
        'investors_train': len(dict_split['investors']['train']),
        'investors_val': len(dict_split['investors']['val']),
        'investors_test': len(dict_split['investors']['test'])
    }
    
    with open(split_dir / 'split_metadata.pickle', 'wb') as f:
        pickle.dump(metadata, f)
    
    pd.DataFrame([metadata]).to_csv(split_dir / 'split_metadata.csv', index=False)
    print(f"✓ Métadonnées sauvegardées")
    
    print(f"\n✓ Tous les splits sauvegardés dans {split_dir}")


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

def clean_organization_data(df):
    """Clean organization dataset"""
    to_drop = [
        'type',
        'permalink',
        'cb_url',   
        'rank',
        'created_at',
        'updated_at',
        'legal_name',
        'roles',
        'domain',
        'homepage_url',
        'country_code',
        'state_code',
        'region',
        'city',
        'address',
        'postal_code',
        'status',
        'category_groups_list',
        'total_funding',
        'total_funding_currency_code',
        'founded_on',
        'last_funding_on',
        'closed_on',
        'employee_count',
        'email',
        'phone',
        'facebook_url',
        'twitter_url',
        'linkedin_url',
        'logo_url',
        'alias1',
        'alias2',
        'alias3',
        'primary_role',
        'num_exits'
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
    # MERGE COMME TU LE VEUX
    df_merged = pd.merge(df_funding, df_investments, on='funding_round_uuid')
    
    #  SAUVEGARDER LE RÉSULTAT DU MERGE COMPLET
    csv_path = f"{SAVE_DIR_CSV}/merged_funding_investments_full.csv"
    df_merged.to_csv(csv_path, index=False)
    print(f"✓ CSV du merge complet sauvegardé : {csv_path}")
    print(f"  Colonnes: {list(df_merged.columns)}")
    print(f"  Shape: {df_merged.shape}")
    
    to_drop = [ ]
    to_rename = {}
    drop_if_nan = []
    to_check_double = {}
    sort_by = ""
    
    df_clean = CB_data_cleaning(df_merged, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)
    
    #  Garder TOUTES les colonnes nécessaires pour le graphe
    required_cols = ["org_name", "investor_name", "org_uuid", "investor_uuid", 
                     "raised_amount_usd", "num_investments", "announced_on"]
    
    # Vérifier quelles colonnes existent
    existing_cols = [col for col in required_cols if col in df_clean.columns]
    
    if 'org_name' in df_clean.columns and 'investor_name' in df_clean.columns:
        df_graph = df_clean[existing_cols].copy()
        df_graph = df_graph.dropna(subset=['org_name', 'investor_name'])
        
        #  SAUVEGARDER AUSSI LE RÉSULTAT FILTRÉ
        csv_path_graph = f"{SAVE_DIR_CSV}/merged_for_graph.csv"
        df_graph.to_csv(csv_path_graph, index=False)
        print(f"✓ CSV pour le graphe sauvegardé : {csv_path_graph}")
        print(f"  Colonnes gardées: {list(df_graph.columns)}")
        
        print("----------en tete de df_graph--------:")
        print(df_graph.head())
        return df_graph
    else:
        raise ValueError("Colonnes 'org_name' ou 'investor_name' manquantes")
    
def filter_merged_by_organizations(df_merged, df_organizations, keywords=FILTER_KEYWORDS):
    """
    Filtre le DataFrame mergé pour ne garder que les entreprises présentes dans df_organizations.
    
    Args:
        df_merged: DataFrame issu du merge entre funding et investments
        df_organizations: DataFrame contenant les organisations filtrées
    
    Returns:
        df_filtered: DataFrame filtré contenant uniquement les org_uuid présents dans df_organizations
    """
    print("\n========================== FILTERING BY ORGANIZATIONS ==========================")
    
    # Afficher les colonnes disponibles pour diagnostic
    print(f"  Colonnes dans df_organizations: {df_organizations.columns.tolist()}")
    print(f"  Colonnes dans df_merged: {df_merged.columns.tolist()}")
    
    # Détecter la colonne UUID dans df_organizations
    # possible_uuid_cols = ['uuid', 'org_uuid', 'organization_uuid', 'id', 'organization_id']
    # uuid_col = None
    
    # for col in possible_uuid_cols:
    #     if col in df_organizations.columns:
    #         uuid_col = col
    #         print(f"  ✓ Colonne UUID détectée: '{uuid_col}'")
    #         break
    
    # if df_organizations["uuid"] is None:
    #     print(f"\n  ERREUR: Aucune colonne UUID trouvée parmi {possible_uuid_cols}")
    #     print(f"  Colonnes disponibles: {df_organizations.columns.tolist()}")
    #     raise ValueError(f"Impossible de trouver une colonne UUID dans df_organizations")
    
    # Vérifier que org_uuid existe dans df_merged
    # if 'org_uuid' not in df_merged.columns:
    #     print(f"\n  ERREUR: Colonne 'org_uuid' manquante dans df_merged")
    #     print(f"  Colonnes disponibles: {df_merged.columns.tolist()}")
    #     raise ValueError("Colonne 'org_uuid' manquante dans df_merged")
    
    # Obtenir la liste des UUIDs à garder
    valid_uuids = set(df_organizations["uuid"].dropna().unique()) # dropna pour éviter les NaN, .unique evite les doublons
    print(f"  Nombre d'organisations de référence: {len(valid_uuids)}")
    
    # Afficher quelques exemples d'UUIDs
    sample_uuids = list(valid_uuids)[:5]
    print(f"  Exemples d'UUIDs: {sample_uuids}")
    
    # Filtrer le DataFrame mergé
    initial_rows = len(df_merged)
    df_filtered = df_merged[df_merged['org_uuid'].isin(valid_uuids)].copy() # pour s'assurer que les organisations sont dans la liste "organizations"
    final_rows = len(df_filtered)

    # Crée un dictionnaire de mapping uuid -> category_list
    uuid_to_cat = df_organizations.set_index('uuid')['category_list'].to_dict()

    # Ajoute une colonne 'category_list' dans df_merged
    df_merged['category_list'] = df_merged['org_uuid'].map(uuid_to_cat)
    # df_merged.to_csv("debug_caca21.csv",index=False)

    # Filtrer chaque liste pour ne garder que les éléments contenant le mot-clé
    def match_category(cat_string):
        # convertir la string en liste
        cats = [c.strip() for c in cat_string.split(",")] if isinstance(cat_string, str) else []
        
        # vérifier si un mot clé apparaît dans un élément de la liste
        return any(
            any(keyword in category for keyword in keywords)
            for category in cats
        )

    # Filtrer le DataFrame
    df_merged_keyworded = df_merged[df_merged["category_list"].apply(match_category)]

    # df_merged_keyworded.to_csv("debug_caca212.csv",index=False)

    
    print(f"  Lignes avant filtrage: {initial_rows:,}")
    print(f"  Lignes après filtrage: {final_rows:,}")
    
    if final_rows == 0:
        print("\n  ⚠️ ATTENTION: Aucune correspondance trouvée!")
        print("  Vérifiez que les UUIDs sont au même format dans les deux fichiers")
        # Afficher des exemples pour comparaison
        sample_merged_uuids = df_merged['org_uuid'].dropna().head(5).tolist()
        print(f"  Exemples d'UUIDs dans df_merged: {sample_merged_uuids}")
    else:
        print(f"  Lignes supprimées: {initial_rows - final_rows:,} ({100*(initial_rows-final_rows)/initial_rows:.1f}%)")
    
    # Statistiques sur les entreprises conservées
    unique_orgs_before = df_merged['org_uuid'].nunique()
    unique_orgs_after = df_filtered['org_uuid'].nunique()
    print(f"  Entreprises uniques avant: {unique_orgs_before:,}")
    print(f"  Entreprises uniques après: {unique_orgs_after:,}")
    
    # Sauvegarder le résultat filtré
    csv_filtered_path = f"{SAVE_DIR_CSV}/merged_filtered_by_organizations.csv"
    df_merged_keyworded.to_csv(csv_filtered_path, index=False)
    print(f"\n✓ CSV filtré sauvegardé: {csv_filtered_path}")
    
    return df_merged_keyworded




def nx_dip_graph_from_pandas(df: pd.DataFrame):
    """
    Creates the bipartite graph from the dataset
    
    bipartite = 0 => company (org_name)
    bipartite = 1 => investor (investor_name)
    """

    import networkx as nx
    from typing import Set

    dict_companies = {}
    dict_invest = {}

    # ======================================================
    # 1) Préparer les technologies par entreprise
    # ======================================================

    def parse_tech_list(cat):
        '''
        transforme: "Electronics,Quantum Computing,Semiconductor"
        en: ["Electronics", "Quantum Computing", "Semiconductor"]
        '''
        if pd.isna(cat):
            return []
        if isinstance(cat, str):
            return [c.strip() for c in cat.split(',') if c.strip()] # c.strip() pour enlever les espaces
        return []

    company_to_techs = {}

    # On parcourt uniquement les colonnes org_name et category_list
    sub_df = df[['org_name', 'category_list']]

    # On supprime les lignes sans nom d’entreprise
    sub_df = sub_df.dropna(subset=['org_name'])

    for index, row in sub_df.iterrows():
        
        # Nom de l'entreprise
        company_name = row['org_name']

        # Liste des technologies à partir de category_list
        tech_list = parse_tech_list(row['category_list'])

        # On ajoute l'entrée dans le dictionnaire
        company_to_techs[company_name] = tech_list


    # ======================================================
    # 2) Création du graphe bipartite
    # ======================================================

    B = nx.Graph()

    def safe_int(x, default=0) -> int:
        try:
            if pd.isna(x): return default
            return int(x)
        except Exception:
            return default

    def safe_float(x, default=0.0) -> float:
        try:
            if pd.isna(x): return default
            return float(x)
        except Exception:
            return default

    all_companies: Set[str] = set(df['org_name'].dropna().unique())
    all_investors: Set[str] = set(df['investor_name'].dropna().unique())

    # ======================================================
    # 3) Ajouter toutes les Company nodes (bipartite=0)
    # ======================================================

    for name in all_companies:
        B.add_node(name, bipartite=0)

        techs = company_to_techs.get(name, [])

        dict_companies[name] = {
            "id": f"org_{name}",
            "name": name,
            "technologies": techs,
            "number_of_tech": len(techs)
        }

    # ======================================================
    # 4) Ajouter Investor nodes (bipartite=1)
    # ======================================================

    for name in all_investors:
        if name not in B.nodes:  # seulement si pas déjà Company
            B.add_node(name, bipartite=1)

        dict_invest[name] = {
            "id": f"inv_{name}",
            "name": name,
            "raised_amount_usd": 0,
            "num_investors": 0
        }

    # ======================================================
    # 5) Construire les arêtes valides (0 -> 1)
    # ======================================================

    for idx, row in df.iterrows():

        invest_name = row.get('investor_name', '') or ''
        comp_name = row.get('org_name', '') or ''

        if not invest_name or not comp_name:
            continue

        comp_b = B.nodes.get(comp_name, {}).get('bipartite')
        inv_b = B.nodes.get(invest_name, {}).get('bipartite')

        if comp_b == 0 and inv_b == 1:
            u, v = comp_name, invest_name
        else:
            continue  # ignore 0-0 ou 1-1

        # Timestamp
        announced_on = row.get('announced_on', '')
        if pd.notna(announced_on):
            try:
                announced_on = str(pd.to_datetime(announced_on).date())
            except Exception:
                announced_on = ''

        raised_amt = safe_float(row.get('raised_amount_usd', 0))

        # Ajouter ou mettre à jour l'arête
        if B.has_edge(u, v):
            data = B[u][v]
            data['funding_rounds'].append({
                'funding_round_uuid': row.get('funding_round_uuid', ''),
                'announced_on': announced_on,
                'raised_amount_usd': raised_amt,
                'investment_type': row.get('investment_type', '')
            })
            data['total_raised_amount_usd'] = safe_float(data.get('total_raised_amount_usd', 0)) + raised_amt
            data['num_funding_rounds'] = data.get('num_funding_rounds', 0) + 1

        else:
            B.add_edge(
                u, v,
                funding_rounds=[{
                    'funding_round_uuid': row.get('funding_round_uuid', ''),
                    'announced_on': announced_on,
                    'raised_amount_usd': raised_amt,
                    'investment_type': row.get('investment_type', '')
                }],
                total_raised_amount_usd=raised_amt,
                num_funding_rounds=1
            )

    # ======================================================
    # 6) Statistiques
    # ======================================================

    print("\nStatistiques des levées de fonds:")
    total_rounds = sum(B[u][v].get('num_funding_rounds', 0) for u, v in B.edges())
    print(f"  - Total arêtes (0-1): {B.number_of_edges()}")
    print(f"  - Total levées de fonds: {total_rounds}")
    print(f"  - Total nœuds: {B.number_of_nodes()}")
    
    # for node, data in B.nodes(data=True):
    #         if data.get('bipartite') == 0:
    #             deg = B.degree(node)
    #             inv_deg = 1 / deg if deg > 0 else 0.0
    #             B.nodes[node]['inv_degree'] = inv_deg
    #             dict_companies[node]['inv_degree'] = inv_deg

    # # Normalisation
    # max_inv = max([B.nodes[n]['inv_degree'] for n in B.nodes if B.nodes[n].get('bipartite')==0])
    # for node, data in B.nodes(data=True):
    #     if data.get('bipartite') == 0:
    #         data['inv_degree_norm'] = data['inv_degree'] / max_inv
    #         dict_companies[node]['inv_degree_norm'] = data['inv_degree_norm']

    # ======================================================
    # 6) Statistiques et Ajout des Node Features de Degré
    # ======================================================

    print("\nStatistiques des levées de fonds:")
    total_rounds = sum(B[u][v].get('num_funding_rounds', 0) for u, v in B.edges())
    print(f"  - Total arêtes (0-1): {B.number_of_edges()}")
    print(f"  - Total levées de fonds: {total_rounds}")
    print(f"  - Total nœuds: {B.number_of_nodes()}")
    
    # -------------------------------------------------------------------
    # Ajout du DEGRÉ comme Node Feature pour les Compagnies (bipartite=0)
    # -------------------------------------------------------------------
    for node, data in B.nodes(data=True):
        if data.get('bipartite') == 0: # C'est une entreprise
            
            # 1. Calculer le degré
            deg = B.degree(node)
            
            # 2. Stocker le degré comme attribut (feature) dans le nœud NetworkX
            B.nodes[node]['degree'] = deg
            
            # 3. Stocker le degré dans le dictionnaire de sortie
            dict_companies[node]['degree'] = deg

    # Vous pouvez aussi ajouter d'autres features ici, comme le degré normalisé
    # ou la feature 'inv_degree' que vous aviez commentée.

                
    return B, dict_companies, dict_invest




# def create_bipartite_from_dataframe(df, df_comp_clean):
#     """Create bipartite graph from org_name and investor_name columns"""
#     B = nx.Graph()
    
#     for _, row in df.iterrows():
#         org = row['org_name']
#         investor = row['investor_name']
        
#         if pd.notna(org) and pd.notna(investor):
#             B.add_node(org, bipartite=0)
#             B.add_node(investor, bipartite=1)
#             B.add_edge(org, investor)
    
#     return B




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
    
    # NOUVEAU: Sauvegarder un CSV avec les informations des arêtes
    edge_data = []
    for u, v in B.edges():
        edge_info = B[u][v]
        edge_data.append({
            'company': u,
            'investor': v,
            'num_funding_rounds': edge_info.get('num_funding_rounds', 1),
            'total_raised_amount_usd': edge_info.get('total_raised_amount_usd', 0)
        })
    
    df_edges = pd.DataFrame(edge_data)
    csv_edges_path = f"{SAVE_DIR_CSV}/edge_funding_info_{limit}.csv"
    df_edges.to_csv(csv_edges_path, index=False)
    print(f"✓ Informations des arêtes sauvegardées : {csv_edges_path}")

    print(f"\n✓ Résultats sauvegardés dans {SAVE_DIR_CLASSES}/ et {SAVE_DIR_NETWORKS}/")


# ===================================================================
# MAIN
# ===================================================================

def main(max_companies_plot=20, max_investors_plot=20, run_techrank_flag=True):
    create_directories()
    
    print("\n========================== LOADING DATA ==========================")
    df_investments = load_data(use_duckdb=USE_DUCKDB, table_name=TABLE_NAME_INVESTMENTS)
    df_funding = load_data(use_duckdb=USE_DUCKDB, table_name=TABLE_NAME_FUNDING)

     # ✅ NOUVEAU: Charger le fichier organizations
    # df_organizations = load_data_from_csv(DATA_PATH_ORGA_CSV)
    df_organizations = load_data(use_duckdb=USE_DUCKDB, table_name=TABLE_NAME_ORGA)

    # Source - https://stackoverflow.com/a
    # Posted by Andy Hayden, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-03, License - CC BY-SA 4.0

    # df_organizations.to_csv("debug_caca.csv",index=False)

    print(f"type de df_organizations: {type(df_organizations)}")

    print(f"Organisations chargées: {len(df_organizations):,} lignes")
    
    print("Données chargées")
    
    print("\n========================== CLEANING DATA ==========================")
    df_investments_clean = clean_investments_data(df_investments)
    df_funding_clean = clean_funding_data(df_funding)
    df_organizations_clean = clean_organization_data(df_organizations)
    # df_organizations_clean.to_csv("debug_caca.csv",index=False)
    print("Données nettoyées")
    
    print("\n========================== MERGING DATA ==========================")
    df_graph_full = merge_and_clean_final(df_funding_clean, df_investments_clean)
    df_graph_full.to_csv("debug_caca.csv",index=False)
    print(f"Données mergées : {len(df_graph_full):,} lignes")

    df_graph_full['announced_on'] = pd.to_datetime(df_graph_full['announced_on'], errors='coerce')
    df_graph_full = df_graph_full.dropna(subset=['announced_on'])
    
    # TIMESPAN print 
    min_date = df_graph_full['announced_on'].min()
    max_date = df_graph_full['announced_on'].max()
    time_span_days = (max_date - min_date).days

    print("\nTEMPORAL INFORMATION (announced_on)")
    print(f"  - First investment date: {min_date.date()}")
    print(f"  - Last investment date:  {max_date.date()}")
    print(f"  - Time span:             {time_span_days} days")

    df_graph_full = filter_merged_by_organizations(df_graph_full, df_organizations, keywords = FILTER_KEYWORDS)
    # df_graph_full.to_csv("debug_filtered_graph_full.csv", index=False)
    
    for limit in LIMITS:
        print(f"\n{'='*70}")
        print(f"TRAITEMENT POUR LIMIT = {limit}")
        print(f"{'='*70}")
        
        # Limiter les données
        df_graph = df_graph_full.head(limit).copy()

        # Sauvegarde temporaire pour inspection
        # df_graph.to_csv("debug_df_graphcaca21.csv", index=False)
        print("✓ Fichier debug_df_graph.csv sauvegardé, tu peux l'ouvrir dans Excel ou VSCode pour vérifier.")
        print("Colonnes :", df_graph.columns.tolist())
        print("nobres de lignes dans df_graph :", len(df_graph))
        print(df_graph.head(5))


        print("=== Vérification de announced_on ===")
        print(df_graph['announced_on'].head(10))
        print("Nb de dates manquantes :", df_graph['announced_on'].isna().sum())
        print("Nb de lignes totales :", len(df_graph))

        
        B_full, dict_companies_full, dict_investors_full = nx_dip_graph_from_pandas(df_graph)
        print(f"nombre de noeuds", B_full.number_of_nodes())

        B_train, B_val, B_test, max_train_time, max_val_time = temporal_split_graph(B_full)
        dict_split = split_dictionaries_by_graph(B_train, B_val, B_test, dict_companies_full, dict_investors_full)
        save_split_data(B_train, B_val, B_test, dict_split, limit, output_dir=None)

        # Préparer les fichiers TGN limités au temps maximal du train
        # Préparer les données TGN
        # df, item_map, user_map = prepare_tgn_input(B_full, output_prefix="crunchbase_filtered")
        df, item_map, user_map = prepare_tgn_input(B_train, output_prefix="crunchbase_filtered_train")
        df, item_map, user_map = prepare_tgn_input(B_val, output_prefix="crunchbase_filtered_val")
        df, item_map, user_map = prepare_tgn_input(B_test, output_prefix="crunchbase_filtered_test")


        # # NOUVEAU: Sauvegarder les degrés pour la weighted loss
        # from data.custom_loss import prepare_degree_weights
        # company_degrees = prepare_degree_weights(B_full, item_map, output_prefix="crunchbase_filtered")

        save_graph_and_dicts(B_full, dict_companies_full, dict_investors_full, limit)


if __name__ == "__main__":
    # Plot seulement 15 entreprises et 15 investisseurs
    main(max_companies_plot=50, max_investors_plot=50,run_techrank_flag=True)