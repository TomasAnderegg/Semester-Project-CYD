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
FILTER_KEYWORDS = ['quantum computing', 'quantum key distribution']  # Keywords pour filtrage optionnel
LIMITS = [10000]  # Nombre d'entrées à traiter

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

    # Sauvegarder le graphe avec pickle directement (évite tout bug NetworkX)
    with open(f"{SAVE_DIR_NETWORKS}/{prefix}bipartite_graph_{limit}.gpickle", "wb") as f:
        pickle.dump(B, f)

    # Sauvegarder le DataFrame
    df_companies.to_csv(f'{SAVE_DIR_CLASSES}/{prefix}companies_ranked_{limit}.csv', index=False)

    print(f"\n✓ Résultats sauvegardés dans {SAVE_DIR_CLASSES}/ et {SAVE_DIR_NETWORKS}/")

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
        for fr in data.get('funding_rounds', []): # cette condtion veut dire: "donne moi la clé 'funding_rounds' dans data, si elle n'existe pas, donne-moi une liste vide []"
            try:
                # Assurez-vous que 'announced_on' est bien une chaîne de format "YYYY-MM-DD"
                announced_on_str = fr['announced_on']
                # Tente de convertir la chaîne de date en timestamp POSIX
                ts_list.append(datetime.strptime(announced_on_str, "%Y-%m-%d").timestamp())
            except ValueError:
                 # Si le format n'est pas bon ou la valeur est manquante
                continue
            except TypeError:
                 # Si fr['announced_on'] n'est pas une string (ex: est pd.NaT ou None)
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
        feat = np.array([data.get('total_raised_amount_usd', 0), data.get('num_funding_rounds', 1)]) # l'argument de la fonctin get veut dire "Donne moi la valeur demandé ou si elle n'existe pas, donne moi 0 (ou 1 pour num_funding_rounds)"
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
    df.to_csv(f"data/{output_prefix}.csv", index=False)
    np.save(f"data/{output_prefix}.npy", feats)
    
    # Création du fichier de features de nœuds (nécessaire par TGN, même vide)
    if not df.empty:
        max_node_id = max(df['u'].max(), df['i'].max())
    else:
        max_node_id = -1
        
    node_feats = np.zeros((max_node_id + 1, 172)) # 172 est la dimension par défaut/conventionnelle
    np.save(f"data/{output_prefix}_node.npy", node_feats)

    # Sauvegarde mappings (PICKLE - format nécessaire pour TGN)
    mapping_dir = Path("data/mappings")
    mapping_dir.mkdir(exist_ok=True)
    with open(mapping_dir / f"{output_prefix}_company_id_map.pickle", "wb") as f:
        pickle.dump(item_map, f)
    with open(mapping_dir / f"{output_prefix}_investor_id_map.pickle", "wb") as f:
        pickle.dump(user_map, f)
        
    print(f"\n✓ Fichiers TGN préparés pour '{output_prefix}'.")
    
    # # =================================================================
    # # VÉRIFICATION DES MAPPINGS EN CSV (AJOUT DEMANDÉ)
    # # =================================================================

    # # 1. Créer le DataFrame de mapping des ENTREPRISES (u = item)
    # df_company_map = pd.DataFrame(
    #     item_map.items(), 
    #     columns=['Company_Name', 'Company_ID_TGN']
    # ).sort_values('Company_ID_TGN')
    
    # csv_company_map_path = mapping_dir / f"{output_prefix}_company_map_verification.csv"
    # df_company_map.to_csv(csv_company_map_path, index=False)
    # print(f"✓ Fichier de vérification Company ID sauvegardé: {csv_company_map_path}")

    # # 2. Créer le DataFrame de mapping des INVESTISSEURS (v = user)
    # df_investor_map = pd.DataFrame(
    #     user_map.items(), 
    #     columns=['Investor_Name', 'Investor_ID_TGN']
    # ).sort_values('Investor_ID_TGN')
    
    # csv_investor_map_path = mapping_dir / f"{output_prefix}_investor_map_verification.csv"
    # df_investor_map.to_csv(csv_investor_map_path, index=False)
    # print(f"✓ Fichier de vérification Investor ID sauvegardé: {csv_investor_map_path}")
    
    # # =================================================================
    # # FIN VÉRIFICATION
    # # =================================================================

    return df, item_map, user_map #, item_inverse, user_inverse

def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
    """Split DataFrame into train/val/test based on timestamp."""
    df_sorted = df.sort_values('announced_on').reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df_sorted.iloc[:train_end].copy()
    df_val = df_sorted.iloc[train_end:val_end].copy()
    df_test = df_sorted.iloc[val_end:].copy()

    print(f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test



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
    
def filter_merged_by_organizations(df_merged, df_organizations):
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
    possible_uuid_cols = ['uuid', 'org_uuid', 'organization_uuid', 'id', 'organization_id']
    uuid_col = None
    
    for col in possible_uuid_cols:
        if col in df_organizations.columns:
            uuid_col = col
            print(f"  ✓ Colonne UUID détectée: '{uuid_col}'")
            break
    
    if uuid_col is None:
        print(f"\n  ERREUR: Aucune colonne UUID trouvée parmi {possible_uuid_cols}")
        print(f"  Colonnes disponibles: {df_organizations.columns.tolist()}")
        raise ValueError(f"Impossible de trouver une colonne UUID dans df_organizations")
    
    # Vérifier que org_uuid existe dans df_merged
    if 'org_uuid' not in df_merged.columns:
        print(f"\n  ERREUR: Colonne 'org_uuid' manquante dans df_merged")
        print(f"  Colonnes disponibles: {df_merged.columns.tolist()}")
        raise ValueError("Colonne 'org_uuid' manquante dans df_merged")
    
    # Obtenir la liste des UUIDs à garder
    valid_uuids = set(df_organizations[uuid_col].dropna().unique())
    print(f"  Nombre d'organisations de référence: {len(valid_uuids)}")
    
    # Afficher quelques exemples d'UUIDs
    sample_uuids = list(valid_uuids)[:5]
    print(f"  Exemples d'UUIDs: {sample_uuids}")
    
    # Filtrer le DataFrame mergé
    initial_rows = len(df_merged)
    df_filtered = df_merged[df_merged['org_uuid'].isin(valid_uuids)].copy()
    final_rows = len(df_filtered)
    
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
    df_filtered.to_csv(csv_filtered_path, index=False)
    print(f"\n✓ CSV filtré sauvegardé: {csv_filtered_path}")
    
    return df_filtered


# ===================================================================
# BIPARTITE GRAPH CREATION
# ===================================================================

def nx_dip_graph_from_pandas(df: pd.DataFrame):
    """
    Creates the bipartite graph from the dataset
    
    bipartite = 0 => company (org_name)
    bipartite = 1 => investor (investor_name)
    
    Correction: Assure que les CVC/Entités mixtes sont correctement labellisées 
    pour le graphe bipartite avant de créer l'arête en priorisant 0 si l'entité est des deux.
    Les arêtes non valides (0-0 ou 1-1) sont ignorées.
    """
    dict_companies = {}
    dict_invest = {}

    B = nx.Graph()

    # helper to safely convert numbers
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

    # =================================================================
    # PRÉ-PASSE 1 : Collecter tous les rôles pour résoudre les conflits (Company vs Investor)
    # =================================================================
    all_companies: Set[str] = set(df['org_name'].dropna().unique())
    all_investors: Set[str] = set(df['investor_name'].dropna().unique())
    
    # Entités qui sont à la fois des Compagnies et des Investisseurs (CVC)
    mixed_entities = all_companies.intersection(all_investors)
    
    # =================================================================
    # PRÉ-PASSE 2 : Définir les nœuds et les labelliser (0 ou 1)
    # Règle de priorité : Si une entité est Comp ET Invest, elle est labellisée 0 (Company)
    # =================================================================

    # 1. Ajouter toutes les entités cibles (Compagnies = 0)
    for name in all_companies:
         B.add_node(name, bipartite=0) # COMPAGNIE = 0
         
         # Création de l'objet classe (pour TechRank)
         try:
             # Assurez-vous que la classe Company existe ou adaptez l'implémentation
             c = classes.Company(id=f'org_{name}', name=name, technologies=[])
         except NameError:
             c = {'name': name}
         dict_companies[name] = c

    # 2. Ajouter les entités sources (Investisseurs = 1) qui ne sont PAS des Compagnies (0)
    for name in all_investors:
         # SI l'entité n'a pas déjà été ajoutée comme Compagnie (bipartite=0),
         # ALORS on l'ajoute comme Investisseur (bipartite=1)
         if name not in B.nodes or B.nodes[name].get('bipartite') != 0:
             B.add_node(name, bipartite=1) # INVESTISSEUR = 1
             
         # Création de l'objet classe (pour TechRank)
         try:
             # Assurez-vous que la classe Investor existe ou adaptez l'implémentation
             i = classes.Investor(investor_id=f'inv_{name}', name=name, announced_on='N/A', raised_amount_usd=0, num_investors=0)
         except NameError:
             i = {'name': name}
         dict_invest[name] = i
    
    # =================================================================
    # DEUXIÈME PASSE : Ajouter les arêtes valides (0-1)
    # =================================================================

    for index, row in df.iterrows():
        invest_name = row.get('investor_name', '') or ''
        comp_name = row.get('org_name', '') or ''
        
        if not invest_name or not comp_name:
            continue
            
        # Récupérer les labels (qui ont été définis dans la pré-passe)
        # On utilise le nom de l'entité comme clé
        comp_bipartite = B.nodes.get(comp_name, {}).get('bipartite')
        invest_bipartite = B.nodes.get(invest_name, {}).get('bipartite')

        # Si l'arête est du type 0 -> 1, elle est valide.
        if (comp_bipartite == 0 and invest_bipartite == 1):
            # Le nom de la Compagnie est u, le nom de l'Investisseur est v
            u, v = comp_name, invest_name 
            
        # Cas des CVC : l'entité (Compagnie 0) investit dans elle-même ou dans une autre Compagnie 0
        # Mais dans le graphe bipartite Compagnie/Investisseur, on considère l'interaction 0-1.
        # Le CVC a été étiqueté 0 (Compagnie), mais dans cette ligne, il agit comme un Investisseur.
        # On doit s'assurer que l'arête est bien entre une Compagnie (0) et un Investisseur (1).
        
        # Cas 1 : Comp (0) <- Inv (1)
        if comp_bipartite == 0 and invest_bipartite == 1:
            u, v = comp_name, invest_name
        
        # Cas 2 : CVC (0, agissant comme Inv) <- Comp (0)
        # (Ex: Palo Alto Networks (Investor) -> Autre Entreprise (Company))
        # Comme Palo Alto Networks est 0, et l'autre Entreprise est 0, c'est 0-0 et ignoré.
        # Si on veut inclure les CVC (Compagnie -> Compagnie/Compagnie), il faut ajuster la règle.
        
        # POUR LA PURETÉ DU GRAPHE BIPARTITE CIBLE(0) <-> SOURCE(1) :
        # On n'ajoute l'arête que si les deux ensembles sont distincts (0 et 1).
        elif comp_bipartite == 0 and invest_name in mixed_entities:
            # Si la compagnie est 0 et l'investisseur est un CVC (qui a été labellisé 0)
            # On ignore l'arête 0-0 (si on veut un graphe strict). 
            # Si on veut la considérer, c'est plus complexe. Restons stricts pour TGN.
            continue
            
        else:
            # Arête invalide (Exemple: 0-0 ou 1-1). Ignorée.
            continue
            
        # Détermination du timestamp (identique à votre code)
        announced_on = row.get('announced_on', '')
        if pd.notna(announced_on):
            if isinstance(announced_on, pd.Timestamp):
                announced_on = str(announced_on.date()) 
            else:
                try:
                    announced_on = str(pd.to_datetime(announced_on).date())
                except Exception:
                    announced_on = ''
        else:
            announced_on = ''
            
        raised_amt = safe_float(row.get('raised_amount_usd', 0))

        # Ajout ou mise à jour de l'arête
        if B.has_edge(u, v):
            existing_data = B[u][v]
            existing_data['funding_rounds'].append({
                'funding_round_uuid': row.get('funding_round_uuid', ''),
                'announced_on': announced_on,
                'raised_amount_usd': raised_amt,
                'investment_type': row.get('investment_type', '')
            })
            existing_data['total_raised_amount_usd'] = safe_float(existing_data.get('total_raised_amount_usd', 0)) + raised_amt
            existing_data['num_funding_rounds'] = existing_data.get('num_funding_rounds', 0) + 1
        else:
            B.add_edge(u, v,
                       funding_rounds=[{
                           'funding_round_uuid': row.get('funding_round_uuid', ''),
                           'announced_on': announced_on,
                           'raised_amount_usd': raised_amt,
                           'investment_type': row.get('investment_type', '')
                       }],
                       total_raised_amount_usd=raised_amt,
                       num_funding_rounds=1
                       )

    # Stats
    print(f"\nStatistiques des levées de fonds:")
    total_funding_rounds = sum([B[u][v].get('num_funding_rounds', 0) for u, v in B.edges()])
    print(f"  - Total arêtes (paires Comp(0)-Inv(1)): {B.number_of_edges()}")
    print(f"  - Total levées de fonds (somme des rounds): {total_funding_rounds}")
    print(f"  - Total nœuds: {B.number_of_nodes()}")

    # Rendu des dictionnaires et du graphe
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
    df_organizations = load_data_from_csv(DATA_PATH_ORGA_CSV)
    print(f"✓ Organisations chargées: {len(df_organizations):,} lignes")
    
    print("✓ Données chargées")
    
    print("\n========================== CLEANING DATA ==========================")
    df_investments_clean = clean_investments_data(df_investments)
    df_funding_clean = clean_funding_data(df_funding)
    
    print("✓ Données nettoyées")
    
    print("\n========================== MERGING DATA ==========================")
    df_graph_full = merge_and_clean_final(df_funding_clean, df_investments_clean)
    print(f"✓ Données mergées : {len(df_graph_full):,} lignes")

    df_graph_full['announced_on'] = pd.to_datetime(df_graph_full['announced_on'], errors='coerce')
    df_graph_full = df_graph_full.dropna(subset=['announced_on'])


    df_graph_full = filter_merged_by_organizations(df_graph_full, df_organizations)

    # --- Découpe temporelle ---
    df_train, df_val, df_test = temporal_split(df_graph_full)
    
    for limit in LIMITS:
        print(f"\n{'='*70}")
        print(f"TRAITEMENT POUR LIMIT = {limit}")
        print(f"{'='*70}")
        
        # Limiter les données
        df_graph = df_graph_full.head(limit).copy()

        # Sauvegarde temporaire pour inspection
        df_graph.to_csv("debug_df_graph.csv", index=False)
        print("✓ Fichier debug_df_graph.csv sauvegardé, tu peux l'ouvrir dans Excel ou VSCode pour vérifier.")
        print("Colonnes :", df_graph.columns.tolist())
        print(df_graph.head(5))


        print("=== Vérification de announced_on ===")
        print(df_graph['announced_on'].head(10))
        print("Nb de dates manquantes :", df_graph['announced_on'].isna().sum())
        print("Nb de lignes totales :", len(df_graph))

        # Créer le graphe bipartite
        B_train, _, _ = nx_dip_graph_from_pandas(df_train)
        prepare_tgn_input(B_train, output_prefix="train")

        B_val, _, _ = nx_dip_graph_from_pandas(df_val)
        prepare_tgn_input(B_val, output_prefix="val")

        B_test, _, _ = nx_dip_graph_from_pandas(df_test)
        prepare_tgn_input(B_test, output_prefix="test")
        
        # --- Graphe complet pour prédiction (forecast) ---
        max_train_time = df_train['announced_on'].max().timestamp()
        # Créer le graphe complet pour forecast
        B_full, dict_companies_full, dict_investors_full = nx_dip_graph_from_pandas(df_graph_full)

        # Préparer les fichiers TGN limités au temps maximal du train
        prepare_tgn_input(
            B_full,
            max_time=max_train_time,
            output_prefix="forecast"
        )

        # # --- Sauvegarde graphique et dictionnaires pour forecast ---
        # save_graph_and_dicts(B_full, dict_companies_full, dict_investors_full, limit="forecast")
        
        # if run_techrank_flag:
        #     print("\n" + "="*70)
        #     print("LANCEMENT DE TECHRANK SUR LE GRAPHE COMPLET")
        #     print("="*70)
            
        #     df_investors_rank, df_companies_rank = run_techrank_on_bipartite(
        #         B=B_full,
        #         dict_companies=dict_companies_full,
        #         dict_investors=dict_investors_full,
        #         limit=limit,
        #         alpha=0.8,  # Tu peux ajuster ces valeurs
        #         beta=-0.6
        #     )
            
        #     # Afficher les résultats si TechRank a réussi
        #     if df_investors_rank is not None and df_companies_rank is not None:
        #         print("\n" + "="*70)
        #         print("TOP 10 INVESTISSEURS (par TechRank)")
        #         print("="*70)
        #         print(df_investors_rank[['TeckRank_int', 'final_configuration', 'techrank']].head(10))
                
        #         print("\n" + "="*70)
        #         print("TOP 10 ENTREPRISES (par TechRank)")
        #         print("="*70)
        #         print(df_companies_rank[['TeckRank_int', 'final_configuration', 'techrank']].head(10))
        
        # # Visualisation sous-graphe
        # companies = [n for n, d in B_full.nodes(data=True) if d['bipartite'] == 0]
        # investors = [n for n, d in B_full.nodes(data=True) if d['bipartite'] == 1]
        # top_companies = sorted(companies, key=lambda n: B_full.degree(n), reverse=True)[:max_companies_plot]
        # top_investors = sorted(investors, key=lambda n: B_full.degree(n), reverse=True)[:max_investors_plot]
        # nodes_to_keep = set(top_companies) | set(top_investors)
        # B_sub = B_full.subgraph(nodes_to_keep).copy()
        # # plot_bipartite_graph(B_sub)



if __name__ == "__main__":
    # Plot seulement 15 entreprises et 15 investisseurs
    main(max_companies_plot=50, max_investors_plot=50,run_techrank_flag=True)