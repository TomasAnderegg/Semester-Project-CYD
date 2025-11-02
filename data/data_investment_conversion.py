"""
Script de traitement des données CrunchBase pour TechRank - Investments & Funding
Auteur: Tomas
Date: 2025

Ce script charge, nettoie et fusionne les données d'investissements et de funding rounds.
Extension du pipeline principal pour gérer les relations d'investissement.
"""

import pandas as pd
import pickle
import os
from pathlib import Path
import duckdb
import networkx as nx
from typing import List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choix de la source de données
USE_DUCKDB = True  # True = utiliser DuckDB, False = utiliser CSV

# Chemins des fichiers
DATA_PATH_DUCKDB = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
DATA_PATH_INVESTMENTS_CSV = "data/data_cb/investments.csv"
DATA_PATH_FUNDING_CSV = "data/data_cb/funding_rounds.csv"

SAVE_DIR_CLASSES = "savings/classes"
SAVE_DIR_NETWORKS = "savings/networks"

# Paramètres de filtrage
FLAG_CYBERSECURITY = True
LIMITS = [10000]
CYBERSECURITY_KEYWORDS = ['quantum computing']

# ============================================================================
# FONCTIONS UTILITAIRES DE NETTOYAGE
# ============================================================================

def CB_data_cleaning(
    df: pd.DataFrame, 
    to_drop: List[str], 
    to_rename: Dict[str, str], 
    to_check_double: Dict[str, str],
    drop_if_nan: List[str], 
    sort_by: str = ""
) -> pd.DataFrame:
    """Performs the Data Cleaning part of the CB dataset

    Args:
        - df: dataset to clean
        - to_drop: columns to drop
        - to_rename: columns to rename and new name
        - to_check_double: columns to check. If they bring additional value and,
                           in case they don't, drop them
        - drop_if_nan: columns where NaN values should cause row deletion
        - sort_by: column by which to sort values

    Return:
        - df: cleaned dataset
    """
    print(f"\n🧹 Nettoyage en cours...")
    print(f"   Lignes avant nettoyage: {len(df):,}")
    
    # Supprimer les colonnes
    df = df.drop(to_drop, axis=1, errors='ignore')
    print(f"   ✓ Colonnes à supprimer traitées")
    
    # Renommer les colonnes
    if to_rename:
        df = df.rename(columns=to_rename)
        print(f"   ✓ Colonnes renommées: {list(to_rename.keys())}")
    
    # Vérifier les doublons
    for key, item in to_check_double.items():
        # Si item n'apporte pas de nouvelle info:
        if key in df.columns and item in df.columns:
            if (df[key] == df[item]).all():
                df = df.drop([item], axis=1)
                print(f"   ✓ Colonne {item} supprimée (doublon de {key})")
    
    # Supprimer les lignes avec NaN dans certaines colonnes
    if len(drop_if_nan) > 0:
        before = len(df)
        for col in drop_if_nan:
            if col in df.columns:
                df = df.dropna(subset=[col])
        print(f"   ✓ {before - len(df):,} lignes supprimées (NaN dans {drop_if_nan})")
    
    # Trier
    if len(sort_by) > 0 and sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)
        print(f"   ✓ Données triées par '{sort_by}'")
    
    print(f"   Lignes après nettoyage: {len(df):,}")
    
    return df


def create_directories():
    """Crée les répertoires de sauvegarde s'ils n'existent pas."""
    Path(SAVE_DIR_CLASSES).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_NETWORKS).mkdir(parents=True, exist_ok=True)


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def load_data_from_duckdb(filepath, table_name):
    """Charge les données depuis une table DuckDB."""
    print(f"\n{'='*60}")
    print(f"CHARGEMENT DE '{table_name.upper()}' (DUCKDB)")
    print(f"{'='*60}")
    
    # Vérifier que le fichier existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier DuckDB introuvable: {filepath}")
    
    conn = duckdb.connect(filepath, read_only=True)
    
    # Vérifier que la table existe
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    
    if table_name not in table_names:
        print(f"⚠️  Table '{table_name}' non trouvée!")
        print(f"   Tables disponibles: {table_names}")
        conn.close()
        return None
    
    # Charger les données
    query = f"SELECT * FROM {table_name}"
    df = conn.execute(query).fetchdf()
    
    conn.close()
    
    print(f"✓ {len(df):,} lignes chargées depuis '{table_name}'")
    print(f"✓ {len(df.columns)} colonnes disponibles")
    
    return df


def load_data_from_csv(filepath):
    """Charge les données depuis un fichier CSV."""
    print(f"\n{'='*60}")
    print(f"CHARGEMENT DE {Path(filepath).name} (CSV)")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier CSV introuvable: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ {len(df):,} lignes chargées depuis {Path(filepath).name}")
    print(f"✓ {len(df.columns)} colonnes disponibles")
    
    return df


# ============================================================================
# NETTOYAGE SPÉCIFIQUE PAR ENTITÉ
# ============================================================================

def clean_investments(df):
    """Nettoie le DataFrame des investments."""
    print(f"\n{'='*60}")
    print("NETTOYAGE DES INVESTMENTS")
    print(f"{'='*60}")
    
    to_drop = [
        'permalink',
        'funding_round_name',
        'cb_url',  
        'is_lead_investor',
        'investor_type',
        'country_code',
        'state_code',
        'region',
        'city',
    ]
    
    to_rename = {}
    to_check_double = {}
    drop_if_nan = []
    sort_by = ""
    
    return CB_data_cleaning(df, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)


def clean_funding_rounds(df):
    """Nettoie le DataFrame des funding rounds."""
    print(f"\n{'='*60}")
    print("NETTOYAGE DES FUNDING ROUNDS")
    print(f"{'='*60}")
    
    to_drop = [
        'type',
        'permalink',
        'cb_url',   
        'rank',
        'funding_round_name',
        'investor_type',
        'raised_amount',
        'is_lead_investor',
        'post_money_valuation_usd',                                      
        'post_money_valuation',                                    
        'post_money_valuation_currency_code',
        'country_code',
        'state_code',
        'region',
        'city',
    ]
    
    to_rename = {
        'category_list': 'category_groups',
        'uuid': 'funding_round_uuid'
    }
    
    to_check_double = {}
    drop_if_nan = ['name']
    sort_by = ""
    
    return CB_data_cleaning(df, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)


def clean_merged_invest_funding(df):
    """Nettoie le DataFrame fusionné investments + funding."""
    print(f"\n{'='*60}")
    print("NETTOYAGE DU DATAFRAME FUSIONNÉ")
    print(f"{'='*60}")
    
    to_drop = [
        'name_x',
        'org_uuid',   
        'lead_investor_uuids',
        'name_y',
        'investor_uuid',
    ]
    
    to_rename = {}
    to_check_double = {}
    drop_if_nan = []
    sort_by = ""
    
    return CB_data_cleaning(df, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)


# ============================================================================
# FUSION DES DONNÉES
# ============================================================================

def merge_investments_funding(df_funding, df_investments):
    """Fusionne les DataFrames funding_rounds et investments."""
    print(f"\n{'='*60}")
    print("FUSION DES DONNÉES")
    print(f"{'='*60}")
    
    print(f"Avant fusion:")
    print(f"  - Funding rounds: {len(df_funding):,} lignes")
    print(f"  - Investments: {len(df_investments):,} lignes")
    
    # Vérifier que la colonne de fusion existe
    if 'funding_round_uuid' not in df_funding.columns:
        print("⚠️  'funding_round_uuid' manquant dans df_funding")
        print(f"   Colonnes disponibles: {df_funding.columns.tolist()}")
        return None
    
    if 'funding_round_uuid' not in df_investments.columns:
        print("⚠️  'funding_round_uuid' manquant dans df_investments")
        print(f"   Colonnes disponibles: {df_investments.columns.tolist()}")
        return None
    
    # Fusion
    df_merged = pd.merge(df_funding, df_investments, on='funding_round_uuid', how='inner')
    
    print(f"\nAprès fusion:")
    print(f"  - Lignes fusionnées: {len(df_merged):,}")
    print(f"  - Colonnes: {len(df_merged.columns)}")
    
    return df_merged


# ============================================================================
# EXTRACTION POUR GRAPHE
# ============================================================================

def extract_investment_graph(df):
    """Extrait un graphe des relations d'investissement."""
    print(f"\n{'='*60}")
    print("EXTRACTION DU GRAPHE D'INVESTISSEMENT")
    print(f"{'='*60}")
    
    G = nx.Graph()
    
    # Identifier les colonnes disponibles
    print(f"Colonnes disponibles: {df.columns.tolist()}")
    
    # Compter les entités
    if 'org_name' in df.columns:
        companies = df['org_name'].dropna().unique()
        print(f"✓ {len(companies):,} companies uniques")
    else:
        print("⚠️  Colonne 'org_name' non trouvée")
        companies = []
    
    if 'investor_name' in df.columns:
        investors = df['investor_name'].dropna().unique()
        print(f"✓ {len(investors):,} investors uniques")
    else:
        print("⚠️  Colonne 'investor_name' non trouvée")
        investors = []
    
    # Créer les nœuds et arêtes
    edges_created = 0
    for idx, row in df.iterrows():
        company = row.get('org_name')
        investor = row.get('investor_name')
        
        if pd.notna(company) and pd.notna(investor):
            if company not in G:
                G.add_node(company, type='company')
            if investor not in G:
                G.add_node(investor, type='investor')
            
            # Arête avec attributs
            G.add_edge(
                company, 
                investor,
                amount=row.get('raised_amount_usd', 0),
                date=row.get('announced_on', None)
            )
            edges_created += 1
    
    print(f"✓ Graphe créé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    
    return G


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale orchestrant tout le pipeline d'investissements."""
    print("\n" + "="*60)
    print(" "*10 + "TECHRANK - TRAITEMENT INVESTMENTS & FUNDING")
    print("="*60)
    print(f"\nSource de données: {'DuckDB' if USE_DUCKDB else 'CSV'}")
    
    # Créer les répertoires
    create_directories()
    
    try:
        # 1. Charger les données
        if USE_DUCKDB:
            df_investments = load_data_from_duckdb(DATA_PATH_DUCKDB, 'investments')
            # df_funding = load_data_from_duckdb(DATA_PATH_DUCKDB, 'funding_rounds')
            
            if df_investments is None :
                print("\n❌ Impossible de charger les données depuis DuckDB")
                return None
        else:
            df_investments = load_data_from_csv(DATA_PATH_INVESTMENTS_CSV)
            # df_funding = load_data_from_csv(DATA_PATH_FUNDING_CSV)
        
        # Afficher les colonnes disponibles
        print(f"\n📋 Colonnes dans investments: {df_investments.columns.tolist()}")
        # print(f"📋 Colonnes dans funding_rounds: {df_funding.columns.tolist()}")
        
        # 2. Nettoyer les données
        df_investments_clean = clean_investments(df_investments)
        # df_funding_clean = clean_funding_rounds(df_funding)
        
        # 3. Fusionner
        # df_merged = merge_investments_funding(df_funding_clean, df_investments_clean)
        
        # if df_merged is None or len(df_merged) == 0:
        #     print("\n⚠️  Fusion échouée ou résultat vide")
        #     return None
        
        # 4. Nettoyer le résultat fusionné
        # df_invest_funding = clean_merged_invest_funding(df_merged)
        
        # 5. Afficher un aperçu
        print(f"\n{'='*60}")
        print("APERÇU DES DONNÉES FINALES")
        print(f"{'='*60}")
        print(f"\nShape: {df_investments_clean.shape}")
        print(f"\nColonnes: {df_investments_clean.columns.tolist()}")
        print(f"\nPremières lignes:")
        print(df_investments_clean.head())
        
        # 6. Extraire le graphe (optionnel)
        G = extract_investment_graph(df_investments_clean)
        
        if G.number_of_nodes() > 0:
            # Sauvegarder le graphe
            graph_file = f"{SAVE_DIR_NETWORKS}/investment_graph.gpickle"
            with open(graph_file, "wb") as f:
                pickle.dump(G, f)
            print(f"\n✓ Graphe d'investissement sauvegardé: {graph_file}")
        
        # 7. Sauvegarder le DataFrame final
        output_file = f"{SAVE_DIR_CLASSES}/df_invest_funding.pickle"
        with open(output_file, "wb") as f:
            pickle.dump(df_investments_clean, f)
        print(f"✓ DataFrame sauvegardé: {output_file}")
        
        # Alternative: sauvegarder en CSV
        csv_file = f"{SAVE_DIR_CLASSES}/df_investments_clean.csv"
        df_investments_clean.to_csv(csv_file, index=False)
        print(f"✓ CSV sauvegardé: {csv_file}")
        
        print(f"\n{'='*60}")
        print(" "*15 + "✓ TRAITEMENT TERMINÉ")
        print(f"{'='*60}\n")
        
        return df_investments_clean
        
    except Exception as e:
        print(f"\n❌ ERREUR: {type(e).__name__}")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    df_result = main()