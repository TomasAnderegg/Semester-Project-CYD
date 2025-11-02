"""
Script de traitement des données CrunchBase pour TechRank
Auteur: Tomas
Date: 2025

Ce script charge, nettoie et traite les données d'organisations CrunchBase,
avec option de filtrage pour les entreprises de cybersécurité.
Supporte CSV et DuckDB comme sources de données.
"""

import pandas as pd
import pickle
import os
from pathlib import Path
import duckdb
import networkx as nx
import random, string, urllib, requests
from typing import List
# ============================================================================
# CONFIGURATION
# ============================================================================

# Choix de la source de données
USE_DUCKDB = True  # True = utiliser DuckDB, False = utiliser CSV

# Chemins des fichiers
DATA_PATH_DUCKDB = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
DATA_PATH_CSV = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Code\TechRank\5-TechRank-main\5-TechRank-main\data\sample CB data\organizations.csv"
ENTITY_NAME_1 = "organizations"  # Nom de l'entité qu'on considère dans DuckDB peut etre faire une structure iterative pour aller dans organization, tech, investiseement pour ne pas le faire a la main !!
ENTITY_NAME_2 = "investments"

SAVE_DIR_CLASSES = "savings/classes"
SAVE_DIR_NETWORKS = "savings/networks"

# Paramètres de filtrage
FLAG_CYBERSECURITY = True  # True = uniquement cybersécurité, False = tous les domaines
LIMITS = [10000]#[2443]  # Nombre de lignes à traiter
CYBERSECURITY_KEYWORDS = ['quantum computing'] #Permet de selectionner la catégorie de cybersécurité

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def visualize_graph(B, max_companies=20, max_technologies=30):
    """Visualize the bipartite graph B using matplotlib."""
    print('\n' + '='*60)
    print('DÉBUT DE VISUALIZE_GRAPH')
    print('='*60)
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from networkx.algorithms import bipartite
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        print(f"✓ Backend: {matplotlib.get_backend()}")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return

    # Vérifier le graphe
    print(f"\n📊 Informations sur le graphe complet:")
    print(f"   - Nœuds: {B.number_of_nodes()}")
    print(f"   - Arêtes: {B.number_of_edges()}")
    
    if B.number_of_nodes() == 0:
        print("⚠️ Graphe vide, impossible de visualiser")
        return

    # Séparer les nœuds en deux ensembles
    print("\n📋 Séparation des nœuds...")
    companies = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 0}
    technologies = set(B) - companies
    
    print(f"   - Companies totales: {len(companies)}")
    print(f"   - Technologies totales: {len(technologies)}")
    
    # ====== LIMITATION DU NOMBRE DE NŒUDS ======
    
    # Sélectionner les top companies (par degré = nombre de connexions)
    company_degrees = [(c, B.degree(c)) for c in companies]
    company_degrees.sort(key=lambda x: x[1], reverse=True)
    top_companies = [c for c, _ in company_degrees[:max_companies]]
    
    # Sélectionner les top technologies (par degré)
    tech_degrees = [(t, B.degree(t)) for t in technologies]
    tech_degrees.sort(key=lambda x: x[1], reverse=True)
    top_technologies = [t for t, _ in tech_degrees[:max_technologies]]
    
    print(f"\n✂️  Réduction du graphe:")
    print(f"   - Companies affichées: {len(top_companies)}/{len(companies)}")
    print(f"   - Technologies affichées: {len(top_technologies)}/{len(technologies)}")
    
    # Créer un sous-graphe avec seulement ces nœuds
    nodes_to_keep = set(top_companies) | set(top_technologies)
    B_sub = B.subgraph(nodes_to_keep).copy()
    
    print(f"   - Nœuds dans le sous-graphe: {B_sub.number_of_nodes()}")
    print(f"   - Arêtes dans le sous-graphe: {B_sub.number_of_edges()}")
    
    if B_sub.number_of_nodes() == 0:
        print("⚠️ Sous-graphe vide après filtrage!")
        return

    # Positionner les nœuds
    print("\n📐 Calcul des positions...")
    pos = dict()
    pos.update((n, (1, i)) for i, n in enumerate(top_companies))
    pos.update((n, (2, i)) for i, n in enumerate(top_technologies))
    print(f"   ✓ {len(pos)} positions calculées")

    # Dessiner le graphe
    print("\n🎨 Création de la figure...")
    try:
        plt.figure(figsize=(16, 12))
        print("   ✓ Figure créée")
        
        # Couleurs
        companies_in_sub = {n for n in top_companies if n in B_sub}
        node_colors = ['lightblue' if n in companies_in_sub else 'lightgreen' for n in B_sub.nodes()]
        print(f"   ✓ {len(node_colors)} couleurs définies")
        
        print("   Drawing graph...")
        nx.draw(
            B_sub, 
            pos=pos, 
            with_labels=True, 
            node_size=800,
            node_color=node_colors,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            alpha=0.7
        )
        print("   ✓ Graphe dessiné")
        
        plt.title(f"Bipartite Graph: Top {len(top_companies)} Companies and Top {len(top_technologies)} Technologies", 
                  fontsize=14, fontweight='bold')
        print("   ✓ Titre ajouté")
        
        # Sauvegarder
        plt.savefig('bipartite_graph_limited.png', dpi=300, bbox_inches='tight')
        print("   ✓ Graphe sauvegardé: bipartite_graph_limited.png")
        
        print("\n📺 Affichage...")
        plt.show()
        print("   ✓ plt.show() appelé")
        
    except Exception as e:
        print(f"   ❌ Erreur pendant le dessin: {e}")
        import traceback
        traceback.print_exc()
    
    plt.close()
    print('\n' + '='*60)
    print('FIN DE VISUALIZE_GRAPH')
    print('='*60 + '\n')

def create_directories():
    """Crée les répertoires de sauvegarde s'ils n'existent pas."""
    Path(SAVE_DIR_CLASSES).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR_NETWORKS).mkdir(parents=True, exist_ok=True)
    # print("✓ Répertoires de sauvegarde créés/vérifiés")


def convert_to_list(string):
    """Convertit une chaîne séparée par des virgules en liste."""
    # Gérer les Series pandas
    if isinstance(string, pd.Series):
        if len(string) == 1:
            string = string.iloc[0]
        else:
            return string.apply(convert_to_list)
    
    if pd.isna(string):
        return []
    return [item.strip() for item in str(string).split(",")]


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def explore_duckdb(filepath, ENTITY_NAME):
    """Explore la structure d'une base DuckDB."""
    # print(f"\n{'='*60}")
    # print("EXPLORATION DE LA BASE DUCKDB")
    # print(f"{'='*60}")
    
    conn = duckdb.connect(filepath, read_only=True)
    
    # Lister les tables
    tables = conn.execute("SHOW TABLES").fetchall()
    # print(f"✓ Tables disponibles: {[t[0] for t in tables]}")
    
    # Si la table existe, afficher ses colonnes
    if any(ENTITY_NAME in t for t in tables):
        columns = conn.execute(f"DESCRIBE {ENTITY_NAME}").fetchall()
        # print(f"\n✓ Colonnes de la table '{TABLE_NAME}':")
        for col in columns[:10]:  # Afficher les 10 premières colonnes
            # print(f"  - {col[0]}: {col[1]}")
            pass
        if len(columns) > 10:
            pass
            # print(f"  ... et {len(columns) - 10} autres colonnes")
        
        # Compter les lignes
        count = conn.execute(f"SELECT COUNT(*) FROM {ENTITY_NAME}").fetchone()[0]
        # print(f"\n✓ Nombre total de lignes: {count:,}")
    else:
        pass
        # print(f"\n⚠ Table '{TABLE_NAME}' non trouvée!")
        # print(f"  Tables disponibles: {[t[0] for t in tables]}")
    
    conn.close()
    return tables


def load_data_from_duckdb(filepath, table_name):
    """Charge les données depuis une base DuckDB."""
    # print(f"\n{'='*60}")
    # print("CHARGEMENT DES DONNÉES (DUCKDB)")
    # print(f"{'='*60}")
    
    # Vérifier que le fichier existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier DuckDB introuvable: {filepath}")
    
    # print(f"✓ Connexion à: {Path(filepath).name}")
    
    # Explorer la base
    explore_duckdb(filepath, table_name)
    
    # Charger les données
    conn = duckdb.connect(filepath, read_only=True)
    
    # Requête pour charger toutes les données
    query = f"SELECT * FROM {table_name}"
    df = conn.execute(query).fetchdf()
    
    conn.close()
    
    # print(f"\n✓ {len(df):,} lignes chargées depuis la table '{table_name}'")
    # print(f"✓ {len(df.columns)} colonnes disponibles")
    # print(f"\n✓ Aperçu des colonnes:")
    for i, col in enumerate(df.columns[:15]):
        # print(f"  {i+1}. {col}")
        pass
    if len(df.columns) > 15:
        pass
        # print(f"  ... et {len(df.columns) - 15} autres colonnes")
    
    return df


def load_data_from_csv(filepath):
    """Charge les données depuis un fichier CSV."""
    # print(f"\n{'='*60}")
    # print("CHARGEMENT DES DONNÉES (CSV)")
    # print(f"{'='*60}")
    
    df = pd.read_csv(filepath)
    # print(f"✓ {len(df):,} lignes chargées depuis {Path(filepath).name}")
    # print(f"✓ {len(df.columns)} colonnes disponibles")
    
    return df


def load_data(use_duckdb=True, entity_name=ENTITY_NAME_1):
    """Charge les données depuis la source configurée.
    Si on a choisi DuckDB, utilise cette source, sinon CSV.
    """
    if use_duckdb:
        return load_data_from_duckdb(DATA_PATH_DUCKDB, entity_name)
    else:
        return load_data_from_csv(DATA_PATH_CSV)


# ============================================================================
# NETTOYAGE DES DONNÉES
# ============================================================================

def clean_data(df):
    """Nettoie et prépare les données."""
    # print(f"\n{'='*60}")
    # print("NETTOYAGE DES DONNÉES")
    # print(f"{'='*60}")
    
    # Colonnes à supprimer
    columns_to_drop = [
        'type', 'permalink', 'cb_url', 'created_at', 'domain',
        'address', 'state_code', 'updated_at', 'legal_name', 'roles',
        'postal_code', 'homepage_url', 'num_funding_rounds',
        'total_funding_currency_code', 'phone', 'email', 'num_exits',
        'alias1', 'alias2', 'alias3', 'logo_url', 'last_funding_on',
        'twitter_url', 'facebook_url', 'linkedin_url', 'crunchbase_url',
        'overview', 'acquisitions', 'city', 'primary_role', 'region', 'founded_on',
        'ipo', 'milestones', 'news_articles', 'status', 'country_code', 'region', 'investment_type',
        'post_money_valuation_usd', 'pre_money_valuation_usd', 'closed_on'
    ]
    
    # Renommer les colonnes (adaptable selon la source)
    """ Sert a renommer les colonnes pour uniformiser les noms entre DuckDB et CSV, 
        donc si categrory_list est present on le renomme en category_groups et 
        ainsi de suite
    """
    rename_mapping = {
        'category_list': 'category_groups',
        'category_groups_list': 'category_groups'  # Au cas où
    }
    
    # Colonnes où NaN n'est pas acceptable
    required_columns = ['category_groups', 'rank', 'short_description']
    
    # Appliquer le nettoyage
    df_clean = df.copy()
    
    # print(f"✓ Colonnes présentes avant nettoyage: {list(df_clean.columns[:10])}...")
    
    # IMPORTANT: Supprimer les colonnes dupliquées
    if df_clean.columns.duplicated().any():
        duplicated_cols = df_clean.columns[df_clean.columns.duplicated()].tolist()
        # print(f"⚠ Colonnes dupliquées détectées: {duplicated_cols}")
        # Garder seulement la première occurrence de chaque colonne
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        # print(f"✓ Colonnes dupliquées supprimées")
    
    # Supprimer les colonnes inutiles
    """ Supprimer seulement les colonnes qu'on a defini dans columns_to_drop et 
        qui existent dans le DataFrame
    """
    cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)
    # print(f"✓ {len(cols_to_drop)} colonnes supprimées")
    
    # Renommer (seulement les colonnes qui existent)
    actual_renames = {k: v for k, v in rename_mapping.items() if k in df_clean.columns}
    if actual_renames:
        df_clean = df_clean.rename(columns=actual_renames)
        # print(f"✓ Colonnes renommées: {actual_renames}")
    
    # Vérifier que les colonnes requises existent
    missing_cols = [col for col in required_columns if col not in df_clean.columns]
    if missing_cols:
        # print(f"\n⚠ ATTENTION: Colonnes manquantes: {missing_cols}")
        # print(f"  Colonnes disponibles: {list(df_clean.columns)}")
        raise ValueError(f"Colonnes requises manquantes: {missing_cols}")
    
    # Supprimer les lignes avec NaN dans les colonnes requises
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=required_columns)
    # print(f"✓ {before - len(df_clean):,} lignes supprimées (valeurs manquantes)")
    
    # Trier par rang
    if 'rank' in df_clean.columns:
        df_clean = df_clean.sort_values('rank').reset_index(drop=True)
        # print(f"✓ Données triées par 'rank'")
    
    return df_clean


def process_category_groups(df):
    """Convertit la colonne category_groups en listes."""
    # print(f"\n{'='*60}")
    # print("TRAITEMENT DES CATÉGORIES")
    # print(f"{'='*60}")
    
    df_proc = df.copy()
    
    # Vérifier que la colonne existe
    if "category_groups" not in df_proc.columns:
        raise ValueError(f"Colonne 'category_groups' introuvable. Colonnes disponibles: {list(df_proc.columns)}")
    
    # S'assurer qu'il n'y a qu'une seule colonne category_groups
    if df_proc.columns.duplicated().any():
        # print(f"⚠ Colonnes dupliquées encore présentes, nettoyage...")
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
    
    # Obtenir la Series (pas DataFrame)
    col_series = df_proc['category_groups']
    
    # Vérifier que c'est bien une Series
    if not isinstance(col_series, pd.Series):
        # print(f"⚠ 'category_groups' est un {type(col_series)}, conversion en Series...")
        col_series = df_proc['category_groups'].squeeze()
    
    # print(f"✓ Type de la colonne: {type(col_series)}")
    
    # Vérifier le type de la première valeur non-nulle
    first_valid_idx = col_series.first_valid_index()
    if first_valid_idx is not None:
        first_valid = col_series.loc[first_valid_idx]
        # print(f"✓ Exemple de valeur: '{first_valid[:50]}...' (type: {type(first_valid)})")
        
        # Convertir en liste si ce n'est pas déjà une liste
        if not isinstance(first_valid, list):
            df_proc['category_groups'] = col_series.apply(convert_to_list)
            # print(f"✓ Conversion des catégories en listes effectuée")
        else:
            pass
            # print(f"✓ Les catégories sont déjà au format liste")
    else:
        pass
        # print(f"⚠ Aucune valeur valide trouvée dans category_groups")
    
    # Statistiques
    # print(f"✓ Valeurs NaN: {df_proc['category_groups'].isna().sum()}")
    # print(f"✓ Exemples de catégories:")
    for i in range(min(3, len(df_proc))):
        cats = df_proc['category_groups'].iloc[i]
        print(f"  {i+1}. {cats}")
    
    return df_proc


def filter_cybersecurity(df, keywords):
    """Filtre les entreprises de cybersécurité."""
    print(f"\n{'='*60}")
    print("FILTRAGE CYBERSÉCURITÉ")
    print(f"{'='*60}")
    
    # Recherche dans category_groups
    mask_cat = df['category_groups'].apply(
        lambda lst: isinstance(lst, list) and 
        any(k.lower() in ' '.join(lst).lower() for k in keywords)
    )
    
    # Recherche dans short_description
    mask_desc = df['short_description'].astype(str).str.contains(
        '|'.join(keywords), case=False, na=False
    )
    
    # Combinaison des masques
    mask_combined = mask_cat | mask_desc
    
    df_filtered = df[mask_combined].reset_index(drop=True)
    
    print(f"✓ Correspondances dans category_groups: {mask_cat.sum():,}")
    print(f"✓ Correspondances dans short_description: {mask_desc.sum():,}")
    print(f"✓ Total d'entreprises cybersécurité: {len(df_filtered):,}")
    
    if len(df_filtered) > 0:
        pass
        # print(f"\n  Exemples d'entreprises filtrées:")
        for i, row in df_filtered.head(3).iterrows():
            pass
            # print(f"  - {row.get('name', 'N/A')}: {row['category_groups']}")
    
    return df_filtered


# ============================================================================
# EXTRACTION ET SAUVEGARDE
# ============================================================================
def extract_classes_company_tech(df):
    """Extracts the dictionaries of Companies and Technologies 
    from the dataset and create the network
    
    Args:
        - df: dataset

    Return:
        - dict_companies: dictionary of companies
        - dict_tech: dictionary of technologies
        - B: graph that links companies and technologies 
    """
 
    # from geopy.geocoders import Nominatim
    import classes  # tes classes Company et Technology
    print('INSIDE EXTRACT FUNCTION')
    print(f"DataFrame shape: {df.shape}")  # ← AJOUTEZ CECI
    print(f"DataFrame columns: {df.columns.tolist()}")  # ← AJOUTEZ CECI
    
    dict_companies = {}
    dict_tech = {}
    B = nx.Graph() #creation d'un graph vide no orienté

    # Boucle sur chaque ligne du DataFrame
    for index, row in df.iterrows():
        # Création du nom de l'entreprise
        comp_name = row['name']

        # Exemple : créer l'objet Company
        c = classes.Company(
            id=row.get('uuid', index),
            name=comp_name,
            technologies=row.get('category_groups', []),

        )

        dict_companies[comp_name] = c # on sauvegarde sous le nom de l'entreprise les infos de l'entreprises (uuid, nom, categories, tot_previous_investments, num_previous_investments)
        B.add_node(comp_name, bipartite=0) #creation d'un noeud avec le nom de la comapagnie et bipartite=0 correspond a la premiere entite (dans ce cas compagnie)

        # Technologies
        categories = row.get('category_groups', [])
        if not isinstance(categories, list):
            categories = [categories]

        for tech in categories:
            if tech not in dict_tech:
                t = classes.Technology(name=tech)
                dict_tech[tech] = t
                B.add_node(tech, bipartite=1)

            # Lien entreprise → technologie
            B.add_edge(comp_name, tech)
    print('INSIDE EXTRACT FUNCTION 2')
    print(f"Total nodes in graph: {B.number_of_nodes()}")  # ← AJOUTEZ CECI
    print(f"Total edges in graph: {B.number_of_edges()}")  # ← AJOUTEZ CECI

    return dict_companies, dict_tech, B

def extract_classes_investment(df_funding_rounds, df_invest):
 
    # from geopy.geocoders import Nominatim
    import classes  # tes classes Company et Technology
    print('INSIDE EXTRACT FUNCTION')
    print(f"DataFrame shape: {df_funding_rounds.shape}")  # ← AJOUTEZ CECI
    print(f"DataFrame columns: {df_funding_rounds.columns.tolist()}")  # ← AJOUTEZ CECI
    
    funding_round_ids = df_invest['funding_round_uuid'].tolist()
    B = nx.Graph() #creation d'un graph vide no orienté

    # Boucle sur chaque ligne du DataFrame
    matching_rows_funding_rounds = df_funding_rounds[df_funding_rounds['uuid'].isin(funding_round_ids)]

    i = classes.Investor(
        name=matching_rows_funding_rounds['orga_name'],
        raised_money_usd=matching_rows_funding_rounds['raised_amount_usd'],
        funding_round_id=funding_round_ids

    )

    for index, row in df_funding_rounds.iterrows():
        # Création du nom de l'entreprise
        funding_round_ids = row['funding_round_uuid']
        
        B.add_node(funding_round_ids, bipartite=2) #creation d'un noeud avec le nom de la comapagnie et bipartite=0 correspond a la premiere entite (dans ce cas compagnie)
        B.add_edge(i.raised_amount_usd, tech)

    print('INSIDE EXTRACT FUNCTION 2')
    print(f"Total nodes in graph: {B.number_of_nodes()}")  # ← AJOUTEZ CECI
    print(f"Total edges in graph: {B.number_of_edges()}")  # ← AJOUTEZ CECI

    return dict_companies, dict_tech, B



def extract_and_save(df, limit, is_cybersecurity):
    """Extrait les classes et sauvegarde les résultats."""
    print(f"\n{'='*60}")
    print(f"EXTRACTION ET SAUVEGARDE (limite: {limit:,} lignes)")
    print(f"{'='*60}")
    
    # Limiter les données
    df_limited = df[:limit]
    print(df_limited.head())
    # print(f"✓ Traitement de {len(df_limited):,} entreprises")
    print(f"Colonnes disponibles: {df_limited.columns.tolist()}")  # ← AJOUTEZ
    
    # Extraction des classes (fonction à importer depuis votre module)
    # Note: Cette fonction doit être définie dans votre code
    dict_companies, dict_tech, B = extract_classes_company_tech(df_limited)
    try:
        # dict_companies, dict_tech, B = extract_classes_company_tech(df_limited)
        # visualize_graph(B)
        visualize_graph(B, max_companies=5, max_technologies=10)
    # except ImportError:
    #     print("⚠ Fonction extract_classes_company_tech non disponible")
    #     print("  Veuillez l'importer depuis votre module")
    #     return
    except Exception as e:
        print(f"⚠️ Erreur lors de la visualisation: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"✓ {len(dict_companies):,} entreprises extraites")
    print(f"✓ {len(dict_tech):,} technologies extraites")
    
    # Générer les noms de fichiers
    suffix = "cybersecurity_" if is_cybersecurity else ""
    
    file_companies = f"{SAVE_DIR_CLASSES}/dict_companies_{suffix}{len(dict_companies)}.pickle"
    file_tech = f"{SAVE_DIR_CLASSES}/dict_tech_{suffix}{len(dict_tech)}.pickle"
    file_graph = f"{SAVE_DIR_NETWORKS}/{suffix}comp_{len(dict_companies)}_tech_{len(dict_tech)}.gpickle"
    
    # Sauvegarder les dictionnaires
    with open(file_companies, "wb") as f:
        pickle.dump(dict_companies, f)
    print(f"✓ Entreprises sauvegardées: {file_companies}")
    
    with open(file_tech, "wb") as f:
        pickle.dump(dict_tech, f)
    print(f"✓ Technologies sauvegardées: {file_tech}")
    
    # Sauvegarder le graphe
    with open(file_graph, "wb") as f:
        pickle.dump(B, f)
    print(f"✓ Graphe sauvegardé: {file_graph}")


# ============================================================================
# Exécution principale 
# ============================================================================

def main():
    """Fonction principale orchestrant tout le pipeline."""
    print("\n" + "="*60)
    print(" "*15 + "TECHRANK - TRAITEMENT CRUNCHBASE")
    print("="*60)
    print(f"\nSource de données: {'DuckDB' if USE_DUCKDB else 'CSV'}")
    
    # Créer les répertoires
    create_directories()
    
    try:
        # 1. Charger les données
        df_comp_tech = load_data(use_duckdb=USE_DUCKDB, entity_name=ENTITY_NAME_1)
        df_invest = load_data(use_duckdb=USE_DUCKDB, entity_name=ENTITY_NAME_2)

        print("=== Aperçu de df (brut) ===")
        print(df_comp_tech.shape)          # nombre de lignes et colonnes
        print(df_comp_tech.columns)        # noms des colonnes
        print(df_comp_tech.head(5))        # les 5 premières lignes
        
        # 2. Nettoyer les données
        df_comp_tech_clean = clean_data(df_comp_tech)
        df_invest_clean = clean_data(df_invest)
        
        # 3. Traiter les catégories
        df_comp_tech_proc = process_category_groups(df_comp_tech_clean)
        df_invest_proc = process_category_groups(df_invest_clean)
        
        # 4. Filtrer si nécessaire
        if FLAG_CYBERSECURITY:
            df_comp_tech_final = filter_cybersecurity(df_comp_tech_proc, CYBERSECURITY_KEYWORDS)
            
            if len(df_comp_tech_final) == 0:
                print("\n⚠ ATTENTION: Aucune entreprise de cybersécurité trouvée!")
                print("  Vérifiez les mots-clés ou les données source")
                return
        else:
            df_comp_tech_final = df_comp_tech_proc
            print(f"\n✓ Mode tous domaines: {len(df_comp_tech_final):,} entreprises")
        
        # 5. Extraire et sauvegarder pour chaque limite
        for limit in LIMITS:
            if limit > len(df_comp_tech_final):
                print(f"\n⚠ Limite {limit:,} > données disponibles ({len(df_comp_tech_final):,})")
                print(f"  Utilisation de {len(df_comp_tech_final):,} lignes")
                limit = len(df_comp_tech_final)
            
            extract_and_save(df_comp_tech_final, limit, FLAG_CYBERSECURITY)
        
        print(f"\n{'='*60}")
        print(" "*20 + "✓ TRAITEMENT TERMINÉ")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {type(e).__name__}")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    main()