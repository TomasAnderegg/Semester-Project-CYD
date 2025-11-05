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
LIMITS = [500]
CYBERSECURITY_KEYWORDS = ['cyber', 'security', 'cybersecurity']

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


def filter_cybersecurity(df: pd.DataFrame, keywords: List[str] = ['cyber', 'security', 'cybersecurity']) -> pd.DataFrame:
    """
    Filtre les entreprises liées à la cybersécurité en se basant sur les colonnes
    'category_groups' et 'short_description'.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entreprises issu de Crunchbase.
    keywords : list of str
        Liste de mots-clés caractéristiques de la cybersécurité.

    Retour
    ------
    df_filtered : pd.DataFrame
        Sous-ensemble du DataFrame contenant uniquement les entreprises
        identifiées comme appartenant au domaine de la cybersécurité.
    """

    print("\n========================== FILTRAGE CYBER ==========================")
    df = df.copy().reset_index(drop=True)

    # --- Vérification des colonnes nécessaires ---
    required_cols = ['category_groups', 'short_description']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le DataFrame : '{col}'")

    # --- 1️⃣ Vérification dans category_groups ---
    def match_in_category_groups(entry):
        """Vérifie si un mot-clé est présent dans la liste de catégories."""
        if isinstance(entry, list):
            joined = ' '.join(map(str, entry)).lower()
        elif isinstance(entry, str):
            joined = entry.lower()
        else:
            return False
        return any(k in joined for k in keywords)

    mask_cat = df['category_groups'].apply(match_in_category_groups) #donne une serie de booléens

    # --- 2️⃣ Vérification dans short_description ---
    pattern = '|'.join([f"\\b{k}\\b" for k in keywords])  # recherche par mot complet
    mask_desc = df['short_description'].astype(str).str.contains(pattern, case=False, na=False, regex=True)

    # --- 3️⃣ Combinaison des deux filtres ---
    mask_combined = mask_cat | mask_desc
    df_filtered = df.loc[mask_combined].reset_index(drop=True)

    # --- 4️⃣ Statistiques ---
    print(f"✓ Entreprises détectées dans category_groups : {mask_cat.sum():,}")
    print(f"✓ Entreprises détectées dans short_description : {mask_desc.sum():,}")
    print(f"➡️  Total unique d'entreprises cybersécurité : {len(df_filtered):,}")

    # --- 5️⃣ Exemple de vérification visuelle ---
    # if not df_filtered.empty:
    #     print("\nExemples d'entreprises cybersécurité détectées :")
    #     print(df_filtered[['name', 'category_groups', 'short_description']].head(5).to_string(index=False))

    return df_filtered



# ===================================================================
# BIPARTITE CREATION FUNCTION
# ===================================================================

def extract_classes_company_tech_all(df):
    """
    Extrait toutes les entreprises et leurs technologies à partir du dataset
    et crée le graphe bipartite.
    
    Args:
        df: pandas DataFrame contenant au moins ['uuid','name','category_groups','rank' ou 'rank_company']
    
    Returns:
        dict_companies: dictionnaire {nom entreprise : Company}
        dict_tech: dictionnaire {nom technologie : Technology}
        B: graphe bipartite NetworkX (avec suppression des arêtes invalides)
    """
    
    dict_companies = {}
    dict_tech = {}
    B = nx.Graph()
    # print("colonne name", row['name'])
    
    for index, row in df.iterrows():
        comp_name = row['name']
        print("colonne name", row['name'])

        # Création de l'objet Company
        c = classes.Company(
            id=row['uuid'],
            name=comp_name,
            technologies=row['category_groups']
        )

        if 'rank_company' in df.columns:
            c.rank_CB = row['rank_company']
        elif 'rank' in df.columns:
            c.rank_CB = row['rank']

        dict_companies[comp_name] = c
        B.add_node(comp_name, bipartite=0)

        # Ajout des technologies au dictionnaire et au graphe
        for tech in row['category_groups']:
            if tech not in dict_tech:
                dict_tech[tech] = classes.Technology(name=tech)
                B.add_node(tech, bipartite=1)
            B.add_edge(comp_name, tech)

    # # ======================
    # # Vérification des arêtes invalides (même ensemble)
    # # ======================
    edges_to_remove = [(u, v) for u, v in B.edges() 
                    if B.nodes[u]['bipartite'] == B.nodes[v]['bipartite']]

    if edges_to_remove:
        print(f"⚠ {len(edges_to_remove)} arêtes invalides détectées. Détails :")
        for u, v in edges_to_remove:
            type_u = 'Company' if B.nodes[u]['bipartite'] == 0 else 'Technology'
            type_v = 'Company' if B.nodes[v]['bipartite'] == 0 else 'Technology'
            # print(f"  - Arête invalide : {u} ({type_u}) ↔ {v} ({type_v})")
        
        B.remove_edges_from(edges_to_remove)
        print("✅ Arêtes invalides supprimées")
    else:
        print("✓ Aucune arête invalide détectée")

    
    print(f"✅ Extraction terminée : {len(dict_companies)} entreprises et {len(dict_tech)} technologies.")
    
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


def plot_bipartite_graph(G, max_companies=10, max_techs=15):
    print("\n========================== PLOTTING BIPARTITE GRAPH ==========================")
    
    # Identifier les composantes connexes
    connected_components = list(nx.connected_components(G))
    connected_components.sort(key=len, reverse=True)
    
    # Prendre la plus grande composante connexe
    if connected_components:
        largest_component = connected_components[0]
        G_connected = G.subgraph(largest_component).copy()
    else:
        G_connected = G
    
    # Séparer les types de nœuds dans la composante connexe
    companies = [n for n, d in G_connected.nodes(data=True) if d['bipartite'] == 0]
    techs = [n for n, d in G_connected.nodes(data=True) if d['bipartite'] == 1]
    
    print(f"Composante principale: {len(companies)} companies, {len(techs)} technologies")
    
    # Si c'est encore trop grand, filtrer intelligemment
    if len(companies) > max_companies or len(techs) > max_techs:
        # Garder les nœuds les plus connectés MAIS avec leurs voisins
        top_companies = sorted(companies, key=lambda n: G_connected.degree(n), reverse=True)[:max_companies]
        
        # Récupérer toutes les technologies connectées à ces companies
        connected_techs = set()
        for company in top_companies:
            connected_techs.update(G_connected.neighbors(company))
        
        # Récupérer toutes les companies connectées à ces technologies (pour complétude)
        all_companies = set(top_companies)
        for tech in connected_techs:
            all_companies.update(G_connected.neighbors(tech))
        
        G_final = G_connected.subgraph(all_companies | connected_techs).copy()
    else:
        G_final = G_connected
    
    # Visualisation
    pos = nx.spring_layout(G_final, k=1, iterations=50)
    plt.figure(figsize=(20, 12))
    
    companies_final = [n for n, d in G_final.nodes(data=True) if d['bipartite'] == 0]
    techs_final = [n for n, d in G_final.nodes(data=True) if d['bipartite'] == 1]
    
    # Taille des nœuds proportionnelle au degré
    company_sizes = [G_final.degree(node) * 200 for node in companies_final]
    tech_sizes = [G_final.degree(node) * 300 for node in techs_final]
    
    # Dessiner
    nx.draw_networkx_nodes(G_final, pos, nodelist=companies_final, 
                          node_color='red', node_size=company_sizes, alpha=0.7, label='Companies')
    nx.draw_networkx_nodes(G_final, pos, nodelist=techs_final, 
                          node_color='blue', node_size=tech_sizes, alpha=0.7, label='Technologies')
    
    # Labels seulement pour les nœuds importants
    labels = {}
    for node in companies_final:
        if G_final.degree(node) >= 2:  # Seulement les companies avec au moins 2 connexions
            labels[node] = node[:15] + "..." if len(node) > 15 else node
    
    for node in techs_final:
        if G_final.degree(node) >= 3:  # Seulement les technologies avec au moins 3 connexions
            labels[node] = node[:15] + "..." if len(node) > 15 else node
    
    nx.draw_networkx_labels(G_final, pos, labels, font_size=8)
    nx.draw_networkx_edges(G_final, pos, alpha=0.3, width=0.5)
    
    plt.legend()
    plt.title(f"Graphe Bipartite - {len(companies_final)} Companies, {len(techs_final)} Technologies")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return pos

def analyze_graph_structure(B):
    """Analyse la structure du graphe pour identifier les problèmes"""
    print("\n" + "="*50)
    print("ANALYSE DU GRAPHE")
    print("="*50)
    
    companies = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    techs = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]
    
    # Degrés
    company_degrees = [B.degree(node) for node in companies]
    tech_degrees = [B.degree(node) for node in techs]
    
    print(f"Nombre total de companies: {len(companies)}")
    print(f"Nombre total de technologies: {len(techs)}")
    print(f"Nombre total d'arêtes: {B.number_of_edges()}")
    print(f"Degré moyen companies: {np.mean(company_degrees):.2f}")
    print(f"Degré moyen technologies: {np.mean(tech_degrees):.2f}")
    
    # Composantes connexes
    connected_components = list(nx.connected_components(B))
    print(f"Nombre de composantes connexes: {len(connected_components)}")
    
    for i, comp in enumerate(connected_components[:5]):  # Afficher les 5 plus grandes
        comp_companies = [n for n in comp if n in companies]
        comp_techs = [n for n in comp if n in techs]
        print(f"  Composante {i+1}: {len(comp_companies)} companies, {len(comp_techs)} technologies")
    
    # Nœuds isolés
    isolated_nodes = list(nx.isolates(B))
    print(f"Nœuds isolés: {len(isolated_nodes)}")
    
    return {
        'companies': companies,
        'techs': techs,
        'components': connected_components
    }


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

def main():
    create_directories()
    df_comp = load_data(use_duckdb=USE_DUCKDB, entity_name=ENTITY_NAME_1)

    df_comp_clean = clean_data(df_comp)
    # print("colonnes de df_comp_clean", df_comp_clean.head())
    df_comp_proc = process_category_groups(df_comp_clean)

    df_comp_filter = filter_cybersecurity(df_comp_proc, CYBERSECURITY_KEYWORDS)
    if len(df_comp_filter) == 0:
        print("Aucune entreprise cybersécurité trouvée")
        return

    for limit in LIMITS:
        dict_companies, dict_tech, B = extract_classes_company_tech_all(df_comp_filter)

        # 3. DIAGNOSTIC CRITIQUE
        if B.number_of_nodes() == 0:
            print("❌ Graphe vide - aucune entreprise avec technologies cybersecurity")
            continue
            
        companies = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
        techs = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]
        
        print(f"📈 MÉTRIQUES RÉSEAU CYBERSECURITY:")
        print(f"  - Companies: {len(companies)}")
        print(f"  - Technologies CYBERSECURITY: {len(techs)}")
        print(f"  - Arêtes: {B.number_of_edges()}")
        
        # Vérifier s'il y a des entreprises sans connexion
        isolated_companies = [node for node in companies if B.degree(node) == 0]
        if isolated_companies:
            print(f"⚠️  Entreprises sans connexion: {len(isolated_companies)}")
            print(f"   Exemples: {isolated_companies[:3]}")
        else:
            print("✅ Toutes les entreprises ont au moins une connexion")
        
        analyze_graph_structure(B)


        # ✅ Appel corrigé
        save_graph_and_dicts(B, df_comp_proc, dict_companies, dict_tech, limit, FLAG_CYBERSECURITY)




if __name__ == "__main__":
    main()
