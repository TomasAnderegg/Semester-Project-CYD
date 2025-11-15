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
CYBERSECURITY_KEYWORDS = ['quantum computing', 'quantum encryption', 'quantum key distribution',]

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


def filter_cybersecurity(df: pd.DataFrame, keywords: List[str] = ['quantum computing', 'quantum encryption', 'quantum key distribution']) -> pd.DataFrame:
    """
    Filtre les entreprises dont AU MOINS UNE catégorie correspond EXACTEMENT à un keyword.
    """
    print("\n========================== FILTRAGE CYBER (EXACT MATCH) ==========================")
    df = df.copy().reset_index(drop=True)

    # Vérification des colonnes nécessaires
    if 'category_groups' not in df.columns:
        raise ValueError(f"Colonne manquante : 'category_groups'")

    # Normaliser les keywords (minuscules, trim)
    normalized_keywords = {k.strip().lower() for k in keywords}
    print(f"🔍 Keywords recherchés (normalisés) : {normalized_keywords}")

    def match_exact_categories(entry):
        """
        Vérifie si AU MOINS UNE catégorie de l'entreprise 
        correspond EXACTEMENT à un keyword.
        """
        if isinstance(entry, list):
            # Normaliser chaque catégorie
            categories = {str(cat).strip().lower() for cat in entry if pd.notna(cat)}
        elif isinstance(entry, str):
            categories = {entry.strip().lower()}
        else:
            return False
        
        # Intersection : y a-t-il au moins une catégorie qui matche ?
        matches = categories & normalized_keywords
        return len(matches) > 0

    # Appliquer le filtre
    mask_cat = df['category_groups'].apply(match_exact_categories)
    df_filtered = df.loc[mask_cat].reset_index(drop=True)

    # Statistiques
    print(f"✓ Entreprises avec match EXACT : {mask_cat.sum():,}")
    print(f"➡️  Total d'entreprises filtrées : {len(df_filtered):,}")

    # Afficher les catégories uniques trouvées
    if not df_filtered.empty:
        all_matching_cats = set()
        for cats in df_filtered['category_groups']:
            if isinstance(cats, list):
                normalized_cats = {str(c).strip().lower() for c in cats if pd.notna(c)}
                # Garder seulement celles qui matchent
                all_matching_cats.update(normalized_cats & normalized_keywords)
        
        print(f"\n🏷️  Catégories qui ont matché : {sorted(all_matching_cats)}")
        
        print("\n📋 Exemples d'entreprises détectées :")
        for idx, row in df_filtered[['name', 'category_groups']].head(10).iterrows():
            print(f"  • {row['name']}: {row['category_groups']}")
    else:
        print("\n⚠️  Aucune entreprise trouvée avec ces keywords exacts !")
        print("Vérifiez que vos keywords correspondent exactement aux catégories Crunchbase.")

    return df_filtered


# ===================================================================
# BIPARTITE CREATION FUNCTION
# ===================================================================

def extract_classes_company_tech_all(df, keywords=None):
    """
    Crée le graphe bipartite en ne gardant QUE les technologies qui matchent les keywords.
    
    Args:
        df: DataFrame filtré des entreprises
        keywords: Liste des keywords exacts à garder (si None, garde toutes les catégories)
    """
    dict_companies = {}
    dict_tech = {}
    B = nx.Graph()
    
    # Normaliser les keywords
    if keywords:
        normalized_keywords = {k.strip().lower() for k in keywords}
    else:
        normalized_keywords = None
    
    print(f"\n🔧 Construction du graphe bipartite...")
    if normalized_keywords:
        print(f"   Technologies autorisées : {normalized_keywords}")
    
    companies_without_tech = []
    
    for index, row in df.iterrows():
        comp_name = row['name']
        
        # Création de l'objet Company
        c = classes.Company(
            id=row['uuid'],
            name=comp_name,
            technologies=row['category_groups']  # Garde toutes pour l'objet
        )

        if 'rank_company' in df.columns:
            c.rank_CB = row['rank_company']
        elif 'rank' in df.columns:
            c.rank_CB = row['rank']

        dict_companies[comp_name] = c
        B.add_node(comp_name, bipartite=0)

        # Technologies : FILTRER selon les keywords
        categories = row['category_groups']
        
        if isinstance(categories, list):
            # Normaliser et filtrer
            if normalized_keywords:
                valid_techs = [
                    tech for tech in categories 
                    if str(tech).strip().lower() in normalized_keywords
                ]
            else:
                valid_techs = categories
            
            # Ajouter les arêtes UNIQUEMENT pour les technologies valides
            for tech in valid_techs:
                tech_normalized = str(tech).strip()
                
                if tech_normalized not in dict_tech:
                    t = classes.Technology(name=tech_normalized)
                    dict_tech[tech_normalized] = t
                    B.add_node(tech_normalized, bipartite=1)
                
                # Ajouter l'arête
                B.add_edge(comp_name, tech_normalized)
            
            if not valid_techs:
                companies_without_tech.append(comp_name)
        
        else:
            # Cas où category_groups n'est pas une liste (normalement ne devrait pas arriver)
            tech_str = str(categories).strip()
            if not normalized_keywords or tech_str.lower() in normalized_keywords:
                if tech_str not in dict_tech:
                    t = classes.Technology(name=tech_str)
                    dict_tech[tech_str] = t
                    B.add_node(tech_str, bipartite=1)
                
                B.add_edge(comp_name, tech_str)
            else:
                companies_without_tech.append(comp_name)
    
    # Rapport
    print(f"✓ Graphe créé :")
    print(f"   - Entreprises : {len(dict_companies)}")
    print(f"   - Technologies (filtrées) : {len(dict_tech)}")
    print(f"   - Arêtes : {B.number_of_edges()}")
    
    if companies_without_tech:
        print(f"\n⚠️  {len(companies_without_tech)} entreprises sans technologie valide")
        print(f"   Exemples : {companies_without_tech[:3]}")

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

    if nx.is_bipartite(B):
        print("✓ Graphe confirmé bipartite")
    else:
        print("❌ Graphe non bipartite")
    
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
                # ✅ PASSER LES KEYWORDS à la fonction
        dict_companies, dict_tech, B = extract_classes_company_tech_all(df_comp_filter,keywords=CYBERSECURITY_KEYWORDS)

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
        save_graph_and_dicts(B, df_comp_filter, dict_companies, dict_tech, limit, FLAG_CYBERSECURITY)




if __name__ == "__main__":
    main()
