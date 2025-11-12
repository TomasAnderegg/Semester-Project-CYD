import pandas as pd
import pickle
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import powerlaw
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
import warnings
from networkx.algorithms.bipartite import is_bipartite
from typing import List
warnings.filterwarnings('ignore')

# ===================================================================
# CONFIGURATION
# ===================================================================

NUM_COMP = 500
NUM_TECH = 500
FLAG_CYBERSECURITY = True

SAVE_DIR_CLASSES = "savings/bipartite_invest_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_invest_comp/networks"
SAVE_DIR_M = "savings/bipartite_tech_comp/M"
SAVE_DIR_ANALYSIS = "analysis/graph_quality"

# ===================================================================
# CHARGEMENT DES DONNÉES
# ===================================================================

def load_graph_and_matrix(num_comp, num_tech, flag_cybersecurity):
    """Charge le graphe et la matrice d'adjacence"""
    prefix = "cybersecurity_" if flag_cybersecurity else ""
    
    # Charger le graphe
    # graph_path = f'{SAVE_DIR_NETWORKS}/{prefix}bipartite_graph_{num_comp}.gpickle'
    graph_path = f'{SAVE_DIR_NETWORKS}/bipartite_graph_{num_comp}.gpickle'
    with open(graph_path, 'rb') as f:
        B = pickle.load(f)
    
    # Charger la matrice
    # matrix_path = f'{SAVE_DIR_M}/{prefix}comp_{num_comp}_tech_{num_tech}.npy'
    # M = np.load(matrix_path)
    M = create_biadjacency_matrix(B)
    
    # Charger les dictionnaires
    # with open(f'{SAVE_DIR_CLASSES}/{prefix}dict_companies_ranked_{num_comp}.pickle', 'rb') as f:
    #     dict_companies = pickle.load(f)
    
    # with open(f'{SAVE_DIR_CLASSES}/{prefix}dict_tech_ranked_{num_tech}.pickle', 'rb') as f:
    #     dict_tech = pickle.load(f)

    with open(f'{SAVE_DIR_CLASSES}/dict_companies_{num_comp}.pickle', 'rb') as f:
        dict_companies = pickle.load(f)
    
    with open(f'{SAVE_DIR_CLASSES}/dict_investors_{num_tech}.pickle', 'rb') as f:
        dict_tech = pickle.load(f)
    
    print(f"✓ Données chargées:")
    print(f"  - Graphe: {B.number_of_nodes()} nœuds, {B.number_of_edges()} arêtes")
    # print(f"  - Matrice: {M.shape}")
    print(f"  - Dictionnaires: {len(dict_companies)} companies, {len(dict_tech)} technologies")
    
    return B, M, dict_companies, dict_tech


# ===================================================================
# ANALYSES STRUCTURELLES DU GRAPHE
# ===================================================================
def extract_nodes(G, bipartite_set) -> List:
    """Extract nodes from the nodes of one of the bipartite sets

    Args:
        - G: graph
        - bipartite_set: select one of the two sets (0 or 1)

    Return:
        - nodes: list of nodes of that set
    """

    nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == bipartite_set]

    return nodes

def create_biadjacency_matrix(B):    
    set0 = extract_nodes(B, 0)
    set1 = extract_nodes(B, 1)
    
    # Vérifier que l'ordre est cohérent
    print(f"Vérification: {len(set0)} companies, {len(set1)} technologies")
    
    # Créer un mapping des noms aux indices
    company_to_idx = {company: i for i, company in enumerate(set0)}
    tech_to_idx = {tech: i for i, tech in enumerate(set1)}
    
    # Construire la matrice manuellement pour vérifier
    M_manual = np.zeros((len(set0), len(set1)), dtype=int)
    
    edges_counted = 0
    for u, v in B.edges():
        if u in company_to_idx and v in tech_to_idx:
            i = company_to_idx[u]
            j = tech_to_idx[v]
            M_manual[i, j] = 1
            edges_counted += 1
        elif v in company_to_idx and u in tech_to_idx:
            i = company_to_idx[v]
            j = tech_to_idx[u]
            M_manual[i, j] = 1
            edges_counted += 1
    
    print(f"Arêtes comptées manuellement: {edges_counted}")
    print(f"Arêtes dans le graphe: {B.number_of_edges()}")
    
    return M_manual

def analyze_graph_structure(B):
    """Analyse détaillée de la structure du graphe bipartite"""
    print("\n" + "="*70)
    print("ANALYSE STRUCTURELLE DU GRAPHE")
    print("="*70)
    
    companies = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
    techs = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]
    
    print(f"\n📊 COMPOSITION:")
    print(f"  - Companies: {len(companies)}")
    print(f"  - Technologies: {len(techs)}")
    print(f"  - Arêtes: {B.number_of_edges()}")
    print(f"  - Densité: {nx.density(B):.4f}")
    
    # Degrés
    company_degrees = [B.degree(node) for node in companies]
    tech_degrees = [B.degree(node) for node in techs]
    
    print(f"\n📈 DEGRÉS:")
    print(f"  Companies:")
    print(f"    - Moyen: {np.mean(company_degrees):.2f}")
    print(f"    - Médian: {np.median(company_degrees):.2f}")
    print(f"    - Min/Max: {np.min(company_degrees)}/{np.max(company_degrees)}")
    print(f"    - Écart-type: {np.std(company_degrees):.2f}")
    
    print(f"  Technologies:")
    print(f"    - Moyen: {np.mean(tech_degrees):.2f}")
    print(f"    - Médian: {np.median(tech_degrees):.2f}")
    print(f"    - Min/Max: {np.min(tech_degrees)}/{np.max(tech_degrees)}")
    print(f"    - Écart-type: {np.std(tech_degrees):.2f}")
    
    # Nœuds isolés
    isolated = list(nx.isolates(B))
    print(f"\n⚠️  NŒUDS ISOLÉS: {len(isolated)}")
    if isolated:
        print(f"  Exemples: {isolated[:5]}")
    
    # Composantes connexes
    components = list(nx.connected_components(B))
    print(f"\n🔗 COMPOSANTES CONNEXES: {len(components)}")
    
    if len(components) > 1:
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        print(f"  Tailles: {comp_sizes[:10]}")
        largest = max(components, key=len)
        print(f"  Plus grande composante: {len(largest)} nœuds ({len(largest)/B.number_of_nodes()*100:.1f}%)")
    
    return {
        'companies': companies,
        'techs': techs,
        'company_degrees': company_degrees,
        'tech_degrees': tech_degrees,
        'components': components
    }


def analyze_matrix_properties(M):
    """Analyse détaillée de la matrice d'adjacence"""
    print("\n" + "="*70)
    print("ANALYSE DE LA MATRICE D'ADJACENCE")
    print("="*70)
    
    print(f"\n📐 DIMENSIONS:")
    print(f"  - Shape: {M.shape}")
    print(f"  - Companies (lignes): {M.shape[0]}")
    print(f"  - Technologies (colonnes): {M.shape[1]}")
    
    print(f"\n🔢 STATISTIQUES:")
    print(f"  - Somme totale: {np.sum(M):.0f}")
    print(f"  - Densité: {np.sum(M) / (M.shape[0] * M.shape[1]):.4f}")
    print(f"  - Valeurs uniques: {np.unique(M)}")
    
    # Distribution des connexions
    row_sums = np.sum(M, axis=1)
    col_sums = np.sum(M, axis=0)
    
    print(f"\n📊 DISTRIBUTION DES CONNEXIONS:")
    print(f"  Par company (lignes):")
    print(f"    - Moyenne: {np.mean(row_sums):.2f}")
    print(f"    - Médiane: {np.median(row_sums):.2f}")
    print(f"    - Min/Max: {np.min(row_sums):.0f}/{np.max(row_sums):.0f}")
    print(f"    - Companies sans connexion: {np.sum(row_sums == 0)}")
    
    print(f"  Par technologie (colonnes):")
    print(f"    - Moyenne: {np.mean(col_sums):.2f}")
    print(f"    - Médiane: {np.median(col_sums):.2f}")
    print(f"    - Min/Max: {np.min(col_sums):.0f}/{np.max(col_sums):.0f}")
    print(f"    - Technologies sans connexion: {np.sum(col_sums == 0)}")
    
    # Vérifications de cohérence
    print(f"\n✅ VÉRIFICATIONS:")
    print(f"  - Matrice binaire: {np.all((M == 0) | (M == 1))}")
    print(f"  - Pas de NaN: {not np.any(np.isnan(M))}")
    print(f"  - Pas d'infini: {not np.any(np.isinf(M))}")
    
    return {
        'row_sums': row_sums,
        'col_sums': col_sums
    }


def check_matrix_graph_consistency(B, M):
    """Vérifie la cohérence entre le graphe et la matrice"""
    print("\n" + "="*70)
    print("VÉRIFICATION COHÉRENCE GRAPHE ↔ MATRICE")
    print("="*70)
    
    companies = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
    techs = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]
    
    # Comparer les dimensions
    print(f"\n📏 DIMENSIONS:")
    print(f"  Graphe: {len(companies)} companies × {len(techs)} technologies")
    print(f"  Matrice: {M.shape[0]} × {M.shape[1]}")
    
    dim_match = (len(companies) == M.shape[0]) and (len(techs) == M.shape[1])
    print(f"  ✓ Dimensions cohérentes: {dim_match}")
    
    # Comparer le nombre d'arêtes
    edges_graph = B.number_of_edges()
    edges_matrix = int(np.sum(M))
    
    print(f"\n🔗 ARÊTES:")
    print(f"  Graphe: {edges_graph}")
    print(f"  Matrice: {edges_matrix}")
    print(f"  Différence: {abs(edges_graph - edges_matrix)}")
    
    edges_match = (edges_graph == edges_matrix)
    print(f"  ✓ Nombre d'arêtes cohérent: {edges_match}")
    
    if edges_match and dim_match:
        print(f"\n✅ GRAPHE ET MATRICE SONT COHÉRENTS")
    else:
        print(f"\n⚠️  INCOHÉRENCE DÉTECTÉE!")
    
    return dim_match and edges_match


# ===================================================================
# VISUALISATIONS DU GRAPHE - NOUVELLES FONCTIONS
# ===================================================================

def filter_nodes_by_degree(G, percentage=10, set1=None, set2=None):
    """Filtre les nœuds avec faible degré"""
    if set1 is None:
        set1 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 0]
    if set2 is None:
        set2 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 1]
    
    # Calculer les degrés
    company_degrees = dict(G.degree(set1))
    tech_degrees = dict(G.degree(set2))
    
    # Trouver les seuils
    comp_threshold = np.percentile(list(company_degrees.values()), percentage)
    tech_threshold = np.percentile(list(tech_degrees.values()), percentage)
    
    # Nœuds à supprimer
    to_delete = []
    to_delete.extend([node for node in set1 if company_degrees[node] <= comp_threshold])
    to_delete.extend([node for node in set2 if tech_degrees[node] <= tech_threshold])
    
    print(f"Filtrage: suppression de {len(to_delete)} nœuds (degré < {percentage}ème percentile)")
    return to_delete

def plot_bipartite_graph(G, small_degree=True, percentage=10, circular=False, figsize=(20, 15)):
    """Plot le graphe bipartite avec options de filtrage"""
    
    set1 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 0]
    set2 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 1]

    if not small_degree:  # Filtrer les nœuds avec faible degré
        to_delete = filter_nodes_by_degree(G, percentage, set1, set2)
        
        G_filtered = G.copy()
        G_filtered.remove_nodes_from(to_delete)
        G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))
        
        print(f"Graphe filtré: {G_filtered.number_of_nodes()} nœuds, {G_filtered.number_of_edges()} arêtes")
        G = G_filtered
        set1 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 0]
        set2 = [node for node in G.nodes() if G.nodes[node]['bipartite'] == 1]

    if circular:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=3/np.sqrt(G.number_of_nodes()), iterations=50)

    # Créer la figure avec une taille adaptative
    plt.figure(figsize=figsize)
    plt.axis('off')

    # Calculer les degrés pour la taille des nœuds
    company_degree = dict(G.degree(set1))
    tech_degree = dict(G.degree(set2))

    # Nœuds - Companies (rouge)
    nx.draw_networkx_nodes(G, pos, nodelist=set1,
                          node_color='red', node_size=[v * 50 + 100 for v in company_degree.values()],
                          alpha=0.7, edgecolors='darkred', linewidths=1)

    # Nœuds - Technologies (bleu)
    nx.draw_networkx_nodes(G, pos, nodelist=set2,
                          node_color='blue', node_size=[v * 30 + 100 for v in tech_degree.values()],
                          alpha=0.7, edgecolors='darkblue', linewidths=1)

    # Labels - seulement pour les nœuds importants
    important_companies = [node for node in set1 if company_degree[node] > np.percentile(list(company_degree.values()), 70)]
    important_techs = [node for node in set2 if tech_degree[node] > np.percentile(list(tech_degree.values()), 70)]
    
    nx.draw_networkx_labels(G, pos, {n: n for n in important_companies}, 
                           font_size=8, font_color='darkred', font_weight='bold')
    nx.draw_networkx_labels(G, pos, {n: n for n in important_techs}, 
                           font_size=8, font_color='darkblue', font_weight='bold')

    # Arêtes
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')

    # Légende
    plt.legend(['Companies', 'Technologies'], loc='upper right')

    plt.title(f'Graphe Bipartite Companies-Technologies\n'
              f'{len(set1)} companies, {len(set2)} technologies, {G.number_of_edges()} connexions',
              fontsize=14, pad=20)
    
    plt.tight_layout()
    return pos, G

def plot_interactive_bipartite(B, max_nodes_for_detailed=100):
    """Crée plusieurs visualisations avec différents niveaux de détail"""
    
    total_nodes = B.number_of_nodes()
    
    if total_nodes <= max_nodes_for_detailed:
        # Graphe complet détaillé
        print("\n📊 Visualisation du graphe complet (détaillé)")
        pos, _ = plot_bipartite_graph(B, small_degree=True, circular=False, figsize=(20, 15))
        plt.savefig(f'{SAVE_DIR_ANALYSIS}/bipartite_graph_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        # Graphe filtré (seulement nœuds importants)
        print("\n📊 Visualisation du graphe filtré (nœuds importants seulement)")
        pos, filtered_G = plot_bipartite_graph(B, small_degree=False, percentage=30, circular=False, figsize=(20, 15))
        plt.savefig(f'{SAVE_DIR_ANALYSIS}/bipartite_graph_filtered.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Graphe très filtré pour voir la structure centrale
        print("\n📊 Visualisation du cœur du graphe (nœuds très connectés)")
        pos, core_G = plot_bipartite_graph(B, small_degree=False, percentage=50, circular=False, figsize=(15, 10))
        plt.savefig(f'{SAVE_DIR_ANALYSIS}/bipartite_graph_core.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_zoomable_plot(B, region='center', focus_nodes=None, figsize=(15, 10)):
    """Crée un plot zoomable sur une région spécifique"""
    
    if focus_nodes is None:
        # Sélectionner automatiquement des nœuds focus
        if region == 'center':
            # Nœuds les plus centraux
            centrality = nx.degree_centrality(B)
            focus_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
            focus_nodes = [node for node, _ in focus_nodes]
        elif region == 'high_degree':
            # Nœuds avec haut degré
            degrees = dict(B.degree())
            focus_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
            focus_nodes = [node for node, _ in focus_nodes]
    
    # Créer un sous-graphe avec les nœuds focus et leurs voisins
    neighbors = set()
    for node in focus_nodes:
        neighbors.update(B.neighbors(node))
    
    subgraph_nodes = set(focus_nodes).union(neighbors)
    H = B.subgraph(subgraph_nodes)
    
    print(f"Sous-graphe de zoom: {H.number_of_nodes()} nœuds, {H.number_of_edges()} arêtes")
    
    # Plot du sous-graphe
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(H, k=1.5/np.sqrt(H.number_of_nodes()), iterations=100)
    
    set1 = [node for node in H.nodes() if H.nodes[node]['bipartite'] == 0]
    set2 = [node for node in H.nodes() if H.nodes[node]['bipartite'] == 1]
    
    # Nœuds
    nx.draw_networkx_nodes(H, pos, nodelist=set1, node_color='red', 
                          node_size=500, alpha=0.8, edgecolors='darkred')
    nx.draw_networkx_nodes(H, pos, nodelist=set2, node_color='blue', 
                          node_size=500, alpha=0.8, edgecolors='darkblue')
    
    # Labels pour tous les nœuds (puisque c'est un sous-ensemble)
    nx.draw_networkx_labels(H, pos, font_size=8, font_weight='bold')
    
    # Arêtes
    nx.draw_networkx_edges(H, pos, width=1.0, alpha=0.5, edge_color='gray')
    
    plt.title(f'Zoom sur {region}\n{H.number_of_nodes()} nœuds, {H.number_of_edges()} connexions', 
              fontsize=12)
    plt.axis('on')  # Garder les axes pour le contexte
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR_ANALYSIS}/bipartite_zoom_{region}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return H, pos

# ===================================================================
# VISUALISATIONS EXISTANTES (gardées pour compatibilité)
# ===================================================================

def plot_degree_distributions(graph_data, matrix_data):
    """Visualise les distributions de degrés"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution des degrés - Companies
    axes[0, 0].hist(graph_data['company_degrees'], bins=30, edgecolor='black', alpha=0.7, color='red')
    axes[0, 0].set_xlabel('Degré')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution des degrés - Companies')
    axes[0, 0].axvline(np.mean(graph_data['company_degrees']), color='black', 
                       linestyle='--', label=f'Moyenne: {np.mean(graph_data["company_degrees"]):.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution des degrés - Technologies
    axes[0, 1].hist(graph_data['tech_degrees'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 1].set_xlabel('Degré')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des degrés - Technologies')
    axes[0, 1].axvline(np.mean(graph_data['tech_degrees']), color='black', 
                       linestyle='--', label=f'Moyenne: {np.mean(graph_data["tech_degrees"]):.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log-log plot - Companies
    hist, bins = np.histogram(graph_data['company_degrees'], bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    axes[1, 0].loglog(bin_centers, hist, 'ro-', alpha=0.6)
    axes[1, 0].set_xlabel('Degré (log)')
    axes[1, 0].set_ylabel('Fréquence (log)')
    axes[1, 0].set_title('Distribution log-log - Companies')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log-log plot - Technologies
    hist, bins = np.histogram(graph_data['tech_degrees'], bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    axes[1, 1].loglog(bin_centers, hist, 'bo-', alpha=0.6)
    axes[1, 1].set_xlabel('Degré (log)')
    axes[1, 1].set_ylabel('Fréquence (log)')
    axes[1, 1].set_title('Distribution log-log - Technologies')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR_ANALYSIS}/degree_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {SAVE_DIR_ANALYSIS}/degree_distributions.png")
    plt.show()

def plot_matrix_visualization(M):
    """Visualise la matrice d'adjacence triée"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Matrice brute
    axes[0].imshow(M, cmap='bone', interpolation='nearest', aspect='auto')
    axes[0].set_xlabel('Technologies')
    axes[0].set_ylabel('Companies')
    axes[0].set_title('Matrice d\'adjacence (brute)')
    
    # Matrice triée (triangularité)
    row_sums = np.sum(M, axis=1)
    col_sums = np.sum(M, axis=0)
    
    row_order = np.argsort(row_sums)
    col_order = np.argsort(col_sums)
    
    M_sorted = M[row_order, :][:, col_order]
    
    axes[1].imshow(M_sorted, cmap='bone', interpolation='nearest', aspect='auto')
    axes[1].set_xlabel('Technologies (triées)')
    axes[1].set_ylabel('Companies (triées)')
    axes[1].set_title('Matrice d\'adjacence (triée par degré)')
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR_ANALYSIS}/matrix_visualization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {SAVE_DIR_ANALYSIS}/matrix_visualization.png")
    plt.show()

def plot_connectivity_heatmap(matrix_data):
    """Heatmap de la connectivité"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribution des connexions par company
    row_sums = matrix_data['row_sums']
    bins_comp = np.linspace(0, np.max(row_sums), 20)
    axes[0].hist(row_sums, bins=bins_comp, edgecolor='black', alpha=0.7, color='red')
    axes[0].set_xlabel('Nombre de technologies')
    axes[0].set_ylabel('Nombre de companies')
    axes[0].set_title('Connexions par company')
    axes[0].axvline(np.mean(row_sums), color='black', linestyle='--', 
                    label=f'Moyenne: {np.mean(row_sums):.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution des connexions par technologie
    col_sums = matrix_data['col_sums']
    bins_tech = np.linspace(0, np.max(col_sums), 20)
    axes[1].hist(col_sums, bins=bins_tech, edgecolor='black', alpha=0.7, color='blue')
    axes[1].set_xlabel('Nombre de companies')
    axes[1].set_ylabel('Nombre de technologies')
    axes[1].set_title('Connexions par technologie')
    axes[1].axvline(np.mean(col_sums), color='black', linestyle='--', 
                    label=f'Moyenne: {np.mean(col_sums):.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR_ANALYSIS}/connectivity_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {SAVE_DIR_ANALYSIS}/connectivity_heatmap.png")
    plt.show()

# ===================================================================
# ANALYSE POUR TECHRANK
# ===================================================================

def assess_techrank_readiness(B, M, graph_data, matrix_data):
    """Évalue si le graphe est prêt pour TechRank"""
    print("\n" + "="*70)
    print("ÉVALUATION DE LA PRÉPARATION POUR TECHRANK")
    print("="*70)
    
    issues = []
    warnings_list = []
    
    # 1. Vérifier les nœuds isolés
    isolated = list(nx.isolates(B))
    if len(isolated) > 0:
        issues.append(f"⚠️  {len(isolated)} nœuds isolés détectés")
    
    # 2. Vérifier les companies sans connexion
    row_sums = matrix_data['row_sums']
    zero_degree_companies = np.sum(row_sums == 0)
    if zero_degree_companies > 0:
        issues.append(f"⚠️  {zero_degree_companies} companies sans connexion")
    
    # 3. Vérifier les technologies sans connexion
    col_sums = matrix_data['col_sums']
    zero_degree_techs = np.sum(col_sums == 0)
    if zero_degree_techs > 0:
        issues.append(f"⚠️  {zero_degree_techs} technologies sans connexion")
    
    # 4. Vérifier la composante connexe principale
    components = graph_data['components']
    if len(components) > 1:
        largest_comp_size = len(max(components, key=len))
        coverage = largest_comp_size / B.number_of_nodes() * 100
        if coverage < 90:
            warnings_list.append(f"⚠️  Composante principale couvre seulement {coverage:.1f}% du graphe")
    
    # 5. Vérifier la densité
    density = nx.density(B)
    if density < 0.001:
        warnings_list.append(f"⚠️  Graphe très sparse (densité: {density:.6f})")
    
    # 6. Vérifier la distribution des degrés
    company_degrees = graph_data['company_degrees']
    tech_degrees = graph_data['tech_degrees']
    
    if np.std(company_degrees) / np.mean(company_degrees) > 2:
        warnings_list.append(f"⚠️  Forte hétérogénéité des degrés companies (CV: {np.std(company_degrees) / np.mean(company_degrees):.2f})")
    
    if np.std(tech_degrees) / np.mean(tech_degrees) > 2:
        warnings_list.append(f"⚠️  Forte hétérogénéité des degrés technologies (CV: {np.std(tech_degrees) / np.mean(tech_degrees):.2f})")
    
    # Afficher les résultats
    if not issues and not warnings_list:
        print("\n✅ GRAPHE PRÊT POUR TECHRANK")
        print("  Aucun problème critique détecté")
    else:
        if issues:
            print("\n❌ PROBLÈMES CRITIQUES:")
            for issue in issues:
                print(f"  {issue}")
        
        if warnings_list:
            print("\n⚠️  AVERTISSEMENTS:")
            for warning in warnings_list:
                print(f"  {warning}")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS:")
    if zero_degree_companies > 0 or zero_degree_techs > 0:
        print("  1. Supprimer les nœuds sans connexion avant d'appliquer TechRank")
    
    if len(components) > 1:
        print("  2. Considérer d'analyser uniquement la composante connexe principale")
    
    if density < 0.001:
        print("  3. Le graphe sparse peut nécessiter plus d'itérations pour converger")
    
    return len(issues) == 0

# ===================================================================
# RAPPORT COMPLET
# ===================================================================

def generate_analysis_report(B, M, dict_companies, dict_tech):
    """Génère un rapport d'analyse complet"""
    Path(SAVE_DIR_ANALYSIS).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GÉNÉRATION DU RAPPORT D'ANALYSE")
    print("="*70)
    
    # 1. Analyse structurelle
    graph_data = analyze_graph_structure(B)
    
    # 2. Analyse de la matrice
    matrix_data = analyze_matrix_properties(M)
    
    # 3. Vérification de cohérence
    is_consistent = check_matrix_graph_consistency(B, M)
    
    # 4. Visualisations
    print("\n📊 Génération des visualisations...")
    plot_degree_distributions(graph_data, matrix_data)
    plot_matrix_visualization(M)
    plot_connectivity_heatmap(matrix_data)
    
    # 5. NOUVEAU: Visualisations du graphe
    print("\n🎨 Génération des visualisations du graphe...")
    plot_interactive_bipartite(B)
    
    # Zoom sur les régions intéressantes
    print("\n🔍 Génération des vues zoomées...")
    create_zoomable_plot(B, region='high_degree')
    create_zoomable_plot(B, region='center')
    
    # 6. Évaluation TechRank
    is_ready = assess_techrank_readiness(B, M, graph_data, matrix_data)
    
    # 7. Sauvegarder un résumé textuel
    summary_path = f'{SAVE_DIR_ANALYSIS}/analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT D'ANALYSE DU GRAPHE BIPARTITE\n")
        f.write("="*70 + "\n\n")
        
        f.write("COMPOSITION:\n")
        f.write(f"  - Companies: {len(graph_data['companies'])}\n")
        f.write(f"  - Technologies: {len(graph_data['techs'])}\n")
        f.write(f"  - Arêtes: {B.number_of_edges()}\n")
        f.write(f"  - Densité: {nx.density(B):.6f}\n\n")
        
        f.write("DEGRÉS MOYENS:\n")
        f.write(f"  - Companies: {np.mean(graph_data['company_degrees']):.2f}\n")
        f.write(f"  - Technologies: {np.mean(graph_data['tech_degrees']):.2f}\n\n")
        
        f.write("COHÉRENCE GRAPHE-MATRICE:\n")
        f.write(f"  - Cohérent: {'OUI' if is_consistent else 'NON'}\n\n")
        
        f.write("PRÊT POUR TECHRANK:\n")
        f.write(f"  - {'OUI' if is_ready else 'NON - voir avertissements ci-dessus'}\n")
    
    print(f"\n✓ Résumé sauvegardé: {summary_path}")
    print(f"\n✅ ANALYSE COMPLÈTE TERMINÉE")
    print(f"   Tous les résultats sont dans: {SAVE_DIR_ANALYSIS}/")

    # 8. Bipartite validation
    print("bipartite validation:", is_bipartite(B))

# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    # Charger les données
    B, M, dict_companies, dict_tech = load_graph_and_matrix(
        NUM_COMP, NUM_TECH, FLAG_CYBERSECURITY
    )
    
    # Générer le rapport complet
    generate_analysis_report(B, M, dict_companies, dict_tech)