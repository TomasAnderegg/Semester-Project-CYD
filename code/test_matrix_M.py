import pandas as pd
import pickle
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from typing import List

warnings.filterwarnings('ignore')

# ===================================================================
# CONFIGURATION
# ===================================================================

NUM_COMP = 50000
NUM_INVESTORS = 50000
FLAG_CYBERSECURITY = True

SAVE_DIR_CLASSES = "savings/bipartite_invest_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_invest_comp/networks"
SAVE_DIR_ANALYSIS = "analysis/graph_quality"

# ===================================================================
# DATA LOADING
# ===================================================================

def load_graph_and_matrix(num_comp, num_investors, flag_cybersecurity):

    graph_path = f'{SAVE_DIR_NETWORKS}/bipartite_graph_{num_comp}.gpickle'
    with open(graph_path, 'rb') as f:
        B = pickle.load(f)

    M = create_biadjacency_matrix(B)

    with open(f'{SAVE_DIR_CLASSES}/dict_companies_{num_comp}.pickle', 'rb') as f:
        dict_companies = pickle.load(f)

    with open(f'{SAVE_DIR_CLASSES}/dict_investors_{num_investors}.pickle', 'rb') as f:
        dict_investors = pickle.load(f)

    print("✓ Data loaded:")
    print(f"  - Graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"  - Dictionaries: {len(dict_companies)} companies, {len(dict_investors)} investors")

    return B, M, dict_companies, dict_investors

# ===================================================================
# GRAPH UTILITIES
# ===================================================================

def extract_nodes(G, bipartite_set) -> List:
    return [n for n, d in G.nodes(data=True) if d["bipartite"] == bipartite_set]

def create_biadjacency_matrix(B):

    companies = extract_nodes(B, 0)
    investors = extract_nodes(B, 1)

    print(f"Check: {len(companies)} companies, {len(investors)} investors")

    company_to_idx = {c: i for i, c in enumerate(companies)}
    investor_to_idx = {i: j for j, i in enumerate(investors)}

    M = np.zeros((len(companies), len(investors)), dtype=int)

    edge_count = 0
    for u, v in B.edges():
        if u in company_to_idx and v in investor_to_idx:
            M[company_to_idx[u], investor_to_idx[v]] = 1
            edge_count += 1
        elif v in company_to_idx and u in investor_to_idx:
            M[company_to_idx[v], investor_to_idx[u]] = 1
            edge_count += 1

    print(f"Edges counted manually: {edge_count}")
    print(f"Edges in graph: {B.number_of_edges()}")

    return M

# ===================================================================
# STRUCTURAL ANALYSIS
# ===================================================================

def analyze_graph_structure(B):

    print("\n" + "=" * 70)
    print("GRAPH STRUCTURAL ANALYSIS")
    print("=" * 70)

    companies = extract_nodes(B, 0)
    investors = extract_nodes(B, 1)

    company_degrees = [B.degree(n) for n in companies]
    investor_degrees = [B.degree(n) for n in investors]

    print("\nCOMPOSITION:")
    print(f"  - Companies: {len(companies)}")
    print(f"  - Investors: {len(investors)}")
    print(f"  - Edges: {B.number_of_edges()}")
    print(f"  - Density: {nx.density(B):.6f}")

    print("\nDEGREE STATISTICS:")
    print("  Companies:")
    print(f"    - Mean: {np.mean(company_degrees):.2f}")
    print(f"    - Median: {np.median(company_degrees):.2f}")
    print(f"    - Min / Max: {np.min(company_degrees)} / {np.max(company_degrees)}")

    print("  Investors:")
    print(f"    - Mean: {np.mean(investor_degrees):.2f}")
    print(f"    - Median: {np.median(investor_degrees):.2f}")
    print(f"    - Min / Max: {np.min(investor_degrees)} / {np.max(investor_degrees)}")

    components = list(nx.connected_components(B))

    return {
        "companies": companies,
        "investors": investors,
        "company_degrees": company_degrees,
        "investor_degrees": investor_degrees,
        "components": components
    }

def analyze_matrix_properties(M):

    print("\n" + "=" * 70)
    print("ADJACENCY MATRIX ANALYSIS")
    print("=" * 70)

    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)

    print("\nMATRIX STATISTICS:")
    print(f"  Shape: {M.shape}")
    print(f"  Density: {M.sum() / (M.shape[0] * M.shape[1]):.6f}")

    print("\nConnections per company:")
    print(f"  Mean: {np.mean(row_sums):.2f}")
    print(f"  Median: {np.median(row_sums):.2f}")
    print(f"  Zero-degree companies: {np.sum(row_sums == 0)}")

    print("\nConnections per investor:")
    print(f"  Mean: {np.mean(col_sums):.2f}")
    print(f"  Median: {np.median(col_sums):.2f}")
    print(f"  Zero-degree investors: {np.sum(col_sums == 0)}")

    return {"row_sums": row_sums, "col_sums": col_sums}

# ===================================================================
# VISUALIZATION
# ===================================================================

def plot_matrix_visualization(M):

    row_order = np.argsort(M.sum(axis=1))
    col_order = np.argsort(M.sum(axis=0))
    M_sorted = M[row_order][:, col_order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(M, cmap="binary", aspect="auto")
    axes[0].set_title("Adjacency Matrix (raw)")
    axes[0].set_xlabel("Investors")
    axes[0].set_ylabel("Companies")

    axes[1].imshow(M_sorted, cmap="binary", aspect="auto")
    axes[1].set_title("Adjacency Matrix (sorted by degree)")
    axes[1].set_xlabel("Investors (sorted)")
    axes[1].set_ylabel("Companies (sorted)")

    Path(SAVE_DIR_ANALYSIS).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{SAVE_DIR_ANALYSIS}/adjacency_matrix.png", dpi=300)
    plt.show()

# ===================================================================
# TECHRANK READINESS
# ===================================================================

def assess_techrank_readiness(B, matrix_data, graph_data):

    print("\n" + "=" * 70)
    print("TECHRANK READINESS ASSESSMENT")
    print("=" * 70)

    issues = []

    if len(list(nx.isolates(B))) > 0:
        issues.append("Isolated nodes detected")

    if np.sum(matrix_data["row_sums"] == 0) > 0:
        issues.append("Companies with zero degree detected")

    if np.sum(matrix_data["col_sums"] == 0) > 0:
        issues.append("Investors with zero degree detected")

    if issues:
        print("⚠️ Issues detected:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("✅ Graph is suitable for TechRank")

# ===================================================================
# REPORT PIPELINE
# ===================================================================

def generate_analysis_report(B, M, dict_companies, dict_investors):

    graph_data = analyze_graph_structure(B)
    matrix_data = analyze_matrix_properties(M)
    plot_matrix_visualization(M)
    assess_techrank_readiness(B, matrix_data, graph_data)

    print("\n✓ Analysis completed.")
    print(f"Results saved in {SAVE_DIR_ANALYSIS}/")

# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":

    B, M, dict_companies, dict_investors = load_graph_and_matrix(
        NUM_COMP, NUM_INVESTORS, FLAG_CYBERSECURITY
    )

    generate_analysis_report(B, M, dict_companies, dict_investors)
