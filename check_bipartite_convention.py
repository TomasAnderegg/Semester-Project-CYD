"""
Script rapide pour vérifier la convention bipartite dans les graphes existants
"""

import pickle
from pathlib import Path
import networkx as nx

# Charger un graphe existant pour vérifier la convention
graph_files = [
    "savings/bipartite_invest_comp/networks/bipartite_graph_1239.gpickle",
    "savings/bipartite_invest_comp/networks/bipartite_graph_199.gpickle"
]

for graph_file in graph_files:
    if not Path(graph_file).exists():
        continue

    print(f"\n{'='*70}")
    print(f"Analysing: {graph_file}")
    print('='*70)

    with open(graph_file, 'rb') as f:
        B = pickle.load(f)

    # Extraire les nodes par bipartite
    nodes_0 = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
    nodes_1 = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]

    print(f"\nNodes avec bipartite=0: {len(nodes_0)}")
    print(f"  Exemples: {nodes_0[:5]}")

    print(f"\nNodes avec bipartite=1: {len(nodes_1)}")
    print(f"  Exemples: {nodes_1[:5]}")

    # Analyser quelques edges
    print(f"\nAnalyse des edges:")
    for i, (u, v, data) in enumerate(B.edges(data=True)):
        if i >= 5:
            break
        u_bip = B.nodes[u].get('bipartite')
        v_bip = B.nodes[v].get('bipartite')
        print(f"  {u} (bip={u_bip}) → {v} (bip={v_bip})")

    # Charger les dictionnaires correspondants
    num_nodes = B.number_of_nodes()
    dict_comp_file = f"savings/bipartite_invest_comp/classes/dict_companies_{num_nodes}.pickle"
    dict_inv_file = f"savings/bipartite_invest_comp/classes/dict_investors_{num_nodes}.pickle"

    if Path(dict_comp_file).exists() and Path(dict_inv_file).exists():
        with open(dict_comp_file, 'rb') as f:
            dict_companies = pickle.load(f)
        with open(dict_inv_file, 'rb') as f:
            dict_investors = pickle.load(f)

        print(f"\nDictionnaires:")
        print(f"  dict_companies: {len(dict_companies)} entries")
        print(f"    Exemples: {list(dict_companies.keys())[:3]}")
        print(f"  dict_investors: {len(dict_investors)} entries")
        print(f"    Exemples: {list(dict_investors.keys())[:3]}")

        # Vérifier quel dictionnaire correspond à quel bipartite
        comp_in_bip0 = sum(1 for name in list(dict_companies.keys())[:10] if name in nodes_0)
        comp_in_bip1 = sum(1 for name in list(dict_companies.keys())[:10] if name in nodes_1)

        inv_in_bip0 = sum(1 for name in list(dict_investors.keys())[:10] if name in nodes_0)
        inv_in_bip1 = sum(1 for name in list(dict_investors.keys())[:10] if name in nodes_1)

        print(f"\nCorrespondance dict → bipartite (sur 10 échantillons):")
        print(f"  dict_companies dans bip=0: {comp_in_bip0}/10")
        print(f"  dict_companies dans bip=1: {comp_in_bip1}/10")
        print(f"  dict_investors dans bip=0: {inv_in_bip0}/10")
        print(f"  dict_investors dans bip=1: {inv_in_bip1}/10")

        print(f"\n{'='*70}")
        print("CONCLUSION:")
        print('='*70)
        if comp_in_bip0 > comp_in_bip1:
            print("✓ dict_companies correspond à bipartite=0")
        else:
            print("✓ dict_companies correspond à bipartite=1")

        if inv_in_bip0 > inv_in_bip1:
            print("✓ dict_investors correspond à bipartite=0")
        else:
            print("✓ dict_investors correspond à bipartite=1")

        break  # Only analyze first found graph
