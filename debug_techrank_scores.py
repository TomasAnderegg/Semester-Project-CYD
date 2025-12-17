"""
Script de debug pour comprendre pourquoi les scores TechRank sont tous à 0
"""

import pickle
from pathlib import Path

# Charger le graphe et les dictionnaires
graph_file = "savings/bipartite_invest_comp/networks/bipartite_graph_1239.gpickle"
dict_comp_file = "savings/bipartite_invest_comp/classes/dict_companies_1239.pickle"
dict_inv_file = "savings/bipartite_invest_comp/classes/dict_investors_1239.pickle"

print("="*70)
print("DEBUG TECHRANK MAPPING")
print("="*70)

try:
    import networkx as nx
    from networkx.algorithms import bipartite

    with open(graph_file, 'rb') as f:
        B = pickle.load(f)

    with open(dict_comp_file, 'rb') as f:
        dict_companies = pickle.load(f)

    with open(dict_inv_file, 'rb') as f:
        dict_investors = pickle.load(f)

    print(f"\nGraphe: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"dict_companies: {len(dict_companies)} entries")
    print(f"dict_investors: {len(dict_investors)} entries")

    # Extraire les nodes par bipartite
    def extract_nodes(B, bipartite_value):
        return [node for node, data in B.nodes(data=True) if data.get('bipartite') == bipartite_value]

    set0 = extract_nodes(B, 0)  # Companies
    set1 = extract_nodes(B, 1)  # Investors

    print(f"\nset0 (bipartite=0): {len(set0)} nodes")
    print(f"  Premiers: {set0[:5]}")

    print(f"\nset1 (bipartite=1): {len(set1)} nodes")
    print(f"  Premiers: {set1[:5]}")

    # Vérifier la correspondance avec les dictionnaires
    print("\n" + "="*70)
    print("CORRESPONDANCE SET0 <-> DICT_COMPANIES")
    print("="*70)

    set0_in_dict = sum(1 for name in set0 if name in dict_companies)
    dict_in_set0 = sum(1 for name in dict_companies.keys() if name in set0)

    print(f"Nodes de set0 présents dans dict_companies: {set0_in_dict}/{len(set0)}")
    print(f"Entrées de dict_companies présentes dans set0: {dict_in_set0}/{len(dict_companies)}")

    if set0_in_dict != len(set0):
        print("\nNODES MANQUANTS dans dict_companies:")
        missing = [n for n in set0 if n not in dict_companies]
        for i, name in enumerate(missing[:10]):
            print(f"  {i+1}. {name}")
        if len(missing) > 10:
            print(f"  ... et {len(missing)-10} autres")

    if dict_in_set0 != len(dict_companies):
        print("\nENTRÉES MANQUANTES dans set0:")
        missing = [n for n in dict_companies.keys() if n not in set0]
        for i, name in enumerate(missing[:10]):
            print(f"  {i+1}. {name}")
        if len(missing) > 10:
            print(f"  ... et {len(missing)-10} autres")

    print("\n" + "="*70)
    print("CORRESPONDANCE SET1 <-> DICT_INVESTORS")
    print("="*70)

    set1_in_dict = sum(1 for name in set1 if name in dict_investors)
    dict_in_set1 = sum(1 for name in dict_investors.keys() if name in set1)

    print(f"Nodes de set1 présents dans dict_investors: {set1_in_dict}/{len(set1)}")
    print(f"Entrées de dict_investors présentes dans set1: {dict_in_set1}/{len(dict_investors)}")

    if set1_in_dict != len(set1):
        print("\nNODES MANQUANTS dans dict_investors:")
        missing = [n for n in set1 if n not in dict_investors]
        for i, name in enumerate(missing[:10]):
            print(f"  {i+1}. {name}")
        if len(missing) > 10:
            print(f"  ... et {len(missing)-10} autres")

    if dict_in_set1 != len(dict_investors):
        print("\nENTRÉES MANQUANTES dans set1:")
        missing = [n for n in dict_investors.keys() if n not in set1]
        for i, name in enumerate(missing[:10]):
            print(f"  {i+1}. {name}")
        if len(missing) > 10:
            print(f"  ... et {len(missing)-10} autres")

    print("\n" + "="*70)
    print("DIAGNOSTIC")
    print("="*70)

    if set0_in_dict == len(set0) and dict_in_set0 == len(dict_companies):
        print("\nOK: set0 et dict_companies correspondent parfaitement")
    else:
        print("\nPROBLEME: set0 et dict_companies ne correspondent pas!")
        print("  -> Les scores ne peuvent pas etre mappes correctement")

    if set1_in_dict == len(set1) and dict_in_set1 == len(dict_investors):
        print("OK: set1 et dict_investors correspondent parfaitement")
    else:
        print("PROBLEME: set1 et dict_investors ne correspondent pas!")
        print("  -> Les scores ne peuvent pas etre mappes correctement")

    # Vérifier l'ordre
    print("\n" + "="*70)
    print("VERIFICATION DE L'ORDRE")
    print("="*70)

    dict_comp_keys = list(dict_companies.keys())
    dict_inv_keys = list(dict_investors.keys())

    print(f"\nPremiers elements de set0: {set0[:5]}")
    print(f"Premiers elements de dict_companies.keys(): {dict_comp_keys[:5]}")
    print(f"Meme ordre? {set0[:5] == dict_comp_keys[:5]}")

    print(f"\nPremiers elements de set1: {set1[:5]}")
    print(f"Premiers elements de dict_investors.keys(): {dict_inv_keys[:5]}")
    print(f"Meme ordre? {set1[:5] == dict_inv_keys[:5]}")

except ImportError:
    print("NetworkX n'est pas disponible")
except Exception as e:
    print(f"Erreur: {e}")
    import traceback
    traceback.print_exc()
