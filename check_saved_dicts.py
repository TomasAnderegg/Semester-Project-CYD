"""
Script simple pour vérifier les dictionnaires sauvegardés
"""

import pickle
from pathlib import Path

dict_comp_file = "savings/bipartite_invest_comp/classes/dict_companies_1239.pickle"
dict_inv_file = "savings/bipartite_invest_comp/classes/dict_investors_1239.pickle"

print("="*70)
print("VÉRIFICATION DES DICTIONNAIRES SAUVEGARDÉS")
print("="*70)

if Path(dict_comp_file).exists():
    with open(dict_comp_file, 'rb') as f:
        dict_companies = pickle.load(f)
    print(f"\ndict_companies: {len(dict_companies)} entries")
    print(f"  Exemples de cles:")
    for i, (key, value) in enumerate(list(dict_companies.items())[:5]):
        print(f"    {i+1}. '{key}'")
else:
    print(f"\nFichier non trouve: {dict_comp_file}")

if Path(dict_inv_file).exists():
    with open(dict_inv_file, 'rb') as f:
        dict_investors = pickle.load(f)
    print(f"\ndict_investors: {len(dict_investors)} entries")
    print(f"  Exemples de cles:")
    for i, (key, value) in enumerate(list(dict_investors.items())[:5]):
        print(f"    {i+1}. '{key}'")
else:
    print(f"\nFichier non trouve: {dict_inv_file}")

# Vérifier un graphe réel (celui créé dans train/val/test)
test_graph = "data/data_split/bipartite_investor_comp_test.gpickle"
if Path(test_graph).exists():
    print(f"\n{'='*70}")
    print(f"VÉRIFICATION DU GRAPHE TEST ORIGINAL")
    print('='*70)

    # On ne peut pas charger sans networkx, mais on peut au moins confirmer qu'il existe
    print(f"\n✓ Graphe test trouvé: {test_graph}")
    print(f"  Taille: {Path(test_graph).stat().st_size / 1024:.2f} KB")
