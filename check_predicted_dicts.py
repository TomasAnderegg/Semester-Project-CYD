"""
Script pour vérifier le contenu des dictionnaires sauvés par TGN_eval.py
"""

import pickle
from pathlib import Path

print("="*70)
print("VÉRIFICATION DES DICTIONNAIRES PRÉDITS")
print("="*70)

dict_comp_file = "dict_companies_crunchbase.pickle"
dict_inv_file = "dict_investors_crunchbase.pickle"

if Path(dict_comp_file).exists():
    with open(dict_comp_file, 'rb') as f:
        dict_companies = pickle.load(f)

    print(f"\n📊 dict_companies_crunchbase.pickle:")
    print(f"   Total entries: {len(dict_companies)}")
    print(f"\n   Premiers 10 noms:")
    for i, (key, value) in enumerate(list(dict_companies.items())[:10]):
        print(f"      {i+1:2d}. {key}")

    # Analyser les préfixes
    with_company_prefix = sum(1 for key in dict_companies.keys() if str(key).startswith("COMPANY_"))
    with_investor_prefix = sum(1 for key in dict_companies.keys() if str(key).startswith("INVESTOR_"))

    print(f"\n   Analyse des préfixes:")
    print(f"      Avec COMPANY_:  {with_company_prefix}/{len(dict_companies)} ({with_company_prefix/len(dict_companies)*100:.1f}%)")
    print(f"      Avec INVESTOR_: {with_investor_prefix}/{len(dict_companies)} ({with_investor_prefix/len(dict_companies)*100:.1f}%)")

    # Chercher des noms suspects (Ventures, Capital, etc.)
    suspicious = [name for name in dict_companies.keys() if any(x in str(name) for x in ["Ventures", "Capital", "Fund", "Investment"])]
    if suspicious:
        print(f"\n   ⚠️  Noms suspects dans dict_companies (possibles investors):")
        for name in suspicious[:5]:
            print(f"      - {name}")
        if len(suspicious) > 5:
            print(f"      ... et {len(suspicious)-5} autres")
else:
    print(f"\n❌ Fichier non trouvé: {dict_comp_file}")

print("\n" + "="*70)

if Path(dict_inv_file).exists():
    with open(dict_inv_file, 'rb') as f:
        dict_investors = pickle.load(f)

    print(f"\n📊 dict_investors_crunchbase.pickle:")
    print(f"   Total entries: {len(dict_investors)}")
    print(f"\n   Premiers 10 noms:")
    for i, (key, value) in enumerate(list(dict_investors.items())[:10]):
        print(f"      {i+1:2d}. {key}")

    # Analyser les préfixes
    with_company_prefix = sum(1 for key in dict_investors.keys() if str(key).startswith("COMPANY_"))
    with_investor_prefix = sum(1 for key in dict_investors.keys() if str(key).startswith("INVESTOR_"))

    print(f"\n   Analyse des préfixes:")
    print(f"      Avec COMPANY_:  {with_company_prefix}/{len(dict_investors)} ({with_company_prefix/len(dict_investors)*100:.1f}%)")
    print(f"      Avec INVESTOR_: {with_investor_prefix}/{len(dict_investors)} ({with_investor_prefix/len(dict_investors)*100:.1f}%)")

    # Chercher des noms suspects (pas de Ventures/Capital/Fund)
    non_investor_like = [name for name in dict_investors.keys()
                         if not any(x in str(name) for x in ["Ventures", "Capital", "Fund", "Investment", "Partners", "Group", "INVESTOR_"])]
    if non_investor_like:
        print(f"\n   ⚠️  Noms suspects dans dict_investors (possibles companies):")
        for name in non_investor_like[:5]:
            print(f"      - {name}")
        if len(non_investor_like) > 5:
            print(f"      ... et {len(non_investor_like)-5} autres")
else:
    print(f"\n❌ Fichier non trouvé: {dict_inv_file}")

print("\n" + "="*70)
print("FIN DE LA VÉRIFICATION")
print("="*70)
