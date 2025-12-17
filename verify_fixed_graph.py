"""
Script de vérification pour confirmer que les self-loops sont éliminés
Exécuter APRÈS avoir lancé TGN_eval.py
"""

import pickle
from pathlib import Path
import networkx as nx

print("="*70)
print("VÉRIFICATION DU GRAPHE APRÈS FIX DES PRÉFIXES")
print("="*70)

# Chercher le fichier graphe le plus récent
graph_dir = Path("savings/bipartite_invest_comp/networks")
if not graph_dir.exists():
    print(f"\n❌ Répertoire non trouvé: {graph_dir}")
    exit(1)

graph_files = list(graph_dir.glob("bipartite_graph_*.gpickle"))
if not graph_files:
    print(f"\n❌ Aucun fichier graphe trouvé dans {graph_dir}")
    exit(1)

# Prendre le plus récent
latest_graph = max(graph_files, key=lambda p: p.stat().st_mtime)
print(f"\n📂 Graphe analysé: {latest_graph.name}")

try:
    with open(latest_graph, 'rb') as f:
        B = pickle.load(f)

    print(f"\n📊 Statistiques du graphe:")
    print(f"   Nœuds: {B.number_of_nodes()}")
    print(f"   Arêtes: {B.number_of_edges()}")

    # Extraire nodes par bipartite
    nodes_bip0 = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
    nodes_bip1 = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]

    print(f"\n   Nodes bipartite=0 (companies): {len(nodes_bip0)}")
    print(f"   Nodes bipartite=1 (investors): {len(nodes_bip1)}")

    # Vérifier les préfixes
    print(f"\n🔍 Vérification des préfixes:")
    company_prefix_count = sum(1 for n in nodes_bip0 if str(n).startswith("COMPANY_"))
    investor_prefix_count = sum(1 for n in nodes_bip1 if str(n).startswith("INVESTOR_"))

    print(f"   Nodes avec COMPANY_ prefix: {company_prefix_count}/{len(nodes_bip0)}")
    print(f"   Nodes avec INVESTOR_ prefix: {investor_prefix_count}/{len(nodes_bip1)}")

    # Exemples de noms
    print(f"\n📋 Exemples de noms de nœuds:")
    print(f"   Companies (bip=0): {nodes_bip0[:3]}")
    print(f"   Investors (bip=1): {nodes_bip1[:3]}")

    # Vérifier les self-loops
    print(f"\n🔍 Vérification des self-loops:")
    self_loops = list(nx.selfloop_edges(B))
    print(f"   Self-loops trouvés: {len(self_loops)}")
    if self_loops:
        print(f"   ❌ PROBLÈME: Il reste des self-loops!")
        for u, v in self_loops[:5]:
            print(f"      {u} → {v}")
    else:
        print(f"   ✅ OK: Aucun self-loop")

    # Vérifier la structure bipartite
    print(f"\n🔍 Vérification de la structure bipartite:")
    invalid_edges = []
    for u, v in B.edges():
        u_bip = B.nodes[u].get('bipartite')
        v_bip = B.nodes[v].get('bipartite')
        if u_bip == v_bip:
            invalid_edges.append((u, v, u_bip))

    print(f"   Arêtes invalides (même bipartite): {len(invalid_edges)}")
    if invalid_edges:
        print(f"   ❌ PROBLÈME: Certaines arêtes connectent le même type!")
        for u, v, bip in invalid_edges[:5]:
            print(f"      {u} (bip={bip}) → {v} (bip={bip})")
    else:
        print(f"   ✅ OK: Toutes les arêtes respectent la structure bipartite")

    # Analyser quelques arêtes
    print(f"\n📋 Exemples d'arêtes:")
    for i, (u, v) in enumerate(B.edges()):
        if i >= 5:
            break
        u_bip = B.nodes[u].get('bipartite')
        v_bip = B.nodes[v].get('bipartite')
        print(f"   {u[:50]}... (bip={u_bip}) → {v[:50]}... (bip={v_bip})")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION:")
    print('='*70)

    if len(self_loops) == 0 and len(invalid_edges) == 0:
        print("✅ SUCCÈS: Le graphe est valide!")
        print("   - Aucun self-loop")
        print("   - Structure bipartite respectée")
        print("   - Les préfixes sont correctement appliqués")
    else:
        print("❌ PROBLÈME: Le graphe contient encore des erreurs")
        if self_loops:
            print(f"   - {len(self_loops)} self-loops détectés")
        if invalid_edges:
            print(f"   - {len(invalid_edges)} arêtes invalides")

except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
