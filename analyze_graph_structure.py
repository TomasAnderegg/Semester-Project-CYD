"""
Script pour analyser la structure du graphe prédit et comprendre
pourquoi TechRank donne des scores de 0.0 pour presque tous les investors
"""

import pickle
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# Essayer d'importer networkx, sinon continuer sans
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️  NetworkX not found, using fallback analysis")
    print("   Install with: pip install networkx")
    print()

def analyze_graph_structure(graph_path='predicted_graph_crunchbase.pkl'):
    """Analyse détaillée de la structure du graphe"""

    if not HAS_NETWORKX:
        print("❌ NetworkX is required for this analysis")
        print("   Please install with: pip install networkx")
        return

    print("="*70)
    print("ANALYSE DE LA STRUCTURE DU GRAPHE")
    print("="*70)

    # Charger le graphe
    if not Path(graph_path).exists():
        print(f"❌ Graph file not found: {graph_path}")
        return

    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    print(f"\n📊 Informations de base:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Type: {type(G)}")
    print(f"   Directed: {G.is_directed()}")

    # Analyser les attributs bipartite
    print(f"\n🔍 Analyse bipartite:")
    bipartite_counts = Counter()
    nodes_by_bipartite = defaultdict(list)

    for node in G.nodes():
        bipartite = G.nodes[node].get('bipartite', None)
        bipartite_counts[bipartite] += 1
        nodes_by_bipartite[bipartite].append(node)

    print(f"   Bipartite distribution:")
    for bp_value, count in sorted(bipartite_counts.items()):
        bp_label = "Companies" if bp_value == 0 else "Investors" if bp_value == 1 else "Unknown"
        print(f"      {bp_label} (bipartite={bp_value}): {count}")

    # Analyser les degrés
    print(f"\n📈 Analyse des degrés:")

    if G.is_directed():
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

        print(f"   In-degree stats:")
        print(f"      Min: {min(in_degrees.values())}")
        print(f"      Max: {max(in_degrees.values())}")
        print(f"      Mean: {np.mean(list(in_degrees.values())):.2f}")
        print(f"      Nodes with in-degree=0: {sum(1 for d in in_degrees.values() if d == 0)}")

        print(f"   Out-degree stats:")
        print(f"      Min: {min(out_degrees.values())}")
        print(f"      Max: {max(out_degrees.values())}")
        print(f"      Mean: {np.mean(list(out_degrees.values())):.2f}")
        print(f"      Nodes with out-degree=0: {sum(1 for d in out_degrees.values() if d == 0)}")

        # Analyser par bipartite
        print(f"\n   Degrés par type de node:")
        for bp_value in sorted(nodes_by_bipartite.keys()):
            bp_label = "Companies" if bp_value == 0 else "Investors" if bp_value == 1 else f"Unknown({bp_value})"
            nodes = nodes_by_bipartite[bp_value]

            in_deg = [in_degrees[n] for n in nodes]
            out_deg = [out_degrees[n] for n in nodes]

            print(f"\n      {bp_label}:")
            print(f"         In-degree:  min={min(in_deg)}, max={max(in_deg)}, mean={np.mean(in_deg):.2f}")
            print(f"         Out-degree: min={min(out_deg)}, max={max(out_deg)}, mean={np.mean(out_deg):.2f}")
            print(f"         Nodes with out-degree=0: {sum(1 for d in out_deg if d == 0)}")
            print(f"         Nodes with in-degree=0: {sum(1 for d in in_deg if d == 0)}")
    else:
        degrees = dict(G.degree())
        print(f"   Degree stats:")
        print(f"      Min: {min(degrees.values())}")
        print(f"      Max: {max(degrees.values())}")
        print(f"      Mean: {np.mean(list(degrees.values())):.2f}")
        print(f"      Nodes with degree=0: {sum(1 for d in degrees.values() if d == 0)}")

    # Analyser les arêtes
    print(f"\n🔗 Analyse des arêtes:")
    edge_types = Counter()

    for u, v in G.edges():
        u_bipartite = G.nodes[u].get('bipartite', None)
        v_bipartite = G.nodes[v].get('bipartite', None)
        edge_type = f"{u_bipartite} -> {v_bipartite}"
        edge_types[edge_type] += 1

    print(f"   Types d'arêtes:")
    for edge_type, count in edge_types.most_common():
        print(f"      {edge_type}: {count}")

    # Connectivité
    print(f"\n🌐 Analyse de connectivité:")
    if G.is_directed():
        weakly_connected = list(nx.weakly_connected_components(G))
        print(f"   Weakly connected components: {len(weakly_connected)}")
        print(f"   Largest component size: {len(max(weakly_connected, key=len))}")

        # Vérifier si certains nodes sont complètement isolés
        isolated_nodes = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
        print(f"   Isolated nodes (in=0, out=0): {len(isolated_nodes)}")

        if len(isolated_nodes) > 0 and len(isolated_nodes) < 20:
            print(f"\n   Isolated nodes:")
            for node in isolated_nodes[:10]:
                name = G.nodes[node].get('name', f'Node {node}')
                bipartite = G.nodes[node].get('bipartite', None)
                print(f"      {name} (bipartite={bipartite})")
    else:
        connected = list(nx.connected_components(G))
        print(f"   Connected components: {len(connected)}")
        print(f"   Largest component size: {len(max(connected, key=len))}")

    # Analyser les attributs des arêtes
    print(f"\n📝 Attributs des arêtes:")
    if G.number_of_edges() > 0:
        sample_edge = list(G.edges(data=True))[0]
        print(f"   Sample edge attributes: {sample_edge[2]}")

        # Statistiques sur les poids/probabilités
        if 'probability' in sample_edge[2]:
            probs = [data.get('probability', 0) for u, v, data in G.edges(data=True)]
            print(f"\n   Probability distribution:")
            print(f"      Min: {min(probs):.4f}")
            print(f"      Max: {max(probs):.4f}")
            print(f"      Mean: {np.mean(probs):.4f}")
            print(f"      Median: {np.median(probs):.4f}")

    # Identifier les nodes avec le plus de connexions
    print(f"\n🏆 Top 10 nodes par degré:")
    if G.is_directed():
        # Top par out-degree (ceux qui investissent le plus)
        top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n   Top out-degree (investisseurs actifs):")
        for node, deg in top_out:
            name = G.nodes[node].get('name', f'Node {node}')
            bipartite = G.nodes[node].get('bipartite', None)
            print(f"      {name} (bipartite={bipartite}): out={deg}, in={in_degrees[node]}")

        # Top par in-degree (ceux qui reçoivent le plus)
        top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n   Top in-degree (companies populaires):")
        for node, deg in top_in:
            name = G.nodes[node].get('name', f'Node {node}')
            bipartite = G.nodes[node].get('bipartite', None)
            print(f"      {name} (bipartite={bipartite}): in={deg}, out={out_degrees[node]}")

    # Vérifier Legend Capital spécifiquement
    print(f"\n🔍 Analyse de Legend Capital (score=1.0):")
    legend_node = None
    for node in G.nodes():
        if G.nodes[node].get('name') == 'Legend Capital':
            legend_node = node
            break

    if legend_node is not None:
        print(f"   Node ID: {legend_node}")
        print(f"   Bipartite: {G.nodes[legend_node].get('bipartite')}")
        if G.is_directed():
            print(f"   In-degree: {G.in_degree(legend_node)}")
            print(f"   Out-degree: {G.out_degree(legend_node)}")

            # Ses voisins
            successors = list(G.successors(legend_node))
            predecessors = list(G.predecessors(legend_node))
            print(f"   Successors (investit dans): {len(successors)}")
            print(f"   Predecessors (reçoit de): {len(predecessors)}")

            if len(successors) > 0 and len(successors) < 20:
                print(f"\n   Companies dans lesquelles Legend Capital investit:")
                for succ in successors[:10]:
                    name = G.nodes[succ].get('name', f'Node {succ}')
                    prob = G[legend_node][succ].get('probability', 0)
                    print(f"      → {name} (prob={prob:.4f})")
        else:
            print(f"   Degree: {G.degree(legend_node)}")
    else:
        print(f"   ❌ Legend Capital not found in graph!")

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC:")
    print(f"{'='*70}")

    # Diagnostic automatique
    if G.is_directed():
        investors_with_zero_out = sum(1 for n in nodes_by_bipartite.get(1, []) if out_degrees[n] == 0)
        total_investors = len(nodes_by_bipartite.get(1, []))

        print(f"\n⚠️  Investors with out-degree=0: {investors_with_zero_out}/{total_investors}")

        if investors_with_zero_out > total_investors * 0.9:
            print(f"\n❌ PROBLÈME IDENTIFIÉ:")
            print(f"   La plupart des investors n'ont AUCUNE arête sortante!")
            print(f"   → TechRank ne peut pas calculer de scores pour eux.")
            print(f"\n💡 CAUSES POSSIBLES:")
            print(f"   1. Le threshold de probabilité est trop élevé")
            print(f"   2. Le modèle prédit des probabilités très basses")
            print(f"   3. Les arêtes sont dans le mauvais sens (company→investor au lieu de investor→company)")
            print(f"\n🔧 SOLUTIONS:")
            print(f"   1. Réduire --prediction_threshold (actuellement {0.0})")
            print(f"   2. Vérifier la direction des arêtes dans generate_predictions_and_graph()")
            print(f"   3. Augmenter le nombre d'edges en gardant plus de prédictions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Analyze graph structure')
    parser.add_argument('--graph_path', type=str, default='predicted_graph_crunchbase.pkl',
                        help='Path to the graph pickle file')
    args = parser.parse_args()

    analyze_graph_structure(args.graph_path)
