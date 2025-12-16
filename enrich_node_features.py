"""
Enrichir les node features avec des statistiques agrégées des edge features

⚠️ IMPORTANT: Éviter le data leakage!
On ne doit utiliser QUE les edges qui ont un timestamp ≤ train_end_time
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def enrich_node_features_safe(node_features, edge_features, train_data, full_data):
    """
    Enrichir les node features avec stats agrégées des edge features

    ⚠️ SANS DATA LEAKAGE: N'utilise que les edges du training set

    Args:
        node_features: Node features originales, shape (num_nodes, node_dim)
        edge_features: Edge features, shape (num_edges, edge_dim)
        train_data: Data object du training set (éviter leakage)
        full_data: Data object complet (pour mapping)

    Returns:
        enriched_features: Node features enrichies, shape (num_nodes, node_dim + stats_dim)
    """

    num_nodes = node_features.shape[0]

    print(f"Enriching node features for {num_nodes} nodes...")
    print(f"Using {len(train_data.sources)} training edges (avoiding data leakage)")

    # ================================================================
    # ÉTAPE 1: Agréger les edge features par destination (investor)
    # ================================================================

    investor_stats = defaultdict(lambda: {
        'edge_features': [],
        'timestamps': [],
        'sources': []
    })

    # ⚠️ CRITIQUE: N'utiliser QUE les edges du training set!
    for i in range(len(train_data.sources)):
        src = train_data.sources[i]
        dst = train_data.destinations[i]
        edge_idx = train_data.edge_idxs[i]
        timestamp = train_data.timestamps[i]

        # Récupérer les edge features
        edge_feat = edge_features[edge_idx]

        # Agréger pour la destination (investor)
        investor_stats[dst]['edge_features'].append(edge_feat)
        investor_stats[dst]['timestamps'].append(timestamp)
        investor_stats[dst]['sources'].append(src)

    # ================================================================
    # ÉTAPE 2: Calculer les statistiques pour chaque node
    # ================================================================

    enriched_features_list = []

    for node_id in range(num_nodes):
        # Features de base
        base_features = node_features[node_id]

        # Vérifier si ce node a un historique
        if node_id in investor_stats and len(investor_stats[node_id]['edge_features']) > 0:
            edge_feats_array = np.array(investor_stats[node_id]['edge_features'])
            timestamps_array = np.array(investor_stats[node_id]['timestamps'])

            # Calculer statistiques agrégées
            stats = compute_aggregated_stats(edge_feats_array, timestamps_array)

            # Concaténer
            enriched = np.concatenate([base_features, stats])
        else:
            # Pas d'historique: padding avec zeros
            stats_dim = get_stats_dimension(edge_features.shape[1])
            enriched = np.concatenate([base_features, np.zeros(stats_dim)])

        enriched_features_list.append(enriched)

    enriched_features = np.array(enriched_features_list)

    print(f"✅ Enriched features shape: {enriched_features.shape}")
    print(f"   Original: {node_features.shape[1]} dims")
    print(f"   Added: {enriched_features.shape[1] - node_features.shape[1]} dims")

    return enriched_features


def compute_aggregated_stats(edge_features_array, timestamps_array):
    """
    Calculer statistiques agrégées des edge features

    Args:
        edge_features_array: Array de shape (num_edges, edge_dim)
        timestamps_array: Array de shape (num_edges,)

    Returns:
        stats: Array de statistiques agrégées
    """

    stats = []

    # 1. Statistiques globales sur les edge features
    stats.append(np.mean(edge_features_array, axis=0))  # Moyenne
    stats.append(np.std(edge_features_array, axis=0))   # Écart-type
    stats.append(np.max(edge_features_array, axis=0))   # Max
    stats.append(np.min(edge_features_array, axis=0))   # Min

    # 2. Statistiques temporelles
    stats.append([
        len(edge_features_array),           # Nombre d'investissements
        timestamps_array.mean(),            # Timestamp moyen
        timestamps_array.std(),             # Écart-type timestamps
        timestamps_array.max(),             # Dernier investissement
        timestamps_array.min(),             # Premier investissement
        timestamps_array.max() - timestamps_array.min()  # Période active
    ])

    # 3. Statistiques récentes (last 3 investments)
    if len(edge_features_array) >= 3:
        # Trier par timestamp
        sorted_indices = np.argsort(timestamps_array)
        recent_3 = edge_features_array[sorted_indices[-3:]]
        stats.append(np.mean(recent_3, axis=0))  # Moyenne des 3 derniers
    else:
        # Pas assez d'historique: utiliser la moyenne globale
        stats.append(np.mean(edge_features_array, axis=0))

    # Concaténer toutes les stats
    return np.concatenate([s.flatten() for s in stats])


def get_stats_dimension(edge_dim):
    """Calculer la dimension des stats agrégées"""
    # mean (edge_dim) + std (edge_dim) + max (edge_dim) + min (edge_dim)
    # + temporal_stats (6) + recent_mean (edge_dim)
    return edge_dim * 5 + 6


def save_enriched_features(enriched_features, output_path):
    """Sauvegarder les features enrichies"""
    np.save(output_path, enriched_features)
    print(f"💾 Saved enriched features to {output_path}")


def test_enrichment():
    """Test de la fonction d'enrichissement"""
    print("Testing enrichment function...")

    # Charger les données
    from utils.data_processing import get_data

    node_features, edge_features, full_data, train_data, val_data, test_data, _, _ = get_data(
        'crunchbase',
        different_new_nodes_between_val_and_test=False
    )

    print(f"\nOriginal node features: {node_features.shape}")
    print(f"Edge features: {edge_features.shape}")
    print(f"Training edges: {len(train_data.sources)}")

    # Enrichir
    enriched = enrich_node_features_safe(
        node_features,
        edge_features,
        train_data,
        full_data
    )

    print(f"\nEnriched features: {enriched.shape}")

    # Vérifier
    assert enriched.shape[0] == node_features.shape[0], "Number of nodes changed!"
    assert enriched.shape[1] > node_features.shape[1], "No features added!"

    # Sauvegarder
    save_enriched_features(
        enriched,
        'data/data_split/crunchbase_filtered_train_node_enriched.npy'
    )

    print("\n✅ Enrichment test passed!")


if __name__ == "__main__":
    test_enrichment()
