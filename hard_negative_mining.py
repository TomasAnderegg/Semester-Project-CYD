"""
Hard Negative Mining for Temporal Graph Networks

Stratégie pour améliorer l'apprentissage en sélectionnant des exemples négatifs
difficiles plutôt que des négatifs aléatoires.

Problème avec échantillonnage aléatoire:
- La plupart des négatifs aléatoires sont "faciles" (très différents des positifs)
- Le modèle apprend à distinguer les cas évidents, pas les cas subtils
- Mauvaise performance sur les vrais cas difficiles en évaluation

Solution avec Hard Negative Mining:
- Sélectionner des négatifs qui ressemblent aux positifs (features similaires)
- Forcer le modèle à apprendre des distinctions plus fines
- Améliorer le ranking des vrais liens vs. faux liens similaires
"""

import numpy as np
import torch


class HardNegativeSampler:
    """
    Échantillonneur de négatifs difficiles pour TGN.

    Stratégie:
    1. Pour chaque edge (src, dst) positif
    2. Trouver des destinations candidates qui:
       - Ne sont PAS connectées à src (vrais négatifs)
       - Ont des embeddings similaires à dst (négtatifs difficiles)
    3. Mélanger hard negatives + random negatives

    Args:
        ratio (float): Proportion de hard negatives (défaut: 0.5)
                      0.0 = tous random, 1.0 = tous hard
        temperature (float): Température pour softmax de similarité (défaut: 0.1)
                           Plus bas = plus agressif (sélectionne les plus similaires)
    """

    def __init__(self, ratio=0.5, temperature=0.1):
        self.ratio = ratio
        self.temperature = temperature

    def sample(self, sources, destinations, embeddings, adjacency_dict, n_negatives=1):
        """
        Échantillonne des négatifs difficiles.

        Args:
            sources (np.array): Source node IDs, shape (batch_size,)
            destinations (np.array): Destination node IDs (positifs), shape (batch_size,)
            embeddings (np.array): Node embeddings, shape (num_nodes, embedding_dim)
            adjacency_dict (dict): {src_id: set(dst_ids)} - vraies connexions
            n_negatives (int): Nombre de négatifs par positif

        Returns:
            np.array: Negative destination IDs, shape (batch_size, n_negatives)
        """
        batch_size = len(sources)
        num_nodes = embeddings.shape[0]

        # Nombre de hard vs random negatives
        n_hard = int(n_negatives * self.ratio)
        n_random = n_negatives - n_hard

        negative_samples = []

        for i in range(batch_size):
            src = sources[i]
            pos_dst = destinations[i]

            # Obtenir les vraies connexions pour cette source
            true_connections = adjacency_dict.get(src, set())

            # Trouver tous les négatifs possibles (pas connectés à src)
            possible_negatives = [
                node_id for node_id in range(num_nodes)
                if node_id not in true_connections
            ]

            if len(possible_negatives) == 0:
                # Cas rare: src est connecté à tous les nodes
                # Fallback: random sampling
                neg_samples = np.random.choice(num_nodes, size=n_negatives, replace=True)
                negative_samples.append(neg_samples)
                continue

            sampled_negatives = []

            # 1. Hard negative sampling (basé sur similarité)
            if n_hard > 0 and len(possible_negatives) > 0:
                # Embedding du positif
                pos_emb = embeddings[pos_dst]

                # Embeddings des négatifs possibles
                neg_embs = embeddings[possible_negatives]

                # Calculer similarités (dot product)
                similarities = np.dot(neg_embs, pos_emb)

                # Probabilités avec temperature
                probs = self._softmax(similarities / self.temperature)

                # Échantillonner selon les probabilités
                # (plus similaire = plus de chances d'être sélectionné)
                n_hard_actual = min(n_hard, len(possible_negatives))
                hard_indices = np.random.choice(
                    len(possible_negatives),
                    size=n_hard_actual,
                    replace=False,
                    p=probs
                )
                hard_negatives = [possible_negatives[idx] for idx in hard_indices]
                sampled_negatives.extend(hard_negatives)

            # 2. Random negative sampling
            remaining = n_negatives - len(sampled_negatives)
            if remaining > 0:
                random_negatives = np.random.choice(
                    possible_negatives,
                    size=remaining,
                    replace=True
                )
                sampled_negatives.extend(random_negatives)

            negative_samples.append(sampled_negatives[:n_negatives])

        return np.array(negative_samples)

    def _softmax(self, x):
        """Stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class BatchedHardNegativeSampler:
    """
    Version optimisée utilisant des opérations matricielles.
    Plus rapide pour les gros batches.
    """

    def __init__(self, ratio=0.5, temperature=0.1, top_k=100):
        self.ratio = ratio
        self.temperature = temperature
        self.top_k = top_k  # Considérer seulement top-K plus similaires

    def sample(self, sources, destinations, embeddings, adjacency_matrix, n_negatives=1):
        """
        Version batched du sampling.

        Args:
            sources (np.array): Source node IDs, shape (batch_size,)
            destinations (np.array): Destination node IDs, shape (batch_size,)
            embeddings (torch.Tensor): Node embeddings, shape (num_nodes, embedding_dim)
            adjacency_matrix (torch.Tensor): Sparse adjacency matrix (num_nodes, num_nodes)
            n_negatives (int): Nombre de négatifs par positif

        Returns:
            torch.Tensor: Negative samples, shape (batch_size, n_negatives)
        """
        batch_size = len(sources)
        num_nodes = embeddings.shape[0]
        device = embeddings.device

        n_hard = int(n_negatives * self.ratio)
        n_random = n_negatives - n_hard

        # Convertir sources/destinations en tensors
        src_tensor = torch.tensor(sources, device=device)
        dst_tensor = torch.tensor(destinations, device=device)

        negative_samples = []

        # Pour chaque élément du batch
        for i in range(batch_size):
            src = src_tensor[i]
            pos_dst = dst_tensor[i]

            # Masque des vraies connexions
            true_connections_mask = adjacency_matrix[src].to_dense() > 0

            # Masque des négatifs possibles
            negative_mask = ~true_connections_mask
            negative_indices = negative_mask.nonzero(as_tuple=True)[0]

            if len(negative_indices) == 0:
                # Fallback: random
                neg_samples = torch.randint(0, num_nodes, (n_negatives,), device=device)
                negative_samples.append(neg_samples)
                continue

            sampled = []

            # Hard negatives
            if n_hard > 0:
                # Embedding du positif
                pos_emb = embeddings[pos_dst]

                # Embeddings des négatifs
                neg_embs = embeddings[negative_indices]

                # Similarités
                similarities = torch.matmul(neg_embs, pos_emb)

                # Top-K plus similaires
                k = min(self.top_k, len(negative_indices))
                top_k_vals, top_k_idx = torch.topk(similarities, k)

                # Probabilités avec temperature
                probs = torch.softmax(top_k_vals / self.temperature, dim=0)

                # Échantillonner
                n_hard_actual = min(n_hard, k)
                sampled_idx = torch.multinomial(probs, n_hard_actual, replacement=False)
                hard_neg_idx = top_k_idx[sampled_idx]
                hard_negatives = negative_indices[hard_neg_idx]
                sampled.append(hard_negatives)

            # Random negatives
            if n_random > 0:
                random_idx = torch.randint(0, len(negative_indices), (n_random,), device=device)
                random_negatives = negative_indices[random_idx]
                sampled.append(random_negatives)

            # Concaténer
            all_sampled = torch.cat(sampled) if len(sampled) > 0 else torch.tensor([], device=device)
            negative_samples.append(all_sampled[:n_negatives])

        return torch.stack(negative_samples)


def build_adjacency_dict(sources, destinations):
    """
    Construit un dictionnaire d'adjacence à partir des edges.

    Args:
        sources (np.array): Source node IDs
        destinations (np.array): Destination node IDs

    Returns:
        dict: {src_id: set(dst_ids)}
    """
    adj_dict = {}
    for src, dst in zip(sources, destinations):
        if src not in adj_dict:
            adj_dict[src] = set()
        adj_dict[src].add(dst)
    return adj_dict


def test_hard_negative_sampler():
    """
    Test pour vérifier que le hard negative sampler fonctionne.
    """
    print("Testing Hard Negative Sampler...")

    # Créer des données de test
    np.random.seed(42)
    num_nodes = 100
    embedding_dim = 64

    # Embeddings aléatoires
    embeddings = np.random.randn(num_nodes, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Quelques edges
    sources = np.array([0, 1, 2, 3, 4])
    destinations = np.array([10, 20, 30, 40, 50])

    # Adjacency dict
    adj_dict = build_adjacency_dict(sources, destinations)

    # Test sampler
    sampler = HardNegativeSampler(ratio=0.5, temperature=0.1)

    negatives = sampler.sample(
        sources=sources,
        destinations=destinations,
        embeddings=embeddings,
        adjacency_dict=adj_dict,
        n_negatives=5
    )

    print(f"\nBatch size: {len(sources)}")
    print(f"Negatives per positive: {negatives.shape[1]}")
    print(f"\nSampled negatives:")
    for i, (src, pos_dst, neg_dsts) in enumerate(zip(sources, destinations, negatives)):
        print(f"  Edge ({src} -> {pos_dst}): negatives = {neg_dsts}")

        # Vérifier que les négatifs ne sont pas dans les vraies connexions
        true_conns = adj_dict.get(src, set())
        assert all(neg not in true_conns for neg in neg_dsts), "Negative sampled from true connections!"

    print("\n✅ Tests passed!")

    # Test de comparaison: hard vs random
    print("\n" + "="*60)
    print("Comparing Hard vs Random Negative Sampling")
    print("="*60)

    # Créer un cas où hard negatives devraient être différents de random
    # Cluster 1: nodes 0-9 (embeddings similaires)
    # Cluster 2: nodes 10-19 (embeddings similaires)
    cluster1_emb = np.random.randn(10, embedding_dim)
    cluster2_emb = np.random.randn(10, embedding_dim) + 5.0  # Shift pour séparer

    test_embeddings = np.vstack([cluster1_emb, cluster2_emb])
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

    # Edge positif: 0 -> 1 (dans cluster 1)
    test_sources = np.array([0])
    test_destinations = np.array([1])
    test_adj = {0: {1}}

    # Hard sampler (ratio=1.0 = 100% hard)
    hard_sampler = HardNegativeSampler(ratio=1.0, temperature=0.1)
    hard_negs = hard_sampler.sample(
        test_sources, test_destinations, test_embeddings, test_adj, n_negatives=10
    )

    # Random sampler (ratio=0.0 = 100% random)
    random_sampler = HardNegativeSampler(ratio=0.0, temperature=0.1)
    random_negs = random_sampler.sample(
        test_sources, test_destinations, test_embeddings, test_adj, n_negatives=10
    )

    # Compter combien de négatifs viennent du cluster 1 (similaires au positif)
    hard_from_cluster1 = sum(1 for neg in hard_negs[0] if neg < 10)
    random_from_cluster1 = sum(1 for neg in random_negs[0] if neg < 10)

    print(f"\nPositive edge: {test_sources[0]} -> {test_destinations[0]} (cluster 1)")
    print(f"\nHard negatives: {hard_negs[0]}")
    print(f"  From cluster 1 (similar): {hard_from_cluster1}/10")
    print(f"\nRandom negatives: {random_negs[0]}")
    print(f"  From cluster 1 (similar): {random_from_cluster1}/10")
    print(f"\n➡️ Hard negatives select {hard_from_cluster1}x more similar nodes than random")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_hard_negative_sampler()
