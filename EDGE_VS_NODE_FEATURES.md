# Edge Features vs Node Features vs Embeddings Temporels

## Question de l'utilisateur

> "mais attends, j'ai des embedding temporel, les informations temporel du raised amount a une date annoncée. C'est contenu dans le edge"

**Excellente observation !** Il y a effectivement des informations temporelles dans les données. Mais il faut clarifier **où** elles sont et **comment** on peut les utiliser.

---

## 🗂️ Les 3 types de représentations

### 1. **Node Features** (features statiques des nodes)

**Fichier**: `data/data_split/crunchbase_filtered_train_node.npy`

**Shape**: `(num_nodes, node_feature_dim)` par exemple `(1016, 172)`

**Contenu**: Attributs fixes de chaque company/investor
```python
# Node 463 (Investor: Sequoia Capital)
node_features[463] = [
    0.5,    # Type: VC
    0.8,    # Focus: Tech
    0.3,    # Stage préféré: Early
    ...     # 172 features au total
]
```

**❌ PAS d'infos temporelles**: Ces features sont les mêmes pour tout le training

---

### 2. **Edge Features** (features des transactions/investissements)

**Fichier**: `data/data_split/crunchbase_filtered_train.npy`

**Shape**: `(num_edges, edge_feature_dim)` par exemple `(1130, X)`

**Contenu**: Informations sur chaque investissement (transaction)
```python
# Edge 42: Company 101 → Investor 463, 2024-01-15
edge_features[42] = [
    15.5,      # raised_amount (log-scale)
    2024.04,   # announced_on (year fraction)
    0.1,       # funding_round_type: Seed
    ...
]

# Edge 43: Company 105 → Investor 463, 2024-03-20
edge_features[43] = [
    17.2,      # raised_amount (log-scale)
    2024.22,   # announced_on (year fraction)
    0.3,       # funding_round_type: Series A
    ...
]
```

**✅ Contient des infos temporelles**: raised_amount, announced_on, etc.

**⚠️ Mais indexé par EDGE, pas par NODE !**

---

### 3. **Embeddings Temporels** (calculés par TGN)

**Pas stockés** - calculés dynamiquement par le modèle

**Shape**: `(num_nodes, embedding_dim)` par exemple `(1016, 100)`

**Contenu**: Représentation contextualisée de chaque node à un timestamp donné
```python
# Investor 463 au timestamp 2024-06-01
temporal_emb = tgn.compute_temporal_embeddings(
    source_nodes=[463],
    timestamps=[2024-06-01],
    ...
)
# temporal_emb[0] = [0.2, -0.5, ..., 0.8]  # 100 dimensions

# Ce embedding AGRÈGE:
# - Node features de l'investor 463
# - Toutes les edge features des investissements passés
# - Structure du graphe (voisins, co-investisseurs)
# - Mémoire de l'état du node
```

**✅ Informations temporelles agrégées**

---

## 🔍 Le problème avec Hard Negative Mining

### Ce qu'on fait actuellement:

```python
# train_self_supervised.py, ligne 315-322
node_features_np = node_features  # ⚠️ Node features STATIQUES

negatives_batch = hard_neg_sampler.sample(
    sources=sources_batch,           # [Company 101, ...]
    destinations=destinations_batch, # [Investor 463, ...]
    embeddings=node_features_np,     # ⚠️ Features statiques!
    ...
)
```

**Problème**: On utilise `node_features` qui **n'ont PAS** les infos temporelles

### Pourquoi on ne peut pas utiliser edge_features?

**Edge features sont indexées par EDGE, pas par NODE !**

```python
# On veut comparer deux investors pour hard negative mining:
investor_A = 463
investor_B = 471

# Mais edge_features nous donne:
edge_features[42]  # Edge: Company 101 → Investor 463, date X
edge_features[43]  # Edge: Company 105 → Investor 463, date Y

# ❌ On ne peut pas faire:
#    similarity(edge_features[investor_A], edge_features[investor_B])
# Parce que investor_A n'est PAS un index valide pour edge_features!
```

**Pour comparer deux investors, il faudrait:**
1. Récupérer TOUS les edges de chaque investor
2. Agréger les edge features de ces edges
3. Créer une représentation du investor basée sur son historique

**C'est exactement ce que fait TGN avec les embeddings temporels !**

---

## 🧩 Comment TGN utilise les edge features

### Dans le forward pass normal:

```python
# model/tgn.py
def compute_temporal_embeddings(self, source_nodes, destination_nodes,
                                negative_nodes, edge_times, edge_idxs, ...):

    # 1. Récupérer les edge features pour cet edge spécifique
    edge_features = self.edge_raw_features[edge_idxs]  # Features de cette transaction

    # 2. Récupérer les voisins temporels de chaque node
    # (tous les nodes avec qui ils ont interagi avant ce timestamp)
    neighbors = self.neighbor_finder.get_temporal_neighbor(...)

    # 3. Récupérer les edge features de TOUTES ces interactions passées
    neighbor_edge_features = self.edge_raw_features[neighbor_edge_idxs]

    # 4. Agréger tout ça avec attention + mémoire
    temporal_embedding = self.embedding_module.compute_embedding(
        node_features=self.node_raw_features[nodes],
        edge_features=neighbor_edge_features,  # ✅ Infos temporelles agrégées!
        memory=self.memory.get_memory(nodes),
        ...
    )

    return temporal_embedding  # Embedding qui capture l'historique temporel
```

**Les edge features sont utilisées**, mais de manière **agrégée** sur tout l'historique d'un node.

---

## 🎯 Exemple concret

### Scénario:

On veut trouver des hard negatives pour:
```
Edge: Company 101 → Investor 463
Timestamp: 2024-06-20
```

### Avec node_features (actuellement):

```python
# Comparer Investor 463 avec d'autres investors
investor_463_features = node_features[463]  # [0.5, 0.8, 0.3, ...]
investor_471_features = node_features[471]  # [0.5, 0.7, 0.4, ...]

similarity = cosine_similarity(investor_463_features, investor_471_features)
# similarity = 0.92 (très similaire!)

# ✅ On peut calculer la similarité
# ❌ Mais elle ignore complètement l'historique d'investissement!
```

### Avec edge_features (ce que tu proposes):

```python
# ❌ Problème: On ne peut pas faire ça directement!

# edge_features est un tableau de shape (num_edges, edge_dim)
# On ne peut pas indexer par node_id:
investor_463_edge_features = edge_features[463]  # ❌ 463 est un edge_idx, pas un node_id!

# Pour utiliser les edge features, il faudrait:
# 1. Trouver tous les edges où Investor 463 apparaît
edges_of_463 = [42, 43, 51, 78, ...]  # Tous les investissements de 463

# 2. Récupérer les edge features de tous ces edges
all_edge_features_463 = edge_features[edges_of_463]  # Shape: (N, edge_dim)

# 3. Agréger ces edge features
# Mais comment? Moyenne? Somme? Attention?
investor_463_aggregated = np.mean(all_edge_features_463, axis=0)

# 4. Faire pareil pour tous les autres investors
# C'est faisable mais TRÈS COÛTEUX!
```

### Avec embeddings temporels (idéal):

```python
# TGN calcule automatiquement les embeddings qui agrègent:
# - Node features
# - Edge features de tous les edges passés (jusqu'au timestamp)
# - Structure du graphe
# - Mémoire

temporal_emb_463 = tgn.compute_temporal_embeddings(
    [463], timestamps=[2024-06-20], ...
)
temporal_emb_471 = tgn.compute_temporal_embeddings(
    [471], timestamps=[2024-06-20], ...
)

similarity = cosine_similarity(temporal_emb_463, temporal_emb_471)

# ✅ Cette similarité capture l'historique complet!
# ❌ Mais c'est TRÈS COÛTEUX à calculer pour tous les nodes
```

---

## 💡 Solution hybride possible

Tu as raison qu'on pourrait utiliser les edge features pour enrichir notre sampling. Voici une approche:

### Option: Enrichir node_features avec statistiques agrégées

```python
def enrich_node_features_with_edge_stats(node_features, edge_features,
                                          train_data):
    """
    Enrichir les node features avec des stats agrégées des edge features
    """
    enriched_features = []

    for node_id in range(len(node_features)):
        # Features statiques de base
        base_features = node_features[node_id]

        # Trouver tous les edges de ce node
        node_edges_mask = (train_data.destinations == node_id)
        node_edge_indices = train_data.edge_idxs[node_edges_mask]

        if len(node_edge_indices) > 0:
            # Récupérer les edge features de tous ces edges
            node_edge_feats = edge_features[node_edge_indices]

            # Agréger: moyenne, std, max
            edge_mean = np.mean(node_edge_feats, axis=0)
            edge_std = np.std(node_edge_feats, axis=0)
            edge_max = np.max(node_edge_feats, axis=0)

            # Concaténer avec base features
            enriched = np.concatenate([base_features, edge_mean, edge_std, edge_max])
        else:
            # Pas d'historique: padding avec zeros
            enriched = np.concatenate([
                base_features,
                np.zeros(edge_features.shape[1] * 3)
            ])

        enriched_features.append(enriched)

    return np.array(enriched_features)
```

**Avantages**:
- ✅ Utilise les edge features (infos temporelles!)
- ✅ Calculé une seule fois au début du training (pas trop coûteux)
- ✅ Capture l'historique d'investissement de chaque investor

**Inconvénients**:
- ⚠️ Agrégation sur TOUT l'historique (ignore le timestamp actuel)
- ⚠️ Pas aussi riche que les embeddings temporels du TGN

---

## 📋 Résumé

**Ta question**: "J'ai des embedding temporel, les infos temporelles sont dans les edge features"

**Réponse**:
1. ✅ **Oui**, les edge features contiennent des infos temporelles (raised_amount, date)
2. ❌ **Mais** edge_features sont indexées par **edge**, pas par **node**
3. 🔧 **Pour hard negative mining**, on a besoin de comparer des **nodes** (investors)
4. 💡 **Solution actuelle**: Utilise node_features (statiques) - ignore edge features
5. 🎯 **Solution idéale**: Utilise embeddings temporels du TGN - agrègent node + edge features
6. ⚖️ **Solution hybride**: Enrichir node_features avec stats agrégées des edge features

**Conclusion**:
- Les edge features sont utilisées **par le TGN** pendant le training
- Mais pas **par le hard negative sampler** (qui travaille sur les nodes, pas les edges)
- On pourrait enrichir node_features avec des stats d'edge features (approche hybride)
- Mais ça reste moins riche que les vrais embeddings temporels du TGN

Veux-tu que j'implémente l'approche hybride (enrichir node_features avec edge stats)?
