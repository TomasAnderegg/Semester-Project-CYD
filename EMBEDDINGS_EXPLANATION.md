# Features Statiques vs Embeddings Temporels - Explication

## 🔍 Le Problème avec Hard Negative Mining

Tu as demandé: "je ne comprends pas quels types de embedding temporels ?"

Excellente question ! Voici la différence cruciale.

## 1️⃣ Features Statiques (utilisées actuellement)

### Qu'est-ce que c'est ?

Les **features statiques** sont les attributs bruts des nodes qui **ne changent JAMAIS** pendant le training:

```python
# Chargées au début du script (train_self_supervised.py, ligne 136)
node_features, edge_features, full_data, train_data, ... = get_data(DATA)

# node_features est un numpy array de shape (num_nodes, feature_dim)
# Par exemple: (1016, 172) = 1016 nodes, 172 features chacun
```

### Exemple concret pour un investisseur:

```
Investor ID 463:
  Features statiques = [
    0.5,    # Type: VC
    0.8,    # Focus: Tech
    0.3,    # Stage préféré: Seed
    1.2,    # Montant moyen investi (log-scale)
    0.1,    # Secteur: Santé
    0.9,    # Secteur: AI
    ...     # 172 features au total
  ]
```

**Ces valeurs sont FIXES** - elles ne changent pas au cours du temps.

### Utilisation actuelle dans Hard Negative Mining:

```python
# train_self_supervised.py, lignes 315-322
node_features_np = node_features  # ⚠️ Features STATIQUES
negatives_batch = hard_neg_sampler.sample(
    sources=sources_batch,
    destinations=destinations_batch,
    embeddings=node_features_np,  # ⚠️ Pas adapté au contexte temporel!
    adjacency_dict=train_adjacency_dict,
    n_negatives=1
)
```

### Problème:

Ces features **ignorent complètement**:
- ✗ L'historique d'investissement de l'investisseur
- ✗ Les connexions récentes dans le graphe
- ✗ L'évolution temporelle des préférences
- ✗ Le contexte du moment (tendances du marché)

**Résultat**: Le sampler peut sélectionner des "hard negatives" qui sont similaires sur papier mais ne sont pas vraiment difficiles dans le contexte temporel.

---

## 2️⃣ Embeddings Temporels (ce qui serait mieux)

### Qu'est-ce que c'est ?

Les **embeddings temporels** sont calculés **dynamiquement** par le modèle TGN en fonction:
- De l'historique des interactions jusqu'à un timestamp donné
- De la structure du graphe au moment T
- De la mémoire du node (si `--use_memory` est activé)

### Comment ils sont calculés:

```python
# model/tgn.py, ligne 101-156
def compute_temporal_embeddings(self, source_nodes, destination_nodes,
                                negative_nodes, edge_times, edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatives.

    Ces embeddings CHANGENT à chaque timestamp!
    """

    # 1. Récupérer la mémoire actuelle (état des nodes)
    if self.use_memory:
        memory = self.get_updated_memory(...)

    # 2. Agréger les voisins temporels (interactions récentes)
    node_embedding = self.embedding_module.compute_embedding(
        memory=memory,
        source_nodes=nodes,
        timestamps=timestamps,
        n_layers=self.n_layers,
        n_neighbors=n_neighbors
    )

    return source_embedding, destination_embedding, negative_embedding
```

### Exemple concret:

**Même investisseur (ID 463) à différents moments:**

```
Timestamp 1 (2023-01-01):
  Embedding temporel = [0.2, -0.5, 0.8, ..., 0.3]  # 100 dimensions
  (Investisseur vient d'investir dans 3 startups AI)

Timestamp 2 (2023-06-01):
  Embedding temporel = [0.7, 0.1, -0.2, ..., 0.9]  # 100 dimensions
  (Investisseur a été actif, nouveau pattern détecté)

Timestamp 3 (2024-01-01):
  Embedding temporel = [-0.1, 0.3, 0.5, ..., -0.4]  # 100 dimensions
  (Investisseur moins actif, préférences ont évolué)
```

**Ces valeurs CHANGENT** en fonction du contexte temporel!

### Schéma de calcul:

```
                    ┌─────────────────┐
                    │  Node Features  │
                    │   (statiques)   │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Historical  │───▶│  TGN Embedding  │◀───│   Memory     │
│ Interactions │    │     Module      │    │  (état)      │
└──────────────┘    └────────┬────────┘    └──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Temporal      │
                    │   Embedding     │
                    │ (contextualisé) │
                    └─────────────────┘
```

---

## ⚖️ Comparaison: Features Statiques vs Embeddings Temporels

| Aspect | Features Statiques | Embeddings Temporels |
|--------|-------------------|---------------------|
| **Source** | Fichiers de données bruts | Calculés par le modèle TGN |
| **Taille** | (1016, 172) fixe | (1016, 100) ou (1016, 200) |
| **Évolution** | ❌ Jamais | ✅ À chaque timestamp |
| **Contexte temporel** | ❌ Non | ✅ Oui (historique inclus) |
| **Mémoire TGN** | ❌ Non utilisée | ✅ Intégrée |
| **Voisinage graphe** | ❌ Non utilisé | ✅ Agrégé |
| **Coût calcul** | ✅ Gratuit (déjà chargé) | ❌ Très coûteux |

---

## 🔴 Pourquoi Hard Negative Mining échoue avec Features Statiques

### Exemple concret:

**Situation**: Nous voulons trouver des hard negatives pour l'edge:
```
Company 101 (Startup AI santé) → Investor 463 (VC tech, actif en AI)
Timestamp: 2024-06-20
```

### Avec Features Statiques:

Le sampler calcule la similarité basée sur les features brutes:

```python
# hard_negative_mining.py, ligne 73
similarities = np.dot(neg_embs, pos_emb)  # Similarité des features statiques

# Top "hard negatives" sélectionnés:
# 1. Investor 471: Features similaires (VC tech, focus AI)
# 2. Investor 498: Features similaires (VC tech, secteur santé)
# 3. Investor 532: Features similaires (VC, early stage)
```

**Problème**: Ces investisseurs ont des **features similaires** mais:
- Investor 471: N'a **jamais investi** dans la santé (seulement fintech)
- Investor 498: N'investit **plus** depuis 2 ans (inactif)
- Investor 532: A déjà investi dans Company 101 au timestamp 2023-12-01 (donc devrait être positif!)

**Pourquoi?** Les features statiques ne capturent pas:
- ✗ L'historique réel d'investissement
- ✗ L'activité récente
- ✗ Les connexions existantes dans le graphe

### Avec Embeddings Temporels (théorique):

```python
# Calculer embeddings temporels au timestamp 2024-06-20
temporal_embeddings = tgn.compute_all_node_embeddings(timestamp=2024-06-20)

# Ces embeddings capturent:
# - Investor 471: Embedding reflète qu'il n'investit QUE dans fintech
# - Investor 498: Embedding montre l'inactivité récente
# - Investor 532: Embedding indique connexion existante avec Company 101

# Top "hard negatives" sélectionnés (basés sur embeddings temporels):
# 1. Investor 555: Profil AI santé, actif, similaire à 463 mais n'a PAS investi dans Company 101
# 2. Investor 602: Même secteur, même stage, co-investit souvent avec 463 mais pas ici
# 3. Investor 644: Pattern d'investissement très proche de 463

# ✅ Ces négatifs sont VRAIMENT difficiles car le modèle devra apprendre
# des distinctions subtiles basées sur le contexte temporel complet
```

---

## 💡 Pourquoi on n'utilise PAS les Embeddings Temporels?

### Coût computationnel prohibitif:

```python
# À CHAQUE batch d'entraînement, il faudrait:

# 1. Calculer embeddings pour TOUS les nodes (1016 nodes)
for batch_idx in range(num_batches):
    # Ça, c'est déjà fait pour le batch actuel
    source_emb, dest_emb, neg_emb = tgn.compute_temporal_embeddings(
        sources_batch, destinations_batch, negatives_batch, ...
    )

    # ❌ Mais pour hard negative mining, il faudrait AUSSI:
    all_node_embeddings = tgn.compute_temporal_embeddings(
        nodes=list(range(1016)),  # TOUS les nodes!
        timestamps=[current_ts] * 1016,
        ...
    )  # ⚠️ TRÈS COÛTEUX!

    # 2. Ensuite seulement, faire le hard sampling
    hard_negatives = hard_sampler.sample(
        embeddings=all_node_embeddings.cpu().numpy()
    )

# Résultat: Training 10-50x plus lent!
```

**Estimation du coût**:
- Sans embeddings temporels: ~2 min/epoch
- Avec embeddings temporels: ~20-100 min/epoch (10-50x plus lent!)

---

## 🎯 Solutions Alternatives

### Option 1: Rester sur Focal Loss seul ✅ (RECOMMANDÉ)

```bash
# Pas de hard negatives, juste Focal Loss
python train_self_supervised.py \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0
```

**Avantages**:
- ✅ Marche bien (9.85x vs random)
- ✅ Rapide (pas de surcoût)
- ✅ Simple

### Option 2: Hard Negatives basés sur le graphe 🔬

Au lieu d'utiliser des features, utiliser la **structure du graphe**:

```python
# Exemple: échantillonner des nodes à distance 2-3
def graph_based_hard_negatives(src, adjacency):
    """
    Hard negatives = "amis d'amis" (distance 2 dans le graphe)
    """
    # Distance 1: voisins directs (positifs)
    neighbors_1hop = adjacency[src]

    # Distance 2: amis d'amis (hard negatives potentiels!)
    neighbors_2hop = set()
    for neighbor in neighbors_1hop:
        neighbors_2hop.update(adjacency[neighbor])

    # Retirer les voisins directs
    hard_negatives = neighbors_2hop - neighbors_1hop

    return random.sample(hard_negatives, k)
```

**Avantages**:
- ✅ Capture la structure du graphe
- ✅ Pas besoin de features ou embeddings
- ✅ Rapide à calculer

**Inconvénients**:
- ⚠️ Ignore les attributs des nodes
- ⚠️ Peut manquer de diversité si le graphe est clairsemé

### Option 3: Embeddings "légers" 💡

Utiliser une approximation rapide:

```python
# Au lieu de recalculer embeddings complets, utiliser:
# 1. Features statiques
# 2. + Degré du node (nombre de connexions)
# 3. + Activité récente (nombre d'interactions dans les N derniers jours)

def enriched_features(node_id, static_features, adjacency, timestamps, current_ts):
    """Features enrichies avec contexte temporel léger"""

    # Features statiques de base
    features = static_features[node_id].copy()

    # Ajouter degré
    degree = len(adjacency[node_id])

    # Ajouter activité récente (30 derniers jours)
    recent_activity = sum(
        1 for ts in timestamps[node_id]
        if current_ts - ts < 30 * 24 * 3600
    )

    return np.concatenate([features, [degree, recent_activity]])
```

---

## 📋 Résumé

**Question**: Quels types d'embedding temporels?

**Réponse**:
1. **Features statiques** (actuelles): Attributs fixes des nodes, ne changent jamais
2. **Embeddings temporels** (idéaux): Représentations calculées par TGN qui évoluent avec le temps

**Pourquoi Hard Negatives échoue**:
- ❌ Utilise features statiques qui ne capturent pas le contexte temporel
- ❌ "Similarité" des features ≠ "difficulté" pour le modèle

**Pourquoi on n'utilise pas embeddings temporels**:
- ❌ Trop coûteux computationnellement (10-50x plus lent)
- ❌ Faudrait recalculer embeddings pour TOUS les nodes à chaque batch

**Recommandation**:
- ✅ **Focal Loss seul** pour l'instant (marche bien, simple, rapide)
- 🔬 Tester **graph-based hard negatives** si tu veux vraiment améliorer
- ⏳ Embeddings temporels seulement si tu as beaucoup de compute et temps
