# Post-Mortem: Hard Negative Mining n'a pas amélioré les résultats

## 📊 Résultats observés

| Métrique | Focal seul | Focal + Hard (ratio=0.5, temp=0.1) | Changement |
|----------|------------|-------------------------------------|------------|
| Precision@1000 | 0.3% (3/1000) | 0.1% (1/1000) | ❌ **-67%** |
| Recall@1000 | 5.77% | 1.92% | ❌ -67% |
| Rank médian | 5,623 | 5,102 | ✅ +9% |
| vs Random | 9.85x | 3.28x | ❌ -67% |

**Conclusion**: Hard Negative Mining a **dégradé** les performances au lieu de les améliorer.

## 🔍 Analyse des causes

### Cause 1: Features statiques inadéquates

**Problème**: Le sampler utilise `node_features` (features brutes) pour calculer la similarité:

```python
# train_self_supervised.py:315-322
node_features_np = node_features  # Features STATIQUES
negatives_batch = hard_neg_sampler.sample(
    embeddings=node_features_np,  # ⚠️ Pas les embeddings temporels!
    ...
)
```

**Pourquoi c'est un problème**:
- `node_features` sont **statiques**: secteur, taille, année fondation, etc.
- Ne capturent **pas** la dynamique temporelle des investissements
- Ne capturent **pas** les patterns de co-investissement
- Ne capturent **pas** les préférences subtiles des investisseurs

**Exemple concret**:
```
Company: Startup AI dans la santé (features: [sector=AI, stage=seed, ...])

Hard negatives sélectionnés par similarité de features:
- Investor A: Spécialisé AI mais n'investit QUE dans fintech (pas santé)
- Investor B: Spécialisé santé mais n'investit QUE dans late-stage (pas seed)

Ces investisseurs sont "similaires" sur papier mais ne sont PAS des hard negatives pertinents!
```

### Cause 2: Hyperparamètres trop agressifs

**Configuration testée**:
- `--hard_neg_ratio 0.5`: 50% de hard negatives
- `--hard_neg_temperature 0.1`: Très agressif (sélectionne top similaires)

**Problème**:
- Avec 50% de négatifs "mal choisis", le modèle apprend des patterns incorrects
- Temperature 0.1 sélectionne presque exclusivement les plus similaires, sans diversité

### Cause 3: Gap entre training et evaluation

**Training**: Negative sampling sur un seul négatif par positif
```python
n_negatives=1  # Seulement 1 négatif par edge
```

**Evaluation temporelle**: Complete ranking sur 170,742 paires
```
199 companies × 858 investors = 170,742 paires
```

**Problème**: Le modèle apprend à distinguer 1 négatif (hard ou random) mais doit ensuite ranker 170K paires. Le gap est énorme.

## 💡 Pourquoi le rang médian s'est amélioré mais pas Precision@1000?

**Observation paradoxale**:
- Rang médian: 5,623 → 5,102 (✅ amélioration +9%)
- Precision@1000: 0.3% → 0.1% (❌ dégradation -67%)

**Explication**:
1. Hard negatives a rendu le modèle **plus conservateur**
2. Le modèle assigne des probabilités **plus uniformes** (range plus étroit)
3. Résultat:
   - Quelques vrais liens ont monté dans le ranking (médiane améliore)
   - Mais beaucoup d'autres ont descendu (moins de hits dans top-1000)
   - La **variance** du ranking a augmenté

**Preuve dans les probabilités**:
- Focal seul: médiane global 0.243, médiane vrais 0.378, **gap = 0.135**
- Focal+Hard: médiane global 0.334, médiane vrais 0.453, **gap = 0.119**

Le gap a **réduit** (0.135 → 0.119), ce qui signifie que le modèle discrimine **moins bien**.

## ✅ Ce qui a fonctionné (à garder)

### Focal Loss seul

**Meilleurs résultats observés**:
- Precision@1000: **0.3%** (3/1000)
- Recall@1000: **5.77%** (3/52)
- vs Random: **9.85x** meilleur que baseline
- Rang médian: **5,623** (top 3.3%)

**Pourquoi ça marche**:
- Focal Loss s'attaque au **vrai** problème: déséquilibre de classes (0.03% positifs)
- Réduit l'importance des négatifs faciles
- Force le modèle à se concentrer sur les exemples difficiles
- **Compatible** avec random sampling (pas besoin de features pour identifier hard negatives)

## 🚫 Ce qui n'a PAS fonctionné (à éviter)

### Hard Negative Mining (avec features statiques)

**Pourquoi ça n'a pas marché**:
1. Features statiques ne capturent pas la dynamique temporelle
2. "Similarité" basée sur features != "difficulté" pour le modèle
3. Gap énorme entre training (1 négatif) et evaluation (170K paires)

## 🎯 Recommandations

### Court terme: Utiliser Focal Loss seul

**Commande recommandée** (best config observée):
```bash
python train_self_supervised.py \
  --use_memory \
  --prefix tgn-focal-final \
  --n_epoch 50 \
  --patience 10 \
  --lr 1e-4 \
  --node_dim 200 \
  --time_dim 200 \
  --memory_dim 200 \
  --message_dim 200 \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --n_runs 1
```

**Résultats attendus**:
- Precision@1000: ~0.3% (stable, reproductible)
- vs Random: ~10x meilleur que baseline
- Suffisant pour TechRank (top-1000 contient ~3 vrais liens)

### Moyen terme: Améliorer Hard Negative Mining

Si tu veux réessayer Hard Negatives, il faut **corriger** l'implémentation:

**Option A: Utiliser embeddings temporels** (au lieu de features statiques)

```python
# Au lieu de:
negatives_batch = hard_neg_sampler.sample(
    embeddings=node_features,  # Static features
    ...
)

# Faire:
# 1. Calculer embeddings temporels pour tous les nodes
with torch.no_grad():
    temporal_embeddings = tgn.compute_all_node_embeddings(current_timestamp)

# 2. Utiliser ces embeddings pour hard sampling
negatives_batch = hard_neg_sampler.sample(
    embeddings=temporal_embeddings.cpu().numpy(),  # Temporal embeddings!
    ...
)
```

**Problème**: Très coûteux computationnellement (calculer embeddings pour tous les nodes à chaque batch)

**Option B: Graph-based hard negatives**

Au lieu de similarité de features, utiliser la **structure du graphe**:

```python
# Hard negatives = nodes à distance 2-3 dans le graphe
# (amis d'amis, mais pas directement connectés)

def sample_graph_hard_negatives(src, adjacency, k=2):
    """Sample nodes at distance 2-k from src"""
    # 1-hop neighbors
    neighbors_1hop = adjacency[src]

    # 2-hop neighbors (friends of friends)
    neighbors_2hop = set()
    for neighbor in neighbors_1hop:
        neighbors_2hop.update(adjacency[neighbor])

    # Remove 1-hop neighbors (they're positives)
    hard_negatives = neighbors_2hop - neighbors_1hop - {src}

    return random.sample(hard_negatives, min(len(hard_negatives), n_samples))
```

**Avantage**: Capture la structure du graphe sans besoin de features

**Option C: Tester hyperparamètres plus conservateurs**

Avant d'abandonner complètement, tester:
- `--hard_neg_ratio 0.2` (20% hard, 80% random)
- `--hard_neg_temperature 1.0` (moins agressif)

Voir [test_hard_neg_hyperparams.sh](test_hard_neg_hyperparams.sh)

### Long terme: Autres approches

Si Focal Loss + ajustements ne suffisent pas:

1. **Curriculum Learning**: Augmenter progressivement la difficulté
   - Epochs 1-10: Random negatives
   - Epochs 11-30: 20% hard negatives
   - Epochs 31-50: 50% hard negatives

2. **Multi-task Learning**: Entraîner sur plusieurs tâches simultanément
   - Tâche 1: Link prediction (comme maintenant)
   - Tâche 2: Node classification (prédire type d'investisseur)
   - Tâche 3: Temporal prediction (prédire délai avant investissement)

3. **Ensemble Methods**: Combiner plusieurs modèles
   - Modèle 1: Focal Loss
   - Modèle 2: Weighted BCE
   - Modèle 3: Different architecture
   - Prédiction finale: moyenne/vote des 3

4. **Graph Augmentation**: Enrichir les features avec structure du graphe
   - Node centrality (PageRank, Betweenness)
   - Community detection (modules de co-investissement)
   - Temporal features (activité récente, tendances)

## 📚 Lessons Learned

1. **Mesurer avant d'optimiser**: Focal Loss a bien marché car il s'attaquait au problème identifié (class imbalance)

2. **Features matter**: Hard negatives basés sur features inadéquates peuvent empirer les résultats

3. **Hyperparamètres matter**: ratio=0.5 était peut-être trop agressif pour un premier essai

4. **Gap training/eval**: Toujours se rappeler de l'écart entre comment on entraîne (1 négatif) et comment on évalue (170K paires)

5. **Itération progressive**: Tester une technique à la fois, mesurer, ajuster avant d'ajouter la suivante

## 🎓 Conclusion

Hard Negative Mining est une technique puissante **en théorie**, mais son efficacité dépend fortement de:
- **Qualité des features** utilisées pour identifier les hard negatives
- **Hyperparamètres** (ratio, temperature)
- **Adéquation** avec la tâche finale (gap training/eval)

Dans notre cas:
- ✅ Focal Loss fonctionne bien (9.85x vs random)
- ❌ Hard Negatives (avec features statiques) dégrade les performances (-67%)
- 💡 Rester sur Focal Loss seul pour maintenant
- 🔬 Améliorer Hard Negatives reste une piste future si on a les bonnes features
