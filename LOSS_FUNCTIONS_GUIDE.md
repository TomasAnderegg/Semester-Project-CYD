# Guide des Loss Functions : BCE vs Focal Loss vs HAR Loss

## Vue d'Ensemble

Votre système TGN supporte maintenant **3 loss functions** pour l'entraînement :

| Loss Function | Problème Ciblé | Quand l'Utiliser |
|---------------|----------------|------------------|
| **BCE (Baseline)** | Aucun (standard) | Baseline de référence |
| **Focal Loss** | Déséquilibre de classes extrême | Dataset avec peu de positifs |
| **HAR Loss** | Degree bias dans les graphes | Favoriser les nœuds à faible degré |

---

## 1. Binary Cross-Entropy (BCE) - Baseline

### Description

La loss function standard pour la classification binaire.

```python
BCE = -[y * log(p) + (1-y) * log(1-p)]
```

### Avantages
- Simple et bien comprise
- Rapide (pas d'overhead)
- Bonne baseline de référence

### Inconvénients
- Sensible au déséquilibre de classes
- Biaisée vers les nœuds populaires (high-degree)
- Les exemples faciles dominent l'entraînement

### Utilisation

```bash
# Option 1: Par défaut (sans flag)
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --prefix tgn-bce \
  --n_epoch 50

# Option 2: Explicite (pour clarté)
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --prefix tgn-bce-baseline \
  --n_epoch 50
```

**Note :** Si aucun flag `--use_focal_loss` ou `--use_har_loss` n'est spécifié, BCE est utilisée par défaut.

---

## 2. Focal Loss - Pour Déséquilibre de Classes

### Description

Focal Loss réduit l'importance des exemples bien classés (easy examples) pour se concentrer sur les exemples difficiles.

```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Paramètres :**
- **gamma (γ)** : Focusing parameter (défaut: 2.0)
  - γ=0 → équivalent à BCE
  - γ=2 → réduction forte des easy examples
  - γ=5 → très agressif

- **alpha (α)** : Poids pour la classe positive (défaut: 0.25)
  - α=0.25 → classe positive a un poids de 25%
  - α=0.5 → poids égal entre positifs et négatifs

### Quand l'Utiliser

✅ **Votre cas (RECOMMANDÉ) :**
```
Dataset: 52 positifs sur 170,742 paires (0.03%)
→ Déséquilibre extrême
→ Focal Loss est idéal
```

✅ **Autres cas :**
- Ratio positifs/négatifs < 1%
- Médiane des probabilités pour vrais liens < 0.3
- Besoin de détecter des patterns rares

❌ **NE PAS utiliser si :**
- Dataset équilibré (ratio ~50/50)
- Tous les exemples sont déjà difficiles

### Utilisation

```bash
# Configuration par défaut (recommandée)
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --prefix tgn-focal \
  --n_epoch 50

# Pour déséquilibre TRÈS extrême
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.1 \
  --focal_gamma 2.0 \
  --prefix tgn-focal-aggressive \
  --n_epoch 50

# Pour focalisation plus forte
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 5.0 \
  --prefix tgn-focal-gamma5 \
  --n_epoch 50
```

### Résultats Attendus

**Avant (BCE) :**
```
Médiane probabilité vrais liens: 0.04
Recall@1000: 7.7%
```

**Après (Focal Loss) :**
```
Médiane probabilité vrais liens: 0.25-0.40 (espéré)
Recall@1000: 15-25% (espéré)
→ Amélioration 2-3x
```

---

## 3. HAR Loss - Pour Degree Bias

### Description

HAR (Hardness Adaptive Reweighted) Loss combat le degree bias en donnant plus de poids aux nœuds à faible degré.

```python
HAR_loss = sum_i [ w(src_i) * w(dst_i) * L_contrastive(i) ]

où w(node) = degree(node)^(-alpha)
```

**Paramètres :**
- **temperature** : Température pour contrastive loss (défaut: 0.07)
  - Plus basse → discrimination plus stricte
  - Plus haute → plus permissive

- **alpha** : Exposant de reweighting par degré (défaut: 0.5)
  - α=0 → pas de correction (équivalent à ignorer le degré)
  - α=0.5 → correction modérée (RECOMMANDÉ)
  - α=1.0 → correction forte

### Mécanisme

```python
# Exemple avec alpha = 0.5
Nœud haut degré (100) → weight = 100^(-0.5) = 0.10  ← Réduit
Nœud bas degré (2)    → weight = 2^(-0.5)   = 0.71  ← Augmenté

→ Les nœuds à faible degré contribuent 7x plus à la loss !
```

### Quand l'Utiliser

✅ **Utilisez HAR Loss si :**
- Vous voulez identifier des **startups émergentes** (low-degree)
- Vous avez détecté un **degree bias** (corrélation degré-performance > 0.5)
- Votre modèle ignore les nœuds rares
- Vous cherchez des "pépites" avant qu'elles deviennent populaires

❌ **NE PAS utiliser si :**
- Vous ciblez surtout les nœuds populaires
- Pas de degree bias détecté (performance uniforme par degré)
- Les low-degree nodes sont peu informatifs (trop de bruit)

### Utilisation

```bash
# Configuration par défaut (recommandée)
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_har_loss \
  --har_temperature 0.07 \
  --har_alpha 0.5 \
  --prefix tgn-har \
  --n_epoch 50

# Correction degree bias plus forte
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_har_loss \
  --har_temperature 0.07 \
  --har_alpha 0.75 \
  --prefix tgn-har-strong \
  --n_epoch 50

# Température plus élevée (plus permissif)
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_har_loss \
  --har_temperature 0.15 \
  --har_alpha 0.5 \
  --prefix tgn-har-temp015 \
  --n_epoch 50
```

### Diagnostic : Avez-vous un Degree Bias ?

**Après entraînement avec BCE/Focal, analysez :**

```python
import pandas as pd
import numpy as np

# Charger prédictions
df = pd.read_csv('predictions.csv')

# Calculer degrés
startup_degrees = {}  # {startup_id: degree}
for src, dst in zip(train_data.sources, train_data.destinations):
    startup_degrees[src] = startup_degrees.get(src, 0) + 1

df['degree'] = df['startup_id'].map(startup_degrees)

# Grouper par quartiles de degré
df['degree_quartile'] = pd.qcut(df['degree'], q=4, labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High'])

# Comparer performance
performance = df.groupby('degree_quartile').agg({
    'probability': 'mean',
    'is_correct': 'mean'
})

print(performance)
```

**Interprétation :**

```
# SI DEGREE BIAS PRÉSENT:
degree_quartile  probability  is_correct
Q1-Low           0.15         0.45        ← Mauvais
Q2               0.35         0.62
Q3               0.58         0.78
Q4-High          0.82         0.91        ← Excellent

→ HAR Loss recommandée

# SI PAS DE DEGREE BIAS:
degree_quartile  probability  is_correct
Q1-Low           0.68         0.83
Q2               0.71         0.85
Q3               0.69         0.84
Q4-High          0.72         0.86

→ HAR Loss pas nécessaire
```

---

## Comparaison Complète

### Tableau Récapitulatif

| Aspect | BCE | Focal Loss | HAR Loss |
|--------|-----|------------|----------|
| **Déséquilibre classes** | ❌ Mauvais | ✅ Excellent | ⚠️ Moyen |
| **Degree bias** | ❌ Pas de correction | ❌ Pas de correction | ✅ Corrige |
| **Exemples faciles** | Dominent | Ignorés | Selon degré |
| **Exemples difficiles** | Standard | Focalisés | + Focalisés si low-degree |
| **Overhead computationnel** | Baseline | +5% | +10% |
| **Complexité** | Simple | Simple | Modérée |
| **Hyperparamètres** | 0 | 2 (alpha, gamma) | 2 (temperature, alpha) |

### Exemple Concret

**Scénario : Prédire investissement**

```
Startup A: "DeepMind" (degré=50, pattern évident)
→ Modèle prédit p=0.95 (facile)

BCE:        loss = 0.05
Focal:      loss = 0.0025    ← Ignoré (facile)
HAR:        loss = 0.007     ← Réduit (haut degré)

----

Startup B: "StealthQuantum" (degré=2, pattern difficile)
→ Modèle prédit p=0.25 (difficile)

BCE:        loss = 1.39
Focal:      loss = 0.78      ← Focalisé (difficile)
HAR:        loss = 0.99      ← TRÈS focalisé (bas degré + difficile)
```

---

## Stratégie de Comparaison Recommandée

### Phase 1 : Baseline

```bash
# Entraîner avec BCE (baseline)
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --prefix tgn-bce \
  --n_epoch 50
```

### Phase 2 : Focal Loss (Votre Priorité)

```bash
# Entraîner avec Focal Loss
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --prefix tgn-focal \
  --n_epoch 50
```

### Phase 3 : HAR Loss (Si Degree Bias)

```bash
# Entraîner avec HAR Loss
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_har_loss \
  --har_temperature 0.07 \
  --har_alpha 0.5 \
  --prefix tgn-har \
  --n_epoch 50
```

### Phase 4 : Évaluation

```bash
# Évaluer tous les modèles
for model in bce focal har; do
  python temporal_validation_diagnostic.py \
    --data crunchbase \
    --model_path saved_models/tgn-${model}-crunchbase.pth \
    --use_memory \
    --auto_detect_params
done
```

### Phase 5 : Comparaison

```python
import pandas as pd

# Charger résultats
results = []
for model in ['bce', 'focal', 'har']:
    df = pd.read_csv(f'results/tgn-{model}-results.csv')
    df['model'] = model
    results.append(df)

results_df = pd.concat(results)

# Comparer métriques
comparison = results_df.groupby('model').agg({
    'recall@1000': 'mean',
    'precision@1000': 'mean',
    'median_rank_true_links': 'median'
})

print(comparison)
```

---

## Métriques à Surveiller

### Pour Focal Loss

| Métrique | Baseline (BCE) | Cible (Focal) |
|----------|----------------|---------------|
| Médiane prob vrais liens | 0.04 | 0.25-0.40 |
| Recall@1000 | 7.7% | 15-25% |
| Rang médian vrais liens | 6,609 | <5,000 |

### Pour HAR Loss

| Métrique | Baseline | Cible (HAR) |
|----------|----------|-------------|
| Performance low-degree | 0.45 | 0.70+ |
| Performance high-degree | 0.91 | 0.80-0.90 (peut baisser) |
| Diversité prédictions | Faible | Élevée |

---

## Troubleshooting

### Problème : Focal Loss donne de moins bons résultats que BCE

**Causes possibles :**
1. Gamma trop élevé (modèle ignore trop d'exemples)
2. Alpha mal calibré
3. Dataset pas assez déséquilibré

**Solutions :**
```bash
# Réduire gamma
--focal_gamma 1.0  # Au lieu de 2.0

# Ajuster alpha
--focal_alpha 0.5  # Au lieu de 0.25
```

### Problème : HAR Loss ne converge pas

**Causes possibles :**
1. Alpha trop élevé (correction trop agressive)
2. Température trop basse
3. Degrés mal calculés

**Solutions :**
```bash
# Réduire alpha
--har_alpha 0.25  # Au lieu de 0.5

# Augmenter température
--har_temperature 0.15  # Au lieu de 0.07
```

### Problème : Pas d'amélioration avec HAR Loss

**Diagnostic :**
- Vérifiez s'il y a vraiment un degree bias (voir section diagnostic)
- Si pas de degree bias → HAR Loss n'est pas nécessaire
- Restez avec Focal Loss

---

## Combinaison Focal + HAR ?

**Actuellement NON supporté**, mais vous pouvez :

1. **Approche séquentielle :**
   ```bash
   # Étape 1: Pré-entraîner avec Focal Loss
   python train_self_supervised.py \
     --use_focal_loss --prefix tgn-focal --n_epoch 30

   # Étape 2: Fine-tuner avec HAR Loss
   python train_self_supervised.py \
     --use_har_loss --prefix tgn-har-finetune --n_epoch 20 \
     --load_checkpoint saved_models/tgn-focal-crunchbase.pth
   ```

2. **Implémenter Hybrid Loss** (nécessite développement)

---

## Références

### Papers

1. **Focal Loss:**
   - Lin et al. (2017), "Focal Loss for Dense Object Detection"
   - https://arxiv.org/abs/1708.02002

2. **HAR Loss:**
   - Zhang et al. (2021), "Graph Contrastive Learning with Adaptive Augmentation"
   - Wang et al. (2022), "Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure"

### Fichiers Code

- `focal_loss.py` : Implémentation Focal Loss
- `har_loss.py` : Implémentation HAR Loss
- `train_self_supervised.py` : Script d'entraînement avec les 3 loss

---

## Résumé : Quelle Loss Choisir ?

```
Votre Dataset: 0.03% positifs, probable degree bias

┌─────────────────────────────────────────┐
│ RECOMMANDATION POUR VOUS :              │
│                                         │
│ 1. Commencer avec FOCAL LOSS ✅         │
│    → Résout votre déséquilibre extrême │
│                                         │
│ 2. Diagnostiquer degree bias           │
│    → Analyser performance par degré    │
│                                         │
│ 3. Si degree bias détecté:             │
│    → Tester HAR LOSS                   │
│                                         │
│ 4. Comparer les 3 approches            │
│    → Choisir la meilleure              │
└─────────────────────────────────────────┘
```

**Prochaine étape :** Lancer `python train_self_supervised.py --use_focal_loss` pour commencer ! 🚀
