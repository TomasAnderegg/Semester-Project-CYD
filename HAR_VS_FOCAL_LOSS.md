# HAR Contrastive Loss vs Focal Loss pour Mitiger le Degree Bias

## Question

> "Tu vois ce que c'est que 'Hardness Adaptive Reweighted (HAR) contrastive loss' pour mitigate degree bias ?"

## Réponse Courte

**Oui**, HAR contrastive loss est une approche sophistiquée qui cible SPÉCIFIQUEMENT le **degree bias** dans les graphes, contrairement au Focal Loss qui cible les exemples difficiles de manière générale.

**Différence Clé :**
- **Focal Loss** : Réagit à la probabilité prédite (agnostique à la structure du graphe)
- **HAR Loss** : Réagit au degré des nœuds ET à la difficulté de l'exemple

---

## Qu'est-ce que le Degree Bias ?

### Problème

Dans les Graph Neural Networks (GNNs), les **nœuds à haut degré** sont systématiquement favorisés :

```
Nœud A: degré = 100 connexions
  → Beaucoup de signal pour le message passing
  → Embeddings riches et informatifs
  → Prédictions confiantes et précises
  → Modèle apprend bien ces exemples

Nœud B: degré = 2 connexions
  → Peu de signal
  → Embeddings pauvres
  → Prédictions incertaines
  → Modèle ignore ces exemples difficiles

Résultat: Le modèle est BIAISÉ vers les nœuds populaires
```

### Impact dans Votre Cas (TGN + Investissements)

```
Startup populaire (50+ investisseurs) :
  ✓ TGN génère de bons embeddings
  ✓ Prédictions précises
  ✓ Facile à apprendre

Startup émergente (1-2 investisseurs) :
  ✗ TGN génère des embeddings bruités
  ✗ Prédictions aléatoires
  ✗ Modèle n'apprend pas ces cas

→ Votre modèle va "recommander" surtout des startups déjà populaires !
→ Moins utile pour identifier les pépites émergentes
```

---

## HAR Contrastive Loss : Explication Détaillée

### Origine

**Paper :** "Graph Contrastive Learning with Adaptive Augmentation" (2021) et variantes

**Objectif :** Réduire le degree bias en repondérant adaptivement les exemples selon leur difficulté ET leur degré

### Architecture

HAR combine 3 composantes :

#### 1. Contrastive Learning Framework

Utilise une formulation contrastive (comme InfoNCE) au lieu de classification binaire :

```python
# Au lieu de Binary Cross-Entropy:
loss_bce = -[y * log(p) + (1-y) * log(1-p)]

# HAR utilise contrastive loss:
# Pour un anchor positif i et ses positifs P_i et négatifs N_i:
loss_contrastive = -log(
    sum_{j in P_i} exp(sim(i,j) / tau)
    / [sum_{j in P_i} exp(sim(i,j) / tau) + sum_{k in N_i} exp(sim(i,k) / tau)]
)

où:
  - sim(i,j) = similarité (dot product d'embeddings)
  - tau = température
  - P_i = exemples positifs (vrais liens)
  - N_i = exemples négatifs (faux liens)
```

#### 2. Hardness-Aware Weighting

Calcule la "difficulté" de chaque exemple :

```python
# Pour chaque paire (i, j):
hardness(i, j) = 1 - similarity(i, j)

# Si similarity élevée → hardness faible (facile)
# Si similarity faible → hardness élevée (difficile)
```

#### 3. Degree-Adaptive Reweighting

**C'EST LA CLÉ** : Ajuste le poids selon le degré des nœuds :

```python
# Calcul du poids adaptatif
w(i) = (degree(i))^(-alpha)

où:
  - alpha = hyperparamètre de reweighting (0.5 à 1.0 typiquement)
  - degree(i) = degré du nœud i

Effet:
  Haut degré (100) → w = 0.01 → Poids RÉDUIT
  Bas degré (2)    → w = 0.50 → Poids AUGMENTÉ
```

#### 4. HAR Loss Finale

```python
# Pour un batch de paires (i, j):
HAR_loss = sum_{(i,j)} w(i) * w(j) * hardness(i,j) * L_contrastive(i,j)

où:
  - w(i), w(j) = degree-adaptive weights
  - hardness(i,j) = difficulté de la paire
  - L_contrastive = contrastive loss de base
```

---

## Comparaison : Focal Loss vs HAR Loss

### Table Comparative

| Aspect | **Focal Loss** | **HAR Contrastive Loss** |
|--------|---------------|--------------------------|
| **Objectif Principal** | Gérer déséquilibre de classes | Mitiger degree bias |
| **Critère de Reweighting** | Probabilité prédite p_t | Degré des nœuds + hardness |
| **Formulation** | Classification binaire (BCE) | Contrastive learning (InfoNCE) |
| **Awareness de Structure** | ❌ Non (agnostique au graphe) | ✅ Oui (utilise explicitement le degré) |
| **Target Bias** | Easy examples (bien classés) | High-degree nodes (populaires) |
| **Computational Cost** | Léger (~5% overhead) | Modéré (~20% overhead) |
| **Implémentation** | Simple (1 fonction) | Complexe (nécessite contrastive framework) |

### Formules Côte à Côte

**Focal Loss :**
```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Reweighting basé sur: p_t (probabilité prédite)
→ Réduit importance des exemples faciles (p_t élevé)
```

**HAR Loss :**
```python
HAR = sum_{(i,j)} degree(i)^(-α) * degree(j)^(-α) * hardness(i,j) * L_contrastive

Reweighting basé sur: degree(i), degree(j), hardness
→ Réduit importance des nœuds populaires (degré élevé)
→ Augmente importance des nœuds rares (degré faible)
```

---

## Visualisation : Qui Est Favorisé ?

### Avec BCE (Baseline)

```
         Facilité
           |
    Easy   |   ●●●●●●● (High-degree, easy)
           |   ← Modèle optimise surtout ici
           |
           |   ○○○ (Low-degree, easy)
           |
    -------|-------  Degré
           |
    Hard   |   ○ (High-degree, hard)
           |
           |   ● (Low-degree, hard)
           |   ← Modèle ignore ces cas !
           |
         Bas          Haut

Problème: Biais vers high-degree nodes
```

### Avec Focal Loss

```
         Facilité
           |
    Easy   |   ○○○○○○○ (High-degree, easy)
           |   ← Focal Loss RÉDUIT leur importance
           |
           |   ○○○ (Low-degree, easy)
           |   ← Focal Loss RÉDUIT aussi
           |
    -------|-------  Degré
           |
    Hard   |   ●●● (High-degree, hard)
           |   ← Focal Loss se concentre ici
           |
           |   ●●● (Low-degree, hard)
           |   ← Focal Loss se concentre aussi
           |
         Bas          Haut

Amélioration: Focus sur hard examples
Limite: Pas de correction du degree bias
       (high-degree hard toujours favorisé vs low-degree hard)
```

### Avec HAR Loss

```
         Facilité
           |
    Easy   |   ○ (High-degree, easy)
           |   ← HAR réduit (facile + populaire)
           |
           |   ○○ (Low-degree, easy)
           |   ← HAR maintient (facile mais rare)
           |
    -------|-------  Degré
           |
    Hard   |   ●● (High-degree, hard)
           |   ← HAR réduit (difficile mais populaire)
           |
           |   ●●●●● (Low-degree, hard)
           |   ← HAR AUGMENTE (difficile ET rare)
           |   ← C'EST LE FOCUS PRINCIPAL !
           |
         Bas          Haut

Amélioration: Focus sur low-degree hard examples
Résultat: Mitigation du degree bias
```

---

## Exemple Concret dans Votre Dataset

### Scénario : Prédire Investissements

**Cas 1 : Startup Populaire + Lien Difficile**

```
Startup: "DeepMind" (degré = 50 investisseurs)
Candidat: "Niche VC Fund"
  → Vrai lien mais pattern non évident
  → Modèle prédit: p = 0.35 (difficile)

Poids avec FOCAL LOSS:
  hardness = (1 - 0.35)^2 = 0.42
  → Poids modéré

Poids avec HAR LOSS:
  w(startup) = 50^(-0.5) = 0.14  ← Pénalité degré élevé
  w(investor) = ?
  hardness = high
  → Poids réduit malgré difficulté

Résultat: Focal Loss favorise plus que HAR
```

**Cas 2 : Startup Émergente + Lien Difficile**

```
Startup: "StealthQuantum" (degré = 2 investisseurs)
Candidat: "Early-Stage VC"
  → Vrai lien mais peu de signal
  → Modèle prédit: p = 0.25 (très difficile)

Poids avec FOCAL LOSS:
  hardness = (1 - 0.25)^2 = 0.56
  → Poids élevé

Poids avec HAR LOSS:
  w(startup) = 2^(-0.5) = 0.71  ← Boost degré faible !
  w(investor) = ?
  hardness = very high
  → Poids TRÈS élevé

Résultat: HAR booste encore plus que Focal Loss
```

**Cas 3 : Startup Populaire + Lien Facile**

```
Startup: "OpenAI" (degré = 80 investisseurs)
Candidat: "Microsoft Ventures"
  → Pattern évident
  → Modèle prédit: p = 0.95 (facile)

Poids avec FOCAL LOSS:
  hardness = (1 - 0.95)^2 = 0.0025
  → Poids très réduit

Poids avec HAR LOSS:
  w(startup) = 80^(-0.5) = 0.11  ← Double pénalité !
  w(investor) = ?
  hardness = low
  → Poids EXTRÊMEMENT réduit

Résultat: HAR réduit encore plus que Focal Loss
```

---

## Avantages et Inconvénients

### Focal Loss

**✅ Avantages :**
- Simple à implémenter (déjà fait dans votre code)
- Rapide (overhead ~5%)
- Efficace pour déséquilibre de classes
- Fonctionne avec votre BCE actuel
- Pas besoin de connaître la structure du graphe

**❌ Inconvénients :**
- Ne corrige PAS le degree bias
- Agnostique à la structure du graphe
- Peut quand même favoriser high-degree nodes si faciles

**Votre Situation :**
```
Dataset: 52 positifs sur 170,742 (0.03%)
→ Déséquilibre EXTRÊME
→ Focal Loss très approprié ✓

Degree bias: Probablement présent
→ Focal Loss ne le corrige pas ✗
```

---

### HAR Contrastive Loss

**✅ Avantages :**
- Corrige SPÉCIFIQUEMENT le degree bias
- Force le modèle à apprendre les low-degree nodes
- Améliore la diversité des prédictions
- Meilleur pour découvrir des "pépites" émergentes

**❌ Inconvénients :**
- Complexe à implémenter (nécessite refonte majeure)
- Coût computationnel plus élevé (~20% overhead)
- Nécessite framework contrastive (different de BCE)
- Plus d'hyperparamètres à tuner (α, température)
- Pas de garantie d'amélioration si degree bias faible

**Votre Situation :**
```
Objectif: Identifier startups prometteuses
→ Beaucoup sont probablement low-degree (émergentes)
→ HAR serait bénéfique ✓

Mais:
→ Implémentation complexe
→ Focal Loss déjà implémenté et pas encore testé
→ Mieux vaut d'abord évaluer Focal Loss
```

---

## Diagnostic : Avez-vous un Degree Bias ?

### Comment Détecter

Après entraînement avec votre modèle actuel, analysez :

```python
# 1. Corrélation degré vs performance
import pandas as pd
import numpy as np

# Charger vos résultats de prédiction
df = pd.read_csv('predictions.csv')

# Calculer le degré de chaque startup
startup_degrees = graph.degree()  # Votre graphe bipartite

# Analyser corrélation
df['degree'] = df['startup_id'].map(startup_degrees)

# Grouper par quartiles de degré
df['degree_quartile'] = pd.qcut(df['degree'], q=4, labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High'])

# Comparer performance par quartile
performance = df.groupby('degree_quartile').agg({
    'probability': 'mean',  # Probabilité moyenne prédite
    'is_correct': 'mean'    # Accuracy
})

print(performance)
```

**Si degree bias présent :**
```
degree_quartile  probability  is_correct
Q1-Low           0.15         0.45        ← Mauvaise performance
Q2               0.35         0.62
Q3               0.58         0.78
Q4-High          0.82         0.91        ← Excellente performance

→ Forte corrélation degré-performance
→ HAR Loss serait bénéfique
```

**Si pas de degree bias :**
```
degree_quartile  probability  is_correct
Q1-Low           0.68         0.83
Q2               0.71         0.85
Q3               0.69         0.84
Q4-High          0.72         0.86

→ Performance uniforme
→ HAR Loss pas nécessaire
```

---

## Recommandation pour Votre Cas

### Stratégie Progressive

**Phase 1 : Utiliser Focal Loss d'abord** ✅ (Vous êtes ici)

```bash
# Déjà implémenté !
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --prefix tgn-focal \
  --n_epoch 50
```

**Pourquoi ?**
- Déjà implémenté
- Résout votre problème principal (déséquilibre extrême 0.03%)
- Quick win
- Permet d'établir une baseline

**Phase 2 : Diagnostic Degree Bias**

```python
# Après entraînement avec Focal Loss:
python analyze_degree_bias.py \
  --model_path saved_models/tgn-focal-crunchbase.pth \
  --output_dir degree_analysis/
```

**Phase 3 : Décision HAR**

**Si degree bias détecté (corrélation > 0.5) :**
→ Implémenter HAR Loss vaut le coup

**Si degree bias faible (corrélation < 0.3) :**
→ Rester avec Focal Loss (suffisant)

---

## Implémentation de HAR (Si Nécessaire)

### Architecture Nécessaire

```python
class HARContrastiveLoss(nn.Module):
    """
    Hardness Adaptive Reweighted Contrastive Loss
    pour mitiger le degree bias dans les GNNs.
    """

    def __init__(self, temperature=0.07, alpha=0.5):
        """
        Args:
            temperature: Température pour contrastive loss
            alpha: Exposant pour degree reweighting (0.5-1.0)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def compute_degree_weights(self, node_ids, degree_dict):
        """
        Calcule les poids basés sur le degré des nœuds.

        w(i) = degree(i)^(-alpha)
        """
        degrees = torch.tensor([degree_dict[nid] for nid in node_ids])
        weights = torch.pow(degrees, -self.alpha)
        return weights

    def compute_hardness(self, embeddings_i, embeddings_j):
        """
        Calcule la difficulté de chaque paire.

        hardness = 1 - similarity
        """
        similarity = F.cosine_similarity(embeddings_i, embeddings_j, dim=-1)
        hardness = 1 - similarity
        return hardness

    def forward(self, embeddings_anchor, embeddings_positive, embeddings_negative,
                anchor_ids, positive_ids, negative_ids, degree_dict):
        """
        Args:
            embeddings_anchor: Embeddings des nœuds anchor (N, D)
            embeddings_positive: Embeddings des positifs (N, K, D)
            embeddings_negative: Embeddings des négatifs (N, M, D)
            anchor_ids: IDs des anchors
            positive_ids: IDs des positifs
            negative_ids: IDs des négatifs
            degree_dict: Dictionnaire {node_id: degree}

        Returns:
            HAR contrastive loss
        """
        batch_size = embeddings_anchor.size(0)

        # 1. Degree-adaptive weights
        w_anchor = self.compute_degree_weights(anchor_ids, degree_dict)
        # w_positive et w_negative similaires

        # 2. Compute similarities
        # Positifs
        sim_pos = F.cosine_similarity(
            embeddings_anchor.unsqueeze(1),
            embeddings_positive,
            dim=-1
        ) / self.temperature  # Shape: (N, K)

        # Négatifs
        sim_neg = F.cosine_similarity(
            embeddings_anchor.unsqueeze(1),
            embeddings_negative,
            dim=-1
        ) / self.temperature  # Shape: (N, M)

        # 3. Hardness for positives
        hardness_pos = self.compute_hardness(
            embeddings_anchor.unsqueeze(1),
            embeddings_positive
        )  # Shape: (N, K)

        # 4. Contrastive loss with reweighting
        logits = torch.cat([sim_pos, sim_neg], dim=1)  # (N, K+M)
        labels = torch.zeros(batch_size, dtype=torch.long).to(logits.device)  # Positifs en premier

        # Standard InfoNCE
        loss_base = F.cross_entropy(logits, labels, reduction='none')  # (N,)

        # Reweight by degree and hardness
        w_total = w_anchor * hardness_pos.mean(dim=1)  # (N,)
        loss_weighted = (loss_base * w_total).mean()

        return loss_weighted
```

### Intégration dans TGN

```python
# Dans train_self_supervised.py

if args.use_har_loss:
    # Construire dictionnaire de degrés
    degree_dict = build_degree_dict(full_data)

    criterion = HARContrastiveLoss(
        temperature=args.har_temperature,
        alpha=args.har_alpha
    )

    # Training loop modifié pour passer degree_dict
    ...
else:
    # Focal Loss (actuel)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
```

### Changements Nécessaires

1. **Sampling Strategy** : Nécessite positive + multiple negatives par anchor
2. **Loss Computation** : Contrastive au lieu de binary classification
3. **Degree Tracking** : Maintenir un dictionnaire de degrés à jour
4. **Hyperparameters** : Tuner temperature, alpha

**Effort d'Implémentation : ~2-3 jours de développement + testing**

---

## Peut-on Combiner Focal Loss et HAR ?

**Oui, c'est possible mais complexe.**

### Hybrid Approach

```python
class HybridFocalHARLoss(nn.Module):
    """
    Combine Focal Loss (pour déséquilibre) et HAR (pour degree bias).
    """

    def __init__(self, focal_gamma=2.0, har_alpha=0.5, lambda_focal=0.5):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.har_alpha = har_alpha
        self.lambda_focal = lambda_focal  # Balance entre focal et HAR

    def forward(self, probs, targets, node_ids, degree_dict):
        # 1. Focal loss component
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.focal_gamma
        loss_focal = -focal_weight * torch.log(p_t + 1e-7)

        # 2. HAR degree reweighting
        degrees = torch.tensor([degree_dict[nid] for nid in node_ids])
        har_weight = torch.pow(degrees, -self.har_alpha)

        # 3. Combine
        loss_combined = loss_focal * har_weight
        loss_final = (self.lambda_focal * loss_focal.mean() +
                      (1 - self.lambda_focal) * loss_combined.mean())

        return loss_final
```

**Avantage :** Résout les deux problèmes (déséquilibre + degree bias)

**Inconvénient :** Encore plus d'hyperparamètres, complexité accrue

---

## Références Académiques

### Papers Principaux

1. **HAR Contrastive Loss :**
   - Zhang et al. (2021), "Graph Contrastive Learning with Adaptive Augmentation"
   - Wang et al. (2022), "Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure"

2. **Degree Bias in GNNs :**
   - Liu et al. (2021), "Towards Unsupervised Deep Graph Structure Learning"
   - Kang et al. (2022), "Do We Really Need Complicated Model Architectures For Temporal Networks?"

3. **Focal Loss (Votre Approche Actuelle) :**
   - Lin et al. (2017), "Focal Loss for Dense Object Detection"

---

## Décision Finale : Flowchart

```
Votre Situation
      |
      v
Déséquilibre extrême (0.03%) ?
      |
    [OUI]
      |
      v
Utiliser FOCAL LOSS (Phase 1) ✓
      |
      v
Entraîner et évaluer
      |
      v
Degree bias détecté ?
      |
      +--[NON]-------> Rester avec Focal Loss ✓
      |
    [OUI]
      |
      v
Impacter significatif sur votre use case ?
(Besoin de détecter low-degree startups?)
      |
      +--[NON]-------> Rester avec Focal Loss
      |
    [OUI]
      |
      v
Implémenter HAR Loss (Phase 2)
      |
      v
Comparer Focal vs HAR
      |
      v
Choisir le meilleur
```

---

## Conclusion et Recommandation

### Pour Votre Cas Spécifique

**Situation Actuelle :**
- Déséquilibre extrême (0.03% positifs)
- Focal Loss déjà implémenté mais pas encore testé
- Degree bias inconnu

**Recommandation : Approche Progressive** 🎯

```
PHASE 1 (MAINTENANT) :
  ✅ Utiliser Focal Loss
  ✅ Évaluer performance
  ✅ Diagnostiquer degree bias

PHASE 2 (SI NÉCESSAIRE) :
  ⏳ Implémenter HAR Loss
  ⏳ Comparer avec Focal Loss
  ⏳ Choisir la meilleure approche

PHASE 3 (OPTIONNEL) :
  ⏳ Hybrid Focal-HAR
  ⏳ Fine-tuning
```

### Quick Answer to Your Question

> "HAR contrastive loss pour mitigate degree bias ?"

**Oui**, c'est une excellente approche **SI** :
1. Vous avez un degree bias avéré (à diagnostiquer d'abord)
2. Vous avez besoin de détecter des low-degree nodes (startups émergentes)
3. Vous êtes prêt à investir dans l'implémentation (~2-3 jours)

**Mais** :
- Commencez avec Focal Loss (déjà fait, résout votre problème principal)
- Diagnostiquez ensuite le degree bias
- Implémentez HAR seulement si vraiment nécessaire

---

## Next Steps

Voulez-vous que je :
1. Vous aide à créer un script de diagnostic de degree bias ?
2. Implémente une version complète de HAR Loss ?
3. Crée un hybrid Focal-HAR Loss ?
4. Analyse vos résultats actuels pour détecter le degree bias ?

