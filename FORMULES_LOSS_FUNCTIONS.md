# Formules Mathématiques : Loss Functions

## Table des Matières

1. [Binary Cross-Entropy (BCE) - Baseline](#1-binary-cross-entropy-bce---baseline)
2. [Focal Loss](#2-focal-loss)
3. [HAR Loss (Hardness Adaptive Reweighted)](#3-har-loss-hardness-adaptive-reweighted)
4. [Hybrid Focal-HAR Loss](#4-hybrid-focal-har-loss)
5. [Comparaison des Composantes](#5-comparaison-des-composantes)

---

## 1. Binary Cross-Entropy (BCE) - Baseline

### Formule

```
L_BCE(p, y) = -[y · log(p) + (1 - y) · log(1 - p)]
```

**Où :**
- `p` : Probabilité prédite par le modèle (sortie sigmoid)
- `y ∈ {0, 1}` : Vérité terrain (label)

### Notation Compacte

```
L_BCE = -log(p_t)

où p_t = {
    p       si y = 1
    1 - p   si y = 0
}
```

### Implémentation PyTorch

```python
criterion = torch.nn.BCELoss()
loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
```

**Problème :** Traite tous les exemples de manière égale → mauvais pour déséquilibre de classes

---

## 2. Focal Loss

### Formule Complète

```
L_Focal(p, y) = -α_t · (1 - p_t)^γ · log(p_t)
```

**Où :**
- `p_t` : Probabilité de la vraie classe (comme BCE)
- `γ ≥ 0` : **Focusing parameter** (typiquement γ = 2)
- `α_t` : **Class balancing weight**

### Décomposition

```
L_Focal = α_t · w_focal · L_BCE

où:
  L_BCE = -log(p_t)                    [Baseline]
  w_focal = (1 - p_t)^γ                [Modulating factor]
  α_t = {α si y=1, (1-α) si y=0}       [Class weights]
```

### Comportement du Modulating Factor

```
Si p_t → 1 (exemple facile, bien classé):
  w_focal = (1 - p_t)^γ → 0
  → Poids réduit, contribue peu à la loss

Si p_t → 0 (exemple difficile, mal classé):
  w_focal = (1 - p_t)^γ → 1
  → Poids maximal, contribue beaucoup à la loss
```

### Exemple Numérique

| p_t (confiance) | BCE      | w_focal (γ=2) | Focal Loss (α=0.25) |
|-----------------|----------|---------------|---------------------|
| 0.95 (facile)   | 0.051    | 0.0025        | **0.0003**          |
| 0.50 (incertain)| 0.693    | 0.25          | **0.043**           |
| 0.05 (difficile)| 2.996    | 0.9025        | **0.676**           |

**Observation :** Focal Loss amplifie les exemples difficiles et réduit drastiquement les faciles.

### Paramètres Standards

```
α = 0.25   (favorise classe minoritaire dans ratio 1:3)
γ = 2.0    (réduction modérée des exemples faciles)
```

### Formule LaTeX

```latex
\mathcal{L}_{\text{Focal}}(p, y) = -\alpha_t (1 - p_t)^\gamma \log(p_t)

\text{où } p_t =
\begin{cases}
p & \text{si } y = 1 \\
1 - p & \text{si } y = 0
\end{cases}
```

### Implémentation ([focal_loss.py:13-36](focal_loss.py#L13-L36))

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # BCE base
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # p_t: probabilité de la vraie classe
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Class balancing weight alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Loss finale
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()
```

**Résout :** Déséquilibre de classes + ignore les exemples faciles

---

## 3. HAR Loss (Hardness Adaptive Reweighted)

### Formule Complète

```
L_HAR = w_degree · L_contrastive

où:
  w_degree(i, j) = [degree(i)]^(-α) · [degree(j)]^(-α)
  L_contrastive = InfoNCE style loss
```

### Poids par Degré (Reweighting Function)

```
w(node) = [degree(node)]^(-α)

où:
  α > 0 : Exposant de reweighting (typiquement α = 0.5)
```

**Comportement :**

```
Nœud haute popularité (degree = 100):
  w = 100^(-0.5) = 0.1    → Poids RÉDUIT

Nœud moyenne popularité (degree = 10):
  w = 10^(-0.5) ≈ 0.316   → Poids modéré

Nœud faible popularité (degree = 2):
  w = 2^(-0.5) ≈ 0.707    → Poids ÉLEVÉ
```

### Poids pour une Paire (src, dst)

```
w_pair(i, j) = w(i) · w(j)
             = [degree(i)]^(-α) · [degree(j)]^(-α)
             = [degree(i) · degree(j)]^(-α)
```

### Contrastive Loss (InfoNCE Style)

```
L_contrastive = -log( exp(s_pos / τ) / [exp(s_pos / τ) + exp(s_neg / τ)] )

où:
  s_pos : Score du lien positif
  s_neg : Score du lien négatif
  τ : Temperature (typiquement τ = 0.07)
```

### Formule Complète HAR

```
L_HAR = w_pair · L_contrastive
      = [degree(i) · degree(j)]^(-α) · (-log[exp(s_pos/τ) / Z])

où Z = exp(s_pos/τ) + exp(s_neg/τ)
```

### Exemple Numérique

| Paire              | degree(i) | degree(j) | w_pair (α=0.5) | L_contrastive | L_HAR   |
|--------------------|-----------|-----------|----------------|---------------|---------|
| DeepMind → Google  | 50        | 100       | 0.014          | 0.5           | **0.007** |
| Startup → VC       | 10        | 20        | 0.071          | 0.5           | **0.035** |
| Emergent → Angel   | 2         | 3         | 0.408          | 0.5           | **0.204** |

**Observation :** HAR booste l'importance des paires low-degree.

### Paramètres Standards

```
α = 0.5         (correction modérée)
τ = 0.07        (temperature pour contrastive loss)
```

### Formule LaTeX

```latex
\mathcal{L}_{\text{HAR}} = w_{\text{degree}}(i, j) \cdot \mathcal{L}_{\text{contrastive}}

\text{où } w_{\text{degree}}(i, j) = [\text{degree}(i)]^{-\alpha} \cdot [\text{degree}(j)]^{-\alpha}

\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(s_{\text{pos}}/\tau)}{\exp(s_{\text{pos}}/\tau) + \exp(s_{\text{neg}}/\tau)}
```

### Implémentation ([har_loss.py:15-66](har_loss.py#L15-L66))

```python
class HARLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        self.temperature = temperature
        self.alpha = alpha

    def compute_degree_weights(self, degrees):
        """w(i) = degree(i)^(-alpha)"""
        degrees_clamped = torch.clamp(degrees, min=1.0)
        weights = torch.pow(degrees_clamped, -self.alpha)
        return weights

    def forward(self, pos_scores, neg_scores, src_degrees, dst_degrees):
        # Degree weights pour src et dst
        w_src = self.compute_degree_weights(src_degrees)
        w_dst = self.compute_degree_weights(dst_degrees)
        w_pair = w_src * w_dst

        # Normalize scores par température
        pos_scores_norm = pos_scores / self.temperature
        neg_scores_norm = neg_scores / self.temperature

        # InfoNCE contrastive loss
        logits = torch.stack([pos_scores_norm, neg_scores_norm], dim=1)
        labels = torch.zeros(pos_scores.size(0), dtype=torch.long)
        loss_base = F.cross_entropy(logits, labels, reduction='none')

        # Apply degree reweighting
        loss_weighted = loss_base * w_pair
        return loss_weighted.mean()
```

**Résout :** Degree bias (favorise nœuds populaires)

---

## 4. Hybrid Focal-HAR Loss

### Formule Complète

```
L_Hybrid = w_degree · w_focal · L_BCE

où:
  w_degree = [degree(i) · degree(j)]^(-α_HAR)       [HAR component]
  w_focal = (1 - p_t)^γ                             [Focal component]
  L_BCE = -log(p_t)                                 [Base BCE]
```

### Décomposition Étape par Étape

```
1. Base BCE:
   L_BCE = -log(p_t)

2. Appliquer Focal modulation (pour hardness):
   L_Focal_partial = (1 - p_t)^γ · L_BCE

3. Appliquer HAR reweighting (pour degree bias):
   L_Hybrid = [degree(i) · degree(j)]^(-α_HAR) · L_Focal_partial
```

### Forme Développée

```
L_Hybrid(p, y, i, j) = α_t · [degree(i)]^(-α_HAR) · [degree(j)]^(-α_HAR)
                       · (1 - p_t)^γ · (-log(p_t))
```

**Où :**
- `α_t` : Class balancing weight de Focal Loss
- `γ` : Focusing parameter de Focal Loss
- `α_HAR` : Degree reweighting exponent de HAR Loss
- `degree(i), degree(j)` : Degrés des nœuds source et destination

### Version avec Lambda (Balance Focal/HAR)

```
L_Hybrid = λ · w_focal · L_BCE + (1 - λ) · w_degree · L_BCE
         = [λ · w_focal + (1 - λ) · w_degree] · L_BCE

où λ ∈ [0, 1] contrôle l'équilibre:
  λ = 0   → Pure HAR
  λ = 0.5 → Balance 50/50 (défaut)
  λ = 1   → Pure Focal
```

**Note :** Dans l'implémentation actuelle, on utilise la version multiplicative (pas λ).

### Exemple Numérique Complet

**Cas : Startup émergente difficile**

```
Données:
  degree(startup) = 2
  degree(investor) = 3
  p (probabilité prédite) = 0.25
  y (label) = 1

Calcul:
  1. BCE:
     L_BCE = -log(0.25) = 1.386

  2. Focal weight (γ = 2.0):
     w_focal = (1 - 0.25)^2 = 0.5625

  3. HAR weight (α_HAR = 0.5):
     w_degree = (2)^(-0.5) · (3)^(-0.5) = 0.707 · 0.577 = 0.408

  4. Class weight (α = 0.25):
     α_t = 0.25

  5. Hybrid Loss:
     L_Hybrid = 0.25 · 0.408 · 0.5625 · 1.386
              = 0.080

Comparaison:
  BCE seul:   1.386
  Focal seul: 0.25 · 0.5625 · 1.386 = 0.195
  HAR seul:   0.408 · 1.386 = 0.565
  Hybrid:     0.080 ← Plus petit car α=0.25 réduit les positifs
```

**Cas : Startup populaire facile**

```
Données:
  degree(startup) = 50
  degree(investor) = 100
  p = 0.95
  y = 1

Calcul:
  1. L_BCE = -log(0.95) = 0.051

  2. w_focal = (1 - 0.95)^2 = 0.0025

  3. w_degree = (50)^(-0.5) · (100)^(-0.5) = 0.141 · 0.1 = 0.0141

  4. α_t = 0.25

  5. L_Hybrid = 0.25 · 0.0141 · 0.0025 · 0.051
              = 0.00000045

Comparaison:
  BCE seul:   0.051
  Focal seul: 0.25 · 0.0025 · 0.051 = 0.000032
  HAR seul:   0.0141 · 0.051 = 0.00072
  Hybrid:     0.00000045 ← TRÈS réduit (doublement pénalisé)
```

### Tableau Récapitulatif des Effets

| Cas                           | Degré | Difficulté | w_degree | w_focal | L_Hybrid |
|-------------------------------|-------|------------|----------|---------|----------|
| Populaire + Facile            | Haut  | Faible     | ↓↓       | ↓↓      | ↓↓↓↓     |
| Populaire + Difficile         | Haut  | Élevée     | ↓↓       | ↑↑      | →        |
| Émergent + Facile             | Bas   | Faible     | ↑↑       | ↓↓      | →        |
| **Émergent + Difficile** ⭐   | Bas   | Élevée     | ↑↑       | ↑↑      | ↑↑↑↑     |

### Paramètres Standards

```
focal_gamma = 2.0       (focusing)
focal_alpha = 0.25      (class balancing)
har_alpha = 0.5         (degree reweighting)
lambda_focal = 0.5      (balance Focal/HAR, non utilisé en version multiplicative)
```

### Formule LaTeX

```latex
\mathcal{L}_{\text{Hybrid}}(p, y, i, j) = \alpha_t \cdot w_{\text{degree}} \cdot w_{\text{focal}} \cdot \mathcal{L}_{\text{BCE}}

\text{où:}

w_{\text{degree}} = [\text{degree}(i)]^{-\alpha_{\text{HAR}}} \cdot [\text{degree}(j)]^{-\alpha_{\text{HAR}}}

w_{\text{focal}} = (1 - p_t)^\gamma

\mathcal{L}_{\text{BCE}} = -\log(p_t)

p_t = \begin{cases}
p & \text{si } y = 1 \\
1 - p & \text{si } y = 0
\end{cases}
```

### Implémentation ([hybrid_loss.py:13-94](hybrid_loss.py#L13-L94))

```python
class HybridFocalHARLoss(nn.Module):
    def __init__(self, focal_gamma=2.0, focal_alpha=0.25,
                 har_alpha=0.5, lambda_focal=0.5):
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.har_alpha = har_alpha
        self.lambda_focal = lambda_focal

    def forward(self, pos_prob, neg_prob, src_degrees, dst_degrees,
                pos_label, neg_label):
        # === POSITIVES ===
        # 1. HAR: degree weights
        w_src = torch.pow(torch.clamp(src_degrees, min=1.0), -self.har_alpha)
        w_dst_pos = torch.pow(torch.clamp(dst_degrees, min=1.0), -self.har_alpha)
        w_degree_pos = w_src * w_dst_pos

        # 2. Focal: hardness weights
        p_t_pos = pos_prob
        focal_weight_pos = torch.pow(1.0 - p_t_pos, self.focal_gamma)

        # 3. BCE base
        bce_pos = -torch.log(torch.clamp(pos_prob, min=1e-7, max=1.0))

        # 4. Combine: alpha_t * w_degree * w_focal * BCE
        loss_pos = self.focal_alpha * focal_weight_pos * w_degree_pos * bce_pos

        # === NEGATIVES === (similaire)
        w_dst_neg = torch.pow(torch.clamp(dst_degrees_neg, min=1.0), -self.har_alpha)
        w_degree_neg = w_src * w_dst_neg

        p_t_neg = 1.0 - neg_prob
        focal_weight_neg = torch.pow(1.0 - p_t_neg, self.focal_gamma)

        bce_neg = -torch.log(torch.clamp(1.0 - neg_prob, min=1e-7, max=1.0))

        loss_neg = (1.0 - self.focal_alpha) * focal_weight_neg * w_degree_neg * bce_neg

        return (loss_pos + loss_neg).mean()
```

**Résout :** Déséquilibre de classes + Degree bias simultanément

---

## 5. Comparaison des Composantes

### Vue d'Ensemble

| Loss     | Formule                                    | Composantes                  |
|----------|--------------------------------------------|------------------------------|
| **BCE**  | `-log(p_t)`                                | Baseline                     |
| **Focal**| `α_t · (1-p_t)^γ · BCE`                    | BCE + hardness modulation    |
| **HAR**  | `[degree]^(-α) · L_contrastive`            | Contrastive + degree weight  |
| **Hybrid**| `α_t · [degree]^(-α) · (1-p_t)^γ · BCE`   | BCE + focal + HAR            |

### Matrice des Effets

```
                    │ Exemples Faciles │ Exemples Difficiles │
────────────────────┼──────────────────┼─────────────────────┤
High-degree nodes   │       ↓↓         │         →           │ (HAR)
Low-degree nodes    │       →          │         ↑↑          │ (HAR)
────────────────────┼──────────────────┼─────────────────────┤
Well-classified     │       ↓↓         │         -           │ (Focal)
Misclassified       │       -          │         ↑↑          │ (Focal)
────────────────────┼──────────────────┼─────────────────────┤
HYBRID:             │                  │                     │
High-degree + Easy  │      ↓↓↓↓        │         ↓           │
High-degree + Hard  │       ↓          │         ↑           │
Low-degree + Easy   │       ↓          │         ↑           │
Low-degree + Hard   │       ↑          │        ↑↑↑↑         │ ⭐
```

### Poids Relatifs (Exemple Numérique)

**Scénario : 4 paires avec différentes caractéristiques**

| Paire | degree | p    | BCE  | Focal | HAR  | Hybrid |
|-------|--------|------|------|-------|------|--------|
| A     | 100    | 0.95 | 0.05 | 0.0003| 0.007| 0.00004|
| B     | 100    | 0.25 | 1.39 | 0.195 | 0.15 | 0.06   |
| C     | 2      | 0.95 | 0.05 | 0.0003| 0.04 | 0.0002 |
| D     | 2      | 0.25 | 1.39 | 0.195 | 0.99 | **1.84**|

**Interprétation :**
- **Paire A** (populaire + facile) : Loss quasi-nulle
- **Paire B** (populaire + difficile) : Focal booste, HAR réduit → modéré
- **Paire C** (émergent + facile) : Focal réduit, HAR booste → modéré
- **Paire D** (émergent + difficile) : **Les deux boostent → MAXIMUM** ⭐

---

## Équations pour Rapport (LaTeX)

### Binary Cross-Entropy

```latex
\mathcal{L}_{\text{BCE}}(p, y) = -\left[ y \log(p) + (1-y) \log(1-p) \right]
```

### Focal Loss

```latex
\mathcal{L}_{\text{Focal}}(p, y) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
```

### HAR Loss

```latex
\mathcal{L}_{\text{HAR}} = w_{\text{degree}}(i, j) \cdot \mathcal{L}_{\text{contrastive}}

\text{where } w_{\text{degree}}(i, j) = \left[\text{degree}(i) \cdot \text{degree}(j)\right]^{-\alpha}
```

### Hybrid Loss

```latex
\mathcal{L}_{\text{Hybrid}} = \alpha_t \cdot \left[\text{degree}(i) \cdot \text{degree}(j)\right]^{-\alpha_{\text{HAR}}} \cdot (1 - p_t)^\gamma \cdot \left(-\log(p_t)\right)
```

---

## Références d'Implémentation

- **Focal Loss** : [focal_loss.py](focal_loss.py)
- **HAR Loss** : [har_loss.py](har_loss.py)
- **Hybrid Loss** : [hybrid_loss.py](hybrid_loss.py)
- **Training Integration** : [train_self_supervised.py:266-430](train_self_supervised.py#L266-L430)

---

## Résumé pour Choix de Loss

| Problème                     | Loss Recommandée |
|------------------------------|------------------|
| Déséquilibre de classes      | **Focal**        |
| Degree bias                  | **HAR**          |
| Les deux                     | **Hybrid**       |
| Dataset équilibré, pas de bias | **BCE**         |

**Pour votre cas (CrunchBase, 0.03% positifs, probable degree bias) :** **Hybrid Loss** ⭐
