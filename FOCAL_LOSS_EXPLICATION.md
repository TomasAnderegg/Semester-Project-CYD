# Focal Loss : Sur Quoi Se Focalise-t-elle ?

## Question

> "Tu as implémenté le focal loss parce qu'elle 'pays more attention to hard examples by calculating a weight factor of predicted probability.' Mais sur quoi elle se focalise ? Les nœuds à faible degré ?"

## Réponse Courte

**Non, le Focal Loss ne se focalise PAS spécifiquement sur les nœuds à faible degré.**

Le Focal Loss se focalise sur les **exemples difficiles à classifier**, indépendamment du degré des nœuds. Ces "hard examples" sont définis par **la probabilité prédite**, pas par les propriétés structurelles du graphe.

---

## Explication Détaillée

### 1. Comment Focal Loss Fonctionne

#### Formule
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

où:
  p_t = probabilité prédite pour la vraie classe
  γ = focusing parameter (gamma)
  α_t = weight factor pour équilibrer les classes
```

#### Mécanisme de Focalisation

Le terme clé est **(1 - p_t)^γ** :

```python
# Exemple avec γ = 2.0

Si p_t = 0.9 (prédiction confiante et correcte) :
    (1 - 0.9)^2 = 0.01  → Loss réduite de 99% !

Si p_t = 0.5 (prédiction incertaine) :
    (1 - 0.5)^2 = 0.25  → Loss réduite de 75%

Si p_t = 0.1 (prédiction mauvaise) :
    (1 - 0.1)^2 = 0.81  → Loss presque normale
```

**Conclusion** : Focal Loss réduit automatiquement l'importance des exemples que le modèle classe déjà bien (easy examples) et **se concentre** sur ceux qu'il classe mal (hard examples).

---

### 2. Qu'est-ce qu'un "Hard Example" dans votre contexte ?

Dans votre système TGN pour prédire les investissements, un **hard example** est :

#### A. **Lien Positif Difficile à Détecter**

**Exemple :**
```
Startup "QuantumTech" + Investor "ABC Ventures"
  → Vrai lien (investissement réel)
  → MAIS le modèle prédit p = 0.15 (très faible confiance)

Pourquoi difficile ?
  - QuantumTech est très récent (peu d'historique)
  - ABC Ventures a un profil atypique
  - Peu de features communes
  - Pattern non évident dans le graphe temporel
```

**Avec BCE classique :**
```
Loss = -log(0.15) = 1.90
→ Traité comme tous les autres exemples
```

**Avec Focal Loss (γ=2) :**
```
Modulating factor = (1 - 0.15)^2 = 0.72
Loss = 0.72 * (-log(0.15)) = 1.37
→ Perte AUGMENTÉE (car p_t faible)
→ Modèle forcé d'apprendre ce cas difficile
```

#### B. **Faux Négatif Difficile (Hard Negative)**

**Exemple :**
```
Startup "DeepMind" + Investor "Google Ventures"
  → Pas encore investi (négatif)
  → MAIS le modèle prédit p = 0.85 (très confiant à tort)

Pourquoi difficile ?
  - Profils très similaires
  - Beaucoup d'investisseurs communs
  - Patterns qui ressemblent à un vrai lien
  - Le modèle est "trompé" par la similarité
```

**Avec BCE classique :**
```
Loss = -log(1 - 0.85) = -log(0.15) = 1.90
→ Pénalité standard
```

**Avec Focal Loss (γ=2) :**
```
Modulating factor = (1 - 0.15)^2 = 0.72  # p_t = 1-p pour négatifs
Loss = 0.72 * 1.90 = 1.37
→ Modèle forcé d'apprendre à ne PAS prédire ce lien
```

---

### 3. Focal Loss vs. Degré des Nœuds

#### Question : Les nœuds à faible degré sont-ils automatiquement des hard examples ?

**Réponse : PAS NÉCESSAIREMENT**

La corrélation n'est pas automatique. Voici les différents cas :

| Type de Nœud | Peut être Easy Example | Peut être Hard Example |
|--------------|------------------------|------------------------|
| **Faible degré** | ✅ OUI (pattern clair malgré peu de connexions) | ✅ OUI (peu de signal) |
| **Haut degré** | ✅ OUI (pattern évident avec beaucoup de données) | ✅ OUI (bruit, sélectivité complexe) |

#### Exemples Concrets

**Cas 1 : Faible Degré = Easy Example**
```
Startup: "BioQuantum" (2 investisseurs actuels)
  - Investisseur 1: "HealthTech Ventures" (spécialisé biotech)
  - Investisseur 2: "Quantum Capital" (spécialisé quantum)

Prédiction: "BioTech Quantum Fund" va investir
  → p = 0.92 (très confiant)
  → Pattern évident malgré faible degré
  → EASY EXAMPLE (focal loss réduit son importance)
```

**Cas 2 : Faible Degré = Hard Example**
```
Startup: "StealthMode Inc." (1 investisseur actuel)
  - Investisseur 1: "Anonymous Angel" (profil inconnu)

Prédiction: "VC XYZ" va investir
  → p = 0.35 (incertain)
  → Pas assez de signal
  → HARD EXAMPLE (focal loss augmente son importance)
```

**Cas 3 : Haut Degré = Easy Example**
```
Startup: "Google DeepMind" (50+ investisseurs)
  - Pattern très clair
  - Beaucoup de features

Prédiction: "Tech Giants Fund" va investir
  → p = 0.98 (très confiant)
  → EASY EXAMPLE (focal loss réduit son importance)
```

**Cas 4 : Haut Degré = Hard Example**
```
Startup: "Versatile Tech" (80+ investisseurs très divers)
  - Secteurs multiples (AI, fintech, biotech, etc.)
  - Pas de pattern clair

Prédiction: "Niche Specialist VC" va investir
  → p = 0.42 (incertain)
  → Trop de bruit malgré beaucoup de données
  → HARD EXAMPLE (focal loss augmente son importance)
```

---

### 4. Mécanismes Indépendants

Votre système combine DEUX mécanismes distincts :

#### A. **Focal Loss (Gestion des Hard Examples)**

**Objectif :** Se concentrer sur les exemples mal classés

**Critère :** Probabilité prédite p_t

**Code :**
```python
# focal_loss.py ligne 66
modulating_factor = (1 - p_t) ** self.gamma

# Si p_t faible → modulating_factor élevé → loss importante
# Si p_t élevé → modulating_factor faible → loss réduite
```

**Indépendant de :** Structure du graphe, degré des nœuds

---

#### B. **Hard Negative Mining (Optionnel dans votre code)**

**Objectif :** Sélectionner des négatifs difficiles basés sur la STRUCTURE

**Critère :** Voisinage dans le graphe

**Code :**
```python
# hard_negative_mining.py
# Sélectionne des négatifs qui sont :
# - À 2-3 sauts du nœud source
# - Ont des voisins communs
# - Sont dans le même "cluster" du graphe
```

**Peut privilégier :** Nœuds avec patterns structurels similaires

---

### 5. Visualisation : Focal Loss en Action

```
Distribution des Prédictions (votre dataset réel):
════════════════════════════════════════════════════

LIENS POSITIFS (vrais investissements) :
  p=0.04  ●●●●●●●●●●●●●●●●●●  (médiane actuelle)
  p=0.25  ●●●●
  p=0.70  ●     (max actuel)
          ↑
    HARD EXAMPLES !
    Focal Loss se concentre ici

LIENS NÉGATIFS (pas d'investissement) :
  p=0.99  ●●●●●●●●●●●●●●●●●●●●●●●●
  p=0.50  ●●●
  p=0.05  ●
          ↑
    EASY EXAMPLES
    Focal Loss réduit leur importance
```

**Analyse :**
- Médiane des vrais liens = 0.04 → **TRÈS HARD EXAMPLES**
- Le modèle ne détecte pas bien les vrais liens
- Focal Loss va **forcer** le modèle à mieux apprendre ces cas

**Résultat Attendu avec Focal Loss :**
```
LIENS POSITIFS (après entraînement avec Focal Loss) :
  p=0.25  ●●●●●●●●●●●●●●●●●●  (nouvelle médiane espérée)
  p=0.60  ●●●●
  p=0.95  ●
          ↑
    Amélioration significative !
```

---

### 6. Ce Que Focal Loss NE Fait PAS

❌ **Ne cible pas spécifiquement les nœuds à faible degré**
- Critère = probabilité prédite, pas structure du graphe

❌ **Ne remplace pas le feature engineering**
- Si le modèle n'a pas accès à des features utiles, focal loss ne peut pas compenser

❌ **Ne résout pas le cold start**
- Nœuds sans historique restent difficiles à prédire

❌ **Ne change pas le modèle sous-jacent**
- TGN reste un TGN, focal loss change juste la fonction de perte

---

### 7. Corrélation Indirecte avec le Degré

**Il PEUT y avoir une corrélation indirecte :**

```
Nœud à faible degré
  → Moins de signal temporel
  → Moins de neighbors pour le message passing
  → Embeddings moins informatifs
  → Probabilités prédites moins confiantes
  → Plus susceptible d'être un HARD EXAMPLE
  → Focal Loss se focalise dessus
```

**MAIS ce n'est pas causal :**
- Focal Loss ne "voit" pas le degré directement
- Elle réagit à la probabilité prédite
- Un nœud à faible degré avec un pattern clair sera un easy example

---

### 8. Comparaison : BCE vs Focal Loss

#### Avec Binary Cross-Entropy (BCE)

```python
# Tous les exemples ont le même poids de base
loss = -[y * log(p) + (1-y) * log(1-p)]

Exemple 1 (easy): y=1, p=0.95 → loss = 0.05
Exemple 2 (hard): y=1, p=0.15 → loss = 1.90

Ratio: 1.90 / 0.05 = 38x
→ L'exemple difficile pèse 38x plus
```

#### Avec Focal Loss (γ=2)

```python
loss = -(1-p_t)^γ * [y * log(p) + (1-y) * log(1-p)]

Exemple 1 (easy): y=1, p=0.95
  → modulating = (1-0.95)^2 = 0.0025
  → loss = 0.0025 * 0.05 = 0.000125

Exemple 2 (hard): y=1, p=0.15
  → modulating = (1-0.15)^2 = 0.7225
  → loss = 0.7225 * 1.90 = 1.37

Ratio: 1.37 / 0.000125 = 10,960x
→ L'exemple difficile pèse 10,960x plus !
```

**Résultat :** Focal Loss amplifie DRAMATIQUEMENT l'importance relative des hard examples.

---

### 9. Dans Votre Contexte Spécifique

#### Votre Dataset
```
Total paires: 170,742
Vrais liens: 52 (0.03%)
Ratio déséquilibre: 3,283:1
```

#### Problème Observé
```
Médiane probabilité vrais liens: 0.04
→ Le modèle ne détecte presque rien
→ TOUS les vrais liens sont des HARD EXAMPLES
```

#### Comment Focal Loss Aide
```python
# AVANT (BCE):
# Modèle optimise surtout pour bien classifier les 170,690 négatifs
# Les 52 positifs sont "noyés" dans la masse

# APRÈS (Focal Loss avec γ=2):
# Les 170,690 négatifs bien classés (p<0.05) ont loss ~0
# Les 52 positifs mal classés (p<0.3) ont loss très élevée
# → Modèle FORCÉ d'apprendre à détecter les positifs
```

---

### 10. Réponse à la Question Initiale

> "Sur quoi elle se focalise ? Les nœuds à faible degré ?"

**Réponse complète :**

Le Focal Loss se focalise sur les **exemples mal classés** (hard examples), définis par :

1. **Critère Direct :** Probabilité prédite faible pour la vraie classe
   - `p_t < 0.5` → Hard example → Loss augmentée

2. **Pas de ciblage structurel :**
   - Ne regarde PAS le degré des nœuds directement
   - Ne connaît PAS la structure du graphe
   - Réagit SEULEMENT à la sortie du modèle

3. **Corrélation Possible mais Indirecte :**
   - Les nœuds à faible degré PEUVENT être des hard examples
   - Mais ce n'est pas systématique
   - Dépend de la qualité des features et des patterns

4. **Dans Votre Cas (TGN + Investment Prediction) :**
   - Hard examples = Liens que TGN prédit mal
   - Souvent : Nouvelles startups, investisseurs atypiques, patterns non-évidents
   - Peut inclure nœuds à faible degré SI le modèle les classe mal
   - Peut aussi inclure nœuds à haut degré avec patterns complexes

---

## Visualisation Finale : Flow de Décision

```
                    FOCAL LOSS
                        |
                        v
            Reçoit: probabilités p du modèle
                        |
                        v
         Calcule: modulating factor (1-p_t)^γ
                        |
                        v
            ┌───────────┴───────────┐
            |                       |
    p_t élevé (>0.7)        p_t faible (<0.3)
    = EASY EXAMPLE          = HARD EXAMPLE
            |                       |
    Loss réduite 99%        Loss maintenue/augmentée
            |                       |
    Gradient faible         Gradient fort
            |                       |
    Peu d'apprentissage     Beaucoup d'apprentissage
            |                       |
            v                       v
    "Ignore" ces exemples   "Focus" sur ces exemples
```

**Conclusion :** Focal Loss est un mécanisme **adaptatif** basé sur la **performance** du modèle, pas sur la **structure** du graphe.

---

## Références

1. **Paper Original :**
   - Lin et al. (2017), "Focal Loss for Dense Object Detection"
   - https://arxiv.org/abs/1708.02002

2. **Votre Implémentation :**
   - `focal_loss.py` lignes 48-81
   - `train_self_supervised.py` lignes 241-244

3. **Documentation :**
   - `FOCAL_LOSS_README.md`
