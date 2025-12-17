# Paragraphe : Validation avec Ground Truth

## Version 1 : Standard (~150 mots)

### Évaluation des Prédictions

Pour évaluer la qualité des prédictions, nous comparons les scores du modèle au **ground truth** constitué des interactions réelles de l'ensemble de test. Pour chaque lien test (u, v, t) où v est l'investisseur qui a effectivement investi dans la startup u au temps t, le modèle calcule un score de prédiction s(u, v, t). Afin de déterminer si cette prédiction est "bonne", nous ne nous contentons pas d'un seuil de classification binaire, mais adoptons une approche de **ranking** : nous échantillonnons N = 100 investisseurs négatifs (n'ayant pas investi dans u), calculons leurs scores s(u, v', t), et classons tous les candidats par ordre décroissant de score. La qualité de la prédiction est alors mesurée par le **rang** r du vrai investisseur v dans ce classement. Le **Mean Reciprocal Rank (MRR = moyenne(1/r))** quantifie la position moyenne des vrais investisseurs, tandis que le **Recall@K** mesure la proportion de vrais investisseurs retrouvés dans les top-K prédictions. Une prédiction est considérée comme "bonne" si le vrai investisseur apparaît dans les premiers rangs (idéalement r ≤ 10), ce qui correspond à un scénario réaliste de recommandation d'investisseurs.

---

## Version 2 : Détaillée avec Exemple (~200 mots)

### Processus de Validation avec Ground Truth

Le **ground truth** est constitué par l'ensemble de test, représentant les investissements réellement survenus dans la période future (15% des interactions les plus récentes). Pour évaluer si une prédiction est correcte, nous procédons comme suit : pour chaque interaction test (u, v, t) où v est l'investisseur qui a effectivement investi dans la startup u, le modèle TGN calcule un score de probabilité s(u, v, t) reflétant la vraisemblance de ce lien. Simultanément, nous échantillonnons N = 100 investisseurs candidats {v₁, v₂, ..., v₁₀₀} n'ayant pas réellement investi dans u (négatifs), et calculons leurs scores respectifs. L'ensemble complet des candidats {v, v₁, ..., v₁₀₀} est ensuite classé par ordre décroissant de score.

La qualité de la prédiction est déterminée par le **rang r** du vrai investisseur v dans ce classement : r = 1 indique une prédiction parfaite (vrai investisseur classé premier), tandis qu'un rang élevé signale une prédiction médiocre. À partir de ces rangs, nous calculons le **Mean Reciprocal Rank (MRR)**, moyenne de 1/r sur tous les exemples test, qui quantifie la position typique du vrai investisseur. Le **Recall@K** mesure la proportion d'exemples où r ≤ K, indiquant combien de vrais investisseurs sont retrouvés dans les top-K recommandations. Par exemple, un Recall@10 = 0.15 signifie que pour 15% des startups, le vrai investisseur futur figure dans les 10 premiers candidats recommandés. Ces métriques fournissent une évaluation objective et quantitative de la capacité prédictive du modèle face aux investissements réellement observés.

---

## Version 3 : Concise (~100 mots)

### Validation des Prédictions

Les prédictions sont évaluées en les comparant au **ground truth**, constitué des investissements réels de l'ensemble de test. Pour chaque lien test (u, v, t), nous calculons le score du vrai investisseur v ainsi que ceux de 100 investisseurs négatifs échantillonnés aléatoirement, puis classons tous les candidats par score décroissant. Le **rang r** du vrai investisseur dans ce classement détermine la qualité de la prédiction. Le **Mean Reciprocal Rank (MRR = moyenne(1/r))** et le **Recall@K** (proportion de vrais investisseurs dans le top-K) quantifient respectivement la position moyenne et le taux de récupération des vrais investisseurs, fournissant une mesure objective de la performance prédictive du modèle.

---

## Version 4 : Avec Exemple Concret (~180 mots)

### Évaluation par Comparaison au Ground Truth

Pour déterminer si une prédiction est correcte, nous la comparons aux investissements réellement observés (ground truth) dans l'ensemble de test. Prenons l'exemple d'une startup "QuantumTech" ayant reçu un investissement de "Sequoia Capital" en 2022. Le modèle TGN calcule un score de prédiction pour ce lien réel ainsi que pour 100 investisseurs candidats n'ayant pas investi (ex: Andreessen Horowitz, Accel Partners, etc.). L'ensemble des 101 candidats est ensuite classé par ordre décroissant de score. Si "Sequoia Capital" apparaît en première position (rang r = 1), la prédiction est parfaite ; s'il est classé 50ème, la prédiction est médiocre.

Le **Mean Reciprocal Rank (MRR)**, calculé comme la moyenne de 1/r sur tous les exemples test, mesure la position typique des vrais investisseurs : un MRR de 0.10 indique qu'en moyenne, le vrai investisseur est classé autour du 10ème rang. Le **Recall@K** quantifie la proportion de cas où le vrai investisseur figure dans le top-K : Recall@10 = 0.15 signifie que pour 15% des startups, le modèle place le vrai investisseur parmi les 10 meilleurs candidats. Ces métriques fournissent une évaluation rigoureuse de la capacité du modèle à identifier les investisseurs réels parmi un large ensemble de candidats potentiels.

---

## Version 5 : Style Scientifique Formel (~150 mots)

### Validation Against Ground Truth

Model predictions are evaluated against the ground truth defined by the test set interactions ℰ_test = {(u, v, t)}. For each test triple (u, v, t), where v denotes the investor that actually invested in startup u at time t, we compute the model's predicted score s(u, v, t). To assess prediction quality, we sample N = 100 negative investors 𝒩_u ⊂ 𝒱 \ {v} and compute their scores {s(u, v', t)}_{v'∈𝒩_u}. Ranking all candidates {v} ∪ 𝒩_u in descending order of score, we obtain the rank r ∈ {1, ..., 101} of the true investor v.

The Mean Reciprocal Rank MRR = 1/|ℰ_test| Σ 1/r quantifies the average position of true investors, while Recall@K = |{(u,v,t) : r ≤ K}| / |ℰ_test| measures the fraction retrieved in top-K predictions. These ranking-based metrics provide an objective assessment of the model's ability to identify real future investments among a pool of candidates, directly reflecting performance in a realistic recommendation scenario.

---

## Version 6 : Français Académique (~160 mots)

### Validation par Rapport à la Vérité Terrain

Les prédictions du modèle sont évaluées en les comparant à la **vérité terrain** (ground truth) constituée des investissements réellement observés dans l'ensemble de test. Pour chaque interaction test (u, v, t), où v représente l'investisseur ayant effectivement investi dans la startup u au temps t, le modèle calcule un score de prédiction s(u, v, t). Afin de déterminer si cette prédiction est correcte, nous échantillonnons aléatoirement N = 100 investisseurs négatifs (n'ayant pas investi dans u), calculons leurs scores respectifs, puis classons l'ensemble des candidats par ordre décroissant de score. Le **rang r** du vrai investisseur v dans ce classement quantifie la qualité de la prédiction : r = 1 correspond à une prédiction parfaite.

Le **Mean Reciprocal Rank (MRR)**, défini comme la moyenne de 1/r sur tous les exemples test, mesure la position moyenne des vrais investisseurs dans les classements prédits. Le **Recall@K** représente la proportion de cas où le vrai investisseur figure dans les K premiers candidats recommandés. Ces métriques fournissent une évaluation objective et quantitative de la capacité du modèle à identifier correctement les investisseurs futurs.

---

## Version 7 : Très Concise pour Abstract (~60 mots)

Predictions are validated against ground truth (real test set investments) using a ranking approach: for each test link (u, v, t), we rank the true investor v among 100 random negatives based on predicted scores. Mean Reciprocal Rank (MRR) measures average position of true investors, while Recall@K quantifies the fraction retrieved in top-K, providing objective performance assessment.

---

## Tableau Comparatif : Quand Utiliser Chaque Version

| Version | Mots | Contexte Recommandé | Points Forts |
|---------|------|---------------------|--------------|
| **Version 1** | ~150 | Section Méthodologie standard | Équilibre détail/concision |
| **Version 2** | ~200 | Méthodologie détaillée | Très complète, couvre tout |
| **Version 3** | ~100 | Introduction/Overview | Concise, va à l'essentiel |
| **Version 4** | ~180 | Avec exemples illustratifs | Pédagogique, facile à comprendre |
| **Version 5** | ~150 | Article scientifique anglais | Notation mathématique formelle |
| **Version 6** | ~160 | Rapport/Thèse français | Style académique français |
| **Version 7** | ~60 | Abstract/Résumé | Ultra-condensé |

---

## Ma Recommandation pour Vous : Version 4

Je recommande la **Version 4 (Avec Exemple Concret)** car elle :

✅ **Explique clairement** le processus ground truth
✅ **Donne un exemple concret** (QuantumTech + Sequoia)
✅ **Lie explicitement** les métriques à leur interprétation
✅ **Répond exactement** à votre question sur comment savoir si c'est "bon"

---

## Intégration dans Votre Rapport

Voici comment l'intégrer :

```markdown
## 4. Méthodologie

### 4.1 Architecture du Modèle
[Votre description du TGN...]

### 4.2 Protocole de Validation

**Division Temporelle.** Les données sont divisées chronologiquement en
ensembles d'entraînement (70%), validation (15%) et test (15%).

**Évaluation par Comparaison au Ground Truth.**  [INSÉRER VERSION 4 ICI]

### 4.3 Fonction de Perte
[Votre section sur Focal/HAR Loss...]
```

---

## Points Clés à Retenir pour Votre Rapport

### Ground Truth = Test Set
```
Test set = Investissements RÉELLEMENT survenus en 2020-2023
         = Ce qu'on veut prédire
         = La "vérité" contre laquelle on compare
```

### Processus de Validation
```
1. Prendre un vrai lien du test set: (QuantumTech, Sequoia, 2022)
                                      ↓
2. Le modèle prédit un score:        s(QuantumTech, Sequoia) = 0.78
                                      ↓
3. Sampler 100 faux investisseurs:   {Accel, a16z, Y Comb, ...}
                                      ↓
4. Prédire leurs scores:             {0.65, 0.23, 0.19, ...}
                                      ↓
5. Classer tous les 101 candidats:   [Sequoia(0.78), Accel(0.65), ...]
                                      ↓
6. Trouver rang de Sequoia:          r = 1 (premier)
                                      ↓
7. Calculer métriques:               MRR = 1/1 = 1.0 ✅
                                     Recall@10 = ✅
```

### Interprétation des Métriques
```
MRR = 0.10  → "En moyenne, vrai investisseur classé ~10ème"
              → Prédiction bonne

MRR = 0.01  → "En moyenne, vrai investisseur classé ~100ème"
              → Prédiction médiocre

Recall@10 = 0.15  → "Pour 15% des startups, vrai investisseur
                     dans top 10"
                  → Utilisable pour recommandations

Recall@10 = 0.00  → "Jamais le vrai investisseur dans top 10"
                  → Pas utilisable
```

---

## Diagramme pour Accompagner le Texte

```
┌─────────────────────────────────────────────────────────────┐
│  PROCESSUS DE VALIDATION                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Ground Truth (Test Set):                                   │
│  ┌──────────────────────────────────────┐                  │
│  │ (QuantumTech, Sequoia Capital, 2022) │ ← Vrai lien      │
│  └──────────────────────────────────────┘                  │
│                    ↓                                         │
│  Prédictions du Modèle:                                     │
│  ┌─────────────────────────────────────────────────┐       │
│  │  Sequoia Capital     → score = 0.78  (VRAI)     │       │
│  │  Accel Partners      → score = 0.65             │       │
│  │  Andreessen Horowitz → score = 0.23             │       │
│  │  Y Combinator        → score = 0.19             │       │
│  │  ...                                            │       │
│  │  Random Angel #100   → score = 0.001            │       │
│  └─────────────────────────────────────────────────┘       │
│                    ↓                                         │
│  Classement (Ranking):                                      │
│  ┌────────────────────┐                                     │
│  │ Rang 1: Sequoia ✅ │ ← MRR = 1/1 = 1.0                  │
│  │ Rang 2: Accel      │   Recall@10 = ✅                   │
│  │ Rang 3: a16z       │                                     │
│  │ ...                │                                     │
│  └────────────────────┘                                     │
│                                                              │
│  Évaluation: BONNE PRÉDICTION                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Formules Mathématiques à Inclure (Optionnel)

Si vous voulez être plus formel :

```latex
% Mean Reciprocal Rank
\text{MRR} = \frac{1}{|\mathcal{E}_{\text{test}}|} \sum_{(u,v,t) \in \mathcal{E}_{\text{test}}} \frac{1}{r_{u,v}}

% Recall@K
\text{Recall@K} = \frac{|\{(u,v,t) \in \mathcal{E}_{\text{test}} : r_{u,v} \leq K\}|}{|\mathcal{E}_{\text{test}}|}

où r_{u,v} est le rang du vrai investisseur v parmi les candidats classés.
```

---

## Checklist pour Votre Paragraphe

- [x] Définir ce qu'est le ground truth (test set = vrais investissements)
- [x] Expliquer le processus de ranking (1 vrai + 100 faux)
- [x] Expliquer ce qu'est le "rang" r
- [x] Définir MRR avec interprétation
- [x] Définir Recall@K avec interprétation
- [x] Donner un exemple concret (optionnel mais recommandé)
- [x] Lier à l'utilité pratique (recommandation)

Tout est prêt pour votre rapport ! 🎓
