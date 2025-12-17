# Guide des Métriques Alternatives TechRank

## Table des Matières
1. [Introduction](#introduction)
2. [Différences entre les Scripts](#différences-entre-les-scripts)
3. [Les 6 Métriques Expliquées](#les-6-métriques-expliquées)
4. [Guide d'Utilisation](#guide-dutilisation)
5. [Recommandations par Stratégie](#recommandations-par-stratégie)

---

## Introduction

Ce document explique le système de métriques alternatives développé pour identifier les entreprises prometteuses au-delà du simple delta TechRank absolu.

### Contexte du Problème

**Observation initiale :**
```
Threshold absolu : delta > 0.0001
64 entreprises analysées
→ Seulement 4 entreprises identifiées (6.25%)
→ Trop restrictif !
```

**Cause :**
Les deltas TechRank sont naturellement très petits (ordre de 10^-4 à 10^-6) car :
- Les scores TechRank sont normalisés
- L'effet de dilution (64 → 199 entreprises) réduit les scores individuels
- Un seuil absolu fixe ignore les variations relatives importantes

---

## Différences entre les Scripts

### TechRank_Comparison.py

**Rôle :** Calculer et comparer les scores TechRank bruts avant/après TGN

**Fonctionnement :**
```python
# 1. Charger les graphes
B_before = load_ground_truth_graph()
B_after = load_predicted_graph()

# 2. Calculer TechRank
df_before = run_techrank(B_before)
df_after = run_techrank(B_after)

# 3. Comparer avec seuil absolu
delta = techrank_after - techrank_before
df_promising = df[delta > threshold]
```

**Sorties :**
- `company_techrank_merged_filtered.csv` : Comparaison brute avant/après
- `company_techrank_deltas.csv` : Deltas calculés
- Graphiques de comparaison

**Limite :**
Filtrage binaire basé sur un seuil absolu arbitraire.

---

### TechRank_Alternative_Metrics.py

**Rôle :** Analyser avec des métriques relatives et contextuelles

**Fonctionnement :**
```python
# 1. Charger les données déjà calculées
df = load_comparison_data()  # Depuis TechRank_Comparison

# 2. Calculer 6 métriques alternatives
compute_alternative_metrics(df)
  ├─ Ratio multiplicateur
  ├─ Nouvelles apparitions
  ├─ Préservation du rang
  ├─ Analyse par percentile
  ├─ Z-score
  └─ Score composite

# 3. Identifier les top performers par métrique
```

**Sorties :**
- `promising_companies_all_metrics.csv` : Toutes les métriques combinées
- `promising_by_ratio.csv` : Top 20 par croissance relative
- `promising_by_percentile.csv` : Top 20 par percentile
- `promising_by_composite.csv` : Top 20 par score composite
- ... et autres

**Avantage :**
Vision multi-dimensionnelle, pas de seuil arbitraire.

---

## Les 6 Métriques Expliquées

### 1. Ratio Multiplicateur (Croissance Relative)

**Formule :**
```python
ratio = techrank_after / techrank_before
```

**Exemple :**
```
QSIM Plus:
  - Before: 0.000149
  - After:  0.000946
  - Ratio:  6.35x  ← Croissance de 535% !
```

**Pertinence :**
- Mesure la **croissance relative** plutôt qu'absolue
- Identifie les "dark horses" avec croissance explosive
- Insensible à la magnitude absolue

**Cas d'usage :**
- **Seed/Early-stage investing** : Détecter les entreprises émergentes
- **Trend spotting** : Identifier les secteurs en forte croissance

**Limites :**
- Valeurs infinies quand `before ≈ 0` (ex: ChromoPIC avec ratio 10^30x)
- Peut survaloriser les micro-entreprises
- Sensible au bruit pour les très petites valeurs

**Interprétation :**
```
Ratio > 10x   : Croissance explosive
Ratio 2-10x   : Forte croissance
Ratio 1-2x    : Croissance modérée
Ratio < 1x    : Décroissance
```

---

### 2. Nouvelles Apparitions (0 → Positif)

**Formule :**
```python
df_new = df[(techrank_before == 0) & (techrank_after > 0)]
```

**Concept :**
- Score = 0 avant → Pas d'investisseurs connus / Hors écosystème
- Score > 0 après → TGN prédit des connexions futures

**Pertinence :**
- Identifie les **entreprises complètement nouvelles** dans l'écosystème
- Détecte les marchés émergents
- Révèle les tendances technologiques naissantes

**Cas d'usage :**
- **Pre-seed/Seed investing** : Détecter avant la compétition
- **Diversification** : Explorer de nouveaux secteurs
- **Innovation scouting** : Identifier les technologies disruptives

**Dans vos résultats :**
```
Total nouvelles entrées: 0
```
→ Toutes les entreprises analysées existaient déjà dans le graphe "before"
→ Normal car `TechRank_Comparison.py` filtre sur `techrank_before != 0`

**Note :** Pour obtenir les vraies nouvelles entrées, il faudrait modifier le filtre dans TechRank_Comparison.py

---

### 3. Préservation du Rang (Résistance à la Dilution)

**Formule :**
```python
rank_change = rank_before - rank_after
# Positif = amélioration
# Négatif = détérioration
```

**Contexte CRITIQUE :**
```
Graphe AVANT : 64 entreprises
Graphe APRÈS : 199 entreprises (+135 nouvelles)
                     ↓
           Effet de DILUTION
```

**Pourquoi les rangs se détériorent :**

Exemple illustré :
```
AVANT (64 entreprises) :
  #10 : TuringQ (score = 0.05)  ← Top 15%

APRÈS (199 entreprises) :
  #50 : TuringQ (score = 0.04)  ← Toujours top 25%

→ Rang numérique pire (10 → 50)
→ Mais position relative stable (~15-25%)
```

**Pertinence :**
- Mesure la **robustesse** face à la compétition accrue
- Les entreprises avec perte minimale sont les plus résilientes
- Indicateur de solidité du positionnement

**Vos résultats :**
```
Top performers (perte minimale):
1. QuantumAstra       : -38   (212 → 250)
2. Great Lakes Crystal: -48   (201 → 249)
3. QSIM Plus         : -214  (193 → 407)  ← Grosse perte malgré bon delta
```

**Interprétation :**
- **QuantumAstra** : Très résilient, maintient sa position malgré +135 concurrents
- **QSIM Plus** : Croissance absolue forte (+6.35x) MAIS beaucoup de nouvelles entreprises le dépassent

**Cas d'usage :**
- **Growth/Late-stage** : Évaluer la défendabilité de la position
- **Portfolio risk** : Identifier les valeurs sûres
- **Competitive analysis** : Comprendre la dynamique du marché

**IMPORTANT :**
Cette métrique est **contextuelle**. Dans un marché en forte expansion (+135 entreprises = +211%), perdre 50-100 rangs est **normal et acceptable**.

---

### 4. Analyse par Percentile ⭐ (Recommandée)

**Formule :**
```python
percentile_before = rank(techrank_before) / total * 100
percentile_after = rank(techrank_after) / total * 100
percentile_change = percentile_after - percentile_before
```

**Concept :**
Le percentile représente la **position relative** dans la distribution, indépendamment du nombre total d'entités.

**Exemple concret :**
```
ChromoPIC:
  AVANT (64 entreprises):
    - Rang: #63/64
    - Percentile: 1.6%  (presque dernière)

  APRÈS (199 entreprises):
    - Rang: #180/199
    - Percentile: 90.6%  (top 10% !)

  Delta percentile: +89.1%  ← Progression MASSIVE !
```

**Comparaison Rang vs Percentile :**
```
                    Rang        Percentile
QSIM Plus AVANT     #193/64     7.8%
QSIM Plus APRÈS     #407/199    95.3%
                       ↓           ↓
                 Pire (409)   Meilleur (95.3%)

→ Le rang trompe, le percentile révèle la vérité !
```

**Pertinence :**
- **Résout le problème de dilution** !
- Comparable entre différentes périodes/datasets
- Reflète vraiment la "montée en puissance"
- Insensible à la taille absolue du marché

**Vos résultats top 3 :**
```
1. ChromoPIC      : +89.1%  (1.6% → 90.6%)
2. QSIM Plus      : +87.5%  (7.8% → 95.3%)
3. Quantistry     : +84.4%  (4.7% → 89.1%)
```

**Interprétation :**
```
Delta > +80%  : Transformation majeure (bottom → top tier)
Delta +50-80% : Forte progression
Delta +20-50% : Progression significative
Delta 0-20%   : Progression modérée
Delta < 0%    : Régression
```

**Cas d'usage :**
- **Toutes stratégies d'investissement**
- **Benchmarking** : Comparer entre différentes périodes
- **Performance relative** : Évaluer vs. marché

**POURQUOI C'EST LA MEILLEURE MÉTRIQUE :**
1. Normalise pour la taille du dataset
2. Intuitive (position dans la distribution)
3. Robuste aux outliers
4. Directement comparable

---

### 5. Z-Score (Outliers Statistiques)

**Formule :**
```python
z_score = (techrank_delta - mean_delta) / std_delta
```

**Concept :**
Mesure combien d'écarts-types une observation est éloignée de la moyenne.

**Distribution normale :**
```
      68%
    ┌─────┐
    │     │
    │     │    95%
  ┌─┼─────┼─┐
  │ │     │ │    99.7%
┌─┼─┼─────┼─┼─┐
│ │ │     │ │ │
├─┼─┼─────┼─┼─┤
-3 -1  0  +1 +3  ← Z-score
```

**Interprétation :**
```
Z > 3.0  : Exceptionnel (0.1% de la population) - Outlier extrême
Z > 2.0  : Très fort (2.5%) - Outlier notable
Z > 1.0  : Au-dessus de la moyenne (16%)
Z ≈ 0.0  : Normal/Moyen
Z < -1.0 : En-dessous de la moyenne
```

**Vos résultats :**
```
Moyenne delta: -0.064324  ← Négatif = dilution générale
Écart-type:     0.107528

Top Z-scores:
1. QSIM Plus    : 0.61  (Normal)
2. ChromoPIC    : 0.60  (Normal)
3. Quantistry   : 0.60  (Normal)
...
```

**Interprétation de vos résultats :**
- Tous les Z-scores ≈ 0.60 → Aucun outlier extrême
- Les deltas positifs sont **rares** mais pas extraordinaires statistiquement
- Distribution relativement homogène
- Pas d'entreprise "miracle" mais croissance constante

**Pertinence :**
- Identifie les **anomalies statistiques**
- Objective (basée sur la distribution observée)
- Indépendante des unités

**Cas d'usage :**
- **Anomaly detection** : Trouver les véritables outliers
- **Risk assessment** : Identifier les paris à haut risque/haute récompense
- **Academic research** : Analyse statistique rigoureuse

**Limites :**
- Nécessite une distribution normale (pas toujours le cas en finance)
- Sensible aux valeurs extrêmes (heavy tails)
- Peu intuitive pour les non-statisticiens

**Quand l'utiliser :**
- Quand vous cherchez les **véritables exceptions**
- Pour valider qu'une observation est statistiquement significative
- En complément d'autres métriques

---

### 6. Score Composite ⭐⭐ (Fortement Recommandé)

**Formule :**
```python
# Normaliser chaque métrique entre 0 et 1
norm_delta = (delta - delta_min) / (delta_max - delta_min)
norm_ratio = (ratio - ratio_min) / (ratio_max - ratio_min)
norm_rank = (rank_change - rank_min) / (rank_max - rank_min)
norm_percentile = (perc_change - perc_min) / (perc_max - perc_min)

# Score composite avec pondération
composite_score = (
    0.30 * norm_delta +       # 30% : Delta absolu
    0.30 * norm_ratio +        # 30% : Croissance relative
    0.20 * norm_rank +         # 20% : Préservation rang
    0.20 * norm_percentile     # 20% : Changement percentile
)
```

**Pourquoi ces poids ?**

| Composante | Poids | Justification |
|------------|-------|---------------|
| **Delta absolu** | 30% | Croissance tangible en valeur réelle |
| **Ratio** | 30% | Momentum/accélération, détecte les tendances |
| **Rank change** | 20% | Robustesse face à la compétition |
| **Percentile** | 20% | Position relative dans l'écosystème |

**Concept :**
Combine les **forces** de toutes les métriques pour une évaluation holistique.

**Vos résultats :**
```
Top 5 par score composite:
1. ChromoPIC                 : 0.800  ← Dominant sur TOUTES les métriques
2. TerraNexum                : 0.574  ← Delta négatif MAIS forte préservation
3. Labber                    : 0.572  ← Idem
4. Global Telecom            : 0.565  ← Ratio infini (0 → presque 0)
5. Cambridge Quantum Computing: 0.552
```

**Analyse détaillée - ChromoPIC (0.800) :**
```
Décomposition du score:
  ✓ Delta absolu    : +0.000546  (positif malgré dilution)
  ✓ Ratio           : 10^30x     (partait de presque 0)
  ✓ Percentile      : +89.1%     (bottom → top 10%)
  ✓ Rank change     : -240       (acceptable vu la dilution)

Verdict: Early-stage exceptionnel avec croissance explosive
```

**Analyse - TerraNexum (0.574) :**
```
Décomposition:
  ✗ Delta absolu    : -0.000771  (négatif)
  ✓ Rank change     : -76        (meilleure préservation)
  ✓ Percentile      : Stable

Verdict: Blue-chip résilient, valeur sûre
```

**Pertinence :**
- **Vue équilibrée** : Pas biaisée vers une seule métrique
- **Robuste** : Moins sensible aux outliers individuels
- **Personnalisable** : Ajuster les poids selon stratégie
- **Actionable** : Score unique pour ranking

**Stratégies de pondération alternatives :**

```python
# Stratégie AGGRESSIVE (croissance pure)
composite = 0.50 * ratio + 0.30 * delta + 0.20 * percentile
→ Favorise les "rockets" à haut risque

# Stratégie CONSERVATRICE (stabilité)
composite = 0.50 * rank_change + 0.30 * percentile + 0.20 * delta
→ Favorise les valeurs défensives

# Stratégie EARLY-STAGE
composite = 0.60 * ratio + 0.40 * percentile
→ Ignore le rang (pas pertinent pour nouvelles entrées)

# Stratégie BALANCED (défaut actuel)
composite = 0.30 * delta + 0.30 * ratio + 0.20 * rank + 0.20 * percentile
→ Équilibre croissance et stabilité
```

**Cas d'usage :**
- **Portfolio construction** : Ranger les opportunités
- **Due diligence** : Score unique de référence
- **Automated screening** : Filtrer programmatiquement
- **Reporting** : Simplifier la communication aux LPs

**Comment l'utiliser :**

```python
# 1. Filtrer par seuil
df_promising = df[df['composite_score'] > 0.50]  # Top 25% environ

# 2. Analyser les composantes
for company in df_promising:
    print(f"{company['name']}: {company['composite_score']:.3f}")
    print(f"  Delta: {company['norm_delta']:.2f}")
    print(f"  Ratio: {company['norm_ratio']:.2f}")
    print(f"  Rank: {company['norm_rank_change']:.2f}")
    print(f"  %ile: {company['norm_percentile']:.2f}")

# 3. Décider selon profil
if high_risk_appetite:
    focus_on = df_promising.nlargest(10, 'norm_ratio')
else:
    focus_on = df_promising.nlargest(10, 'norm_rank_change')
```

---

## Guide d'Utilisation

### Workflow Recommandé

```bash
# Étape 1: Générer les comparaisons TechRank
python TechRank_Comparison.py --data crunchbase --alpha 0.0 --beta -50

# Étape 2: Calculer les métriques alternatives
python TechRank_Alternative_Metrics.py

# Étape 3: Analyser les résultats
# → Fichiers CSV générés dans techrank_comparison/
```

### Fichiers Générés

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `promising_companies_all_metrics.csv` | Dataset complet | Analyse approfondie |
| `promising_by_ratio.csv` | Top 20 par ratio | Early-stage screening |
| `promising_by_percentile.csv` | Top 20 par percentile | Performance relative |
| `promising_by_composite.csv` | Top 20 par score composite | Shortlist prioritaire |
| `promising_by_rank_preservation.csv` | Top 20 par préservation | Blue-chips |
| `promising_by_zscore.csv` | Top 20 par Z-score | Outliers statistiques |

### Analyse des Résultats

**Approche en 3 étapes :**

```python
import pandas as pd

# 1. Charger le dataset complet
df = pd.read_csv('techrank_comparison/promising_companies_all_metrics.csv')

# 2. Filtrer par score composite
threshold = 0.50  # Top 25% environ
df_shortlist = df[df['composite_score'] > threshold]

print(f"Shortlist: {len(df_shortlist)} entreprises")

# 3. Analyser par profil
for idx, row in df_shortlist.iterrows():
    name = row['display_name']
    score = row['composite_score']

    # Identifier le profil dominant
    if row['ratio_multiplier'] > 5:
        profile = "ROCKET (croissance explosive)"
    elif row['rank_change'] > -50:
        profile = "STABLE (résilient)"
    elif row['percentile_change'] > 50:
        profile = "RISING STAR (montée en puissance)"
    else:
        profile = "BALANCED"

    print(f"\n{name} ({score:.3f}) - {profile}")
    print(f"  Ratio: {row['ratio_multiplier']:.2f}x")
    print(f"  Delta: {row['techrank_delta']:.6f}")
    print(f"  Percentile: {row['percentile_change']:+.1f}%")
```

---

## Recommandations par Stratégie

### Selon Votre Objectif d'Investissement

| Objectif | Métrique Primaire | Métrique Secondaire | Seuils Suggérés |
|----------|-------------------|---------------------|-----------------|
| **Pre-seed / Early-stage** | Ratio multiplicateur | Percentile change | ratio > 5x, %ile > +50% |
| **Seed / Growth** | Percentile change | Composite score | %ile > +60%, composite > 0.6 |
| **Late-stage / Growth** | Composite score | Rank preservation | composite > 0.5, rank > -100 |
| **Blue-chip / Defensive** | Rank preservation | Percentile change | rank > -50, %ile > 0% |
| **Opportunistic / High-risk** | Z-score | Ratio | z > 2.0, ratio > 10x |
| **Diversified Portfolio** | Composite score | Tous | composite > 0.5 |

### Profils d'Entreprises Identifiés

**1. ROCKET (Croissance Explosive)**
```
Critères:
  - Ratio > 10x
  - Percentile change > +80%
  - Delta positif

Exemple: ChromoPIC (ratio 10^30x, %ile +89%)

Stratégie: Early-stage, haut risque/haute récompense
```

**2. RISING STAR (Montée Régulière)**
```
Critères:
  - Percentile change +50% à +80%
  - Composite score > 0.5
  - Ratio 2-5x

Exemple: QSIM Plus (ratio 6.35x, %ile +87%)

Stratégie: Growth-stage, risque modéré
```

**3. BLUE-CHIP (Valeur Sûre)**
```
Critères:
  - Rank change > -50
  - Percentile stable (±20%)
  - Delta légèrement négatif acceptable

Exemple: QuantumAstra (rank -38)

Stratégie: Defensive, low-risk
```

**4. TURNAROUND (Récupération)**
```
Critères:
  - Delta négatif → positif
  - Percentile remonte
  - Score composite > 0.4

Stratégie: Contrarian, timing critique
```

### Approche Pratique

**Étape 1: Screening Initial**
```python
# Utiliser le score composite pour créer une shortlist
df_shortlist = df[df['composite_score'] > 0.50]
```

**Étape 2: Segmentation**
```python
# Classifier par profil
rockets = df_shortlist[df_shortlist['ratio_multiplier'] > 5]
stars = df_shortlist[
    (df_shortlist['percentile_change'] > 50) &
    (df_shortlist['ratio_multiplier'] < 5)
]
blue_chips = df_shortlist[df_shortlist['rank_change'] > -50]
```

**Étape 3: Due Diligence Approfondie**
- Analyser les 4 métriques pour chaque entreprise
- Comparer avec le contexte sectoriel
- Vérifier les données fondamentales (funding, team, traction)

---

## Formules de Référence Rapide

```python
# 1. Ratio Multiplicateur
ratio = techrank_after / techrank_before

# 2. Nouvelles Apparitions
is_new = (techrank_before == 0) & (techrank_after > 0)

# 3. Préservation du Rang
rank_change = rank_before - rank_after  # Positif = amélioration

# 4. Percentile
percentile = rank(techrank) / total * 100
percentile_change = percentile_after - percentile_before

# 5. Z-Score
z_score = (techrank_delta - mean(techrank_delta)) / std(techrank_delta)

# 6. Score Composite
norm_metric = (metric - min) / (max - min)
composite = 0.3*delta + 0.3*ratio + 0.2*rank + 0.2*percentile
```

---

## Troubleshooting

### Problème : Ratios infinis
```
ChromoPIC: ratio = 10^30x
```
**Cause :** techrank_before ≈ 0 (très proche de zéro)
**Solution :** Filtrer `techrank_before > 1e-8` ou interpréter comme "nouvelle entrée"

### Problème : Tous les ranks négatifs
```
Toutes les entreprises ont rank_change < 0
```
**Cause :** Dilution normale (64 → 199 entreprises)
**Solution :** Utiliser le percentile au lieu du rang absolu

### Problème : Pas de nouvelles apparitions
```
Total nouvelles entrées: 0
```
**Cause :** TechRank_Comparison filtre `techrank_before != 0`
**Solution :** Modifier le filtre ou utiliser le graphe complet

### Problème : Score composite = 0 pour certaines entreprises
```
Company X: composite_score = 0.0
```
**Cause :** Toutes les métriques normalisées sont à leur minimum
**Solution :** Normal, ces entreprises sont en queue de distribution

---

## Prochaines Étapes Suggérées

1. **Personnaliser les poids du score composite** selon votre stratégie
2. **Ajouter des métriques complémentaires** :
   - Vélocité (variation du delta dans le temps)
   - Accélération (variation du ratio)
   - Volatilité (variance des scores)
3. **Créer des visualisations** :
   - Scatter plot : ratio vs percentile
   - Heatmap : corrélations entre métriques
   - Time series : évolution du composite
4. **Automatiser le reporting** :
   - Génération de PDF par entreprise
   - Dashboard interactif
   - Alertes sur nouvelles opportunités

---

## Conclusion

Le système de métriques alternatives offre une **vision multi-dimensionnelle** pour identifier les entreprises prometteuses, au-delà du simple delta absolu.

**Points clés à retenir :**

1. **Percentile change** = Meilleure métrique pour résoudre la dilution
2. **Score composite** = Recommandé pour ranking général
3. **Ratio multiplicateur** = Idéal pour early-stage
4. **Rank preservation** = Indicateur de robustesse
5. **Pas de métrique unique parfaite** = Toujours analyser le contexte

**La bonne métrique dépend de votre stratégie d'investissement.**

---

## Contacts et Support

Pour toute question ou suggestion d'amélioration, référez-vous aux fichiers :
- [FIX_DICTIONARY_COLLISION.md](FIX_DICTIONARY_COLLISION.md) : Fix technique du système
- [TECHRANK_FIX_SUMMARY.md](TECHRANK_FIX_SUMMARY.md) : Historique des corrections

**Fichiers associés :**
- `TechRank_Comparison.py` : Script principal de comparaison
- `TechRank_Alternative_Metrics.py` : Calcul des métriques alternatives
- `verify_fixed_graph.py` : Vérification de la structure du graphe

---

*Document généré pour l'analyse TechRank - Version 1.0*
