# Paragraphes pour Rapport : Validation du Modèle

## Version 1 : Courte (1 paragraphe, ~150 mots)

### Pour section "Méthodologie" ou "Expérimentation"

Pour évaluer notre modèle TGN, nous adoptons une approche de **validation temporelle** stricte, où les données sont divisées chronologiquement en ensembles d'entraînement (70%), validation (15%) et test (15%). Cette stratégie garantit que le modèle prédit uniquement des interactions futures, reflétant ainsi un scénario d'utilisation réaliste. L'évaluation se base sur des **métriques de ranking** plutôt que sur une simple classification binaire : pour chaque lien positif du test set, nous calculons son rang parmi un ensemble de candidats négatifs échantillonnés aléatoirement. Nous mesurons le **Mean Reciprocal Rank (MRR)**, qui quantifie la position moyenne du vrai lien dans le classement, ainsi que le **Recall@K**, représentant la proportion de vrais liens retrouvés dans les top-K prédictions. Nous reportons également l'**Average Precision (AP)** et l'**AUC-ROC** pour permettre la comparaison avec les approches de classification binaire traditionnelles. Cette méthodologie multi-métrique permet d'évaluer à la fois la capacité discriminative du modèle et sa pertinence pratique pour des systèmes de recommandation.

---

## Version 2 : Moyenne (2 paragraphes, ~250 mots)

### Pour section "Méthodologie Expérimentale"

**Validation Temporelle.** Afin de respecter la nature dynamique des graphes temporels et d'éviter toute fuite d'information (*temporal leakage*), nous adoptons une stratégie de validation temporelle stricte. Les interactions sont divisées chronologiquement selon les quantiles temporels : les 70% premières interactions constituent l'ensemble d'entraînement, les 15% suivantes l'ensemble de validation pour le réglage des hyperparamètres, et les 15% finales l'ensemble de test pour l'évaluation finale. Cette séparation garantit que le modèle ne prédit jamais des événements passés, simulant ainsi un déploiement réaliste où seules les informations historiques sont disponibles au moment de la prédiction.

**Métriques d'Évaluation.** Plutôt qu'une simple classification binaire, nous évaluons le modèle selon une approche de **ranking**, plus représentative des applications pratiques en recommandation d'investisseurs. Pour chaque lien positif (u, v, t) du test set, nous calculons son score ainsi que les scores de N = 100 candidats négatifs échantillonnés aléatoirement. Le **Mean Reciprocal Rank (MRR)** mesure la position moyenne du vrai lien dans ce classement, tandis que le **Recall@K** quantifie la proportion de vrais liens présents dans les top-K prédictions (nous reportons K ∈ {10, 50, 1000}). Nous complétons cette évaluation par l'**Average Precision (AP)** et l'**AUC-ROC**, métriques standard pour la prédiction de liens, permettant ainsi une comparaison directe avec les travaux antérieurs. L'ensemble de ces métriques offre une vue multidimensionnelle de la performance du modèle, évaluant à la fois sa précision discriminative et son utilité pour un système de recommandation en production.

---

## Version 3 : Longue (3-4 paragraphes, ~400 mots)

### Pour section "Méthodologie" détaillée

**Protocole de Validation Temporelle.** La validation de modèles sur des graphes temporels nécessite une attention particulière pour éviter le *temporal leakage*, où des informations futures contamineraient l'apprentissage. Nous adoptons donc une stratégie de **division temporelle stricte** basée sur les timestamps des interactions. Soit T l'ensemble des timestamps, nous définissons t₇₀ et t₈₅ comme les 70ème et 85ème quantiles de T. Les interactions avec t < t₇₀ constituent l'ensemble d'entraînement (70% des données), celles avec t₇₀ ≤ t < t₈₅ forment l'ensemble de validation (15%), et les interactions avec t ≥ t₈₅ constituent l'ensemble de test (15%). Cette partition garantit que le modèle, entraîné sur le passé, est évalué exclusivement sur sa capacité à prédire le futur, reflétant ainsi fidèlement un scénario de déploiement réel.

**Évaluation par Ranking.** Contrairement aux approches de classification binaire traditionnelles qui évaluent la capacité du modèle à distinguer un lien positif d'un négatif arbitraire, nous adoptons une méthodologie de **ranking** plus représentative des applications pratiques. Pour chaque interaction test (u, v, t), nous générons un ensemble de N candidats comprenant le nœud destination réel v et N - 1 nœuds négatifs échantillonnés uniformément parmi tous les nœuds possibles (excluant v). Le modèle calcule un score pour chaque candidat, et nous mesurons le rang r du nœud positif v dans le classement décroissant de ces scores. Cette approche simule directement une tâche de recommandation où le système doit identifier le bon candidat parmi un large ensemble de possibilités.

**Métriques Utilisées.** Nous reportons quatre familles de métriques complémentaires. Le **Mean Reciprocal Rank (MRR = 1/|Test| Σᵢ 1/rᵢ)** quantifie la position moyenne des vrais liens dans le classement, avec des valeurs proches de 1 indiquant que les vrais liens sont systématiquement bien classés. Le **Recall@K** mesure la proportion de vrais liens présents dans les top-K prédictions : Recall@K = |{i : rᵢ ≤ K}| / |Test|. Nous reportons K ∈ {10, 50, 1000} pour capturer différents régimes de précision. L'**Average Precision (AP)**, définie comme l'aire sous la courbe Precision-Recall, évalue la qualité globale du classement tout en étant robuste au déséquilibre de classes. Enfin, l'**AUC-ROC** mesure la capacité du modèle à distinguer les classes positives et négatives. Ces métriques, couramment utilisées dans la littérature sur la prédiction de liens [Rossi et al., 2020; Kumar et al., 2020], permettent une comparaison directe avec les approches de l'état de l'art.

**Baseline Aléatoire.** Pour contextualiser nos résultats, nous comparons systématiquement avec une baseline aléatoire. Dans notre dataset CrunchBase, avec 52 liens positifs sur 170,742 paires possibles (ratio 0.03%), un classement aléatoire atteindrait un Recall@1000 de 0.6%. Tout modèle significativement au-dessus de ce seuil démontre une capacité d'apprentissage réelle. Nous mesurons l'amélioration relative comme le ratio entre le Recall@K du modèle et celui de la baseline aléatoire, fournissant ainsi une interprétation intuitive de la performance.

---

## Version 4 : Très Formelle (Style Article Scientifique, ~300 mots)

### Pour article de conférence (NeurIPS, ICML, KDD, etc.)

**Experimental Protocol.** Following standard practices in temporal graph learning [Rossi et al., 2020; Kumar et al., 2020], we employ a strict temporal validation protocol to prevent information leakage. Let ℰ = {(u, v, t)} denote the set of all interactions. We partition ℰ into train, validation, and test sets based on temporal quantiles: ℰ_train = {(u, v, t) : t < q₀.₇₀}, ℰ_val = {(u, v, t) : q₀.₇₀ ≤ t < q₀.₈₅}, and ℰ_test = {(u, v, t) : t ≥ q₀.₈₅}, where q_p denotes the p-th quantile of all timestamps. This ensures the model is trained on historical data and evaluated exclusively on future predictions, mirroring real-world deployment scenarios.

**Evaluation Metrics.** We adopt a ranking-based evaluation framework rather than binary classification. For each test interaction (u, v, t) ∈ ℰ_test, we sample N = 100 negative nodes 𝒩_u uniformly from 𝒱 \ {v}, compute scores s(u, v', t) for all v' ∈ {v} ∪ 𝒩_u, and determine the rank r of the true node v in descending order of scores. We report:

- **Mean Reciprocal Rank (MRR)**: MRR = 1/|ℰ_test| Σ_(u,v,t)∈ℰ_test 1/r, measuring the average inverse rank of true links.
- **Recall@K**: Recall@K = |{(u,v,t) ∈ ℰ_test : r ≤ K}| / |ℰ_test|, quantifying the fraction of true links retrieved in the top-K predictions. We report K ∈ {10, 50, 1000}.
- **Average Precision (AP)**: The area under the precision-recall curve, robust to class imbalance.
- **AUC-ROC**: The area under the receiver operating characteristic curve, measuring binary classification performance.

These metrics provide complementary perspectives: MRR and Recall@K evaluate ranking quality relevant to recommendation systems, while AP and AUC enable comparison with prior work on link prediction [Hamilton et al., 2017; Xu et al., 2020]. Given the extreme class imbalance in our CrunchBase dataset (0.03% positive rate), we report the improvement factor over a random baseline, which achieves Recall@1000 = 0.6%.

---

## Version 5 : En Français Académique (~250 mots)

### Pour rapport de Master/Thèse en français

**Protocole de Validation Temporelle.** Afin de respecter la causalité temporelle inhérente aux graphes dynamiques, nous adoptons un protocole de validation strictement chronologique. Les interactions sont partitionnées selon leurs timestamps : 70% des interactions les plus anciennes constituent l'ensemble d'entraînement, 15% servent à la validation des hyperparamètres, et les 15% restantes, les plus récentes, forment l'ensemble de test. Cette stratégie garantit que le modèle prédit exclusivement des événements futurs, évitant ainsi toute fuite d'information (*temporal leakage*) et simulant fidèlement un scénario de déploiement réel.

**Métriques d'Évaluation.** Plutôt qu'une simple classification binaire, nous évaluons le modèle selon une approche de **ranking** plus pertinente pour les systèmes de recommandation. Pour chaque lien test (u, v, t), le modèle classe le nœud destination réel v parmi 100 candidats négatifs échantillonnés aléatoirement. Nous mesurons le **Mean Reciprocal Rank (MRR)**, qui quantifie la position moyenne du vrai lien dans ce classement, ainsi que le **Recall@K**, représentant la proportion de vrais liens retrouvés dans les top-K prédictions (K ∈ {10, 50, 1000}). Nous complétons par l'**Average Precision (AP)** et l'**AUC-ROC**, métriques standard de prédiction de liens. Dans notre jeu de données CrunchBase, caractérisé par un fort déséquilibre (0,03% de liens positifs), un classement aléatoire atteindrait un Recall@1000 de 0,6%. Nos résultats sont donc systématiquement rapportés avec le facteur d'amélioration par rapport à cette baseline, fournissant ainsi une mesure intuitive de la performance du modèle.

---

## Version 6 : Condensée pour Abstract/Résumé (~80 mots)

We evaluate our TGN model using temporal validation, where data is split chronologically into train (70%), validation (15%), and test (15%) sets, ensuring predictions are made exclusively on future interactions. Performance is measured using ranking-based metrics: Mean Reciprocal Rank (MRR), Recall@K (K ∈ {10, 50, 1000}), Average Precision (AP), and AUC-ROC. Given the extreme class imbalance (0.03% positive links), we report improvement factors over a random baseline to contextualize results.

---

## Recommandations d'Utilisation

| Section du Rapport | Version Recommandée | Pourquoi |
|-------------------|---------------------|----------|
| **Abstract/Résumé** | Version 6 (Condensée) | Très concis, capture l'essentiel |
| **Introduction** | Version 1 (Courte) | Introduit la méthodologie sans détails |
| **Méthodologie** | Version 2 (Moyenne) ou 3 (Longue) | Détails suffisants sans surcharger |
| **Article Scientifique** | Version 4 (Formelle) | Notation mathématique, références |
| **Rapport Master/Thèse FR** | Version 5 (Français) | Style académique français |
| **Supplementary Material** | Version 3 (Longue) | Tous les détails techniques |

---

## Éléments à Personnaliser

Selon vos résultats finaux, remplacez les valeurs suivantes :

```
[VALEURS ACTUELLES - À METTRE À JOUR]

- AP: 0.35 → [votre valeur après Focal/HAR]
- AUC: 0.75 → [votre valeur]
- MRR: 0.02 → [votre valeur]
- Recall@10: 0.00 → [votre valeur]
- Recall@50: 0.05 → [votre valeur]
- Recall@1000: 0.077 → [votre valeur]
- Baseline aléatoire: 0.6% → [confirmer avec votre dataset final]
- Amélioration vs baseline: 13x → [recalculer]
```

---

## Citations à Ajouter (Optionnel)

Si vous utilisez la Version 4 (formelle), ajoutez ces références :

```bibtex
@article{rossi2020temporal,
  title={Temporal graph networks for deep learning on dynamic graphs},
  author={Rossi, Emanuele and Chamberlain, Ben and Frasca, Fabrizio and Eynard, Danica and Monti, Federico and Bronstein, Michael},
  journal={ICML Workshop on Graph Representation Learning},
  year={2020}
}

@inproceedings{kumar2020predicting,
  title={Predicting dynamic embedding trajectory in temporal interaction networks},
  author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
  booktitle={KDD},
  year={2020}
}

@inproceedings{hamilton2017inductive,
  title={Inductive representation learning on large graphs},
  author={Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  booktitle={NeurIPS},
  year={2017}
}

@article{xu2020tgat,
  title={Inductive representation learning on temporal graphs},
  author={Xu, Da and Ruan, Chuanwei and Korpeoglu, Evren and Kumar, Sushant and Achan, Kannan},
  journal={ICLR},
  year={2020}
}
```

---

## Figure Suggérée

Pour accompagner le texte, créez une figure montrant :

```
┌─────────────────────────────────────────────────────────┐
│  Timeline des Interactions (CrunchBase)                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [████████████████ TRAIN 70% ████████] [VAL] [TEST]    │
│  2000              2010            2018 2020      2023   │
│                                      ↑    ↑         ↑    │
│                                      │    │         │    │
│                              Entraînement │    Évaluation│
│                                           │              │
│                                    Validation            │
│                                                          │
│  Prédiction: Investissements 2020-2023                  │
│  basée sur: Historique 2000-2018                        │
└─────────────────────────────────────────────────────────┘
```

---

## Exemple d'Intégration Complète

Voici comment intégrer le paragraphe dans votre structure de rapport :

```markdown
## 4. Méthodologie Expérimentale

### 4.1 Architecture du Modèle
[Votre description du TGN...]

### 4.2 Stratégie de Validation

[INSÉRER VERSION 2 OU 3 ICI]

### 4.3 Fonction de Perte
Pour gérer le déséquilibre de classes extrême (0.03% de liens positifs),
nous comparons trois fonctions de perte : Binary Cross-Entropy (BCE) comme
baseline, Focal Loss pour...

### 4.4 Hyperparamètres
[Vos hyperparamètres...]

## 5. Résultats

### 5.1 Performance des Fonctions de Perte
Le tableau 1 présente les résultats de validation pour les trois
fonctions de perte testées...

[TABLE avec résultats]
```

---

## Checklist pour Votre Rapport

- [ ] Expliquer le split temporel (70/15/15)
- [ ] Justifier pourquoi ranking > classification binaire
- [ ] Définir MRR avec formule mathématique
- [ ] Définir Recall@K avec formule
- [ ] Mentionner AP et AUC pour comparaison
- [ ] Donner baseline aléatoire (0.6%)
- [ ] Calculer amélioration vs baseline
- [ ] Ajouter figure du timeline temporel
- [ ] Citer au moins 2-3 références pertinentes

Bonne chance avec votre rapport ! 🎓
