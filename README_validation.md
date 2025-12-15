# Validation Temporelle - Documentation

## Vue d'ensemble

Ce module étend votre pipeline TGN-TechRank existant avec des **métriques de validation temporelle robustes** pour évaluer la capacité du modèle à prédire les entreprises prometteuses **AVANT** qu'elles ne deviennent évidentes.

### Principe

```
Timeline :
├─────────────┼─────────────┼──────────→
2018          2020          2022      temps
│             │             │
│◄─Training──►│◄Validation─►│

Question clé : Les entreprises classées en tête en 2020 (par TGN+TechRank)
               ont-elles RÉELLEMENT connu une croissance en 2020-2022?
```

## Installation

```bash
# Aucune nouvelle dépendance - utilise vos packages existants
pip install pandas numpy networkx scipy matplotlib seaborn
```

## Utilisation Rapide

### Option 1: Pipeline complet (recommandé)

```python
from temporal_validation import run_temporal_validation_pipeline
import pandas as pd
import pickle

# 1. Charger les résultats existants
df_delta = pd.read_csv('techrank_comparison/company_techrank_deltas.csv')

# 2. Charger les graphes
with open('B_train.pkl', 'rb') as f:
    B_train = pickle.load(f)
with open('B_test.pkl', 'rb') as f:
    B_test = pickle.load(f)

# 3. Lancer la validation
metrics = run_temporal_validation_pipeline(
    df_delta=df_delta,
    B_before=B_train,
    B_after=B_test,
    top_k_list=[10, 20, 50],
    growth_threshold=2.0,  # Doublement du degré
    prediction_horizon_days=730,  # 2 ans
    output_dir='validation_results',
    create_plots=True,
    export_latex=True
)

# 4. Résultats
print(f"Precision@20: {metrics.precision_at_k[20]:.2%}")
print(f"Spearman ρ: {metrics.spearman_rho:.3f}")
print(f"EDR@50: {metrics.edr_at_k[50]:.2%}")
print(f"Lift@20: {metrics.lift_at_k[20]:.2f}x")
```

### Option 2: Métriques seulement (sans plots)

```python
from temporal_validation import compute_validation_metrics

metrics = compute_validation_metrics(
    df_delta=df_delta,
    B_before=B_train,
    B_after=B_test,
    top_k_list=[10, 20, 50],
    growth_threshold=2.0
)
```

## Métriques Calculées

### 1. Precision@K

**Définition**: Pour les K entreprises les mieux classées par le modèle, combien ont RÉELLEMENT eu une croissance positive?

```python
Precision@K = (Nb d'entreprises top-K avec croissance) / K
```

**Interprétation**:
- `Precision@20 = 0.70` → 14/20 entreprises du top-20 ont effectivement grandi
- ✅ **Bon**: > 0.6
- ⚠️ **Modéré**: 0.3 - 0.6
- ❌ **Faible**: < 0.3

### 2. Rank Correlation (Spearman ρ)

**Définition**: Corrélation entre le classement prédit et la croissance réelle observée.

```python
ρ = spearmanr(predicted_ranks, actual_growth)
```

**Interprétation**:
- `ρ > 0.7` → Corrélation forte ⭐
- `ρ > 0.4` → Corrélation modérée ✓
- `ρ > 0.2` → Corrélation faible
- `ρ < 0.2` → Pas de corrélation ❌

**Exemple**: `ρ = 0.58, p < 0.001` → Corrélation modérée et hautement significative

### 3. Early Detection Rate (EDR@K)

**Définition**: Parmi les entreprises ayant connu une **forte croissance** (doublement), combien étaient dans le top-K?

```python
EDR@K = (Nb détectées dans top-K) / (Total entreprises forte croissance)
```

**Interprétation**:
- `EDR@50 = 0.45` → Le modèle a détecté 45% des entreprises à forte croissance dans le top-50

### 4. Lift Score

**Définition**: Amélioration par rapport à une sélection aléatoire.

```python
Lift@K = (Taux de succès modèle) / (Taux de succès baseline)
```

**Interprétation**:
- `Lift@20 = 8.5x` → Le modèle est 8.5× meilleur que le hasard
- ✅ **Excellent**: > 5x
- ✓ **Bon**: 2-5x
- ⚠️ **Faible**: < 2x

### 5. Lead Time

**Définition**: Délai moyen entre la prédiction et l'observation réelle de la croissance.

**Exemple**: `Lead time = 730 jours (24 mois)` → Le modèle prédit 2 ans à l'avance

## Structure des Fichiers Générés

```
validation_results/
├── validation_metrics.json           # Toutes les métriques (réutilisable)
├── validation_report.tex             # Rapport LaTeX prêt à inclure
├── precision_at_k_comparison.png     # Model vs Baselines
├── predicted_vs_actual_scatter.png   # Scatter plot avec corrélation
├── top_20_companies_validation.png   # Top-20 avec croissance réelle
└── edr_lift_summary.png              # EDR et Lift visualisés
```

## Workflow Complet

### Étape 1: Préparer les données

```python
# Votre code existant (déjà fait!)
from data.bipartite_investor_comp import main, temporal_split_graph

# Split temporel (déjà implémenté dans votre code)
B_train, B_val, B_test, max_train_time, max_val_time = temporal_split_graph(
    B_full,
    train_ratio=0.85,
    val_ratio=0.0
)
```

### Étape 2: Calculer TechRank AVANT et APRÈS

```python
# Votre code existant (TechRank_Comparison.py)
from code.TechRank import run_techrank

# TechRank sur graphe AVANT (train)
_, df_companies_before = run_techrank(
    B=B_train,
    dict_investors=dict_inv_train,
    dict_comp=dict_comp_train,
    alpha=0.8,
    beta=-0.6
)

# TechRank sur graphe APRÈS (test = "future réel")
_, df_companies_after = run_techrank(
    B=B_test,
    dict_investors=dict_inv_test,
    dict_comp=dict_comp_test,
    alpha=0.8,
    beta=-0.6
)

# Calculer les deltas (déjà implémenté!)
from TechRank_Comparison import analyze_company_deltas

df_delta, df_promising = analyze_company_deltas(
    df_companies_before,
    df_companies_after,
    threshold=0.01,
    top_k=50
)
```

### Étape 3: Validation temporelle (NOUVEAU)

```python
from temporal_validation import run_temporal_validation_pipeline

metrics = run_temporal_validation_pipeline(
    df_delta=df_delta,
    B_before=B_train,
    B_after=B_test,
    prediction_horizon_days=(max_val_time - max_train_time).days
)
```

## Exemples de Résultats

### Résultat Excellent

```
📊 VALIDATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 PRECISION@K:
   @10: 0.800 (80.0%)
   @20: 0.750 (75.0%)
   @50: 0.660 (66.0%)

📈 RANK CORRELATION:
   Spearman ρ: 0.6234 (p=0.0001)
   ⭐ Highly significant correlation!

🔍 EARLY DETECTION RATE (≥2.0x growth):
   @50: 0.550 (11/20 high-growth companies detected)

📊 LIFT SCORE:
   @20: 9.38x (baseline: 0.08)

⏱️ LEAD TIME:
   Average: 730 days (24.3 months)
```

**Interprétation**: Le modèle détecte efficacement les entreprises prometteuses 2 ans à l'avance, avec un taux de succès 9× supérieur au hasard.

### Résultat Modéré

```
🎯 PRECISION@K:
   @20: 0.450 (45.0%)

📈 RANK CORRELATION:
   Spearman ρ: 0.3521 (p=0.0123)
   ✓ Significant correlation

📊 LIFT SCORE:
   @20: 3.75x
```

**Interprétation**: Le modèle capture certains signaux prédictifs mais pourrait bénéficier d'améliorations (features, hyperparamètres).

## Bonnes Pratiques

### 1. Validation stricte temporelle

✅ **CORRECT**: Split strict par date
```python
train_end = "2020-12-31"
test_start = "2021-01-01"
# Aucune donnée du futur dans l'entraînement
```

❌ **INCORRECT**: Split aléatoire
```python
train_test_split(random_state=42)  # Leakage temporel!
```

### 2. Interprétation des métriques

- **Precision@K** → Qualité des top-K prédictions (most important)
- **Spearman ρ** → Qualité globale du classement
- **EDR@K** → Capacité à détecter les "pépites"
- **Lift** → Amélioration vs baselines

### 3. Choix du seuil de croissance

```python
# Tester plusieurs seuils
for threshold in [1.5, 2.0, 2.5, 3.0]:
    metrics = compute_validation_metrics(..., growth_threshold=threshold)
    print(f"Threshold {threshold}x: EDR@50={metrics.edr_at_k[50]:.2%}")
```

Recommandation:
- **2.0x** (doublement) → Standard, équilibré
- **1.5x** → Plus permissif (plus d'entreprises qualifiées)
- **3.0x** → Très strict (seulement croissances exceptionnelles)

## Comparaison avec d'autres Approches

### vs Random Baseline
Sélection aléatoire de K entreprises.

### vs Degree Baseline
Classement naïf par degré initial seulement (sans TGN, sans TechRank).

**Le modèle doit TOUJOURS surpasser les deux baselines.**

## Intégration avec votre Rapport LaTeX

Le fichier `validation_report.tex` est prêt à inclure:

```latex
\section{Résultats}

\input{validation_results/validation_report.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{validation_results/precision_at_k_comparison.png}
    \caption{Comparaison des performances: Modèle vs Baselines}
\end{figure}
```

## Dépannage

### Problème: Corrélation faible (ρ < 0.2)

**Causes possibles**:
1. Features de nœuds insuffisantes (vecteurs zéro)
2. Hyperparamètres TGN non optimisés
3. Paramètres TechRank (α, β) inadaptés

**Solutions**:
```python
# 1. Enrichir les features
# Ajouter descriptions textuelles (BERT), métadonnées

# 2. Optimiser TGN
# Sweep sur num_layers, memory_dim, etc.

# 3. Tester différents α, β
for alpha in [0.0, 0.3, 0.8]:
    for beta in [-5.0, -2.0, -0.6]:
        # Recalculer TechRank et évaluer
```

### Problème: Precision@20 faible (< 0.4)

**Diagnostic**:
```python
# Analyser les faux positifs
top_20 = df_delta.nlargest(20, 'techrank_delta')
false_positives = top_20[top_20['degree_growth'] <= 0]
print(false_positives[['final_configuration', 'techrank_delta', 'degree_before']])
```

**Vérifier**:
- Les entreprises ont-elles un degré initial trop faible? → Filtrer par `degree_before > 2`
- Problème de hard negative mining? → Améliorer le sampling TGN

### Problème: "FileNotFoundError: df_delta.csv"

**Solution**: Lancer d'abord votre pipeline existant:
```bash
python TechRank_Comparison.py --data crunchbase --alpha 0.8 --beta -0.6 --plot
```

Puis la validation:
```bash
python -c "from temporal_validation import run_temporal_validation_pipeline; ..."
```

## API Reference

### Classes

#### `ValidationMetrics`
Dataclass contenant toutes les métriques calculées.

**Attributs**:
- `precision_at_k: Dict[int, float]`
- `spearman_rho: float`
- `spearman_p_value: float`
- `edr_at_k: Dict[int, float]`
- `lift_at_k: Dict[int, float]`
- `avg_lead_time_days: float`

**Méthodes**:
- `to_dict() -> Dict`: Convertit en dictionnaire
- `save_json(filepath)`: Sauvegarde en JSON

### Fonctions

#### `compute_validation_metrics()`
Calcule toutes les métriques de validation.

**Args**:
- `df_delta (pd.DataFrame)`: DataFrame avec deltas TechRank
- `B_before (nx.Graph)`: Graphe initial
- `B_after (nx.Graph)`: Graphe futur réel
- `top_k_list (List[int])`: Liste des K (défaut: [10, 20, 50])
- `growth_threshold (float)`: Seuil de forte croissance (défaut: 2.0)
- `prediction_horizon_days (float)`: Horizon en jours

**Returns**:
- `ValidationMetrics`

#### `create_validation_plots()`
Génère toutes les visualisations.

**Args**:
- `df_delta (pd.DataFrame)`
- `metrics (ValidationMetrics)`
- `save_dir (str)`: Répertoire de sortie
- `top_k_viz (int)`: Nombre d'entreprises à visualiser (défaut: 20)

#### `generate_latex_report()`
Génère un rapport LaTeX.

**Args**:
- `metrics (ValidationMetrics)`
- `output_path (str)`: Chemin du fichier .tex

**Returns**:
- `str`: Contenu LaTeX

## FAQ

**Q: Quelle est la différence avec les métriques TGN standard (AUROC, AP)?**

A: Les métriques TGN mesurent la capacité à distinguer vrais/faux liens. La validation temporelle mesure si les entreprises bien classées connaissent RÉELLEMENT une croissance future. C'est une validation business, pas seulement technique.

**Q: Pourquoi Precision@K et pas seulement AUROC?**

A: En pratique, on ne regarde que le top-K (ex: top-20 entreprises). Precision@K mesure directement l'utilité business.

**Q: Comment interpréter un Spearman ρ = 0.5?**

A: Corrélation modérée. Le modèle capture des patterns prédictifs mais pas parfaitement. Chercher à améliorer les features ou hyperparamètres.

**Q: EDR@50 = 0.3, c'est bon?**

A: Dépend du contexte. Si seulement 10 entreprises ont eu une forte croissance, détecter 3/10 (30%) est déjà utile. Comparer avec le Lift pour contextualiser.

**Q: Mon modèle a Precision@20 = 0.2, que faire?**

A:
1. Vérifier que le split temporel est correct (pas de leakage)
2. Analyser les faux positifs (pourquoi sont-ils mal prédits?)
3. Enrichir les features de nœuds
4. Optimiser les hyperparamètres (TGN + TechRank)
5. Tester d'autres architectures (GCN, GraphSAGE)

## Citation

Si vous utilisez ce module dans vos travaux, merci de citer:

```bibtex
@misc{temporal_validation_tgn,
  author = {Your Name},
  title = {Temporal Validation for TGN-TechRank Disruption Detection},
  year = {2025},
  publisher = {EPFL - CYD Campus},
  howpublished = {\url{https://github.com/...}}
}
```

## Support

Pour toute question:
- Issues GitHub: https://github.com/.../issues
- Email: your.email@epfl.ch

## License

MIT License - Voir LICENSE file
