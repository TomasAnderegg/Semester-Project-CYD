# Guide: Comparaison des Loss Functions

Ce guide explique comment entraîner ton modèle TGN avec différentes loss functions et comparer leurs performances.

## 1. Entraîner avec Différentes Loss Functions

### BCE Loss (Baseline)
```bash
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory
```
Résultats sauvegardés: `results/tgn-attn_bce.json`

### Focal Loss
```bash
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory \
  --use_focal_loss --focal_alpha 0.25 --focal_gamma 2.0
```
Résultats sauvegardés: `results/tgn-attn_focal.json`

### HAR Loss (Hardness Adaptive Reweighted)
```bash
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory \
  --use_har_loss --har_alpha 0.5 --har_temperature 0.07
```
Résultats sauvegardés: `results/tgn-attn_har.json`

### Hybrid Loss (Focal + HAR)
```bash
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory \
  --use_focal_loss --focal_alpha 0.25 --focal_gamma 2.0 \
  --use_har_loss --har_alpha 0.5 --har_temperature 0.07
```
Résultats sauvegardés: `results/tgn-attn_hybrid.json`

## 2. Comparer les Résultats

Une fois que tu as entraîné plusieurs modèles, lance le script de comparaison:

```bash
python plot_loss_comparison.py
```

## 3. Outputs Générés

Le script génère automatiquement dans `loss_comparison_plots/`:

### 📊 `training_loss_comparison.png`
- Courbes de training loss par epoch
- Compare BCE vs Focal vs HAR vs Hybrid
- Permet de voir quelle loss converge le mieux

### 📊 `test_metrics_comparison.png`
- Bar charts des métriques finales de test
- AUROC, AP, MRR, Recall@10, Recall@50
- Comparaison directe des performances

### 📊 `validation_metrics_over_epochs.png`
- Évolution des métriques de validation
- 4 subplots: MRR, AP, Recall@10, Recall@50
- Montre la stabilité de l'entraînement

### 📄 `summary_table.csv`
- Tableau récapitulatif de toutes les métriques
- Inclut les hyperparamètres utilisés
- Format CSV facile à importer dans Excel

## 4. Structure des Fichiers de Résultats

Chaque entraînement génère 2 fichiers:

### `results/{prefix}_{loss}.pkl` (Pickle)
Format binaire Python contenant toutes les données brutes.

### `results/{prefix}_{loss}.json` (JSON)
Format lisible contenant:
```json
{
  "loss_function": "har",
  "config": {
    "focal_alpha": null,
    "focal_gamma": null,
    "har_alpha": 0.5,
    "har_temperature": 0.07
  },
  "validation": {
    "ap": [0.782, 0.795, ...],
    "mrr": [0.306, 0.315, ...],
    "recall_10": [0.382, 0.391, ...],
    "recall_50": [0.852, 0.861, ...]
  },
  "test": {
    "ap": 0.807,
    "auc": 0.767,
    "mrr": 0.531,
    "recall_10": 0.611,
    "recall_50": 0.788
  },
  "training": {
    "losses": [0.642, 0.589, 0.521, ...],
    "epoch_times": [45.2, 44.8, 45.1, ...]
  }
}
```

## 5. Interprétation des Résultats

### Training Loss
- **Plus bas = meilleur** (convergence)
- Vérifie qu'il n'y a pas d'overfitting (écart train/val)

### AUROC & AP
- **Classification metrics**
- Plus haut = meilleure discrimination positif/négatif
- Optimal: > 0.75

### MRR (Mean Reciprocal Rank)
- **Ranking metric**
- Plus haut = meilleur classement du vrai investisseur
- Optimal: > 0.5

### Recall@K
- **Ranking metric**
- Recall@10: % de vrais investisseurs dans top-10
- Recall@50: % de vrais investisseurs dans top-50
- Optimal: > 0.60 pour @10, > 0.80 pour @50

## 6. Recommandations

### Pour le Degree Bias
Si ton dataset a un **fort déséquilibre de degrés** (quelques nœuds très connectés):
- ✅ Utilise **HAR Loss** ou **Hybrid Loss**
- 📊 Compare avec BCE pour quantifier l'amélioration

### Pour le Class Imbalance
Si tu as **beaucoup plus de négatifs que de positifs**:
- ✅ Utilise **Focal Loss** ou **Hybrid Loss**
- ⚙️ Tune `focal_gamma` (2.0-5.0) pour ajuster l'agressivité

### Pour les Deux Problèmes
Si tu as **degree bias ET class imbalance**:
- ✅ Utilise **Hybrid Loss**
- ⚙️ Ajuste `lambda_focal` dans hybrid_loss.py (0.5 = équilibré)

## 7. Troubleshooting

### Problème: "Aucun fichier de résultats trouvé"
**Solution**: Entraîne d'abord au moins un modèle (voir section 1)

### Problème: Training loss n'apparaît pas sur le plot
**Solution**: Vérifie que l'entraînement s'est terminé complètement (pas d'interruption)

### Problème: Métriques de test manquantes
**Solution**: Assure-toi que l'évaluation de test s'est bien exécutée après l'entraînement

### Problème: Courbes trop bruitées
**Solution**: Augmente `--n_epoch` pour avoir plus de données

## 8. Exemple Complet

```bash
# 1. Entraîner les 4 configurations
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory --n_epoch 50
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory --n_epoch 50 --use_focal_loss
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory --n_epoch 50 --use_har_loss
python train_self_supervised.py --data crunchbase --prefix tgn-attn --use_memory --n_epoch 50 --use_focal_loss --use_har_loss

# 2. Comparer les résultats
python plot_loss_comparison.py

# 3. Visualiser les plots
# Les fichiers sont dans loss_comparison_plots/
```

## 9. Références

- **BCE Loss**: Binary Cross-Entropy (baseline PyTorch)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **HAR Loss**: Adaptée de "Graph Contrastive Learning with Adaptive Augmentation" (2021)
- **Hybrid Loss**: Combinaison custom Focal + HAR
