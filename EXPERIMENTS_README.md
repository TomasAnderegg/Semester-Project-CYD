# Guide d'Exécution des Expériences

Ce guide explique comment lancer toutes les configurations d'entraînement pour comparer les loss functions.

## 📋 Configurations Testées

Les scripts lancent automatiquement **4 configurations** :

1. **BCE (Baseline)** : Binary Cross-Entropy standard
2. **Focal Loss** : Focus sur les exemples difficiles (α=0.25, γ=2.0)
3. **DCL Loss** : Correction du biais de degré (α=0.5, τ=0.07)
4. **Hybrid** : Focal + DCL combinés

Chaque configuration est exécutée avec **6 runs** de **50 epochs** chacun.

## 🚀 Lancement Rapide

### Option 1 : Script Python (Recommandé)

```bash
python run_all_experiments.py
```

**Avantages** :
- ✅ Multiplateforme (Windows, Linux, Mac)
- ✅ Gestion des erreurs
- ✅ Résumé détaillé en fin d'exécution
- ✅ Estimation du temps

### Option 2 : Script Batch (Windows)

```bash
run_all_experiments.bat
```

### Option 3 : Script Bash (Linux/Mac)

```bash
bash run_all_experiments.sh
```

### Option 4 : Lancer Manuellement

Si tu veux lancer une seule configuration :

```bash
# BCE (Baseline)
python train_self_supervised.py --use_memory --prefix tgn-bce \
    --n_epoch 50 --patience 10 --lr 1e-4 \
    --node_dim 200 --time_dim 200 --memory_dim 200 --message_dim 200 \
    --n_runs 6 --use_wandb

# Focal Loss
python train_self_supervised.py --use_memory --prefix tgn-focal \
    --n_epoch 50 --patience 10 --lr 1e-4 \
    --node_dim 200 --time_dim 200 --memory_dim 200 --message_dim 200 \
    --n_runs 6 --use_wandb \
    --use_focal_loss --focal_alpha 0.25 --focal_gamma 2.0

# DCL Loss
python train_self_supervised.py --use_memory --prefix tgn-dcl \
    --n_epoch 50 --patience 10 --lr 1e-4 \
    --node_dim 200 --time_dim 200 --memory_dim 200 --message_dim 200 \
    --n_runs 6 --use_wandb \
    --use_dcl_loss --dcl_alpha 0.5 --dcl_temperature 0.07

# Hybrid (Focal + DCL)
python train_self_supervised.py --use_memory --prefix tgn-hybrid \
    --n_epoch 50 --patience 10 --lr 1e-4 \
    --node_dim 200 --time_dim 200 --memory_dim 200 --message_dim 200 \
    --n_runs 6 --use_wandb \
    --use_focal_loss --use_dcl_loss \
    --focal_alpha 0.25 --focal_gamma 2.0 \
    --dcl_alpha 0.5 --dcl_temperature 0.07
```

## ⏱️ Durée Estimée

- **Par run** : ~10-30 minutes (dépend du GPU et du dataset)
- **Par configuration** : ~1-3 heures (6 runs)
- **Total (4 configurations)** : **~4-12 heures**

## 📊 Visualisation des Résultats

Une fois toutes les expériences terminées, visualise les résultats :

```bash
python plot_loss_comparison.py
```

Cela génère :
- `loss_comparison_plots/training_loss_comparison.png` : Courbes de training loss
- `loss_comparison_plots/test_metrics_comparison.png` : Comparaison des métriques de test
- `loss_comparison_plots/validation_metrics_over_epochs.png` : Evolution des métriques de validation
- `loss_comparison_plots/summary_table.csv` : Tableau récapitulatif

## 📁 Fichiers Générés

Après l'exécution, tu trouveras :

```
results/
├── tgn-bce_0.json          # Résultats BCE run 0
├── tgn-bce_1.json          # Résultats BCE run 1
├── ...
├── tgn-focal_0.json        # Résultats Focal run 0
├── ...
├── tgn-dcl_0.json          # Résultats DCL run 0
├── ...
└── tgn-hybrid_0.json       # Résultats Hybrid run 0

saved_models/
├── tgn-bce-crunchbase.pth
├── tgn-focal-crunchbase.pth
├── tgn-dcl-crunchbase.pth
└── tgn-hybrid-crunchbase.pth
```

## 🔧 Paramètres des Configurations

| Configuration | Focal Loss | DCL Loss | Paramètres |
|---------------|------------|----------|------------|
| **BCE** | ❌ | ❌ | - |
| **Focal** | ✅ | ❌ | α=0.25, γ=2.0 |
| **DCL** | ❌ | ✅ | α=0.5, τ=0.07 |
| **Hybrid** | ✅ | ✅ | Focal: α=0.25, γ=2.0<br>DCL: α=0.5, τ=0.07 |

### Paramètres Communs

- **Memory** : Activée
- **Epochs** : 50
- **Early Stopping** : Patience = 10
- **Learning Rate** : 0.0001 (1e-4)
- **Dimensions** : node=200, time=200, memory=200, message=200
- **Runs** : 6 (pour moyenner les résultats)
- **WandB** : Activé pour logging

## 🐛 Dépannage

### Erreur "CUDA out of memory"

Réduis la batch size :
```bash
python train_self_supervised.py --bs 100 ...
```

### Les runs prennent trop de temps

Réduis le nombre de runs ou d'epochs :
```bash
# 3 runs au lieu de 6
--n_runs 3

# 30 epochs au lieu de 50
--n_epoch 30
```

### WandB ne fonctionne pas

Désactive WandB :
```bash
# Retire simplement --use_wandb de la commande
```

## 📈 Analyse des Résultats

Pour ton rapport, concentre-toi sur :

1. **Training Loss** : Quelle loss converge le plus vite ?
2. **Test Metrics** : Quelle loss donne les meilleures performances finales ?
   - MRR (Mean Reciprocal Rank)
   - Recall@10, Recall@50
   - AP (Average Precision)
3. **Validation Curves** : Quelle loss est la plus stable ?
4. **New Nodes** : Quelle loss généralise le mieux aux nouveaux nœuds ?

## 💡 Conseils

- ✅ Lance les expériences **overnight** (elles peuvent prendre plusieurs heures)
- ✅ Vérifie que ton **GPU est disponible** avant de lancer
- ✅ **Surveille WandB** pour voir la progression en temps réel
- ✅ Garde une **copie de sauvegarde** de `results/` avant de relancer
- ✅ Compare les **moyennes sur 6 runs** plutôt qu'un seul run

## 🎯 Objectif

À la fin, tu auras des **résultats statistiquement robustes** (6 runs par config) pour comparer :
- BCE (baseline)
- Focal Loss (gestion du déséquilibre de classes)
- DCL Loss (correction du biais de degré)
- Hybrid (combinaison des deux)

Cela te permettra de **quantifier l'apport** de chaque technique pour ton rapport ! 📊
