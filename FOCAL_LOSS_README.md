# Guide d'utilisation de Focal Loss

## 📚 Qu'est-ce que Focal Loss?

Focal Loss est une fonction de perte spécialement conçue pour gérer les **déséquilibres de classes extrêmes** (Lin et al., 2017).

Dans votre cas: **52 vrais liens sur 170,742 paires (0.03%)**

### Formule

```
FL(p) = -α(1-p)^γ * log(p)    pour classe positive
FL(p) = -α * p^γ * log(1-p)   pour classe négative
```

**Paramètres:**
- **gamma (γ)**: Facteur de focalisation (défaut: 2.0)
  - γ=0 → équivalent à BCE classique
  - γ=2 → réduit fortement l'importance des exemples faciles
  - γ=5 → très agressif

- **alpha (α)**: Poids pour la classe positive (défaut: 0.25)
  - α=0.25 → classe positive a un poids de 0.25
  - α=0.5 → poids égal entre positifs et négatifs

## 🚀 Comment utiliser

### Option 1: Entraînement avec Focal Loss (RECOMMANDÉ)

```bash
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --prefix tgn-focal \
  --n_epoch 50
```

### Option 2: Entraînement avec BCE classique (ANCIEN)

```bash
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --prefix tgn-bce \
  --n_epoch 50
```

## ⚙️ Hyperparamètres recommandés

### Configuration par défaut (bonne pour commencer)
```bash
--use_focal_loss --focal_alpha 0.25 --focal_gamma 2.0
```

### Pour données très déséquilibrées (votre cas)
```bash
--use_focal_loss --focal_alpha 0.1 --focal_gamma 2.0
```
↳ Donne encore plus d'importance aux rares exemples positifs

### Pour focalisation plus agressive
```bash
--use_focal_loss --focal_alpha 0.25 --focal_gamma 5.0
```
↳ Ignore encore plus les exemples faciles

## 📊 Résultats attendus

### Avec BCE (actuel):
- Probabilités: médiane ~0.04, max ~0.70
- Recall@1000: 7.7% des vrais liens
- Modèle très conservateur

### Avec Focal Loss (attendu):
- Probabilités mieux calibrées
- Recall@1000: **15-25%** des vrais liens (amélioration 2-3x)
- Meilleur ranking des vrais liens
- Top prédictions plus pertinentes pour TechRank

## 🧪 Test de la Focal Loss

Pour vérifier que Focal Loss fonctionne correctement:

```bash
python focal_loss.py
```

Ce script de test affiche:
- Comparaison BCE vs Focal Loss
- Impact du paramètre gamma
- Comportement sur exemples faciles vs difficiles

## 📁 Fichiers modifiés

1. **focal_loss.py** (NOUVEAU)
   - Implémentation de FocalLoss
   - Classe FocalLoss avec paramètres alpha et gamma
   - Tests unitaires

2. **train_self_supervised.py** (MODIFIÉ)
   - Ligne 20: Import de FocalLoss
   - Lignes 84-89: Nouveaux arguments --use_focal_loss, --focal_alpha, --focal_gamma
   - Lignes 211-238: Configuration conditionnelle BCE vs Focal Loss
   - **L'ancienne BCE est COMMENTÉE, pas supprimée**

## 🔄 Comment revenir en arrière?

Si Focal Loss ne donne pas de bons résultats:

1. **Ne PAS utiliser** le flag `--use_focal_loss` lors de l'entraînement
2. Le code utilisera automatiquement BCE (ligne 225)
3. Aucun changement de code nécessaire!

```bash
# Revenir à BCE (enlever --use_focal_loss)
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --prefix tgn-bce-fallback \
  --n_epoch 50
```

## 📈 Comparaison des modèles

Pour comparer BCE vs Focal Loss, entraînez deux modèles:

```bash
# Modèle 1: BCE classique
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --prefix tgn-bce \
  --n_epoch 50

# Modèle 2: Focal Loss
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --prefix tgn-focal \
  --n_epoch 50
```

Puis évaluez avec validation temporelle:

```bash
# Évaluer modèle BCE
python temporal_validation_diagnostic.py \
  --data crunchbase \
  --model_path saved_models/tgn-bce-crunchbase.pth \
  --use_memory \
  --auto_detect_params

# Évaluer modèle Focal Loss
python temporal_validation_diagnostic.py \
  --data crunchbase \
  --model_path saved_models/tgn-focal-crunchbase.pth \
  --use_memory \
  --auto_detect_params
```

Comparez les résultats:
- Precision@K
- Recall@K
- Rang moyen des vrais liens
- Distribution des probabilités

## 🎯 Métriques à surveiller

Avec Focal Loss, vous devriez voir:

1. **Probabilités des vrais liens plus élevées**
   - Médiane devrait passer de 0.25 à 0.40+

2. **Meilleur ranking**
   - Rang médian devrait descendre de 6,609 à <5,000

3. **Meilleur recall**
   - Recall@1000 devrait passer de 7.7% à 15-20%

4. **Amélioration vs baseline aléatoire**
   - Devrait passer de 13x à 20-30x meilleur que le hasard

## ⚠️ Points d'attention

1. **Temps d'entraînement**: Focal Loss est légèrement plus lent (~5-10%)
2. **Convergence**: Peut nécessiter plus d'epochs pour converger
3. **Hyperparamètres**: Commencez avec les valeurs par défaut, ajustez ensuite

## 📚 Référence

Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
"Focal loss for dense object detection."
In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
