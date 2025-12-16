#!/bin/bash
# Script pour comparer toutes les approches

echo "================================================"
echo "TRAINING 4 MODELS FOR COMPARISON"
echo "================================================"

# 1. Baseline (Random sampling + BCE)
echo ""
echo "1/4 Training Baseline (Random + BCE)..."
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --prefix baseline \
  --n_epoch 50

# 2. Focal Loss seul
echo ""
echo "2/4 Training Focal Loss..."
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --prefix focal \
  --n_epoch 50

# 3. Hard Negatives seul
echo ""
echo "3/4 Training Hard Negatives..."
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_hard_negatives \
  --hard_neg_ratio 0.5 \
  --hard_neg_temperature 0.1 \
  --prefix hardneg \
  --n_epoch 50

# 4. Focal + Hard Negatives
echo ""
echo "4/4 Training Focal + Hard Negatives..."
python train_self_supervised.py \
  --data crunchbase \
  --use_memory \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --use_hard_negatives \
  --hard_neg_ratio 0.5 \
  --hard_neg_temperature 0.1 \
  --prefix focal-hardneg \
  --n_epoch 50

echo ""
echo "================================================"
echo "EVALUATING ALL 4 MODELS"
echo "================================================"

# Evaluate all models
for model in baseline focal hardneg focal-hardneg
do
  echo ""
  echo "Evaluating $model..."
  python temporal_validation_diagnostic.py \
    --data crunchbase \
    --model_path saved_models/${model}-crunchbase.pth \
    --use_memory \
    --auto_detect_params \
    > results/${model}_temporal_validation.log 2>&1
done

echo ""
echo "================================================"
echo "DONE! Check results/ directory for logs"
echo "================================================"
