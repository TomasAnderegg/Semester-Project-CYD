#!/bin/bash
# Test différentes configurations de Hard Negative Mining

# Configuration de base
BASE_CMD="python train_self_supervised.py \
  --use_memory \
  --n_epoch 50 \
  --patience 10 \
  --lr 1e-4 \
  --node_dim 200 \
  --time_dim 200 \
  --memory_dim 200 \
  --message_dim 200 \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0"

echo "Testing different Hard Negative Mining configurations..."
echo ""

# Test 1: Ratio plus bas (30% hard)
echo "1/3 Testing ratio=0.3, temperature=0.1..."
$BASE_CMD \
  --use_hard_negatives \
  --hard_neg_ratio 0.3 \
  --hard_neg_temperature 0.1 \
  --prefix focal-hard-r0.3-t0.1 \
  --n_runs 1

# Test 2: Temperature plus haute (moins agressif)
echo "2/3 Testing ratio=0.3, temperature=0.5..."
$BASE_CMD \
  --use_hard_negatives \
  --hard_neg_ratio 0.3 \
  --hard_neg_temperature 0.5 \
  --prefix focal-hard-r0.3-t0.5 \
  --n_runs 1

# Test 3: Très conservateur
echo "3/3 Testing ratio=0.2, temperature=1.0..."
$BASE_CMD \
  --use_hard_negatives \
  --hard_neg_ratio 0.2 \
  --hard_neg_temperature 1.0 \
  --prefix focal-hard-r0.2-t1.0 \
  --n_runs 1

echo ""
echo "Done! Evaluate with:"
echo "python temporal_validation_diagnostic.py --data crunchbase --model_path saved_models/focal-hard-r0.3-t0.1-crunchbase.pth --use_memory --auto_detect_params"
echo "python temporal_validation_diagnostic.py --data crunchbase --model_path saved_models/focal-hard-r0.3-t0.5-crunchbase.pth --use_memory --auto_detect_params"
echo "python temporal_validation_diagnostic.py --data crunchbase --model_path saved_models/focal-hard-r0.2-t1.0-crunchbase.pth --use_memory --auto_detect_params"
