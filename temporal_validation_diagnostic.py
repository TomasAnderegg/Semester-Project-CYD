#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic détaillé de la validation temporelle
"""

import argparse
import logging
import torch
import numpy as np
from pathlib import Path

from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

def parse_args():
    parser = argparse.ArgumentParser('Temporal Validation Diagnostic')
    parser.add_argument('-d', '--data', type=str, default='crunchbase', help='Dataset name')
    parser.add_argument('--bs', type=int, default=200, help='Batch size')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of layers')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--node_dim', type=int, default=100, help='Node dimensions')
    parser.add_argument('--time_dim', type=int, default=100, help='Time dimensions')
    parser.add_argument('--use_memory', action='store_true', help='Use memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention")
    parser.add_argument('--message_function', type=str, default="identity")
    parser.add_argument('--memory_updater', type=str, default="gru")
    parser.add_argument('--aggregator', type=str, default="last")
    parser.add_argument('--memory_update_at_end', action='store_true')
    parser.add_argument('--message_dim', type=int, default=200)
    parser.add_argument('--memory_dim', type=int, default=None)
    parser.add_argument('--different_new_nodes', action='store_true')
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--randomize_features', action='store_true')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true')
    parser.add_argument('--use_source_embedding_in_message', action='store_true')
    parser.add_argument('--dyrep', action='store_true')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--temporal_split', type=float, default=0.7, help='Split ratio')
    parser.add_argument('--auto_detect_params', action='store_true')
    return parser.parse_args()

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger()

def detect_model_params(checkpoint_path, logger):
    """Détecte les paramètres du modèle"""
    logger.info("🔍 Detecting model parameters...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    params = {}

    if 'memory.memory' in checkpoint:
        params['memory_dim'] = checkpoint['memory.memory'].shape[1]
        logger.info(f"   memory_dim: {params['memory_dim']}")

    if 'message_function.message_linear_1.weight' in checkpoint:
        params['message_dim'] = checkpoint['message_function.message_linear_1.weight'].shape[0]
        logger.info(f"   message_dim: {params['message_dim']}")

    return params

def main():
    args = parse_args()
    logger = setup_logger()

    logger.info("="*70)
    logger.info("TEMPORAL VALIDATION - DIAGNOSTIC DÉTAILLÉ")
    logger.info("="*70)

    # Auto-detect parameters
    if args.auto_detect_params or args.memory_dim is None:
        detected = detect_model_params(args.model_path, logger)
        if args.memory_dim is None and 'memory_dim' in detected:
            args.memory_dim = detected['memory_dim']

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("\nLoading data...")
    node_features, edge_features, full_data, train_data, val_data, test_data, \
    new_node_val_data, new_node_test_data = get_data(
        args.data,
        different_new_nodes_between_val_and_test=args.different_new_nodes,
        randomize_features=args.randomize_features
    )

    logger.info(f"✅ Data loaded: {len(test_data.sources)} test interactions")

    # Neighbor finder
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    # Initialize model
    logger.info("\nInitializing model...")
    tgn = TGN(
        neighbor_finder=full_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dropout=args.drop_out,
        use_memory=args.use_memory,
        message_dimension=args.message_dim,
        memory_dimension=args.memory_dim,
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=args.n_degree,
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message,
        dyrep=args.dyrep
    )

    tgn = tgn.to(device)

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    tgn.load_state_dict(checkpoint, strict=False)
    tgn.eval()
    logger.info("✅ Model loaded")

    # Run temporal validation with diagnostics
    run_temporal_validation_with_diagnostics(tgn, test_data, full_data, full_ngh_finder, args, logger)

def run_temporal_validation_with_diagnostics(tgn, test_data, full_data, full_ngh_finder, args, logger):
    """Validation temporelle avec diagnostic détaillé"""

    logger.info("\n" + "="*70)
    logger.info("TEMPORAL VALIDATION WITH DIAGNOSTICS")
    logger.info("="*70)

    # Préparer les données
    test_sources = np.array(test_data.sources)
    test_destinations = np.array(test_data.destinations)
    test_timestamps = np.array(test_data.timestamps)
    test_edge_idxs = np.array(test_data.edge_idxs)

    # Trier
    sorted_idx = np.argsort(test_timestamps)
    test_sources = test_sources[sorted_idx]
    test_destinations = test_destinations[sorted_idx]
    test_timestamps = test_timestamps[sorted_idx]
    test_edge_idxs = test_edge_idxs[sorted_idx]

    # Split
    split_idx = int(len(test_timestamps) * args.temporal_split)
    split_timestamp = test_timestamps[split_idx]

    logger.info(f"Test set: {len(test_timestamps)} interactions")
    logger.info(f"Split: {args.temporal_split:.0%} history, {1-args.temporal_split:.0%} ground truth")
    logger.info(f"Split timestamp: {split_timestamp:.2f}")
    logger.info(f"History: {split_idx} interactions")
    logger.info(f"Future: {len(test_timestamps) - split_idx} interactions")

    # History & ground truth
    history_sources = test_sources[:split_idx]
    history_destinations = test_destinations[:split_idx]
    history_timestamps = test_timestamps[:split_idx]
    history_edge_idxs = test_edge_idxs[:split_idx]

    future_sources = test_sources[split_idx:]
    future_destinations = test_destinations[split_idx:]

    true_future_links = set((int(s), int(d)) for s, d in zip(future_sources, future_destinations))
    logger.info(f"Unique true future links: {len(true_future_links)}")

    # Update memory with history
    tgn.set_neighbor_finder(full_ngh_finder)
    if args.use_memory:
        tgn.memory.__init_memory__()
        logger.info("\nUpdating memory with history...")
        for i in range(0, len(history_sources), args.bs):
            s = history_sources[i:i+args.bs]
            d = history_destinations[i:i+args.bs]
            ts = history_timestamps[i:i+args.bs]
            e = history_edge_idxs[i:i+args.bs]

            with torch.no_grad():
                _ = tgn.compute_temporal_embeddings(s, d, d, ts, e, args.n_degree)

    # Generate predictions
    logger.info("\n" + "="*70)
    logger.info("GENERATING PREDICTIONS")
    logger.info("="*70)

    all_sources = sorted(set(full_data.sources))
    all_destinations = sorted(set(full_data.destinations))

    logger.info(f"Companies: {len(all_sources)}")
    logger.info(f"Investors: {len(all_destinations)}")
    logger.info(f"Total pairs: {len(all_sources) * len(all_destinations):,}")

    predictions_list = []
    neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=99)

    for company_id in all_sources:
        src_batch = np.full(len(all_destinations), company_id, dtype=np.int32)
        dst_batch = np.array(all_destinations, dtype=np.int32)
        times_batch = np.full(len(dst_batch), split_timestamp, dtype=np.float32)
        idx_batch = np.zeros(len(dst_batch), dtype=np.int32)

        neg_tuple = neg_sampler.sample(len(src_batch))
        neg_batch = np.array(neg_tuple[1])
        if len(neg_batch.shape) > 1:
            neg_batch = neg_batch[:, 0]

        with torch.no_grad():
            try:
                pos_prob, _ = tgn.compute_edge_probabilities(
                    src_batch, dst_batch, neg_batch,
                    times_batch, idx_batch, args.n_degree
                )

                prob_scores = pos_prob.cpu().numpy()

                for d, prob in zip(dst_batch, prob_scores):
                    predictions_list.append((company_id, int(d), float(prob)))

            except Exception as e:
                logger.warning(f"Error for company {company_id}: {e}")
                continue

    logger.info(f"✅ Generated {len(predictions_list):,} predictions")

    # Sort predictions
    predictions_list.sort(key=lambda x: x[2], reverse=True)

    # ================================================================
    # DIAGNOSTIC DÉTAILLÉ
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC - DISTRIBUTION DES PROBABILITÉS")
    logger.info("="*70)

    all_probs = np.array([p[2] for p in predictions_list])

    logger.info(f"\n📊 Statistiques globales ({len(all_probs):,} prédictions):")
    logger.info(f"   Min:        {np.min(all_probs):.6f}")
    logger.info(f"   Q1 (25%):   {np.percentile(all_probs, 25):.6f}")
    logger.info(f"   Médiane:    {np.median(all_probs):.6f}")
    logger.info(f"   Q3 (75%):   {np.percentile(all_probs, 75):.6f}")
    logger.info(f"   Max:        {np.max(all_probs):.6f}")
    logger.info(f"   Moyenne:    {np.mean(all_probs):.6f}")
    logger.info(f"   Std:        {np.std(all_probs):.6f}")

    # Histogramme
    logger.info(f"\n📊 Distribution des probabilités:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(all_probs, bins=bins)
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(all_probs) * 100
        logger.info(f"   [{bins[i]:.1f} - {bins[i+1]:.1f}): {count:8,} ({pct:5.2f}%)")

    # ================================================================
    # ANALYSE DES VRAIS LIENS
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC - VRAIS LIENS FUTURS")
    logger.info("="*70)

    # Trouver les vrais liens dans les prédictions
    pred_dict = {(s, d): (prob, rank) for rank, (s, d, prob) in enumerate(predictions_list)}

    true_link_probs = []
    true_link_ranks = []

    for s, d in true_future_links:
        if (s, d) in pred_dict:
            prob, rank = pred_dict[(s, d)]
            true_link_probs.append(prob)
            true_link_ranks.append(rank)

    if len(true_link_probs) > 0:
        logger.info(f"\n📊 Probabilités des vrais liens ({len(true_link_probs)} liens):")
        logger.info(f"   Min:     {np.min(true_link_probs):.6f}")
        logger.info(f"   Q1:      {np.percentile(true_link_probs, 25):.6f}")
        logger.info(f"   Médiane: {np.median(true_link_probs):.6f}")
        logger.info(f"   Q3:      {np.percentile(true_link_probs, 75):.6f}")
        logger.info(f"   Max:     {np.max(true_link_probs):.6f}")
        logger.info(f"   Moyenne: {np.mean(true_link_probs):.6f}")

        logger.info(f"\n📊 Rangs des vrais liens (sur {len(predictions_list):,}):")
        logger.info(f"   Meilleur rang:  {np.min(true_link_ranks):,} (top {np.min(true_link_ranks)/len(predictions_list)*100:.2f}%)")
        logger.info(f"   Rang médian:    {int(np.median(true_link_ranks)):,} (top {np.median(true_link_ranks)/len(predictions_list)*100:.2f}%)")
        logger.info(f"   Rang moyen:     {int(np.mean(true_link_ranks)):,} (top {np.mean(true_link_ranks)/len(predictions_list)*100:.2f}%)")
        logger.info(f"   Pire rang:      {np.max(true_link_ranks):,} (top {np.max(true_link_ranks)/len(predictions_list)*100:.2f}%)")

        # Distribution des rangs
        logger.info(f"\n📊 Distribution des rangs des vrais liens:")
        rank_bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, len(predictions_list)]
        for i in range(len(rank_bins)-1):
            count = sum(1 for r in true_link_ranks if rank_bins[i] <= r < rank_bins[i+1])
            pct = count / len(true_link_ranks) * 100 if len(true_link_ranks) > 0 else 0
            logger.info(f"   Top {rank_bins[i]:6,} - {rank_bins[i+1]:6,}: {count:3,} liens ({pct:5.1f}%)")

        # Top exemples
        logger.info(f"\n📋 Meilleurs vrais liens (top 10 par probabilité):")
        true_links_with_prob = [(s, d, prob, rank) for (s, d), (prob, rank) in pred_dict.items() if (s, d) in true_future_links]
        true_links_with_prob.sort(key=lambda x: x[2], reverse=True)

        for i, (s, d, prob, rank) in enumerate(true_links_with_prob[:10], 1):
            logger.info(f"   #{i:2d} Company {s:3d} → Investor {d:3d}  |  Prob: {prob:.6f}  |  Rank: {rank:6,}")

    # ================================================================
    # PRECISION@K
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("PRECISION@K")
    logger.info("="*70)

    k_values = [10, 20, 50, 100, 200, 500, 1000, 5000, 10000]

    logger.info(f"\nTotal vrais liens: {len(true_future_links)}")

    for k in k_values:
        if k > len(predictions_list):
            continue

        top_k = predictions_list[:k]
        true_in_top_k = sum(1 for s, d, _ in top_k if (s, d) in true_future_links)
        precision = true_in_top_k / k
        recall = true_in_top_k / len(true_future_links) if len(true_future_links) > 0 else 0

        logger.info(f"\nPrecision@{k:6,}:  {precision:.4f}  ({true_in_top_k}/{k})")
        logger.info(f"  Recall@{k:6,}:  {recall:.4f}  ({true_in_top_k}/{len(true_future_links)})")

    # ================================================================
    # BASELINE ALÉATOIRE
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("COMPARAISON AVEC BASELINE ALÉATOIRE")
    logger.info("="*70)

    random_precision = len(true_future_links) / len(predictions_list)
    logger.info(f"\nBaseline aléatoire (expected precision): {random_precision:.6f} ({random_precision*100:.4f}%)")

    logger.info(f"\nComparaison Precision@1000:")
    if len(predictions_list) >= 1000:
        top_1000 = predictions_list[:1000]
        true_in_1000 = sum(1 for s, d, _ in top_1000 if (s, d) in true_future_links)
        model_precision = true_in_1000 / 1000
        random_expected = 1000 * random_precision

        logger.info(f"   Modèle:     {model_precision:.6f} ({true_in_1000}/1000 liens)")
        logger.info(f"   Aléatoire:  {random_precision:.6f} ({random_expected:.1f}/1000 liens attendus)")

        if model_precision > random_precision:
            improvement = model_precision / random_precision
            logger.info(f"   ✅ Amélioration: {improvement:.2f}x meilleur que aléatoire")
        else:
            logger.info(f"   ❌ Le modèle fait MOINS bien que le hasard!")

    logger.info("\n✅ Diagnostic terminé!")

if __name__ == "__main__":
    main()
