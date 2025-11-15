#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGN evaluation + prediction (single-file, functions + main)
Comportement identique à la version précédente "premium":
 - Reset mémoire avant chaque phase
 - Tri chronologique centralisé
 - Filtre les arêtes pour la prédiction à partir du max timestamp du train
 - Exports mapping / graph / top-k predictions
"""

import argparse
import logging
import sys
from pathlib import Path
import pickle
import csv

import torch
import numpy as np
import networkx as nx

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

# ----------------------------
# Setup / Defaults
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser('TGN Evaluation and Prediction (modular single-file)')
    parser.add_argument('-d', '--data', type=str, default='crunchbase', help='Dataset name')
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--node_dim', type=int, default=100, help='Node embedding dimensions')
    parser.add_argument('--time_dim', type=int, default=100, help='Time embedding dimensions')
    parser.add_argument('--use_memory', action='store_true', help='Use memory module')
    parser.add_argument('--embedding_module', type=str, default="graph_attention")
    parser.add_argument('--message_function', type=str, default="identity")
    parser.add_argument('--memory_updater', type=str, default="gru")
    parser.add_argument('--aggregator', type=str, default="last")
    parser.add_argument('--memory_update_at_end', action='store_true')
    parser.add_argument('--message_dim', type=int, default=100)
    parser.add_argument('--memory_dim', type=int, default=172)
    parser.add_argument('--different_new_nodes', action='store_true')
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--randomize_features', action='store_true')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true')
    parser.add_argument('--use_source_embedding_in_message', action='store_true')
    parser.add_argument('--dyrep', action='store_true')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--circular', action='store_true', help='Use circular layout')
    parser.add_argument('--mapping_dir', type=str, default='data/mappings')
    parser.add_argument('--top_k_export', type=int, default=100)
    return parser.parse_args()

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger()

# ----------------------------
# Utilities
# ----------------------------
def initialize_model(args, device, node_features, edge_features, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst):
    tgn = TGN(neighbor_finder=None, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=args.n_layer, n_heads=args.n_head, dropout=args.drop_out,
              use_memory=args.use_memory, message_dimension=args.message_dim,
              memory_dimension=args.memory_dim,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=args.n_degree,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep)
    return tgn

def load_mappings(mapping_dir, logger):
    mapping_dir = Path(mapping_dir)
    candidate_company = [
        "crunchbase_tgn_company_id_map.pickle",
        "investment_bipartite_company_id_map.pickle",
    ]
    candidate_investor = [
        "crunchbase_tgn_investor_id_map.pickle",
        "investment_bipartite_investor_id_map.pickle",
    ]

    company_map_path = None
    investor_map_path = None

    for cand in candidate_company:
        p = mapping_dir / cand
        if p.exists():
            company_map_path = p
            break

    for cand in candidate_investor:
        p = mapping_dir / cand
        if p.exists():
            investor_map_path = p
            break

    id_to_company = {}
    id_to_investor = {}
    if company_map_path and investor_map_path:
        try:
            with open(company_map_path, "rb") as f:
                company_map = pickle.load(f)
            with open(investor_map_path, "rb") as f:
                investor_map = pickle.load(f)
            id_to_company = {int(v): k for k, v in company_map.items()}
            id_to_investor = {int(v): k for k, v in investor_map.items()}
            logger.info(f"Mappings loaded: {len(id_to_company)} companies, {len(id_to_investor)} investors")
        except Exception as e:
            logger.warning("Could not load mapping pickles (%s). Falling back to numeric ids. Error: %s", mapping_dir, e)
    else:
        logger.warning("Mappings not found in %s. Using numeric IDs.", mapping_dir)

    return id_to_company, id_to_investor

def process_data_chronologically(data, tgn_model, ngh_finder, batch_size, n_neighbors, logger):
    """
    Parcours les interactions de 'data' dans l'ordre chronologique strict et appelle
    compute_temporal_embeddings pour mettre à jour la mémoire.
    """
    tgn_model.set_neighbor_finder(ngh_finder)

    sources = np.asarray(data.sources)
    destinations = np.asarray(data.destinations)
    timestamps = np.asarray(data.timestamps)
    edge_idxs = np.asarray(data.edge_idxs)

    if len(timestamps) == 0:
        logger.debug("process_data_chronologically: no interactions to process")
        return

    sorted_idx = np.argsort(timestamps)
    sources = sources[sorted_idx]
    destinations = destinations[sorted_idx]
    timestamps = timestamps[sorted_idx]
    edge_idxs = edge_idxs[sorted_idx]

    for i in range(0, len(sources), batch_size):
        s = sources[i:i+batch_size]
        d = destinations[i:i+batch_size]
        ts = timestamps[i:i+batch_size]
        e = edge_idxs[i:i+batch_size]
        with torch.no_grad():
            try:
                _ = tgn_model.compute_temporal_embeddings(s, d, d, ts, e, n_neighbors)
            except AssertionError as ex:
                logger.error("AssertionError during compute_temporal_embeddings in process_data_chronologically: %s", ex)
                raise

def build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger):
    """
    Reset memory (if enabled) and build it from train_data chronologically.
    """
    if args.use_memory:
        tgn.memory.__init_memory__()
    process_data_chronologically(train_data, tgn, train_ngh_finder, args.bs, args.n_degree, logger)

def filter_edges_after_training(full_data, train_data, logger):
    """
    Retourne filtered arrays (sources, destinations, edge_times, edge_idxs)
    contenant uniquement les arêtes avec timestamp >= max(train.timestamps).
    """
    sources = np.array(full_data.sources)
    destinations = np.array(full_data.destinations)
    edge_times = np.array(full_data.timestamps)
    edge_idxs = np.array(full_data.edge_idxs)

    sorted_indices = np.argsort(edge_times)
    sources = sources[sorted_indices]
    destinations = destinations[sorted_indices]
    edge_times = edge_times[sorted_indices]
    edge_idxs = edge_idxs[sorted_indices]

    if len(train_data.timestamps) > 0:
        train_max_ts = np.max(np.asarray(train_data.timestamps))
    else:
        train_max_ts = -np.inf

    mask_after_train = edge_times >= train_max_ts
    count_total = len(edge_times)
    count_after = mask_after_train.sum()
    logger.info("Edges total in full_data: %d, edges at/after train max timestamp: %d", count_total, int(count_after))

    return sources[mask_after_train], destinations[mask_after_train], edge_times[mask_after_train], edge_idxs[mask_after_train]

def generate_predictions_and_graph(tgn, id_to_company, id_to_investor, full_data, train_data, args, logger):
    """
    Génère le graphe prédit et une liste de prédictions (uid, vid, prob).
    Filtre les edges pour timestamps >= max(train.timestamps).
    """
    # Filter edges after train
    sources, destinations, edge_times, edge_idxs = filter_edges_after_training(full_data, train_data, logger)

    neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)
    pred_graph = nx.Graph()
    predictions = []

    logger.info("Computing edge probabilities for filtered dataset (post-train timestamps)...")
    for start in range(0, len(sources), args.bs):
        end = min(start + args.bs, len(sources))
        src_batch = sources[start:end]
        dst_batch = destinations[start:end]
        times_batch = edge_times[start:end]
        idx_batch = edge_idxs[start:end]

        neg_tuple = neg_sampler.sample(len(src_batch))
        neg_batch = np.array(neg_tuple[1])
        if len(neg_batch.shape) > 1:
            neg_batch = neg_batch[:, 0]

        with torch.no_grad():
            pos_prob, _ = tgn.compute_edge_probabilities(src_batch, dst_batch, neg_batch, times_batch, idx_batch)

        for s, d, p in zip(src_batch, dst_batch, pos_prob.cpu().numpy()):
            inv_name = id_to_investor.get(s, f"investor_{s}")
            comp_name = id_to_company.get(d, f"company_{d}")
            pred_graph.add_node(inv_name, bipartite=0)
            pred_graph.add_node(comp_name, bipartite=1)
            pred_graph.add_edge(inv_name, comp_name, weight=float(p))
            predictions.append((s, d, float(p)))

        if (start // args.bs) % 10 == 0:
            logger.info("Processed %d/%d edges...", end, len(sources))

    logger.info("Graph created: %d nodes, %d edges", pred_graph.number_of_nodes(), pred_graph.number_of_edges())
    return pred_graph, predictions

def save_graph_and_top(pred_graph, predictions, args, logger):
    # Save graph
    graph_path = Path(f'predicted_graph_{args.data}.pkl')
    with open(graph_path, 'wb') as f:
        pickle.dump(pred_graph, f)
    logger.info("Graph saved to %s", graph_path)

    # Export top-k predictions
    if args.top_k_export > 0 and len(predictions) > 0:
        sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)[:args.top_k_export]
        csv_path = Path(f"top_predictions_{args.data}.csv")
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["investor_id", "company_id", "investor_name", "company_name", "probability"])
            for uid, vid, prob in sorted_preds:
                writer.writerow([uid, vid, f"investor_{uid}", f"company_{vid}", prob])
        logger.info("Top %d predictions saved to %s", args.top_k_export, csv_path)
    else:
        logger.info("No predictions to export or top_k_export <= 0")

# ----------------------------
# main()
# ----------------------------
def main():
    args = parse_args()
    logger = setup_logger()

    logger.info("Evaluating TGN model on dataset: %s", args.data)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load data
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
        args.data,
        different_new_nodes_between_val_and_test=args.different_new_nodes,
        randomize_features=args.randomize_features
    )

    logger.info("Dataset loaded: %d interactions (full)", len(full_data.sources))

    # Neighbor finders
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Negative samplers
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

    # Time stats
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps
    )

    # Initialize model
    tgn = initialize_model(args, device, node_features, edge_features,
                           mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst)

    # Load weights
    if Path(args.model_path).exists():
        logger.info("Loading model from %s", args.model_path)
        tgn.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        logger.error("Model file not found: %s", args.model_path)
        sys.exit(1)

    tgn.to(device)
    tgn.eval()

    # Build memory and evaluation phases
    logger.info("Starting evaluation pipeline (memory reset before each phase)")

    # Phase 1: Validation (seen nodes)
    logger.info("Phase 1: Validation (seen nodes)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)
    val_ap, val_auc = eval_edge_prediction(tgn, val_rand_sampler, val_data, args.n_degree)
    logger.info("Validation (seen nodes) - AUC: %.4f, AP: %.4f", val_auc, val_ap)

    # Phase 2: Validation (new nodes)
    logger.info("Phase 2: Validation (new nodes)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(train_ngh_finder)
    nn_val_ap, nn_val_auc = eval_edge_prediction(tgn, nn_val_rand_sampler, new_node_val_data, args.n_degree)
    logger.info("Validation (new nodes) - AUC: %.4f, AP: %.4f", nn_val_auc, nn_val_ap)

    # Phase 3: Test (seen nodes)
    logger.info("Phase 3: Test (seen nodes)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)
    test_ap, test_auc = eval_edge_prediction(tgn, test_rand_sampler, test_data, args.n_degree)
    logger.info("Test (seen nodes) - AUC: %.4f, AP: %.4f", test_auc, test_ap)

    # Phase 4: Test (new nodes)
    logger.info("Phase 4: Test (new nodes)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(train_ngh_finder)
    nn_test_ap, nn_test_auc = eval_edge_prediction(tgn, nn_test_rand_sampler, new_node_test_data, args.n_degree)
    logger.info("Test (new nodes) - AUC: %.4f, AP: %.4f", nn_test_auc, nn_test_ap)

    # Load mappings for naming
    id_to_company, id_to_investor = load_mappings(args.mapping_dir, logger)

    # Prepare model for prediction graph
    logger.info("Generating predicted graph (using memory built from train)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)

    pred_graph, predictions = generate_predictions_and_graph(tgn, id_to_company, id_to_investor, full_data, train_data, args, logger)

    save_graph_and_top(pred_graph, predictions, args, logger)

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
