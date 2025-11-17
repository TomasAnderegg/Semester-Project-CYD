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
    parser.add_argument('--run_techrank', action='store_true', help='Run TechRank after evaluation')
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
        # "crunchbase_tgn_company_id_map.pickle",
        # "investment_bipartite_company_id_map.pickle",
        # "investments_10000_investor_id_map.pickle"
        "forecast_company_id_map.pickle"
    ]
    candidate_investor = [
        # "crunchbase_tgn_investor_id_map.pickle",
        # "investment_bipartite_investor_id_map.pickle",
        "forecast_investor_id_map.pickle"
        # "investments_10000_company_id_map.pickle"
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

    mask_after_train = edge_times >= train_max_ts # on ne garde que les arretes après le temps max du train
    # mask_after_train = np.ones_like(edge_times, dtype=bool)  # garde toutes les arêtes

    count_total = len(edge_times)
    count_after = mask_after_train.sum()
    logger.info("Edges total in full_data: %d, edges at/after train max timestamp: %d", count_total, int(count_after))

    return sources[mask_after_train], destinations[mask_after_train], edge_times[mask_after_train], edge_idxs[mask_after_train]

def generate_predictions_and_graph(tgn, id_to_company, id_to_investor, full_data, train_data, args, logger):
    """
    Génère le graphe prédit et une liste de prédictions (uid, vid, prob).
    Correction robuste:
     - construit node_type/node_name à partir des mappings fournis
     - résout systématiquement qui est investor / company
     - ignore les arêtes ambigües au lieu de polluer le graphe
     - ajoute explicitement tous les nodes connus au graphe
    """
    # Build a reliable node_type / node_name table from mappings
    node_type = {}   # tgn_id -> "company" or "investor"
    node_name = {}   # tgn_id -> human name

    # id_to_company and id_to_investor are expected to map numeric_id -> name
    for nid, name in (id_to_company or {}).items():
        try:
            node_type[int(nid)] = "company"
            node_name[int(nid)] = name
        except Exception:
            continue
    for nid, name in (id_to_investor or {}).items():
        try:
            node_type[int(nid)] = "investor"
            node_name[int(nid)] = name
        except Exception:
            continue

    # Filter edges after train
    sources, destinations, edge_times, edge_idxs = filter_edges_after_training(full_data, train_data, logger)

    neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)
    pred_graph = nx.Graph()
    predictions = []

    dict_companies = {}
    dict_investors = {}
    edge_funding_info = {}

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

        for s, d, p, t in zip(src_batch, dst_batch, pos_prob.cpu().numpy(), times_batch):
            s = int(s); d = int(d)
            prob_score = float(p)
            timestamp = float(t)

            # Resolve types/names deterministically using node_type/node_name
            s_type = node_type.get(s, None)
            d_type = node_type.get(d, None)
            s_name = node_name.get(s, None)
            d_name = node_name.get(d, None)

            # Try best-effort if one side missing: use id_to_* fallback
            if s_name is None:
                s_name = id_to_company.get(s) or id_to_investor.get(s)
            if d_name is None:
                d_name = id_to_company.get(d) or id_to_investor.get(d)

            # Determine investor/company using types
            investor_name = None
            company_name = None

            if s_type == "investor" and d_type == "company":
                investor_name = s_name
                company_name = d_name
            elif s_type == "company" and d_type == "investor":
                investor_name = d_name
                company_name = s_name
            else:
                # If one type known and other unknown, try to deduce:
                if s_type == "investor" and d_type is None:
                    investor_name = s_name
                    company_name = d_name
                elif s_type == "company" and d_type is None:
                    investor_name = d_name
                    company_name = s_name
                elif d_type == "investor" and s_type is None:
                    investor_name = d_name
                    company_name = s_name
                elif d_type == "company" and s_type is None:
                    investor_name = s_name
                    company_name = d_name
                else:
                    # Ambiguous: both types unknown OR both same -> skip this edge
                    logger.debug(f"Skipping ambiguous edge (cannot resolve roles): s={s}({s_type}), d={d}({d_type}), s_name={s_name}, d_name={d_name})")
                    continue  # skip ambiguous mapping to avoid corruption

            # If still missing a name, build fallback string
            if investor_name is None:
                investor_name = f"investor_{s}_{d}"
            if company_name is None:
                company_name = f"company_{s}_{d}"

            # Create/update dict entries (consistent structure)
            if company_name not in dict_companies:
                # prefer numeric id of the company (if we can decide)
                comp_id_for_dict = d if node_type.get(d) == "company" else (s if node_type.get(s) == "company" else None)
                dict_companies[company_name] = {
                    'id': int(comp_id_for_dict) if comp_id_for_dict is not None else None,
                    'name': company_name,
                    'technologies': [],
                    'total_funding': 0.0,
                    'num_funding_rounds': 0
                }

            if investor_name not in dict_investors:
                inv_id_for_dict = s if node_type.get(s) == "investor" else (d if node_type.get(d) == "investor" else None)
                dict_investors[investor_name] = {
                    'investor_id': int(inv_id_for_dict) if inv_id_for_dict is not None else None,
                    'name': investor_name,
                    'num_investments': 0,
                    'total_invested': 0.0
                }

            # Tracking funding rounds per pair
            edge_key = (company_name, investor_name)
            if edge_key not in edge_funding_info:
                edge_funding_info[edge_key] = {
                    'funding_rounds': [],
                    'total_raised_amount_usd': 0.0,
                    'num_funding_rounds': 0
                }

            edge_funding_info[edge_key]['funding_rounds'].append({
                'timestamp': timestamp,
                'probability': prob_score
            })
            edge_funding_info[edge_key]['total_raised_amount_usd'] += prob_score
            edge_funding_info[edge_key]['num_funding_rounds'] += 1

            # Update global stats
            dict_companies[company_name]['total_funding'] += prob_score
            dict_companies[company_name]['num_funding_rounds'] += 1
            dict_investors[investor_name]['num_investments'] += 1
            dict_investors[investor_name]['total_invested'] += prob_score

            # Add nodes (with correct bipartite)
            pred_graph.add_node(company_name, bipartite=0)
            pred_graph.add_node(investor_name, bipartite=1)

            # Save raw prediction mapping (keep numeric ids for reference)
            predictions.append((s, d, prob_score))

        if (start // args.bs) % 10 == 0:
            logger.info("Processed %d/%d edges...", end, len(sources))

    # Add the edges to the graph (company -> investor)
    for (comp_name, inv_name), funding_info in edge_funding_info.items():
        pred_graph.add_edge(
            comp_name,
            inv_name,
            weight=funding_info['total_raised_amount_usd'],
            funding_rounds=funding_info['funding_rounds'],
            total_raised_amount_usd=funding_info['total_raised_amount_usd'],
            num_funding_rounds=funding_info['num_funding_rounds']
        )

    # Ensure all mapped nodes (from id_to_company/id_to_investor) are present in the graph
    # This avoids "in dict but not in graph" for nodes without predicted edges
    for nid, name in (id_to_company or {}).items():
        if name not in pred_graph:
            pred_graph.add_node(name, bipartite=0)
            if name not in dict_companies:
                dict_companies[name] = {'id': int(nid), 'name': name, 'technologies': [], 'total_funding': 0.0, 'num_funding_rounds': 0}
    for nid, name in (id_to_investor or {}).items():
        if name not in pred_graph:
            pred_graph.add_node(name, bipartite=1)
            if name not in dict_investors:
                dict_investors[name] = {'investor_id': int(nid), 'name': name, 'num_investments': 0, 'total_invested': 0.0}

    logger.info("Graph created: %d nodes, %d edges", pred_graph.number_of_nodes(), pred_graph.number_of_edges())
    logger.info("Dictionaries created: %d companies, %d investors", len(dict_companies), len(dict_investors))

    # Stats
    logger.info("\nStatistiques des levées de fonds:")
    total_funding_rounds = sum([data.get('num_funding_rounds', 0) for u, v, data in pred_graph.edges(data=True)])
    logger.info(f"  - Total levées de fonds: {total_funding_rounds}")

    return pred_graph, predictions, dict_companies, dict_investors


def save_graph_and_top(pred_graph, predictions, dict_companies, dict_investors, args, logger, id_to_company, id_to_investor):
    """Sauvegarde compatible TechRank"""
    # Save graph
    graph_path = Path(f'predicted_graph_{args.data}.pkl')
    with open(graph_path, 'wb') as f:
        pickle.dump(pred_graph, f)
    logger.info("Graph saved to %s", graph_path)
    
    # Save dictionaries (format TechRank)
    dict_comp_path = Path(f'dict_companies_{args.data}.pickle')
    with open(dict_comp_path, 'wb') as f:
        pickle.dump(dict_companies, f)
    logger.info("Companies dict saved to %s", dict_comp_path)
    
    dict_inv_path = Path(f'dict_investors_{args.data}.pickle')
    with open(dict_inv_path, 'wb') as f:
        pickle.dump(dict_investors, f)
    logger.info("Investors dict saved to %s", dict_inv_path)

    # Export top-k predictions
    if args.top_k_export > 0 and len(predictions) > 0:
        sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)[:args.top_k_export]
        csv_path = Path(f"top_predictions_{args.data}.csv")
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["investor_id", "company_id", "investor_name", "company_name", "probability"])
            for uid, vid, prob in sorted_preds:
                investor_name = id_to_investor.get(uid, f"investor_{uid}")
                company_name = id_to_company.get(vid, f"company_{vid}")
                writer.writerow([uid, vid, investor_name, company_name, prob])
        logger.info("Top %d predictions saved to %s", args.top_k_export, csv_path)

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

     # MODIFICATION: Récupérer aussi les dictionnaires
    pred_graph, predictions, dict_companies, dict_investors = generate_predictions_and_graph(
        tgn, id_to_company, id_to_investor, full_data, train_data, args, logger
    )

    # MODIFICATION: Passer les dictionnaires
    save_graph_and_top(pred_graph, predictions, dict_companies, dict_investors, 
                       args, logger, id_to_company, id_to_investor)

    logger.info("Evaluation complete!")

    # NOUVEAU: Lancer TechRank si demandé
    if args.run_techrank:
        logger.info("\n" + "="*70)
        logger.info("LANCEMENT DE TECHRANK SUR LE GRAPHE PRÉDIT")
        logger.info("="*70)
        
        # Sauvegarder d'abord dans le format attendu (au cas où)
        num_nodes = pred_graph.number_of_nodes()
        save_dir_classes = Path("savings/bipartite_invest_comp/classes")
        save_dir_networks = Path("savings/bipartite_invest_comp/networks")
        save_dir_classes.mkdir(parents=True, exist_ok=True)
        save_dir_networks.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir_classes / f'dict_companies_{num_nodes}.pickle', 'wb') as f:
            pickle.dump(dict_companies, f)
        with open(save_dir_classes / f'dict_investors_{num_nodes}.pickle', 'wb') as f:
            pickle.dump(dict_investors, f)
        with open(save_dir_networks / f'bipartite_graph_{num_nodes}.gpickle', 'wb') as f:
            pickle.dump(pred_graph, f)
        
        logger.info(f"✓ Données sauvegardées pour TechRank (limit={num_nodes})")
        
        # Importer et lancer TechRank avec les données directement
        try:
            from code.TechRank import run_techrank
            
            # ✅ CORRECTION: Passer les données directement
            df_investors_rank, df_companies_rank, _, _ = run_techrank(
                num_comp=num_nodes,
                num_tech=num_nodes,
                flag_cybersecurity=False,
                alpha=0.8,
                beta=-0.6,
                do_plot=False,
                dict_investors=dict_investors,  # ✅ Passer directement
                dict_comp=dict_companies,        # ✅ Passer directement
                B=pred_graph                     # ✅ Passer directement
            )
            
            logger.info("\n✓ TechRank terminé avec succès!")
            logger.info("\nTop 5 Investors (par TechRank):")
            logger.info(df_investors_rank[['TeckRank_int', 'final_configuration', 'techrank']].head().to_string())
            
            logger.info("\nTop 5 Companies (par TechRank):")
            logger.info(df_companies_rank[['TeckRank_int', 'final_configuration', 'techrank']].head().to_string())
            
        except ImportError as e:
            logger.error(f"Impossible d'importer TechRank: {e}")
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de TechRank: {e}", exc_info=True)


    if args.run_techrank:
        logger.info("\n" + "="*70)
        logger.info("DIAGNOSTIC DU GRAPHE AVANT TECHRANK")
        logger.info("="*70)
        
        # Vérifier la structure bipartite
        investors = [n for n, d in pred_graph.nodes(data=True) if d.get('bipartite') == 1]
        companies = [n for n, d in pred_graph.nodes(data=True) if d.get('bipartite') == 0]
        
        logger.info(f" Structure du graphe:")
        logger.info(f"   Total nodes: {pred_graph.number_of_nodes()}")
        logger.info(f"   Total edges: {pred_graph.number_of_edges()}")
        logger.info(f"   Investors (bipartite=0): {len(investors)}")
        logger.info(f"   Companies (bipartite=1): {len(companies)}")
        logger.info(f"   Dict investors: {len(dict_investors)}")
        logger.info(f"   Dict companies: {len(dict_companies)}")
        
        # Exemples de noms
        logger.info(f"\ Exemples de noms:")
        logger.info(f"   Investors: {investors[:3]}")
        logger.info(f"   Companies: {companies[:3]}")
        logger.info(f"   Dict investors keys: {list(dict_investors.keys())[:3]}")
        logger.info(f"   Dict companies keys: {list(dict_companies.keys())[:3]}")
        
        # Vérifier les nœuds isolés
        isolated = list(nx.isolates(pred_graph))
        logger.info(f"\  Nœuds isolés: {len(isolated)}")
        if isolated:
            logger.info(f"   Exemples: {isolated[:5]}")
        
        # Vérifier la cohérence dict ↔ graphe
        missing_in_graph_inv = set(dict_investors.keys()) - set(investors)
        missing_in_graph_comp = set(dict_companies.keys()) - set(companies)
        
        if missing_in_graph_inv:
            logger.info(f"\n❌ {len(missing_in_graph_inv)} investors dans dict mais pas dans graphe")
            logger.info(f"   Exemples: {list(missing_in_graph_inv)[:5]}")
        
        if missing_in_graph_comp:
            logger.info(f"❌ {len(missing_in_graph_comp)} companies dans dict mais pas dans graphe")
            logger.info(f"   Exemples: {list(missing_in_graph_comp)[:5]}")
        
        # Vérifier un edge exemple
        if pred_graph.number_of_edges() > 0:
            sample_edge = list(pred_graph.edges(data=True))[0]
            logger.info(f"\n🔗 Exemple d'arête:")
            logger.info(f"   {sample_edge[0]} → {sample_edge[1]}")
            logger.info(f"   Attributs: {sample_edge[2]}")
            
            # Vérifier le sens de l'arête
            u, v = sample_edge[0], sample_edge[1]
            u_bipartite = pred_graph.nodes[u].get('bipartite', 'MISSING')
            v_bipartite = pred_graph.nodes[v].get('bipartite', 'MISSING')
            logger.info(f"   {u} (bipartite={u_bipartite}) → {v} (bipartite={v_bipartite})")

if __name__ == "__main__":
    main()
