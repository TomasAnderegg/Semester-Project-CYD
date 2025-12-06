#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGN evaluation + prediction - VERSION CORRIGÉE FINALE
Compatible avec le pipeline prepare_tgn_input où:
  - u (sources) = companies (bipartite=0)
  - i (destinations) = investors (bipartite=1)
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
import pandas as pd

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
    """
    Charge les mappings depuis les CSV de vérification.
    IMPORTANT: Dans prepare_tgn_input, la convention est:
      - item_map = companies (colonnes 'u' dans TGN)
      - user_map = investors (colonnes 'i' dans TGN)
    """
    mapping_dir = Path(mapping_dir)
    
    id_to_company = {}
    id_to_investor = {}
    
    # CSV de vérification
    csv_company = mapping_dir / "crunchbase_filtered_company_map_verification.csv"
    csv_investor = mapping_dir / "crunchbase_filtered_investor_map_verification.csv"
    
    if csv_company.exists() and csv_investor.exists():
        try:
            df_comp = pd.read_csv(csv_company)
            df_inv = pd.read_csv(csv_investor)
            
            # Construire les mappings avec conversion en int
            id_to_company = {int(row['Company_ID_TGN']): str(row['Company_Name']) 
                            for _, row in df_comp.iterrows()}
            id_to_investor = {int(row['Investor_ID_TGN']): str(row['Investor_Name']) 
                             for _, row in df_inv.iterrows()}
            
            logger.info(f"✅ Mappings chargés depuis CSV:")
            logger.info(f"   {len(id_to_company):,} companies")
            logger.info(f"   {len(id_to_investor):,} investors")
            logger.info(f"   Company IDs: [{min(id_to_company.keys())} - {max(id_to_company.keys())}]")
            logger.info(f"   Investor IDs: [{min(id_to_investor.keys())} - {max(id_to_investor.keys())}]")
            
            return id_to_company, id_to_investor
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors du chargement des CSV: {e}")
            logger.info("Basculement sur les fichiers pickle...")
    else:
        logger.warning(f"⚠️  Fichiers CSV introuvables:")
        logger.warning(f"   - {csv_company}")
        logger.warning(f"   - {csv_investor}")
    
    # Fallback sur pickle
    candidate_company = ["crunchbase_filtered_company_id_map.pickle", "forecast_company_id_map.pickle"]
    candidate_investor = ["crunchbase_filtered_investor_id_map.pickle", "forecast_investor_id_map.pickle"]
    
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
    
    if company_map_path and investor_map_path:
        try:
            with open(company_map_path, "rb") as f:
                company_map = pickle.load(f)
            with open(investor_map_path, "rb") as f:
                investor_map = pickle.load(f)
            
            # Dans prepare_tgn_input: item_map = {company_name: id}, donc on inverse
            # Si le pickle est {name: id}, on l'inverse en {id: name}
            if company_map and isinstance(list(company_map.keys())[0], str):
                id_to_company = {int(v): str(k) for k, v in company_map.items()}
            else:
                id_to_company = {int(k): str(v) for k, v in company_map.items()}
                
            if investor_map and isinstance(list(investor_map.keys())[0], str):
                id_to_investor = {int(v): str(k) for k, v in investor_map.items()}
            else:
                id_to_investor = {int(k): str(v) for k, v in investor_map.items()}
            
            logger.info(f"✅ Mappings chargés depuis pickle:")
            logger.info(f"   {len(id_to_company):,} companies")
            logger.info(f"   {len(id_to_investor):,} investors")
            
            return id_to_company, id_to_investor
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors du chargement des pickle: {e}")
    
    logger.error("❌ Impossible de charger les mappings!")
    return id_to_company, id_to_investor

def process_data_chronologically(data, tgn_model, ngh_finder, batch_size, n_neighbors, logger):
    """Parcours chronologique pour mise à jour mémoire"""
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
                logger.error("AssertionError during compute_temporal_embeddings: %s", ex)
                raise

def build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger):
    """Reset memory and build from train_data"""
    if args.use_memory:
        tgn.memory.__init_memory__()
    process_data_chronologically(train_data, tgn, train_ngh_finder, args.bs, args.n_degree, logger)

def filter_edges_after_training(full_data, train_data, logger):
    """Filtre les arêtes après le timestamp max du train"""
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
    CORRECTION FINALE BASÉE SUR prepare_tgn_input():
    - sources (u) = companies → bipartite=0
    - destinations (i) = investors → bipartite=1
    """
    
    logger.info(f"\n{'='*70}")
    logger.info(f"CONSTRUCTION DU GRAPHE DE PRÉDICTION")
    logger.info(f"{'='*70}")
    logger.info(f"Convention TGN (depuis prepare_tgn_input):")
    logger.info(f"  - sources (u/item_map) = COMPANIES → bipartite=0")
    logger.info(f"  - destinations (i/user_map) = INVESTORS → bipartite=1")
    
    # ================================================================
    # ÉTAPE 1: Construire node_type/node_name SELON LA CONVENTION TGN
    # ================================================================
    node_type = {}
    node_name = {}
    
    all_sources = set(full_data.sources)
    all_destinations = set(full_data.destinations)
    
    logger.info(f"\nPlages d'IDs dans les données TGN:")
    logger.info(f"  Sources (companies): [{min(all_sources)}, {max(all_sources)}] ({len(all_sources)} uniques)")
    logger.info(f"  Destinations (investors): [{min(all_destinations)}, {max(all_destinations)}] ({len(all_destinations)} uniques)")
    logger.info(f"  id_to_company: [{min(id_to_company.keys()) if id_to_company else 'N/A'}, {max(id_to_company.keys()) if id_to_company else 'N/A'}]")
    logger.info(f"  id_to_investor: [{min(id_to_investor.keys()) if id_to_investor else 'N/A'}, {max(id_to_investor.keys()) if id_to_investor else 'N/A'}]")
    
    # Sources = COMPANIES (selon prepare_tgn_input)
    companies_mapped = 0
    for src_id in all_sources:
        src_id = int(src_id)
        if src_id in id_to_company:
            node_type[src_id] = "company"
            node_name[src_id] = id_to_company[src_id]
            companies_mapped += 1
    
    # Destinations = INVESTORS (selon prepare_tgn_input)
    investors_mapped = 0
    for dst_id in all_destinations:
        dst_id = int(dst_id)
        if dst_id in id_to_investor:
            node_type[dst_id] = "investor"
            node_name[dst_id] = id_to_investor[dst_id]
            investors_mapped += 1
    
    logger.info(f"\nNode mappings construits:")
    logger.info(f"  Companies (sources, bipartite=0): {companies_mapped}")
    logger.info(f"  Investors (destinations, bipartite=1): {investors_mapped}")
    logger.info(f"  Total nodes mappés: {len(node_type)}")
    
    if companies_mapped == 0:
        logger.error("❌ ERREUR CRITIQUE: Aucune company mappée!")
        logger.error("   Vérifiez que id_to_company contient les bons IDs")
    
    if investors_mapped == 0:
        logger.error("❌ ERREUR CRITIQUE: Aucun investor mappé!")
        logger.error("   Vérifiez que id_to_investor contient les bons IDs")
    
    logger.info(f"{'='*70}\n")
    
    # Convention TechRank: companies=0, investors=1
    COMPANY_BIPARTITE = 0
    INVESTOR_BIPARTITE = 1
    
    # ================================================================
    # ÉTAPE 2: Filtrer et générer les prédictions
    # ================================================================
    sources, destinations, edge_times, edge_idxs = filter_edges_after_training(full_data, train_data, logger)
    
    neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)
    pred_graph = nx.Graph()
    predictions = []
    
    dict_companies = {}
    dict_investors = {}
    edge_funding_info = {}
    
    logger.info("Computing edge probabilities for filtered dataset...")
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
            s = int(s)  # Company ID (source)
            d = int(d)  # Investor ID (destination)
            prob_score = float(p)
            timestamp = float(t)
            
            # Selon notre convention: s=company, d=investor
            company_id = s
            investor_id = d
            company_name = node_name.get(s, f"company_{s}")
            investor_name = node_name.get(d, f"investor_{d}")
            
            # Créer/mettre à jour les dictionnaires
            if company_name not in dict_companies:
                dict_companies[company_name] = {
                    'id': company_id,
                    'name': company_name,
                    'technologies': [],
                    'total_funding': 0.0,
                    'num_funding_rounds': 0
                }
            
            if investor_name not in dict_investors:
                dict_investors[investor_name] = {
                    'investor_id': investor_id,
                    'name': investor_name,
                    'num_investments': 0,
                    'total_invested': 0.0
                }
            
            # Tracking funding rounds
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
            
            # Update stats
            dict_companies[company_name]['total_funding'] += prob_score
            dict_companies[company_name]['num_funding_rounds'] += 1
            dict_investors[investor_name]['num_investments'] += 1
            dict_investors[investor_name]['total_invested'] += prob_score
            
            # Add nodes avec bipartite correct
            pred_graph.add_node(company_name, bipartite=COMPANY_BIPARTITE)
            pred_graph.add_node(investor_name, bipartite=INVESTOR_BIPARTITE)
            
            predictions.append((s, d, prob_score))
        
        if (start // args.bs) % 10 == 0:
            logger.info("Processed %d/%d edges...", end, len(sources))
    
    # Add edges
    for (comp_name, inv_name), funding_info in edge_funding_info.items():
        pred_graph.add_edge(
            comp_name,
            inv_name,
            weight=funding_info['total_raised_amount_usd'],
            funding_rounds=funding_info['funding_rounds'],
            total_raised_amount_usd=funding_info['total_raised_amount_usd'],
            num_funding_rounds=funding_info['num_funding_rounds']
        )
    
    # Ensure all mapped nodes are in graph
    for nid, name in (id_to_company or {}).items():
        if name not in pred_graph:
            pred_graph.add_node(name, bipartite=COMPANY_BIPARTITE)
            if name not in dict_companies:
                dict_companies[name] = {
                    'id': int(nid),
                    'name': name,
                    'technologies': [],
                    'total_funding': 0.0,
                    'num_funding_rounds': 0
                }
    
    for nid, name in (id_to_investor or {}).items():
        if name not in pred_graph:
            pred_graph.add_node(name, bipartite=INVESTOR_BIPARTITE)
            if name not in dict_investors:
                dict_investors[name] = {
                    'investor_id': int(nid),
                    'name': name,
                    'num_investments': 0,
                    'total_invested': 0.0
                }
    
    logger.info("Graph created: %d nodes, %d edges", pred_graph.number_of_nodes(), pred_graph.number_of_edges())
    logger.info("Dictionaries created: %d companies, %d investors", len(dict_companies), len(dict_investors))
    
    # Stats
    total_funding_rounds = sum([data.get('num_funding_rounds', 0) for u, v, data in pred_graph.edges(data=True)])
    logger.info(f"Total funding rounds: {total_funding_rounds}")
    
    # Vérifier les poids des arêtes
    if pred_graph.number_of_edges() > 0:
        edge_weights = [data.get('weight', 0) for u, v, data in pred_graph.edges(data=True)]
        logger.info(f"\n📊 Statistiques des poids d'arêtes:")
        logger.info(f"   Min: {min(edge_weights):.6f}")
        logger.info(f"   Max: {max(edge_weights):.6f}")
        logger.info(f"   Moyenne: {np.mean(edge_weights):.6f}")
        logger.info(f"   Arêtes avec poids > 0: {sum(1 for w in edge_weights if w > 0)}")
    
    return pred_graph, predictions, dict_companies, dict_investors

def save_graph_and_top(pred_graph, predictions, dict_companies, dict_investors, args, logger, id_to_company, id_to_investor):
    """Sauvegarde compatible TechRank"""
    graph_path = Path(f'predicted_graph_{args.data}.pkl')
    with open(graph_path, 'wb') as f:
        pickle.dump(pred_graph, f)
    logger.info("Graph saved to %s", graph_path)
    
    dict_comp_path = Path(f'dict_companies_{args.data}.pickle')
    with open(dict_comp_path, 'wb') as f:
        pickle.dump(dict_companies, f)
    logger.info("Companies dict saved to %s", dict_comp_path)
    
    dict_inv_path = Path(f'dict_investors_{args.data}.pickle')
    with open(dict_inv_path, 'wb') as f:
        pickle.dump(dict_investors, f)
    logger.info("Investors dict saved to %s", dict_inv_path)

    if args.top_k_export > 0 and len(predictions) > 0:
        sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)[:args.top_k_export]
        csv_path = Path(f"top_predictions_{args.data}.csv")
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["company_id", "investor_id", "company_name", "investor_name", "probability"])
            for src_id, dst_id, prob in sorted_preds:
                comp_name = id_to_company.get(src_id, f"company_{src_id}")
                inv_name = id_to_investor.get(dst_id, f"investor_{dst_id}")
                writer.writerow([src_id, dst_id, comp_name, inv_name, prob])
        logger.info("Top %d predictions saved to %s", args.top_k_export, csv_path)

# ----------------------------
# main()
# ----------------------------
def main():
    args = parse_args()
    logger = setup_logger()

    logger.info("Evaluating TGN model on dataset: %s", args.data)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
        args.data,
        different_new_nodes_between_val_and_test=args.different_new_nodes,
        randomize_features=args.randomize_features
    )

    logger.info("Dataset loaded: %d interactions (full)", len(full_data.sources))

    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps
    )

    tgn = initialize_model(args, device, node_features, edge_features,
                           mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst)

    if Path(args.model_path).exists():
        logger.info("Loading model from %s", args.model_path)
        checkpoint = torch.load(args.model_path, map_location=device)
        for key in list(checkpoint.keys()):
            if "memory" in key:
                checkpoint.pop(key)
        tgn.load_state_dict(checkpoint, strict=False)
    else:
        logger.error("Model file not found: %s", args.model_path)
        sys.exit(1)

    tgn.to(device)
    tgn.eval()
    
    logger.info("Starting evaluation pipeline")

    # Phase 1: Validation (seen nodes)
    logger.info("Phase 1: Validation (seen nodes)")
    if args.use_memory:
        build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)
    val_ap, val_auc,_,_,_ = eval_edge_prediction(tgn, val_rand_sampler, val_data, args.n_degree)
    logger.info("Validation (seen nodes) - AUC: %.4f, AP: %.4f", val_auc, val_ap)

    # Phase 2: Validation (new nodes)
    logger.info("Phase 2: Validation (new nodes)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)
    nn_val_ap, nn_val_auc,_,_,_ = eval_edge_prediction(tgn, nn_val_rand_sampler, new_node_val_data, args.n_degree)
    logger.info("Validation (new nodes) - AUC: %.4f, AP: %.4f", nn_val_auc, nn_val_ap)

    logger.info("Phase 3: Test (seen nodes)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)
    test_ap, test_auc,_,_,_ = eval_edge_prediction(tgn, test_rand_sampler, test_data, args.n_degree)
    logger.info("Test (seen nodes) - AUC: %.4f, AP: %.4f", test_auc, test_ap)

    # Phase 4: Test (new nodes)
    logger.info("Phase 4: Test (new nodes)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)
    nn_test_ap, nn_test_auc,_,_,_ = eval_edge_prediction(tgn, nn_test_rand_sampler, new_node_test_data, args.n_degree)
    logger.info("Test (new nodes) - AUC: %.4f, AP: %.4f", nn_test_auc, nn_test_ap)

    # Load mappings
    id_to_company, id_to_investor = load_mappings(args.mapping_dir, logger)

    # Generate predictions
    logger.info("Generating predicted graph (using memory built from train)")
    build_memory_from_train(args, tgn, train_data, train_ngh_finder, logger)
    tgn.set_neighbor_finder(full_ngh_finder)

    pred_graph, predictions, dict_companies, dict_investors = generate_predictions_and_graph(
        tgn, id_to_company, id_to_investor, full_data, train_data, args, logger
    )

    save_graph_and_top(pred_graph, predictions, dict_companies, dict_investors, 
                       args, logger, id_to_company, id_to_investor)

    logger.info("Evaluation complete!")

    # DIAGNOSTIC AVANT TECHRANK
    if args.run_techrank:
        logger.info("\n" + "="*70)
        logger.info("DIAGNOSTIC DU GRAPHE AVANT TECHRANK")
        logger.info("="*70)
        
        # Vérifier la structure bipartite
        nodes_0 = [n for n, d in pred_graph.nodes(data=True) if d.get('bipartite') == 0]
        nodes_1 = [n for n, d in pred_graph.nodes(data=True) if d.get('bipartite') == 1]
        
        logger.info(f"📊 Structure du graphe:")
        logger.info(f"   Total nodes: {pred_graph.number_of_nodes()}")
        logger.info(f"   Total edges: {pred_graph.number_of_edges()}")
        logger.info(f"   Companies (bipartite=0): {len(nodes_0)}")
        logger.info(f"   Investors (bipartite=1): {len(nodes_1)}")
        logger.info(f"   Dict companies: {len(dict_companies)}")
        logger.info(f"   Dict investors: {len(dict_investors)}")
        
        # Exemples de noms
        logger.info(f"\n📝 Exemples de noms:")
        logger.info(f"   Companies (bipartite=0): {nodes_0[:3]}")
        logger.info(f"   Investors (bipartite=1): {nodes_1[:3]}")
        logger.info(f"   Dict companies keys: {list(dict_companies.keys())[:3]}")
        logger.info(f"   Dict investors keys: {list(dict_investors.keys())[:3]}")
        
        # Vérifier les nœuds isolés
        isolated = list(nx.isolates(pred_graph))
        logger.info(f"\n🔍 Nœuds isolés: {len(isolated)}")
        if isolated:
            logger.info(f"   Exemples: {isolated[:5]}")
        
        # Vérifier la cohérence dict ↔ graphe
        nodes_0_set = set(nodes_0)
        nodes_1_set = set(nodes_1)
        
        companies_in_0 = len(set(dict_companies.keys()) & nodes_0_set)
        companies_in_1 = len(set(dict_companies.keys()) & nodes_1_set)
        investors_in_0 = len(set(dict_investors.keys()) & nodes_0_set)
        investors_in_1 = len(set(dict_investors.keys()) & nodes_1_set)
        
        logger.info(f"\n🔍 Cohérence dict ↔ graphe:")
        logger.info(f"   Companies dans bipartite=0: {companies_in_0}/{len(dict_companies)}")
        logger.info(f"   Companies dans bipartite=1: {companies_in_1}/{len(dict_companies)} (devrait être 0)")
        logger.info(f"   Investors dans bipartite=0: {investors_in_0}/{len(dict_investors)} (devrait être 0)")
        logger.info(f"   Investors dans bipartite=1: {investors_in_1}/{len(dict_investors)}")
        
        # Vérifier des exemples d'arêtes
        if pred_graph.number_of_edges() > 0:
            sample_edges = list(pred_graph.edges(data=True))[:3]
            logger.info(f"\n🔗 Exemples d'arêtes:")
            for u, v, data in sample_edges:
                u_bipartite = pred_graph.nodes[u].get('bipartite', 'MISSING')
                v_bipartite = pred_graph.nodes[v].get('bipartite', 'MISSING')
                weight = data.get('weight', 0)
                logger.info(f"   {u} (bipartite={u_bipartite}) → {v} (bipartite={v_bipartite})")
                logger.info(f"      Weight: {weight:.6f}, Rounds: {data.get('num_funding_rounds', 0)}")
                
                # Vérifier la cohérence
                if u_bipartite == v_bipartite:
                    logger.error(f"      ❌ ERREUR: Les deux nœuds ont le même bipartite={u_bipartite}!")
                else:
                    logger.info(f"      ✅ OK: Arête bipartite valide")
        
        # Vérifier les degrés
        if pred_graph.number_of_edges() > 0:
            degrees = dict(pred_graph.degree())
            if nodes_0:
                company_degrees = [degrees[c] for c in nodes_0[:10]]
                logger.info(f"\n📊 Degrés des companies (10 premiers): {company_degrees}")
            if nodes_1:
                investor_degrees = [degrees[i] for i in nodes_1[:10]]
                logger.info(f"📊 Degrés des investors (10 premiers): {investor_degrees}")

    # LANCEMENT DE TECHRANK
    if args.run_techrank:
        logger.info("\n" + "="*70)
        logger.info("LANCEMENT DE TECHRANK SUR LE GRAPHE PRÉDIT")
        logger.info("="*70)
        
        # Vérification finale avant TechRank
        companies_in_graph = set(n for n in pred_graph.nodes() if n in dict_companies)
        investors_in_graph = set(n for n in pred_graph.nodes() if n in dict_investors)
        
        logger.info(f"\n📊 Données pour TechRank:")
        logger.info(f"   Companies dans le graphe: {len(companies_in_graph)}")
        logger.info(f"   Investors dans le graphe: {len(investors_in_graph)}")
        logger.info(f"   Total edges: {pred_graph.number_of_edges()}")
        
        # Sauvegarder dans le format attendu
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
        
        logger.info(f"\n✓ Données sauvegardées pour TechRank (limit={num_nodes})")
        
        # Importer et lancer TechRank
        try:
            from code.TechRank import run_techrank
            
            logger.info("\n🚀 Lancement de TechRank...")
            logger.info(f"   Alpha: 0.8, Beta: -0.6")
            logger.info(f"   Companies: {len(dict_companies)}")
            logger.info(f"   Investors: {len(dict_investors)}")
            
            df_investors_rank, df_companies_rank, _, _ = run_techrank(
                num_comp=num_nodes,
                num_tech=num_nodes,
                flag_cybersecurity=False,
                alpha=0.8,
                beta=-0.6,
                do_plot=False,
                dict_investors=dict_investors,
                dict_comp=dict_companies,
                B=pred_graph
            )
            
            logger.info("\n✓ TechRank terminé avec succès!")
            
            # Vérifier si les scores sont tous à zéro
            if df_investors_rank is not None and len(df_investors_rank) > 0:
                non_zero_inv = (df_investors_rank['techrank'] > 0).sum()
                max_score_inv = df_investors_rank['techrank'].max()
                logger.info(f"\n📊 Résultats Investors:")
                logger.info(f"   Total: {len(df_investors_rank)}")
                logger.info(f"   Scores > 0: {non_zero_inv}")
                logger.info(f"   Score max: {max_score_inv:.6f}")
                
                if non_zero_inv > 0:
                    logger.info("\n📊 Top 10 Investors (par TechRank):")
                    top_inv = df_investors_rank.nlargest(10, 'techrank')[['final_configuration', 'techrank']]
                    for idx, (_, row) in enumerate(top_inv.iterrows(), 1):
                        logger.info(f"   #{idx:2d} {row['final_configuration']:40s} → Score: {row['techrank']:.6f}")
                else:
                    logger.error("\n❌ TOUS les scores investors sont à zéro!")
                    logger.error("   Causes possibles:")
                    logger.error("   1. Structure bipartite incorrecte (arêtes 0-0 ou 1-1)")
                    logger.error("   2. Poids des arêtes tous nuls")
                    logger.error("   3. Graphe déconnecté")
            
            if df_companies_rank is not None and len(df_companies_rank) > 0:
                non_zero_comp = (df_companies_rank['techrank'] > 0).sum()
                max_score_comp = df_companies_rank['techrank'].max()
                logger.info(f"\n📊 Résultats Companies:")
                logger.info(f"   Total: {len(df_companies_rank)}")
                logger.info(f"   Scores > 0: {non_zero_comp}")
                logger.info(f"   Score max: {max_score_comp:.6f}")
                
                if non_zero_comp > 0:
                    logger.info("\n📊 Top 10 Companies (par TechRank):")
                    top_comp = df_companies_rank.nlargest(10, 'techrank')[['final_configuration', 'techrank']]
                    for idx, (_, row) in enumerate(top_comp.iterrows(), 1):
                        logger.info(f"   #{idx:2d} {row['final_configuration']:40s} → Score: {row['techrank']:.6f}")
                else:
                    logger.error("\n❌ TOUS les scores companies sont à zéro!")
                    logger.error("   Causes possibles:")
                    logger.error("   1. Structure bipartite incorrecte (arêtes 0-0 ou 1-1)")
                    logger.error("   2. Poids des arêtes tous nuls")
                    logger.error("   3. Graphe déconnecté")
            
        except ImportError as e:
            logger.error(f"❌ Impossible d'importer TechRank: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de TechRank: {e}", exc_info=True)

if __name__ == "__main__":
    main()