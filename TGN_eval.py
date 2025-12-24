#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGN Evaluation Script - Version Corrigée pour Reproduire les Métriques du Training
"""

import argparse
import logging
import sys
from pathlib import Path
import pickle
import csv

import torch
import numpy as np
import pandas as pd

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, Data

def parse_args():
    parser = argparse.ArgumentParser('TGN Evaluation - Reproduction exacte du training')
    parser.add_argument('-d', '--data', type=str, default='crunchbase', help='Dataset name')
    parser.add_argument('--bs', type=int, default=200, help='Batch_size')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--node_dim', type=int, default=200, help='Node embedding dimensions')
    parser.add_argument('--time_dim', type=int, default=200, help='Time embedding dimensions')
    parser.add_argument('--use_memory', action='store_true', help='Use memory module')
    parser.add_argument('--embedding_module', type=str, default="graph_attention")
    parser.add_argument('--message_function', type=str, default="identity")
    parser.add_argument('--memory_updater', type=str, default="gru")
    parser.add_argument('--aggregator', type=str, default="last")
    parser.add_argument('--memory_update_at_end', action='store_true')
    parser.add_argument('--message_dim', type=int, default=200)
    parser.add_argument('--memory_dim', type=int, default=None, help='Memory dimensions (auto-detect from model if not specified)')
    parser.add_argument('--different_new_nodes', action='store_true')
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--randomize_features', action='store_true')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true')
    parser.add_argument('--use_source_embedding_in_message', action='store_true')
    parser.add_argument('--dyrep', action='store_true')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--mapping_dir', type=str, default='data/mappings')
    parser.add_argument('--top_k_export', type=int, default=100)
    parser.add_argument('--run_techrank', action='store_true', help='Run TechRank after evaluation')
    parser.add_argument('--auto_detect_params', action='store_true',
                        help='Auto-detect model parameters from checkpoint')
    parser.add_argument('--temporal_validation', action='store_true',
                        help='Run temporal validation on test set')
    parser.add_argument('--temporal_split', type=float, default=0.5,
                        help='Fraction of test set to use as history (default: 0.5)')
    parser.add_argument('--prediction_threshold', type=float, default=0.0,
                        help='Probability threshold for predictions (default: 0.0 = no threshold)')
    return parser.parse_args()

def detect_model_params_from_checkpoint(checkpoint_path, logger):
    """
    Détecte automatiquement les hyperparamètres du modèle depuis le checkpoint
    """
    logger.info("🔍 Auto-detecting model parameters from checkpoint...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    params = {}
    
    # Détecter memory_dim depuis la shape de memory.memory
    if 'memory.memory' in checkpoint:
        memory_shape = checkpoint['memory.memory'].shape
        params['memory_dim'] = memory_shape[1]
        logger.info(f"   ✓ Detected memory_dim: {params['memory_dim']}")
    
    # Détecter message_dim depuis message_function si disponible
    if 'message_function.message_linear_1.weight' in checkpoint:
        msg_shape = checkpoint['message_function.message_linear_1.weight'].shape
        params['message_dim'] = msg_shape[0]
        logger.info(f"   ✓ Detected message_dim: {params['message_dim']}")
    
    # Détecter node_dim depuis embedding_module
    if 'embedding_module.linear.weight' in checkpoint:
        node_shape = checkpoint['embedding_module.linear.weight'].shape
        params['node_dim'] = node_shape[1]
        logger.info(f"   ✓ Detected node_dim: {params['node_dim']}")
    
    # Détecter n_heads depuis attention layers
    if 'embedding_module.attention_models.0.attention.out_proj.weight' in checkpoint:
        out_proj_shape = checkpoint['embedding_module.attention_models.0.attention.out_proj.weight'].shape
        # Le nombre de têtes peut être déduit de la structure
        logger.info(f"   ⚠️  n_heads detection: check manually (typically 2)")
    
    return params

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger()

def process_data_chronologically(data, tgn_model, ngh_finder, batch_size, n_neighbors, logger):
    """
    ⚠️ CRITIQUE: Parcours chronologique pour construire la mémoire
    Cette fonction doit être appelée dans le même ordre qu'au training
    """
    tgn_model.set_neighbor_finder(ngh_finder)

    sources = np.asarray(data.sources)
    destinations = np.asarray(data.destinations)
    timestamps = np.asarray(data.timestamps)
    edge_idxs = np.asarray(data.edge_idxs)

    if len(timestamps) == 0:
        logger.debug("No interactions to process")
        return

    # ⚠️ IMPORTANT: Trier chronologiquement (comme au training)
    sorted_idx = np.argsort(timestamps)
    sources = sources[sorted_idx]
    destinations = destinations[sorted_idx]
    timestamps = timestamps[sorted_idx]
    edge_idxs = edge_idxs[sorted_idx]

    logger.info(f"Processing {len(sources)} interactions chronologically...")
    for i in range(0, len(sources), batch_size):
        s = sources[i:i+batch_size]
        d = destinations[i:i+batch_size]
        ts = timestamps[i:i+batch_size]
        e = edge_idxs[i:i+batch_size]
        
        with torch.no_grad():
            try:
                _ = tgn_model.compute_temporal_embeddings(s, d, d, ts, e, n_neighbors)
            except AssertionError as ex:
                logger.error(f"AssertionError during memory update: {ex}")
                raise
        
        if (i // batch_size) % 50 == 0:
            logger.info(f"  Processed {i}/{len(sources)} interactions...")

def load_mappings(mapping_dir, logger):
    """Charge les mappings pour les prédictions"""
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
            
            id_to_company = {int(row['Company_ID_TGN']): str(row['Company_Name']) 
                            for _, row in df_comp.iterrows()}
            id_to_investor = {int(row['Investor_ID_TGN']): str(row['Investor_Name']) 
                             for _, row in df_inv.iterrows()}
            
            logger.info(f"✅ Mappings loaded: {len(id_to_company)} companies, {len(id_to_investor)} investors")
            return id_to_company, id_to_investor
            
        except Exception as e:
            logger.warning(f"⚠️  Error loading CSV mappings: {e}")
    
    # Fallback sur pickle
    logger.info("Trying pickle fallback...")
    company_pickle = mapping_dir / "crunchbase_filtered_company_id_map.pickle"
    investor_pickle = mapping_dir / "crunchbase_filtered_investor_id_map.pickle"
    
    if company_pickle.exists() and investor_pickle.exists():
        with open(company_pickle, "rb") as f:
            company_map = pickle.load(f)
        with open(investor_pickle, "rb") as f:
            investor_map = pickle.load(f)
        
        # Inversion si nécessaire
        if company_map and isinstance(list(company_map.keys())[0], str):
            id_to_company = {int(v): str(k) for k, v in company_map.items()}
        else:
            id_to_company = {int(k): str(v) for k, v in company_map.items()}
            
        if investor_map and isinstance(list(investor_map.keys())[0], str):
            id_to_investor = {int(v): str(k) for k, v in investor_map.items()}
        else:
            id_to_investor = {int(k): str(v) for k, v in investor_map.items()}
        
        logger.info(f"✅ Mappings loaded from pickle: {len(id_to_company)} companies, {len(id_to_investor)} investors")
        return id_to_company, id_to_investor
    
    logger.error("❌ No mapping files found!")
    return id_to_company, id_to_investor

def main():
    args = parse_args()
    logger = setup_logger()

    logger.info("="*70)
    logger.info("TGN EVALUATION - Reproduction exacte du training")
    logger.info("="*70)
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Model: {args.model_path}")
    
    logger.info("\n" + "="*70)
    logger.info("HYPERPARAMETERS")
    logger.info("="*70)
    logger.info(f"⚠️  IMPORTANT: Use the SAME hyperparameters as training!")
    logger.info(f"   batch_size: {args.bs}")
    logger.info(f"   n_degree: {args.n_degree}")
    logger.info(f"   n_head: {args.n_head}")
    logger.info(f"   n_layer: {args.n_layer}")
    logger.info(f"   node_dim: {args.node_dim}")
    logger.info(f"   time_dim: {args.time_dim}")
    logger.info(f"   message_dim: {args.message_dim}")
    logger.info(f"   memory_dim: {args.memory_dim if args.memory_dim else 'auto-detect'}")
    logger.info(f"   embedding_module: {args.embedding_module}")
    logger.info(f"   use_memory: {args.use_memory}")

    # ================================================================
    # Auto-detect parameters from checkpoint if requested
    # ================================================================
    if args.auto_detect_params or args.memory_dim is None:
        if not Path(args.model_path).exists():
            logger.error(f"❌ Model file not found: {args.model_path}")
            sys.exit(1)
        
        logger.info("\n" + "="*70)
        logger.info("AUTO-DETECTING PARAMETERS FROM CHECKPOINT")
        logger.info("="*70)
        
        detected_params = detect_model_params_from_checkpoint(args.model_path, logger)
        
        # Override args with detected parameters
        if args.memory_dim is None and 'memory_dim' in detected_params:
            args.memory_dim = detected_params['memory_dim']
            logger.info(f"✅ Using detected memory_dim: {args.memory_dim}")
        
        if args.auto_detect_params:
            if 'message_dim' in detected_params:
                args.message_dim = detected_params['message_dim']
                logger.info(f"✅ Using detected message_dim: {args.message_dim}")
            if 'node_dim' in detected_params:
                args.node_dim = detected_params['node_dim']
                logger.info(f"✅ Using detected node_dim: {args.node_dim}")
    
    # Fallback si toujours None
    if args.memory_dim is None:
        logger.error("❌ Could not detect memory_dim from checkpoint!")
        logger.error("   Please specify --memory_dim manually")
        logger.error("   Check your training logs or pickle file for the correct value")
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nDevice: {device}")

    # ================================================================
    # ✅ ÉTAPE 1: Charger LES MÊMES DONNÉES qu'au training
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Loading FULL dataset (train+val+test)")
    logger.info("="*70)
    
    node_features, edge_features, full_data, train_data, val_data, test_data, \
    new_node_val_data, new_node_test_data = get_data(
        args.data,
        different_new_nodes_between_val_and_test=args.different_new_nodes,
        randomize_features=args.randomize_features
    )
    
    logger.info(f"✅ Dataset loaded:")
    logger.info(f"   Full data: {len(full_data.sources)} interactions")
    logger.info(f"   Train: {len(train_data.sources)} interactions")
    logger.info(f"   Val: {len(val_data.sources)} interactions")
    logger.info(f"   Test: {len(test_data.sources)} interactions")
    logger.info(f"   New nodes test: {len(new_node_test_data.sources)} interactions")

    # ================================================================
    # ✅ ÉTAPE 2: Créer LES MÊMES neighbor finders qu'au training
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Creating neighbor finders")
    logger.info("="*70)
    
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
    
    logger.info("✅ Neighbor finders created (train + full)")

    # ================================================================
    # ✅ ÉTAPE 3: Créer LES MÊMES samplers qu'au training
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Creating random samplers")
    logger.info("="*70)
    
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)
    
    logger.info("✅ Random samplers created")

    # ================================================================
    # ✅ ÉTAPE 4: Calculer LES MÊMES time statistics qu'au training
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Computing time statistics")
    logger.info("="*70)
    
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    
    logger.info(f"✅ Time statistics computed:")
    logger.info(f"   Source: mean={mean_time_shift_src:.2f}, std={std_time_shift_src:.2f}")
    logger.info(f"   Dest: mean={mean_time_shift_dst:.2f}, std={std_time_shift_dst:.2f}")

    # ================================================================
    # ✅ ÉTAPE 5: Initialiser le modèle EXACTEMENT comme au training
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Initializing model")
    logger.info("="*70)
    
    tgn = TGN(
        neighbor_finder=train_ngh_finder,  # ⚠️ IMPORTANT: Commencer avec train_ngh_finder
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
    logger.info("✅ Model initialized")

    # ================================================================
    # ✅ ÉTAPE 6: Charger les poids du modèle
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 6: Loading model weights")
    logger.info("="*70)
    
    if not Path(args.model_path).exists():
        logger.error(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # ⚠️ IMPORTANT: Ne PAS supprimer les memory states si use_memory=True
    # Car on veut reconstruire l'état exact du training
    if not args.use_memory:
        for key in list(checkpoint.keys()):
            if "memory" in key:
                checkpoint.pop(key)
    
    tgn.load_state_dict(checkpoint, strict=False)
    tgn.eval()
    logger.info("✅ Model weights loaded")

    # ================================================================
    # ✅ ÉTAPE 7: Reconstruire la mémoire EXACTEMENT comme au training
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 7: Rebuilding memory from training data")
    logger.info("="*70)
    
    if args.use_memory:
        logger.info("Resetting memory...")
        tgn.memory.__init_memory__()
        
        logger.info("Processing training data chronologically...")
        process_data_chronologically(train_data, tgn, train_ngh_finder, args.bs, args.n_degree, logger)
        
        logger.info("✅ Memory rebuilt from training data")
        train_memory_backup = tgn.memory.backup_memory()
    else:
        logger.info("⚠️  No memory module used")
        train_memory_backup = None

    # ================================================================
    # ✅ ÉTAPE 8: Validation (pour comparaison)
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 8: Validation evaluation")
    logger.info("="*70)
    
    tgn.set_neighbor_finder(full_ngh_finder)  # ⚠️ IMPORTANT: Utiliser full_ngh_finder pour val/test
    
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
    
    if args.use_memory:
        # Sauvegarder l'état après train
        train_memory_state = tgn.memory.backup_memory()
    
    val_ap, val_auc, val_mrr, val_recall_10, val_recall_50 = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=val_rand_sampler,
        data=val_data,
        n_neighbors=args.n_degree
    )
    
    logger.info("📊 Validation Results (old nodes):")
    logger.info(f"   AUROC: {val_auc:.4f}")
    logger.info(f"   AP: {val_ap:.4f}")
    logger.info(f"   MRR: {val_mrr:.4f}")
    logger.info(f"   Recall@10: {val_recall_10:.4f}")
    logger.info(f"   Recall@50: {val_recall_50:.4f}")
    
    if args.use_memory:
        val_memory_backup = tgn.memory.backup_memory()
        tgn.memory.restore_memory(train_memory_state)
    
    nn_val_ap, nn_val_auc, nn_val_mrr, nn_val_recall_10, nn_val_recall_50 = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=nn_val_rand_sampler,
        data=new_node_val_data,
        n_neighbors=args.n_degree
    )
    
    logger.info("📊 Validation Results (new nodes):")
    logger.info(f"   AUROC: {nn_val_auc:.4f}")
    logger.info(f"   AP: {nn_val_ap:.4f}")
    logger.info(f"   MRR: {nn_val_mrr:.4f}")
    logger.info(f"   Recall@10: {nn_val_recall_10:.4f}")
    logger.info(f"   Recall@50: {nn_val_recall_50:.4f}")
    
    if args.use_memory:
        tgn.memory.restore_memory(val_memory_backup)

    # ================================================================
    # ✅ ÉTAPE 9: Test (metrics principales)
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 9: Test evaluation")
    logger.info("="*70)
    
    test_ap, test_auc, test_mrr, test_recall_10, test_recall_50 = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=test_rand_sampler,
        data=test_data,
        n_neighbors=args.n_degree
    )
    
    logger.info("📊 Test Results (old nodes):")
    logger.info(f"   AUROC: {test_auc:.4f}")
    logger.info(f"   AP: {test_ap:.4f}")
    logger.info(f"   MRR: {test_mrr:.4f}")
    logger.info(f"   Recall@10: {test_recall_10:.4f}")
    logger.info(f"   Recall@50: {test_recall_50:.4f}")
    
    if args.use_memory:
        tgn.memory.restore_memory(val_memory_backup)
    
    nn_test_ap, nn_test_auc, nn_test_mrr, nn_test_recall_10, nn_test_recall_50 = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data,
        n_neighbors=args.n_degree
    )
    
    logger.info("📊 Test Results (new nodes):")
    logger.info(f"   AUROC: {nn_test_auc:.4f}")
    logger.info(f"   AP: {nn_test_ap:.4f}")
    logger.info(f"   MRR: {nn_test_mrr:.4f}")
    logger.info(f"   Recall@10: {nn_test_recall_10:.4f}")
    logger.info(f"   Recall@50: {nn_test_recall_50:.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info("\n📊 Test Results (Old Nodes):")
    logger.info(f"   AUROC: {test_auc:.4f}")
    logger.info(f"   AP: {test_ap:.4f}")
    logger.info(f"   MRR: {test_mrr:.4f}")
    logger.info(f"   Recall@10: {test_recall_10:.4f}")
    logger.info(f"   Recall@50: {test_recall_50:.4f}")
    
    logger.info("\n📊 Test Results (New Nodes):")
    logger.info(f"   AUROC: {nn_test_auc:.4f}")
    logger.info(f"   AP: {nn_test_ap:.4f}")
    logger.info(f"   MRR: {nn_test_mrr:.4f}")
    logger.info(f"   Recall@10: {nn_test_recall_10:.4f}")
    logger.info(f"   Recall@50: {nn_test_recall_50:.4f}")

    # ================================================================
    # ✅ ÉTAPE 10: Temporal Validation (optional)
    # ================================================================
    if args.temporal_validation:
        # Restaurer la mémoire à l'état après validation avant de faire la validation temporelle
        if args.use_memory and val_memory_backup is not None:
            tgn.memory.restore_memory(val_memory_backup)

        temporal_validation(tgn, test_data, full_data, full_ngh_finder, args, logger, train_data, val_data)

    # ================================================================
    # ✅ ÉTAPE 11: Predictions & TechRank
    # ================================================================
    if args.run_techrank:
        logger.info("\n" + "="*70)
        logger.info("STEP 11: GENERATING PREDICTIONS FOR TECHRANK")
        logger.info("="*70)
        
        id_to_company, id_to_investor = load_mappings(args.mapping_dir, logger)
        
        # ⚠️ IMPORTANT: Restaurer la mémoire à l'état après validation
        # pour éviter les erreurs de timestamps
        if args.use_memory and val_memory_backup is not None:
            logger.info("Restoring memory to post-validation state for predictions...")
            tgn.memory.restore_memory(val_memory_backup)
        
        # Générer le graphe de prédictions
        logger.info("Generating prediction graph...")
        pred_graph, predictions, dict_companies, dict_investors = generate_predictions_and_graph(
            tgn, id_to_company, id_to_investor, full_data, train_data, args, logger, full_ngh_finder
        )
        
        # Sauvegarder le graphe et les top prédictions
        save_graph_and_top(pred_graph, predictions, dict_companies, dict_investors, 
                          args, logger, id_to_company, id_to_investor)
        
        # Diagnostic avant TechRank
        diagnostic_graph_before_techrank(pred_graph, dict_companies, dict_investors, logger)
        
        # Lancer TechRank
        run_techrank_analysis(pred_graph, dict_companies, dict_investors, logger)

    logger.info("\n✅ Evaluation complete!")

def temporal_validation(tgn, test_data, full_data, full_ngh_finder, args, logger, train_data, val_data):
    """
    Validation temporelle: divise le test set en deux parties et évalue les prédictions.

    ✅ CORRECTION DU DATA LEAKAGE:
    Cette fonction crée un neighbor finder qui contient SEULEMENT train + val + history
    (pas les liens futurs du test set). Cela évite que TGN puisse "voir le futur" lors
    des prédictions.

    Workflow:
    1. Split test set temporellement: HISTORY (50%) + FUTURE (50%)
    2. Créer history_ngh_finder = train + val + HISTORY (PAS FUTURE!)
    3. Mettre à jour mémoire TGN avec HISTORY en utilisant history_ngh_finder
    4. Générer prédictions à split_timestamp en utilisant history_ngh_finder
    5. Comparer prédictions avec FUTURE (ground truth)

    Args:
        tgn: Modèle TGN
        test_data: Données de test
        full_data: Données complètes
        full_ngh_finder: Neighbor finder (DEPRECATED - non utilisé, gardé pour compatibilité)
        args: Arguments
        logger: Logger
        train_data: Données d'entraînement (pour créer history_ngh_finder)
        val_data: Données de validation (pour créer history_ngh_finder)
    """
    logger.info("\n" + "="*70)
    logger.info("TEMPORAL VALIDATION")
    logger.info("="*70)

    # Récupérer les données du test set
    test_sources = np.array(test_data.sources)
    test_destinations = np.array(test_data.destinations)
    test_timestamps = np.array(test_data.timestamps)
    test_edge_idxs = np.array(test_data.edge_idxs)

    # Trier par timestamp
    sorted_idx = np.argsort(test_timestamps)
    test_sources = test_sources[sorted_idx]
    test_destinations = test_destinations[sorted_idx]
    test_timestamps = test_timestamps[sorted_idx]
    test_edge_idxs = test_edge_idxs[sorted_idx]

    # Calculer le point de split temporel
    split_idx = int(len(test_timestamps) * args.temporal_split)
    split_timestamp = test_timestamps[split_idx]

    logger.info(f"Test set: {len(test_timestamps)} interactions")
    logger.info(f"Split ratio: {args.temporal_split:.1%}")
    logger.info(f"Split timestamp: {split_timestamp:.2f}")
    logger.info(f"History (before split): {split_idx} interactions")
    logger.info(f"Ground truth (after split): {len(test_timestamps) - split_idx} interactions")

    # Partie 1: Historique (pour construire la mémoire)
    history_sources = test_sources[:split_idx]
    history_destinations = test_destinations[:split_idx]
    history_timestamps = test_timestamps[:split_idx]
    history_edge_idxs = test_edge_idxs[:split_idx]

    # Partie 2: Ground truth (vrais liens futurs)
    future_sources = test_sources[split_idx:]
    future_destinations = test_destinations[split_idx:]

    # Créer un set des vrais liens futurs
    true_future_links = set((int(s), int(d)) for s, d in zip(future_sources, future_destinations))
    logger.info(f"Unique true future links: {len(true_future_links)}")

    # Sauvegarder l'état de la mémoire actuel
    if args.use_memory:
        current_memory_backup = tgn.memory.backup_memory()

    # ================================================================
    # ✅ CORRECTION DU DATA LEAKAGE
    # ================================================================
    # Créer un neighbor finder qui contient SEULEMENT train + val + history
    # (PAS les liens futurs du test set!)
    logger.info("\n🔧 Creating history-only neighbor finder (fixing data leakage)...")

    # Combiner train + val + history
    train_val_length = len(train_data.sources) + len(val_data.sources)
    train_val_sources = full_data.sources[:train_val_length]
    train_val_destinations = full_data.destinations[:train_val_length]
    train_val_timestamps = full_data.timestamps[:train_val_length]
    train_val_edge_idxs = full_data.edge_idxs[:train_val_length]

    # Ajouter l'historique (concaténer les numpy arrays)
    history_data_sources = np.concatenate([train_val_sources, history_sources])
    history_data_destinations = np.concatenate([train_val_destinations, history_destinations])
    history_data_timestamps = np.concatenate([train_val_timestamps, history_timestamps])
    history_data_edge_idxs = np.concatenate([train_val_edge_idxs, history_edge_idxs])

    # Créer l'objet Data pour l'historique
    history_data = Data(
        sources=history_data_sources,
        destinations=history_data_destinations,
        timestamps=history_data_timestamps,
        edge_idxs=history_data_edge_idxs,
        labels=np.ones(len(history_data_sources))
    )

    # Créer le neighbor finder SANS les liens futurs
    history_ngh_finder = get_neighbor_finder(history_data, args.uniform)

    logger.info(f"✅ History neighbor finder created:")
    logger.info(f"   Train+Val+History edges: {len(history_data_sources)}")
    logger.info(f"   Full data edges: {len(full_data.sources)}")
    logger.info(f"   Excluded future edges: {len(full_data.sources) - len(history_data_sources)}")

    # Mettre à jour la mémoire avec l'historique en utilisant history_ngh_finder
    tgn.set_neighbor_finder(history_ngh_finder)  # ✅ Utiliser history_ngh_finder au lieu de full_ngh_finder
    if args.use_memory and len(history_sources) > 0:
        logger.info("Updating memory with history (using history_ngh_finder)...")
        for i in range(0, len(history_sources), args.bs):
            s = history_sources[i:i+args.bs]
            d = history_destinations[i:i+args.bs]
            ts = history_timestamps[i:i+args.bs]
            e = history_edge_idxs[i:i+args.bs]

            with torch.no_grad():
                _ = tgn.compute_temporal_embeddings(s, d, d, ts, e, args.n_degree)

    # Générer les prédictions
    logger.info("\nGenerating predictions (using history_ngh_finder)...")
    # ✅ IMPORTANT: TGN utilise maintenant history_ngh_finder (défini ligne 645)
    # Les prédictions ne peuvent PAS voir les liens futurs!

    # Identifier tous les nodes disponibles
    all_sources = set(full_data.sources)
    all_destinations = set(full_data.destinations)

    # Créer toutes les paires possibles
    all_company_ids = sorted(all_sources)
    all_investor_ids = sorted(all_destinations)

    predictions_list = []

    # Prédire par batches
    neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=99)

    logger.info(f"Predicting for {len(all_company_ids)} companies x {len(all_investor_ids)} investors...")

    total_pairs = 0
    for company_id in all_company_ids:
        # Batch de prédictions pour cette company avec tous les investors
        src_batch = np.full(len(all_investor_ids), company_id, dtype=np.int32)
        dst_batch = np.array(all_investor_ids, dtype=np.int32)
        times_batch = np.full(len(dst_batch), split_timestamp, dtype=np.float32)
        idx_batch = np.zeros(len(dst_batch), dtype=np.int32)

        # Negative samples (dummy)
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
                    total_pairs += 1

            except Exception as e:
                logger.warning(f"Error predicting for company {company_id}: {e}")
                continue

        if (total_pairs // len(all_investor_ids)) % 50 == 0:
            logger.info(f"  Predicted {total_pairs:,} pairs...")

    logger.info(f"Total predictions generated: {len(predictions_list):,}")

    # Trier les prédictions par probabilité décroissante
    predictions_list.sort(key=lambda x: x[2], reverse=True)

    # Calculer Precision@K pour différentes valeurs de K
    k_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    logger.info("\n" + "="*70)
    logger.info("PRECISION@K RESULTS")
    logger.info("="*70)
    logger.info(f"Prediction threshold: {args.prediction_threshold}")
    logger.info(f"Total true future links: {len(true_future_links)}")

    for k in k_values:
        if k > len(predictions_list):
            continue

        # Prendre les top K prédictions
        top_k_predictions = predictions_list[:k]

        # Filtrer par seuil de probabilité
        top_k_above_threshold = [(s, d, p) for s, d, p in top_k_predictions if p >= args.prediction_threshold]

        # Compter combien sont vrais
        true_positives = sum(1 for s, d, _ in top_k_above_threshold if (s, d) in true_future_links)

        # Calculer precision
        if len(top_k_above_threshold) > 0:
            precision = true_positives / len(top_k_above_threshold)
        else:
            precision = 0.0

        # Calculer aussi sans seuil pour comparaison
        true_positives_no_threshold = sum(1 for s, d, _ in top_k_predictions if (s, d) in true_future_links)
        precision_no_threshold = true_positives_no_threshold / k

        logger.info(f"\nPrecision@{k:4d}:")
        logger.info(f"  With threshold {args.prediction_threshold}: {precision:.4f} ({true_positives}/{len(top_k_above_threshold)} predictions)")
        logger.info(f"  Without threshold:  {precision_no_threshold:.4f} ({true_positives_no_threshold}/{k} predictions)")

    # Restaurer la mémoire
    if args.use_memory:
        tgn.memory.restore_memory(current_memory_backup)

    logger.info("\n✅ Temporal validation complete!")

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

    mask_after_train = edge_times > train_max_ts

    count_total = len(edge_times)
    count_after = mask_after_train.sum()
    logger.info("Edges total in full_data: %d, edges after train max timestamp: %d", count_total, int(count_after))

    return sources[mask_after_train], destinations[mask_after_train], edge_times[mask_after_train], edge_idxs[mask_after_train]

def generate_predictions_and_graph(tgn, id_to_company, id_to_investor, full_data, train_data, args, logger, full_ngh_finder):
    """
    Génère le graphe de prédictions UNIQUEMENT pour les liens FUTURS (après train).

    ✅ MÉTHODOLOGIE CORRECTE:
    1. Prendre TOUTES les paires possibles (company, investor)
    2. Calculer la probabilité pour chaque paire
    3. Construire un graphe avec les paires ayant une probabilité > seuil

    ⚠️ IMPORTANT: On ne doit PAS utiliser full_data car ça inclut les vrais liens!
    On doit prédire sur TOUTES les paires possibles, y compris celles qui n'existent pas.

    Args:
        tgn: Modèle TGN chargé
        id_to_company: Mapping ID -> nom de company
        id_to_investor: Mapping ID -> nom d'investisseur
        full_data: Données complètes (pour neighbor finder)
        train_data: Données d'entraînement (pour timestamp de prédiction)
        args: Arguments
        logger: Logger
        full_ngh_finder: Neighbor finder sur full data
    """

    logger.info(f"\n{'='*70}")
    logger.info(f"CONSTRUCTION DU GRAPHE DE PRÉDICTION")
    logger.info(f"{'='*70}")
    logger.info(f"✅ MÉTHODOLOGIE CORRECTE:")
    logger.info(f"   1. Prédire probabilités pour TOUTES les paires possibles")
    logger.info(f"   2. Construire graphe avec paires prob > seuil")
    logger.info(f"   3. Comparer avec graphe TEST réel pour validation")
    logger.info(f"\nConvention TGN:")
    logger.info(f"  - sources (u) = COMPANIES")
    logger.info(f"  - destinations (i) = INVESTORS")

    # Mettre le modèle en mode évaluation
    tgn.eval()
    tgn.set_neighbor_finder(full_ngh_finder)

    # Convention bipartite (doit être cohérente avec bipartite_investor_comp.py):
    # bipartite=0 => Companies (sources dans TGN)
    # bipartite=1 => Investors (destinations dans TGN)
    # Edges: Company → Investor (comme dans le graphe original)
    COMPANY_BIPARTITE = 0   # Companies ont bipartite=0
    INVESTOR_BIPARTITE = 1  # Investors ont bipartite=1

    # ================================================================
    # ÉTAPE 1: Identifier TOUTES les paires possibles
    # ================================================================
    logger.info(f"\n🔍 Étape 1: Identifier toutes les paires possibles")

    # ⚠️ IMPORTANT: Récupérer d'abord les IDs qui existent dans les données TGN
    all_sources = set(full_data.sources)
    all_destinations = set(full_data.destinations)

    logger.info(f"\nPlages d'IDs dans les données TGN:")
    logger.info(f"  Sources (companies): [{min(all_sources)}, {max(all_sources)}] ({len(all_sources)} uniques)")
    logger.info(f"  Destinations (investors): [{min(all_destinations)}, {max(all_destinations)}] ({len(all_destinations)} uniques)")

    # ⚠️ CRITIQUE: Ne garder que les IDs qui existent DANS LES DONNÉES TGN
    # Sinon on aura des IndexError quand le modèle essaiera d'accéder aux node_features
    company_ids = sorted([cid for cid in id_to_company.keys() if cid in all_sources])
    investor_ids = sorted([iid for iid in id_to_investor.keys() if iid in all_destinations])

    logger.info(f"\nIDs disponibles après filtrage TGN:")
    logger.info(f"   Companies (dans mappings): {len(id_to_company)}")
    logger.info(f"   Companies (dans TGN data): {len(company_ids)}")
    logger.info(f"   Investors (dans mappings): {len(id_to_investor)}")
    logger.info(f"   Investors (dans TGN data): {len(investor_ids)}")
    logger.info(f"   Total paires possibles: {len(company_ids) * len(investor_ids):,}")

    # ================================================================
    # ÉTAPE 2: Préparer les structures de données
    # ================================================================
    # ⚠️ CRITIQUE: Utiliser des dictionnaires SÉPARÉS car les IDs se chevauchent!
    # Les IDs des companies et des investors commencent tous deux à 0
    company_id_to_name = {}
    investor_id_to_name = {}

    # Sources = COMPANIES
    companies_mapped = 0
    for src_id in all_sources:
        src_id = int(src_id)
        if src_id in id_to_company:
            company_id_to_name[src_id] = id_to_company[src_id]
            companies_mapped += 1

    # Destinations = INVESTORS
    investors_mapped = 0
    for dst_id in all_destinations:
        dst_id = int(dst_id)
        if dst_id in id_to_investor:
            investor_id_to_name[dst_id] = id_to_investor[dst_id]
            investors_mapped += 1

    logger.info(f"\nNode mappings construits:")
    logger.info(f"  Companies (sources, bipartite=0): {companies_mapped}")
    logger.info(f"  Investors (destinations, bipartite=1): {investors_mapped}")
    
    # ================================================================
    # ÉTAPE 3: Générer TOUTES les paires possibles et prédire
    # ================================================================
    logger.info(f"\n🚀 Étape 3: Génération des prédictions pour TOUTES les paires")

    import networkx as nx

    # Timestamp de prédiction = fin du train set
    if len(train_data.timestamps) > 0:
        prediction_timestamp = float(np.max(train_data.timestamps))
    else:
        prediction_timestamp = float(np.max(full_data.timestamps))

    logger.info(f"   Timestamp de prédiction: {prediction_timestamp:.2f}")

    # Créer toutes les paires possibles
    all_pairs = []
    for company_id in company_ids:
        for investor_id in investor_ids:
            all_pairs.append((company_id, investor_id))

    logger.info(f"   Total paires à prédire: {len(all_pairs):,}")

    # Prédire par batches
    pred_graph = nx.Graph()
    predictions = []
    dict_companies = {}
    dict_investors = {}
    edge_funding_info = {}

    # Seuil de probabilité pour créer une arête
    # Utiliser args.prediction_threshold pour la cohérence
    PROBABILITY_THRESHOLD = args.prediction_threshold

    logger.info(f"   Probability threshold: {PROBABILITY_THRESHOLD}")
    logger.info(f"   Batch size: {args.bs}")
    logger.info(f"\n   Computing edge probabilities...")

    num_batches = (len(all_pairs) + args.bs - 1) // args.bs

    # Negative sampler (dummy, pas vraiment utilisé)
    neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)

    # Sauvegarder l'état de la mémoire pour éviter les erreurs de timestamp
    # Chaque batch va restaurer cette sauvegarde pour repartir du même état
    if args.use_memory:
        logger.info("   Backing up memory state before predictions...")
        memory_backup = tgn.memory.backup_memory()

    for batch_idx in range(num_batches):
        # Restaurer la mémoire pour ce batch (évite "time in the past" errors)
        if args.use_memory:
            tgn.memory.restore_memory(memory_backup)
        start_idx = batch_idx * args.bs
        end_idx = min(len(all_pairs), start_idx + args.bs)

        batch_pairs = all_pairs[start_idx:end_idx]
        src_batch = np.array([p[0] for p in batch_pairs], dtype=np.int32)
        dst_batch = np.array([p[1] for p in batch_pairs], dtype=np.int32)

        # Timestamps (tous identiques pour la prédiction)
        times_batch = np.full(len(src_batch), prediction_timestamp, dtype=np.float32)

        # Edge indices (dummy, pas utilisés pour l'inférence)
        idx_batch = np.zeros(len(src_batch), dtype=np.int32)

        # Negative samples (dummy)
        neg_tuple = neg_sampler.sample(len(src_batch))
        neg_batch = np.array(neg_tuple[1])
        if len(neg_batch.shape) > 1:
            neg_batch = neg_batch[:, 0]

        # ⚠️ IMPORTANT: torch.no_grad() pour ne pas mettre à jour la mémoire
        with torch.no_grad():
            try:
                pos_prob, _ = tgn.compute_edge_probabilities(
                    src_batch, dst_batch, neg_batch,
                    times_batch, idx_batch, args.n_degree
                )

                prob_scores = pos_prob.cpu().numpy()

                # Stocker les prédictions et construire le graphe
                for s, d, prob in zip(src_batch, dst_batch, prob_scores):
                    s = int(s)
                    d = int(d)
                    prob_score = float(prob)
                    timestamp = prediction_timestamp

                    # Ne garder que les arêtes avec probabilité > seuil
                    if prob_score < PROBABILITY_THRESHOLD:
                        continue

                    company_id = s
                    investor_id = d

                    # Récupérer les noms de base depuis les dictionnaires SÉPARÉS
                    # ⚠️ CRITIQUE: Ne PAS utiliser un seul dictionnaire car les IDs se chevauchent!
                    company_base_name = company_id_to_name.get(s, f"company_{s}")
                    investor_base_name = investor_id_to_name.get(d, f"investor_{d}")

                    # ⚠️ CRITIQUE: Préfixer les noms avec leur rôle pour éviter les collisions
                    # Car une même entité (ex: "Legend Capital") peut être à la fois company et investor
                    company_name = f"COMPANY_{company_base_name}"
                    investor_name = f"INVESTOR_{investor_base_name}"

                    if company_name not in dict_companies:
                        dict_companies[company_name] = {
                            'id': company_id,
                            'name': company_name,  # Nom avec préfixe (pour graphe)
                            'base_name': company_base_name,  # Nom sans préfixe (pour affichage)
                            'technologies': [],
                            'total_funding': 0.0,
                            'num_funding_rounds': 0
                        }

                    if investor_name not in dict_investors:
                        dict_investors[investor_name] = {
                            'investor_id': investor_id,
                            'name': investor_name,  # Nom avec préfixe (pour graphe)
                            'base_name': investor_base_name,  # Nom sans préfixe (pour affichage)
                            'num_investments': 0,
                            'total_invested': 0.0
                        }

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

                    dict_companies[company_name]['total_funding'] += prob_score
                    dict_companies[company_name]['num_funding_rounds'] += 1
                    dict_investors[investor_name]['num_investments'] += 1
                    dict_investors[investor_name]['total_invested'] += prob_score

                    pred_graph.add_node(company_name, bipartite=COMPANY_BIPARTITE)
                    pred_graph.add_node(investor_name, bipartite=INVESTOR_BIPARTITE)

                    predictions.append((s, d, prob_score))

            except AssertionError as e:
                logger.error(f"❌ AssertionError at batch {batch_idx}: {e}")
                logger.error(f"   Timestamps range: [{times_batch.min():.2f}, {times_batch.max():.2f}]")
                logger.error("   Skipping this batch...")
                continue

        if (batch_idx % 100) == 0:
            logger.info(f"      Processed {end_idx}/{len(all_pairs):,} pairs...")

    logger.info(f"\n✅ Prédictions terminées:")
    logger.info(f"   Total paires prédites: {len(all_pairs):,}")
    logger.info(f"   Arêtes retenues (prob > {PROBABILITY_THRESHOLD}): {len(edge_funding_info)}")
    logger.info(f"   Taux de rétention: {len(edge_funding_info)/len(all_pairs)*100:.2f}%")

    # ================================================================
    # ÉTAPE 4: Construire le graphe final
    # ================================================================
    logger.info(f"\n🔨 Étape 4: Construction du graphe final")

    # Add edges
    # ⚠️ IMPORTANT: Convention du graphe original (bipartite_investor_comp.py:934-935):
    # Edges vont de Company (bipartite=0) → Investor (bipartite=1)
    for (comp_name, inv_name), funding_info in edge_funding_info.items():
        pred_graph.add_edge(
            comp_name,     # Company (source, bipartite=0)
            inv_name,      # Investor (destination, bipartite=1)
            weight=funding_info['total_raised_amount_usd'],
            funding_rounds=funding_info['funding_rounds'],
            total_raised_amount_usd=funding_info['total_raised_amount_usd'],
            num_funding_rounds=funding_info['num_funding_rounds']
        )
    
    # Ensure all mapped nodes are in graph
    for nid, name in id_to_company.items():
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
    
    for nid, name in id_to_investor.items():
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

    # Restaurer l'état de la mémoire original
    if args.use_memory:
        logger.info("   Restoring original memory state...")
        tgn.memory.restore_memory(memory_backup)

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

def diagnostic_graph_before_techrank(pred_graph, dict_companies, dict_investors, logger):
    """Diagnostic du graphe avant TechRank"""
    import networkx as nx
    
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC DU GRAPHE AVANT TECHRANK")
    logger.info("="*70)
    
    nodes_0 = [n for n, d in pred_graph.nodes(data=True) if d.get('bipartite') == 0]
    nodes_1 = [n for n, d in pred_graph.nodes(data=True) if d.get('bipartite') == 1]
    
    logger.info(f"📊 Structure du graphe:")
    logger.info(f"   Total nodes: {pred_graph.number_of_nodes()}")
    logger.info(f"   Total edges: {pred_graph.number_of_edges()}")
    logger.info(f"   Companies (bipartite=0): {len(nodes_0)}")
    logger.info(f"   Investors (bipartite=1): {len(nodes_1)}")
    logger.info(f"   Dict companies: {len(dict_companies)}")
    logger.info(f"   Dict investors: {len(dict_investors)}")
    
    logger.info(f"\n📝 Exemples de noms:")
    logger.info(f"   Companies (bipartite=0): {nodes_0[:3]}")
    logger.info(f"   Investors (bipartite=1): {nodes_1[:3]}")
    
    isolated = list(nx.isolates(pred_graph))
    logger.info(f"\n🔍 Nœuds isolés: {len(isolated)}")
    if isolated:
        logger.info(f"   Exemples: {isolated[:5]}")
    
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
    
    if pred_graph.number_of_edges() > 0:
        sample_edges = list(pred_graph.edges(data=True))[:3]
        logger.info(f"\n🔗 Exemples d'arêtes:")
        for u, v, data in sample_edges:
            u_bipartite = pred_graph.nodes[u].get('bipartite', 'MISSING')
            v_bipartite = pred_graph.nodes[v].get('bipartite', 'MISSING')
            weight = data.get('weight', 0)
            logger.info(f"   {u} (bipartite={u_bipartite}) → {v} (bipartite={v_bipartite})")
            logger.info(f"      Weight: {weight:.6f}, Rounds: {data.get('num_funding_rounds', 0)}")
            
            if u_bipartite == v_bipartite:
                logger.error(f"      ❌ ERREUR: Les deux nœuds ont le même bipartite={u_bipartite}!")
            else:
                logger.info(f"      ✅ OK: Arête bipartite valide")
        
        edge_weights = [data.get('weight', 0) for u, v, data in pred_graph.edges(data=True)]
        logger.info(f"\n📊 Statistiques des poids d'arêtes:")
        logger.info(f"   Min: {min(edge_weights):.6f}")
        logger.info(f"   Max: {max(edge_weights):.6f}")
        logger.info(f"   Moyenne: {np.mean(edge_weights):.6f}")
        logger.info(f"   Arêtes avec poids > 0: {sum(1 for w in edge_weights if w > 0)}")

def run_techrank_analysis(pred_graph, dict_companies, dict_investors, logger):
    """Lance TechRank sur le graphe prédit"""
    logger.info("\n" + "="*70)
    logger.info("LANCEMENT DE TECHRANK SUR LE GRAPHE PRÉDIT")
    logger.info("="*70)
    
    companies_in_graph = set(n for n in pred_graph.nodes() if n in dict_companies)
    investors_in_graph = set(n for n in pred_graph.nodes() if n in dict_investors)

    logger.info(f"\n📊 Données pour TechRank:")
    logger.info(f"   Companies dans le graphe: {len(companies_in_graph)}")
    logger.info(f"   Investors dans le graphe: {len(investors_in_graph)}")
    logger.info(f"   Total edges: {pred_graph.number_of_edges()}")

    # ⚠️ CRITIQUE: Filtrer les dictionnaires pour ne garder QUE les nodes dans le graphe!
    # Sinon le mapping scores<->noms sera incorrect
    dict_companies_filtered = {name: dict_companies[name] for name in companies_in_graph}
    dict_investors_filtered = {name: dict_investors[name] for name in investors_in_graph}

    logger.info(f"\n🔧 Dictionnaires filtrés:")
    logger.info(f"   dict_companies: {len(dict_companies)} -> {len(dict_companies_filtered)}")
    logger.info(f"   dict_investors: {len(dict_investors)} -> {len(dict_investors_filtered)}")

    num_nodes = pred_graph.number_of_nodes()
    save_dir_classes = Path("savings/bipartite_invest_comp/classes")
    save_dir_networks = Path("savings/bipartite_invest_comp/networks")
    save_dir_classes.mkdir(parents=True, exist_ok=True)
    save_dir_networks.mkdir(parents=True, exist_ok=True)

    # Sauvegarder les dictionnaires FILTRÉS
    with open(save_dir_classes / f'dict_companies_{num_nodes}.pickle', 'wb') as f:
        pickle.dump(dict_companies_filtered, f)
    with open(save_dir_classes / f'dict_investors_{num_nodes}.pickle', 'wb') as f:
        pickle.dump(dict_investors_filtered, f)
    with open(save_dir_networks / f'bipartite_graph_{num_nodes}.gpickle', 'wb') as f:
        pickle.dump(pred_graph, f)
    
    logger.info(f"\n✓ Données sauvegardées pour TechRank (limit={num_nodes})")
    
    try:
        from code.TechRank import run_techrank
        
        logger.info("\n🚀 Lancement de TechRank...")
        logger.info(f"   Alpha: 0.8, Beta: -0.6")
        logger.info(f"   Companies: {len(dict_companies_filtered)}")
        logger.info(f"   Investors: {len(dict_investors_filtered)}")

        # ⚠️ IMPORTANT: run_techrank retourne (df_companies, df_investors, dict_comp, dict_investors)
        # Ordre corrigé pour correspondre à la convention: bipartite=0=Companies, bipartite=1=Investors
        # Passer les dictionnaires FILTRÉS pour avoir le bon mapping
        df_companies_rank, df_investors_rank, _, _ = run_techrank(
            num_comp=num_nodes,
            num_tech=num_nodes,
            flag_cybersecurity=False,
            alpha=0.8,
            beta=-0.6,
            do_plot=False,
            dict_investors=dict_investors_filtered,
            dict_comp=dict_companies_filtered,
            B=pred_graph
        )
        
        logger.info("\n✓ TechRank terminé avec succès!")
        
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
                    # Enlever le préfixe "INVESTOR_" pour l'affichage
                    display_name = row['final_configuration'].replace("INVESTOR_", "")
                    logger.info(f"   #{idx:2d} {display_name:40s} → Score: {row['techrank']:.6f}")
            else:
                logger.error("\n❌ TOUS les scores investors sont à zéro!")
        
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
                    # Enlever le préfixe "COMPANY_" pour l'affichage
                    display_name = row['final_configuration'].replace("COMPANY_", "")
                    logger.info(f"   #{idx:2d} {display_name:40s} → Score: {row['techrank']:.6f}")
            else:
                logger.error("\n❌ TOUS les scores companies sont à zéro!")
        
    except ImportError as e:
        logger.error(f"❌ Impossible d'importer TechRank: {e}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution de TechRank: {e}", exc_info=True)

if __name__ == "__main__":
    main()