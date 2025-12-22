import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import wandb  # AJOUT: Import wandb
import shutil
import pandas as pd
import json  # AJOUT: Pour sauvegarder les résultats en JSON

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from tqdm import tqdm
# from data.custom_loss import create_business_aware_loss  # AJOUT: Nouvelle loss
from focal_loss import FocalLoss  # AJOUT: Import Focal Loss
from dcl_loss import DCLLoss, build_degree_dict  # AJOUT: Import DCL Loss
from hybrid_loss import HybridFocalDCLLoss  # AJOUT: Import Hybrid Loss
from hard_negative_mining import HardNegativeSampler, build_adjacency_dict  # AJOUT: Import Hard Negative Mining

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='crunchbase')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs') 
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", 
                    choices=["graph_attention", "graph_sum", "identity", "time"], 
                    help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", 
                    choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", 
                    choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", 
                    help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, 
                    help='Dimensions of the memory for each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
# AJOUT: Arguments pour wandb
parser.add_argument('--use_wandb', action='store_true',
                    help='Whether to use Weights & Biases for logging')
parser.add_argument('--wandb_project', type=str, default='tgn-training',
                    help='Wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='Wandb entity (username or team)')

parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Use inverse-degree weighted loss')
parser.add_argument('--weight_alpha', type=float, default=1.0,
                    help='Exponent for inverse degree weighting (1.0=linear, 2.0=quadratic)')
# AJOUT: Arguments pour Focal Loss
parser.add_argument('--use_focal_loss', action='store_true',
                    help='Use Focal Loss instead of BCE')
parser.add_argument('--focal_alpha', type=float, default=0.25,
                    help='Alpha parameter for Focal Loss (default: 0.25)')
parser.add_argument('--focal_gamma', type=float, default=2.0,
                    help='Gamma parameter for Focal Loss (default: 2.0)')
# AJOUT: Arguments pour DCL Loss
parser.add_argument('--use_dcl_loss', action='store_true',
                    help='Use DCL (Degree Contrastive Loss) to mitigate degree bias')
parser.add_argument('--dcl_temperature', type=float, default=0.07,
                    help='Temperature parameter for DCL Loss (default: 0.07)')
parser.add_argument('--dcl_alpha', type=float, default=0.5,
                    help='Degree reweighting exponent for DCL Loss (default: 0.5)')
# AJOUT: Arguments pour Hard Negative Mining
parser.add_argument('--use_hard_negatives', action='store_true',
                    help='Use hard negative mining instead of random sampling')
parser.add_argument('--hard_neg_ratio', type=float, default=0.5,
                    help='Ratio of hard negatives (0.0=all random, 1.0=all hard, default: 0.5)')
parser.add_argument('--hard_neg_temperature', type=float, default=0.1,
                    help='Temperature for hard negative sampling (lower=more aggressive, default: 0.1)')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

# Configuration des variables globales
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA, different_new_nodes_between_val_and_test=args.different_new_nodes, 
                              randomize_features=args.randomize_features)

# Initialize neighbor finders
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# Initialize hard negative sampler if needed
if args.use_hard_negatives:
  logger.info(f"Using Hard Negative Mining with ratio={args.hard_neg_ratio}, temperature={args.hard_neg_temperature}")
  hard_neg_sampler = HardNegativeSampler(ratio=args.hard_neg_ratio, temperature=args.hard_neg_temperature)
  # Build adjacency dict for training data
  train_adjacency_dict = build_adjacency_dict(train_data.sources, train_data.destinations)
else:
  hard_neg_sampler = None
  train_adjacency_dict = None

# Build degree dictionary if DCL Loss or Hybrid Loss is used
if args.use_dcl_loss or (args.use_focal_loss and args.use_dcl_loss):
  logger.info("Building degree dictionary for DCL/Hybrid Loss...")
  degree_dict = build_degree_dict(train_data)
  logger.info(f"Degree dictionary built: {len(degree_dict)} nodes")
  # Convert to tensor for faster lookup during training
  max_node_id = max(degree_dict.keys())
  degree_tensor = torch.zeros(max_node_id + 1, dtype=torch.float32)
  for node_id, degree in degree_dict.items():
    degree_tensor[node_id] = degree
else:
  degree_tensor = None

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  # AJOUT: Initialiser wandb pour chaque run
  if args.use_wandb:
    run_name = f"{args.prefix}_{args.data}_run{i}" if args.prefix else f"{args.data}_run{i}"
    wandb.init(
      project=args.wandb_project,
      entity=args.wandb_entity,
      name=run_name,
      config={
        "dataset": args.data,
        "batch_size": BATCH_SIZE,
        "num_neighbors": NUM_NEIGHBORS,
        "num_epochs": NUM_EPOCH,
        "num_heads": NUM_HEADS,
        "dropout": DROP_OUT,
        "num_layers": NUM_LAYER,
        "learning_rate": LEARNING_RATE,
        "node_dim": NODE_DIM,
        "time_dim": TIME_DIM,
        "use_memory": USE_MEMORY,
        "message_dim": MESSAGE_DIM,
        "memory_dim": MEMORY_DIM,
        "embedding_module": args.embedding_module,
        "message_function": args.message_function,
        "memory_updater": args.memory_updater,
        "aggregator": args.aggregator,
        "use_focal_loss": args.use_focal_loss,
        "focal_alpha": args.focal_alpha if args.use_focal_loss else None,
        "focal_gamma": args.focal_gamma if args.use_focal_loss else None,
        "use_dcl_loss": args.use_dcl_loss,
        "dcl_temperature": args.dcl_temperature if args.use_dcl_loss else None,
        "dcl_alpha": args.dcl_alpha if args.use_dcl_loss else None,
        "use_hard_negatives": args.use_hard_negatives,
        "hard_neg_ratio": args.hard_neg_ratio if args.use_hard_negatives else None,
        "hard_neg_temperature": args.hard_neg_temperature if args.use_hard_negatives else None,
        "run": i
      },
      reinit=True
    )
  
  # Déterminer le nom de la loss function pour nommer les fichiers
  loss_name = "bce"  # Default
  if args.use_focal_loss and args.use_dcl_loss:
    loss_name = "hybrid"
  elif args.use_dcl_loss:
    loss_name = "dcl"
  elif args.use_focal_loss:
    loss_name = "focal"

  # Créer des noms de fichiers identifiants la loss
  prefix_with_loss = f"{args.prefix}_{loss_name}" if args.prefix else loss_name
  results_path = "results/{}_{}.pkl".format(prefix_with_loss, i) if i > 0 else "results/{}.pkl".format(prefix_with_loss)
  results_json_path = "results/{}_{}.json".format(prefix_with_loss, i) if i > 0 else "results/{}.json".format(prefix_with_loss)

  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  
  # ================================================================
  # CONFIGURATION DE LA LOSS FUNCTION
  # ================================================================

  # ⚠️ IMPORTANT: Choisir la loss function à utiliser

  if args.use_focal_loss and args.use_dcl_loss:
    # ✅ HYBRID LOSS: Combine Focal Loss (class imbalance) + DCL (degree bias)
    logger.info(f"Using HYBRID Focal-DCL Loss:")
    logger.info(f"  - Focal: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    logger.info(f"  - DCL: alpha={args.dcl_alpha}")
    criterion = HybridFocalDCLLoss(
      focal_gamma=args.focal_gamma,
      focal_alpha=args.focal_alpha,
      dcl_alpha=args.dcl_alpha,
      lambda_focal=0.5,  # Balanced: 50% Focal, 50% DCL
      reduction='mean'
    )
    # Move degree tensor to device
    if degree_tensor is not None:
      degree_tensor = degree_tensor.to(device)
  elif args.use_dcl_loss:
    # ✅ DCL LOSS: Hardness Adaptive Reweighted Loss pour mitiger le degree bias
    logger.info(f"Using DCL Loss with temperature={args.dcl_temperature}, alpha={args.dcl_alpha}")
    criterion = DCLLoss(temperature=args.dcl_temperature, alpha=args.dcl_alpha, reduction='mean')
    # Move degree tensor to device
    if degree_tensor is not None:
      degree_tensor = degree_tensor.to(device)
  elif args.use_focal_loss:
    # ✅ FOCAL LOSS: Pour gérer le déséquilibre de classes
    logger.info(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')
  else:
    # 📌 BASELINE: Binary Cross-Entropy (BCE)
    logger.info("Using standard Binary Cross-Entropy Loss")
    criterion = torch.nn.BCELoss()

  # Alternative: Weighted Loss (commentée pour l'instant)
  # from data.custom_loss import load_weighted_loss
  # with open(f"data/mappings/{DATA}_filtered_company_id_map.pickle", "rb") as f:
  #     item_map = pickle.load(f)
  # criterion = load_weighted_loss(
  #     data_name=DATA,
  #     item_map=item_map,
  #     alpha=1.0,      # 1.0 = linéaire, 2.0 = quadratique (plus agressif)
  #     normalize=False
  # )

  criterion = criterion.to(device)  # Déplacer sur GPU si nécessaire
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='max', factor=0.5, patience=3, verbose=True
  )

  tgn = tgn.to(device)

  # AJOUT: Watch model avec wandb
  if args.use_wandb:
    wandb.watch(tgn, log="all", log_freq=100)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []
  val_mrrs = []
  val_recall_10s = []
  val_recall_50s = []
  new_nodes_val_mrrs = []
  new_nodes_val_recall_10s = []
  new_nodes_val_recall_50s = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  
  # for epoch in range(NUM_EPOCH):
  for epoch in tqdm(range(NUM_EPOCH), desc='Training Progress', position=0):
  
    start_epoch = time.time()

    ### Training
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))

    # Barre de progression pour les batches
    # pbar = tqdm(range(0, num_batch, args.backprop_every), 
    #             desc=f'Epoch {epoch}/{NUM_EPOCH}',
    #             total=math.ceil(num_batch / args.backprop_every))
    
    for k in range(0, num_batch, args.backprop_every):
    # for k in pbar:
      loss = 0
      optimizer.zero_grad()

      for j in range(args.backprop_every):
        batch_idx = k + j
        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch = train_data.sources[start_idx:end_idx]
        destinations_batch = train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)

        # Sample negatives: hard or random
        if args.use_hard_negatives:
          # Use hard negative mining
          node_features_np = node_features  # Already numpy array
          dst_negatives_batch = hard_neg_sampler.sample(
            sources=sources_batch,
            destinations=destinations_batch,
            embeddings=node_features_np,
            adjacency_dict=train_adjacency_dict,
            n_negatives=1
          ).flatten()
        else:
          # Use random sampling
          _, dst_negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        pos_prob, neg_prob = tgn.compute_edge_probabilities(
          sources_batch, destinations_batch, dst_negatives_batch,
          timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        # Compute loss based on criterion type
        if args.use_focal_loss and args.use_dcl_loss:
          # HYBRID LOSS: Needs both probabilities and degrees
          src_degrees = degree_tensor[sources_batch]
          dst_degrees_pos = degree_tensor[destinations_batch]
          dst_degrees_neg = degree_tensor[dst_negatives_batch]

          loss += criterion(pos_prob.squeeze(), neg_prob.squeeze(),
                           src_degrees, dst_degrees_pos, dst_degrees_neg,
                           pos_label, neg_label)
        elif args.use_dcl_loss:
          # DCL Loss needs degrees of source and destination nodes
          # Get degrees for positive pairs
          src_degrees = degree_tensor[sources_batch]
          dst_degrees_pos = degree_tensor[destinations_batch]
          # Get degrees for negative pairs
          dst_degrees_neg = degree_tensor[dst_negatives_batch]

          # DCL Loss expects scores, so convert probs back to scores (logit)
          # score = log(p / (1-p))
          pos_scores = torch.log(pos_prob.squeeze() / (1 - pos_prob.squeeze() + 1e-7))
          neg_scores = torch.log(neg_prob.squeeze() / (1 - neg_prob.squeeze() + 1e-7))

          loss += criterion(pos_scores, neg_scores, src_degrees, dst_degrees_pos, dst_degrees_neg)
        else:
          # Standard BCE or Focal Loss
          loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
        # sources_tensor = torch.from_numpy(sources_batch).long().to(device)

        # if isinstance(criterion, InverseDegreeWeightedBCELoss):
        #     from data.custom_loss import InverseDegreeWeightedBCELoss
        #     pos_loss = criterion(pos_prob.squeeze(), pos_label, sources_tensor)
        #     neg_loss = criterion(neg_prob.squeeze(), neg_label, sources_tensor)
        #     loss += pos_loss + neg_loss
        # else:
        #     loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every
      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Mise à jour de la barre avec la loss actuelle
      # pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{np.mean(m_loss):.4f}'})

      # AJOUT: Log batch loss à wandb
      if args.use_wandb:
        wandb.log({"batch_loss": loss.item(), "batch": k})

      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc, val_mrr, val_recall_10, val_recall_50 = eval_edge_prediction(
      model=tgn, negative_edge_sampler=val_rand_sampler,
      data=val_data, n_neighbors=NUM_NEIGHBORS)
    
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      tgn.memory.restore_memory(train_memory_backup)

    nn_val_ap, nn_val_auc, nn_val_mrr, nn_val_recall_10, nn_val_recall_50 = eval_edge_prediction(
      model=tgn, negative_edge_sampler=val_rand_sampler,
      data=new_node_val_data, n_neighbors=NUM_NEIGHBORS)
    
    if USE_MEMORY:
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))
    val_mrrs.append(val_mrr)
    val_recall_10s.append(val_recall_10)
    val_recall_50s.append(val_recall_50)
    new_nodes_val_mrrs.append(nn_val_mrr)
    new_nodes_val_recall_10s.append(nn_val_recall_10)
    new_nodes_val_recall_50s.append(nn_val_recall_50)
    # Scheduler step
    scheduler.step(val_ap)  # tu peux aussi utiliser val_auc si tu préfères

    # AJOUT: Log toutes les métriques à wandb
    if args.use_wandb:
      wandb.log({
        "epoch": epoch,
        "train_loss": np.mean(m_loss),
        "epoch_time": epoch_time,
        "val_ap": val_ap,
        "val_auc": val_auc,
        "val_mrr": val_mrr,
        "val_recall@10": val_recall_10,
        "val_recall@50": val_recall_50,
        "new_nodes_val_ap": nn_val_ap,
        "new_nodes_val_auc": nn_val_auc,
        "new_nodes_val_mrr": nn_val_mrr,
        "new_nodes_val_recall@10": nn_val_recall_10,
        "new_nodes_val_recall@50": nn_val_recall_50
      })

    pickle.dump({
      "val_aps": val_aps,
      "val_mrrs": val_mrrs,
      "val_recall_10s": val_recall_10s,
      "val_recall_50s": val_recall_50s,
      "new_nodes_val_aps": new_nodes_val_aps,
      "new_nodes_val_mrrs": new_nodes_val_mrrs,
      "new_nodes_val_recall_10s": new_nodes_val_recall_10s,
      "new_nodes_val_recall_50s": new_nodes_val_recall_50s,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    logger.info('val mrr: {:.4f}, new node val mrr: {:.4f}'.format(val_mrr, nn_val_mrr))
    logger.info('val recall@10: {:.4f}, new node val recall@10: {:.4f}'.format(
        val_recall_10, nn_val_recall_10))
    logger.info('val recall@50: {:.4f}, new node val recall@50: {:.4f}'.format(
        val_recall_50, nn_val_recall_50))

    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_mrr, test_recall_10, test_recall_50 = eval_edge_prediction(
    model=tgn, negative_edge_sampler=test_rand_sampler,
    data=test_data, n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  nn_test_ap, nn_test_auc, nn_test_mrr, nn_test_recall_10, nn_test_recall_50 = eval_edge_prediction(
    model=tgn, negative_edge_sampler=nn_test_rand_sampler,
    data=new_node_test_data, n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}, mrr: {:.4f}, recall@10: {:.4f}, recall@50: {:.4f}'.format(
      test_auc, test_ap, test_mrr, test_recall_10, test_recall_50))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}, mrr: {:.4f}, recall@10: {:.4f}, recall@50: {:.4f}'.format(
      nn_test_auc, nn_test_ap, nn_test_mrr, nn_test_recall_10, nn_test_recall_50))
  
  # AJOUT: Log résultats finaux de test à wandb
  if args.use_wandb:
    wandb.log({
      "test_ap": test_ap,
      "test_auc": test_auc,
      "test_mrr": test_mrr,
      "test_recall@10": test_recall_10,
      "test_recall@50": test_recall_50,
      "new_nodes_test_ap": nn_test_ap,
      "new_nodes_test_auc": nn_test_auc,
      "new_nodes_test_mrr": nn_test_mrr,
      "new_nodes_test_recall@10": nn_test_recall_10,
      "new_nodes_test_recall@50": nn_test_recall_50
    })
    
    # Sauvegarder le modèle dans wandb (avec policy=now pour Windows)
    # wandb.save(MODEL_SAVE_PATH, policy="now", base_path=None)
    # shutil.copy(MODEL_SAVE_PATH, wandb.run.dir)
    # wandb.save("saved_models/tgn-attn-crunchbase.pth")

  
  # Save results (pickle format)
  results_dict = {
    "val_aps": val_aps,
    "val_mrrs": val_mrrs,
    "val_recall_10s": val_recall_10s,
    "val_recall_50s": val_recall_50s,
    "new_nodes_val_aps": new_nodes_val_aps,
    "new_nodes_val_mrrs": new_nodes_val_mrrs,
    "new_nodes_val_recall_10s": new_nodes_val_recall_10s,
    "new_nodes_val_recall_50s": new_nodes_val_recall_50s,
    "test_ap": test_ap,
    "test_mrr": test_mrr,
    "test_recall_10": test_recall_10,
    "test_recall_50": test_recall_50,
    "new_node_test_ap": nn_test_ap,
    "new_node_test_mrr": nn_test_mrr,
    "new_node_test_recall_10": nn_test_recall_10,
    "new_node_test_recall_50": nn_test_recall_50,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }

  pickle.dump(results_dict, open(results_path, "wb"))

  # AJOUT: Sauvegarder aussi en JSON pour faciliter la lecture et la comparaison
  results_json = {
    "loss_function": loss_name,
    "config": {
      "focal_alpha": args.focal_alpha if args.use_focal_loss else None,
      "focal_gamma": args.focal_gamma if args.use_focal_loss else None,
      "dcl_alpha": args.dcl_alpha if args.use_dcl_loss else None,
      "dcl_temperature": args.dcl_temperature if args.use_dcl_loss else None,
    },
    "validation": {
      "ap": val_aps,
      "mrr": val_mrrs,
      "recall_10": val_recall_10s,
      "recall_50": val_recall_50s,
    },
    "test": {
      "ap": float(test_ap),
      "auc": float(test_auc),
      "mrr": float(test_mrr),
      "recall_10": float(test_recall_10),
      "recall_50": float(test_recall_50),
    },
    "new_nodes_test": {
      "ap": float(nn_test_ap),
      "auc": float(nn_test_auc),
      "mrr": float(nn_test_mrr),
      "recall_10": float(nn_test_recall_10),
      "recall_50": float(nn_test_recall_50),
    },
    "training": {
      "losses": train_losses,
      "epoch_times": epoch_times,
    }
  }

  with open(results_json_path, 'w') as f:
    json.dump(results_json, f, indent=2)

  logger.info(f'Results saved to {results_path} and {results_json_path}')

  logger.info('Saving TGN model')
  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')
  
  # AJOUT: Terminer le run wandb
  if args.use_wandb:
    wandb.finish()

def extract_features(B, edges):
    """Build features for edges: ONLY temporal features (no degrees)."""
    X, y = [], []
    for u, v, label, edge_ts in edges:
        # NO DEGREES — only temporal features
        total_raised_raw = 0.0
        num_rounds = 0
        edge_data = B.get_edge_data(u, v)
        if edge_data:
            for fr in edge_data.get("funding_rounds", []):
                ann = fr.get("announced_on", None)
                if not ann:
                    continue
                try:
                    ann_ts = pd.to_datetime(ann).timestamp()
                except:
                    continue
                if ann_ts <= edge_ts:
                    raised = fr.get("raised_amount_usd", fr.get("raised_amount", 0)) or 0
                    try:
                        total_raised_raw += float(raised)
                    except:
                        continue
                    num_rounds += 1

        total_raised = np.log1p(total_raised_raw)
        # Only 2 features now: [total_raised, num_rounds]
        feat = [total_raised, num_rounds]
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)


def compute_mrr_recall_at_k(B, model, test_edges, k_list=[10, 50]):
    """Ranking: for each investor, compute features for all candidate companies (NO DEGREES)."""
    inv_to_true = {}
    for u, v, label, first_ts in test_edges:
        inv_to_true.setdefault(v, []).append((u, first_ts))

    nodes_comp = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]

    MRRs = []
    recalls = {k: [] for k in k_list}

    for inv, true_list in tqdm(inv_to_true.items(), desc="Computing MRR/Recall"):
        cutoff_ts = min(ts for _, ts in true_list)

        X_candidates = []
        comps_list = []
        for comp in nodes_comp:
            # NO DEGREES — only temporal features
            edge_data = B.get_edge_data(comp, inv)

            total_raised_raw = 0.0
            num_rounds = 0
            if edge_data:
                for fr in edge_data.get("funding_rounds", []):
                    ann = fr.get("announced_on", None)
                    if not ann:
                        continue
                    try:
                        ann_ts = pd.to_datetime(ann).timestamp()
                    except:
                        continue
                    if ann_ts <= cutoff_ts:
                        raised = fr.get("raised_amount_usd", fr.get("raised_amount", 0)) or 0
                        try:
                            total_raised_raw += float(raised)
                        except:
                            continue
                        num_rounds += 1

            total_raised = np.log1p(total_raised_raw)
            # Only 2 features: [total_raised, num_rounds]
            feat = [total_raised, num_rounds]
            X_candidates.append(feat)
            comps_list.append(comp)

        scores = model.predict_proba(np.array(X_candidates))[:, 1]
        ranking = [x for _, x in sorted(zip(scores, comps_list), reverse=True)]

        true_comps = [tc for tc, _ in true_list]
        ranks = [ranking.index(tc) + 1 for tc in true_comps if tc in ranking]
        MRRs.append(1.0 / min(ranks) if ranks else 0.0)

        for k in k_list:
            top_k = set(ranking[:k])
            hits = len(set(true_comps) & top_k)
            recalls[k].append(hits / len(true_comps))

    return np.mean(MRRs), {k: np.mean(v) for k, v in recalls.items()}

# print(f"\nFeature Importances (temporal only): {rf.feature_importances_}")
# print(f"Feature names: ['total_raised', 'num_rounds']")