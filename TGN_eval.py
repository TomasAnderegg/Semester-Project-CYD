import argparse
import logging
import torch
import numpy as np
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument parser
parser = argparse.ArgumentParser('TGN model evaluation')
parser.add_argument('-d', '--data', type=str, help='Dataset name',
                    default='crunchbase')
parser.add_argument('--model_path', type=str, help='Path to saved model',
                    default='./saved_models/-crunchbase.pth')
parser.add_argument('--prefix', type=str, default='', help='Prefix used during training')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
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

args = parser.parse_args()

# Parameters
DATA = args.data
NUM_NEIGHBORS = args.n_degree
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
NUM_LAYER = args.n_layer
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

### Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info(f"Evaluating model on dataset: {DATA}")
logger.info(f"Model path: {args.model_path}")

### Load data
node_features, edge_features, full_data, train_data, val_data, test_data, \
new_node_val_data, new_node_test_data = get_data(
    DATA,
    different_new_nodes_between_val_and_test=args.different_new_nodes,
    randomize_features=args.randomize_features
)

# Initialize neighbor finder with full data
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize negative samplers with same seeds as training
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(
    new_node_val_data.sources, new_node_val_data.destinations, seed=1
)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(
    new_node_test_data.sources, new_node_test_data.destinations, seed=3
)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
logger.info(f"Using device: {device}")

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

### Initialize model with same architecture as training
tgn = TGN(
    neighbor_finder=train_ngh_finder,
    node_features=node_features,
    edge_features=edge_features,
    device=device,
    n_layers=NUM_LAYER,
    n_heads=NUM_HEADS,
    dropout=DROP_OUT,
    use_memory=USE_MEMORY,
    message_dimension=MESSAGE_DIM,
    memory_dimension=MEMORY_DIM,
    memory_update_at_start=False,
    embedding_module_type=args.embedding_module,
    message_function=args.message_function,
    aggregator_type=args.aggregator,
    memory_updater_type=args.memory_updater,
    n_neighbors=NUM_NEIGHBORS,
    mean_time_shift_src=mean_time_shift_src,
    std_time_shift_src=std_time_shift_src,
    mean_time_shift_dst=mean_time_shift_dst,
    std_time_shift_dst=std_time_shift_dst,
    use_destination_embedding_in_message=args.use_destination_embedding_in_message,
    use_source_embedding_in_message=args.use_source_embedding_in_message,
    dyrep=args.dyrep
)

### Load trained model weights
if Path(args.model_path).exists():
    logger.info(f"Loading model from {args.model_path}")
    tgn.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.info("Model loaded successfully")
else:
    logger.error(f"Model file not found: {args.model_path}")
    exit(1)

tgn = tgn.to(device)
tgn.eval()

### Evaluation on different datasets

logger.info("\n" + "="*50)
logger.info("Starting Evaluation")
logger.info("="*50)

# If using memory, initialize it
if USE_MEMORY:
    tgn.memory.__init_memory__()

# Set neighbor finder to full graph
tgn.set_neighbor_finder(full_ngh_finder)

# 1. Validation set (seen nodes)
logger.info("\n[1/4] Evaluating on Validation Set (Seen Nodes)...")
val_ap, val_auc = eval_edge_prediction(
    model=tgn,
    negative_edge_sampler=val_rand_sampler,
    data=val_data,
    n_neighbors=NUM_NEIGHBORS
)
logger.info(f"Validation - AUC: {val_auc:.4f}, AP: {val_ap:.4f}")

if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

# 2. Validation set (new nodes)
logger.info("\n[2/4] Evaluating on Validation Set (New Nodes)...")
if USE_MEMORY:
    tgn.memory.__init_memory__()
    
nn_val_ap, nn_val_auc = eval_edge_prediction(
    model=tgn,
    negative_edge_sampler=nn_val_rand_sampler,
    data=new_node_val_data,
    n_neighbors=NUM_NEIGHBORS
)
logger.info(f"New Node Validation - AUC: {nn_val_auc:.4f}, AP: {nn_val_ap:.4f}")

if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

# 3. Test set (seen nodes)
logger.info("\n[3/4] Evaluating on Test Set (Seen Nodes)...")
test_ap, test_auc = eval_edge_prediction(
    model=tgn,
    negative_edge_sampler=test_rand_sampler,
    data=test_data,
    n_neighbors=NUM_NEIGHBORS
)
logger.info(f"Test - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

if USE_MEMORY:
    test_memory_backup = tgn.memory.backup_memory()

# 4. Test set (new nodes)
logger.info("\n[4/4] Evaluating on Test Set (New Nodes)...")
if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)
    
nn_test_ap, nn_test_auc = eval_edge_prediction(
    model=tgn,
    negative_edge_sampler=nn_test_rand_sampler,
    data=new_node_test_data,
    n_neighbors=NUM_NEIGHBORS
)
logger.info(f"New Node Test - AUC: {nn_test_auc:.4f}, AP: {nn_test_ap:.4f}")

### Summary
logger.info("\n" + "="*50)
logger.info("EVALUATION SUMMARY")
logger.info("="*50)
logger.info("\nSeen Nodes (Transductive):")
logger.info(f"  Validation - AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
logger.info(f"  Test       - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
logger.info("\nNew Nodes (Inductive):")
logger.info(f"  Validation - AUC: {nn_val_auc:.4f}, AP: {nn_val_ap:.4f}")
logger.info(f"  Test       - AUC: {nn_test_auc:.4f}, AP: {nn_test_ap:.4f}")
logger.info("="*50)

### Save results
results = {
    'val_auc': val_auc,
    'val_ap': val_ap,
    'nn_val_auc': nn_val_auc,
    'nn_val_ap': nn_val_ap,
    'test_auc': test_auc,
    'test_ap': test_ap,
    'nn_test_auc': nn_test_auc,
    'nn_test_ap': nn_test_ap,
}

# Print in a format easy to copy-paste
logger.info("\nResults (CSV format):")
logger.info("metric,seen_val,seen_test,new_val,new_test")
logger.info(f"AUC,{val_auc:.4f},{test_auc:.4f},{nn_val_auc:.4f},{nn_test_auc:.4f}")
logger.info(f"AP,{val_ap:.4f},{test_ap:.4f},{nn_val_ap:.4f},{nn_test_ap:.4f}")