import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import networkx as nx

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument parser
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
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
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
parser.add_argument('--model_path', type=str, default='', help='Path to the trained model')


args = parser.parse_args()

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

# Device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(f"Evaluating TGN model on dataset: {args.data}")

# Load data
node_features, edge_features, full_data, train_data, val_data, test_data, \
new_node_val_data, new_node_test_data = get_data(
    args.data,
    different_new_nodes_between_val_and_test=args.different_new_nodes,
    randomize_features=args.randomize_features
)

# Neighbor finders
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Negative samplers
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Initialize model
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,#False,#not args.memory_update_at_end, ajoute par Moi lol
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

# Load model weights
if Path(args.model_path).exists():
    logger.info(f"Loading model from {args.model_path}")
    tgn.load_state_dict(torch.load(args.model_path, map_location=device))
else:
    logger.error(f"Model file not found: {args.model_path}")
    exit(1)

tgn.to(device)
tgn.eval()

# ---------------- Evaluation ---------------- #
logger.info("Starting evaluation...")

# if args.use_memory:
#     tgn.memory.__init_memory__()

# 1️⃣ Validation - Seen nodes
tgn.set_neighbor_finder(full_ngh_finder)
val_ap, val_auc = eval_edge_prediction(tgn, val_rand_sampler, val_data, args.n_degree)
logger.info(f"Validation (seen nodes) - AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
if args.use_memory:
    val_memory_backup = tgn.memory.backup_memory()

# 2️⃣ Validation - New nodes
tgn.set_neighbor_finder(train_ngh_finder)  # Important: only neighbors seen in train
# if args.use_memory:
#     tgn.memory.__init_memory__()
nn_val_ap, nn_val_auc = eval_edge_prediction(tgn, nn_val_rand_sampler, new_node_val_data, args.n_degree)
logger.info(f"Validation (new nodes) - AUC: {nn_val_auc:.4f}, AP: {nn_val_ap:.4f}")
if args.use_memory:
    tgn.memory.restore_memory(val_memory_backup)

# 3️⃣ Test - Seen nodes
tgn.set_neighbor_finder(full_ngh_finder)
test_ap, test_auc = eval_edge_prediction(tgn, test_rand_sampler, test_data, args.n_degree)
logger.info(f"Test (seen nodes) - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
if args.use_memory:
    test_memory_backup = tgn.memory.backup_memory()

# 4️⃣ Test - New nodes
tgn.set_neighbor_finder(train_ngh_finder)
if args.use_memory:
    tgn.memory.restore_memory(val_memory_backup)
nn_test_ap, nn_test_auc = eval_edge_prediction(tgn, nn_test_rand_sampler, new_node_test_data, args.n_degree)
logger.info(f"Test (new nodes) - AUC: {nn_test_auc:.4f}, AP: {nn_test_ap:.4f}")

# ---------------- Summary ---------------- #
logger.info("Generating predicted graph...")

# Création du graphe (graph non orienté pour TGN, orienté si dyrep)
pred_graph = nx.DiGraph() if args.dyrep else nx.Graph()

sources = full_data.sources
destinations = full_data.destinations
edge_times = full_data.timestamps
edge_idxs = full_data.edge_idxs

# Sampler négatif avec 1 négatif par arête
neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)

batch_size = 200
num_edges = len(sources)

for start in range(0, num_edges, batch_size):
    end = min(start + batch_size, num_edges)

    src_batch = sources[start:end]
    dst_batch = destinations[start:end]
    times_batch = edge_times[start:end]
    idx_batch = edge_idxs[start:end]

    # Negative sampling (1 négatif par source)
    neg_tuple = neg_sampler.sample(len(src_batch))  # retourne un tuple
    neg_batch = neg_tuple[1]  # on prend les destinations négatives

    # s'assurer que c'est 1D
    if isinstance(neg_batch, list):
        neg_batch = np.array(neg_batch)
    elif len(neg_batch.shape) > 1:
        neg_batch = neg_batch[:, 0]

    # Conversion en tenseurs CPU Long
    src_tensor = torch.tensor(src_batch, dtype=torch.long, device='cpu')
    dst_tensor = torch.tensor(dst_batch, dtype=torch.long, device='cpu')
    neg_tensor = torch.tensor(neg_batch, dtype=torch.long, device='cpu')
    times_tensor = torch.tensor(times_batch, dtype=torch.long, device='cpu')
    idx_tensor = torch.tensor(idx_batch, dtype=torch.long, device='cpu')

    # Calcul des probabilités sans gradient
    with torch.no_grad():
        pos_prob, _ = tgn.compute_edge_probabilities(
            source_nodes=src_tensor,
            destination_nodes=dst_tensor,
            negative_nodes=neg_tensor,
            edge_times=times_tensor,
            edge_idxs=idx_tensor,
            n_neighbors=args.n_degree
        )

    # Ajouter les arêtes avec leur poids dans le graphe
    for s, d, p in zip(src_batch, dst_batch, pos_prob.cpu().numpy()):
        pred_graph.add_edge(int(s), int(d), weight=float(p))

logger.info(f"Predicted graph has {pred_graph.number_of_nodes()} nodes and {pred_graph.number_of_edges()} edges.")

# Sauvegarde du graphe prédit
nx.write_gpickle(pred_graph, f"predicted_graph_{args.data}.gpickle")
logger.info(f"Predicted graph saved to predicted_graph_{args.data}.gpickle")