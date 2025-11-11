import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import networkx as nx
import pickle
import matplotlib.pyplot as plt

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument parser
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
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
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
parser.add_argument('--circular', action='store_true',
                    help='Use circular layout for bipartite graph')

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

# 1. Validation - Seen nodes
tgn.set_neighbor_finder(full_ngh_finder)
val_ap, val_auc = eval_edge_prediction(tgn, val_rand_sampler, val_data, args.n_degree)
logger.info(f"Validation (seen nodes) - AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
if args.use_memory:
    val_memory_backup = tgn.memory.backup_memory()

# 2. Validation - New nodes
tgn.set_neighbor_finder(train_ngh_finder)
nn_val_ap, nn_val_auc = eval_edge_prediction(tgn, nn_val_rand_sampler, new_node_val_data, args.n_degree)
logger.info(f"Validation (new nodes) - AUC: {nn_val_auc:.4f}, AP: {nn_val_ap:.4f}")
if args.use_memory:
    tgn.memory.restore_memory(val_memory_backup)

# 3. Test - Seen nodes
tgn.set_neighbor_finder(full_ngh_finder)
test_ap, test_auc = eval_edge_prediction(tgn, test_rand_sampler, test_data, args.n_degree)
logger.info(f"Test (seen nodes) - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
if args.use_memory:
    test_memory_backup = tgn.memory.backup_memory()

# 4. Test - New nodes
tgn.set_neighbor_finder(train_ngh_finder)
if args.use_memory:
    tgn.memory.restore_memory(val_memory_backup)
nn_test_ap, nn_test_auc = eval_edge_prediction(tgn, nn_test_rand_sampler, new_node_test_data, args.n_degree)
logger.info(f"Test (new nodes) - AUC: {nn_test_auc:.4f}, AP: {nn_test_ap:.4f}")

# ---------------- Generate Predicted Graph ---------------- #
logger.info("Generating predicted graph...")

# Reset memory before generating predictions
if args.use_memory:
    tgn.memory.__init_memory__()

# Set to full neighbor finder for graph generation
tgn.set_neighbor_finder(full_ngh_finder)

# Create empty graph
pred_graph = nx.DiGraph() if args.dyrep else nx.Graph()

# Get all edges and sort by timestamp
sources = np.array(full_data.sources)
destinations = np.array(full_data.destinations)
edge_times = np.array(full_data.timestamps)
edge_idxs = np.array(full_data.edge_idxs)

# Sort by timestamp to maintain temporal order
sorted_indices = np.argsort(edge_times)
sources = sources[sorted_indices]
destinations = destinations[sorted_indices]
edge_times = edge_times[sorted_indices]
edge_idxs = edge_idxs[sorted_indices]

# Negative sampler
neg_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=42)

batch_size = 200
num_edges = len(sources)

for start in range(0, num_edges, batch_size):
    end = min(start + batch_size, num_edges)

    src_batch = sources[start:end]
    dst_batch = destinations[start:end]
    times_batch = edge_times[start:end]
    idx_batch = edge_idxs[start:end]

    neg_tuple = neg_sampler.sample(len(src_batch))
    neg_batch = np.array(neg_tuple[1])

    # Ensure 1D array
    if len(neg_batch.shape) > 1:
        neg_batch = neg_batch[:, 0]

    with torch.no_grad():
        pos_prob, _ = tgn.compute_edge_probabilities(
            src_batch,
            dst_batch,
            neg_batch,
            times_batch,
            idx_batch
        )

    # Add edges to graph with probabilities
    for s, d, p in zip(src_batch, dst_batch, pos_prob.cpu().numpy()):
        pred_graph.add_edge(int(s), int(d), weight=float(p))
    
    # Log progress
    if (start // batch_size) % 10 == 0:
        logger.info(f"Processed {end}/{num_edges} edges...")

logger.info(f"Predicted graph has {pred_graph.number_of_nodes()} nodes and {pred_graph.number_of_edges()} edges.")

# Save graph
with open(f'predicted_graph_{args.data}.pkl', 'wb') as f:
    pickle.dump(pred_graph, f, pickle.HIGHEST_PROTOCOL)
logger.info(f"Predicted graph saved to predicted_graph_{args.data}.pkl")

# ---------------- Visualization Functions ---------------- #

def plot_bipartite_graph(G, circular=False):
    """Plots the bipartite network similar to the original style"""
    print("\n========================== PLOTTING BIPARTITE GRAPH ==========================")

    # Identify the two sets of the bipartite graph
    # Assuming sources are set 0 (companies) and destinations are set 1 (investors)
    set1 = [node for node in G.nodes() if G.nodes[node].get('bipartite', 0) == 0]
    set2 = [node for node in G.nodes() if G.nodes[node].get('bipartite', 1) == 1]
    
    # If bipartite attribute doesn't exist, split by degree or node ID
    if len(set1) == 0 or len(set2) == 0:
        all_nodes = list(G.nodes())
        mid = len(all_nodes) // 2
        set1 = all_nodes[:mid]
        set2 = all_nodes[mid:]
        # Add bipartite attribute
        for node in set1:
            G.nodes[node]['bipartite'] = 0
        for node in set2:
            G.nodes[node]['bipartite'] = 1

    if circular:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)

    if len(set1) >= 20:
        plt.figure(figsize=(25, 15))
    else:
        plt.figure(figsize=(19, 13))

    plt.ion()
    plt.axis('on')

    companies = set1
    investors = set2

    # Calculate degree centrality
    companyDegree = nx.degree(G, companies) 
    investorDegree = nx.degree(G, investors)

    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=companies,
                           node_color='r',
                           node_size=[v * 100 for v in dict(companyDegree).values()],
                           alpha=0.6,
                           label='Companies')

    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=investors,
                           node_color='g',
                           node_size=[v * 200 for v in dict(investorDegree).values()],
                           alpha=0.6,
                           label='Investors')

    nx.draw_networkx_labels(G, pos, {n: n for n in companies}, font_size=10)
    nx.draw_networkx_labels(G, pos, {n: n for n in investors}, font_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bipartite_graph_{args.data}.png", dpi=300, bbox_inches='tight')
    logger.info(f"Bipartite graph saved to bipartite_graph_{args.data}.png")
    plt.show(block=True)
    return pos


def plot_predicted_graph(graph, dataset_name, save_path=None, figsize=(15, 10)):
    """Visualize the predicted graph with edge weights."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'TGN Predicted Graph Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Main graph visualization
    ax1 = axes[0, 0]
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    
    # Get edge weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    
    # Normalize weights for visualization
    weights_normalized = np.array(weights)
    weights_normalized = (weights_normalized - weights_normalized.min()) / (weights_normalized.max() - weights_normalized.min() + 1e-8)
    
    # Draw nodes
    node_sizes = [graph.degree(node) * 50 + 100 for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.7, ax=ax1)
    
    # Draw edges with varying thickness based on weight
    nx.draw_networkx_edges(graph, pos, width=[w * 3 + 0.5 for w in weights_normalized],
                          alpha=0.6, edge_color=weights_normalized, 
                          edge_cmap=plt.cm.RdYlGn, ax=ax1)
    
    # Draw labels for nodes with high degree
    high_degree_nodes = [node for node in graph.nodes() if graph.degree(node) > 3]
    labels = {node: str(node) for node in high_degree_nodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax1)
    
    ax1.set_title('Graph Structure (node size = degree, edge color/width = probability)')
    ax1.axis('off')
    
    # 2. Edge weight distribution
    ax2 = axes[0, 1]
    ax2.hist(weights, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(weights), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(weights):.3f}')
    ax2.axvline(np.median(weights), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(weights):.3f}')
    ax2.set_xlabel('Edge Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Edge Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Degree distribution
    ax3 = axes[1, 0]
    degrees = [graph.degree(node) for node in graph.nodes()]
    ax3.hist(degrees, bins=max(degrees) if max(degrees) < 50 else 50, 
             color='coral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Node Degree')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Degree Distribution (avg: {np.mean(degrees):.2f})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats = [
        ['Metric', 'Value'],
        ['─' * 30, '─' * 15],
        ['Number of Nodes', f'{graph.number_of_nodes()}'],
        ['Number of Edges', f'{graph.number_of_edges()}'],
        ['Density', f'{nx.density(graph):.4f}'],
        ['Avg. Degree', f'{np.mean(degrees):.2f}'],
        ['Max Degree', f'{max(degrees)}'],
        ['Min Degree', f'{min(degrees)}'],
        ['─' * 30, '─' * 15],
        ['Edge Probabilities:', ''],
        ['  Mean', f'{np.mean(weights):.4f}'],
        ['  Std', f'{np.std(weights):.4f}'],
        ['  Min', f'{min(weights):.4f}'],
        ['  Max', f'{max(weights):.4f}'],
        ['─' * 30, '─' * 15],
    ]
    
    # Add connectivity info if not directed
    if not graph.is_directed():
        stats.append(['Connected Components', f'{nx.number_connected_components(graph)}'])
    
    table = ax4.table(cellText=stats, cellLoc='left', loc='center',
                     colWidths=[0.7, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graph visualization saved to {save_path}")
    
    plt.show()


def plot_top_edges(graph, k=20, dataset_name=""):
    """Plot a subgraph containing only the top-k most confident edges."""
    edges_with_weights = [(u, v, data['weight']) for u, v, data in graph.edges(data=True)]
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    
    top_edges = edges_with_weights[:k]
    
    # Create subgraph
    subgraph = nx.Graph() if not graph.is_directed() else nx.DiGraph()
    for u, v, w in top_edges:
        subgraph.add_edge(u, v, weight=w)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='lightcoral', alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, width=3, alpha=0.6, 
                          edge_color=weights, edge_cmap=plt.cm.RdYlGn)
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold')
    
    # Add edge labels with weights
    edge_labels = {(u, v): f'{subgraph[u][v]["weight"]:.3f}' for u, v in subgraph.edges()}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=8)
    
    plt.title(f'Top {k} Most Confident Edges - {dataset_name}', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"top_edges_{dataset_name}.png", dpi=300, bbox_inches='tight')
    logger.info(f"Top edges plot saved to top_edges_{dataset_name}.png")
    plt.show()


# Generate visualizations
logger.info("Visualizing predicted graph (bipartite style)...")
plot_bipartite_graph(pred_graph, circular=args.circular)

logger.info("Visualizing predicted graph (detailed analysis)...")
plot_predicted_graph(pred_graph, args.data, save_path=f"predicted_graph_{args.data}.png")

logger.info("Plotting top confident edges...")
plot_top_edges(pred_graph, k=20, dataset_name=args.data)

logger.info("Evaluation complete!")