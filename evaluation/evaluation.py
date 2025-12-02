import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  val_mrr, val_recall_10, val_recall_50 = [], [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

      # ADDED MRR AND RECALL@K COMPUTATION
      # Sample multiple negatives for MRR and Recall@K
      num_negatives = 100
      neg_probs_list = []
      
      for _ in range(num_negatives):
        _, neg_batch = negative_edge_sampler.sample(size)
        _, neg_prob_i = model.compute_edge_probabilities(
            sources_batch, 
            destinations_batch,
            neg_batch, 
            timestamps_batch, 
            edge_idxs_batch, 
            n_neighbors
        )
        neg_probs_list.append(neg_prob_i)
      
      # Stack all negative probabilities: shape (size, num_negatives)
      all_neg_probs = torch.stack(neg_probs_list, dim=1)
      
      # Compute ranking metrics for this batch
      batch_mrr, batch_recall_dict = compute_ranking_metrics(
          pos_prob.squeeze(), 
          all_neg_probs.squeeze()
      )
      
      val_mrr.append(batch_mrr)
      val_recall_10.append(batch_recall_dict['recall@10'])
      val_recall_50.append(batch_recall_dict['recall@50'])

  return (np.mean(val_ap), np.mean(val_auc), 
          np.mean(val_mrr), np.mean(val_recall_10), np.mean(val_recall_50))


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
# MRR, Recall@10, Recall@50
def compute_ranking_metrics(pos_scores, neg_scores, k_list=[10, 50]):
    """
    Compute MRR and Recall@K metrics
    
    Args:
        pos_scores: scores for positive edges (batch_size,)
        neg_scores: scores for negative edges (batch_size, num_negatives)
        k_list: list of K values for Recall@K
    
    Returns:
        mrr: Mean Reciprocal Rank
        recall_at_k_dict: Dictionary with Recall@K for each K
    """
    # Ensure pos_scores has the right shape
    if len(pos_scores.shape) == 1:
        pos_scores = pos_scores.unsqueeze(1)  # (batch_size, 1)
    
    # Combine positive and negative scores
    # Shape: (batch_size, 1 + num_negatives)
    all_scores = torch.cat([pos_scores, neg_scores], dim=1)
    
    # Get rankings (argsort in descending order)
    # rankings contains indices sorted by score (highest first)
    rankings = torch.argsort(all_scores, dim=1, descending=True)
    
    # Find where the positive sample (index 0) is ranked
    positive_ranks = (rankings == 0).nonzero(as_tuple=True)[1] + 1  # +1 because ranks start at 1
    
    # Compute MRR
    mrr = (1.0 / positive_ranks.float()).mean().item()
    
    # Compute Recall@K
    recall_at_k = {}
    for k in k_list:
        recall_at_k[f'recall@{k}'] = (positive_ranks <= k).float().mean().item()
    
    return mrr, recall_at_k
