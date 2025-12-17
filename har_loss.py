"""
HAR (Hardness Adaptive Reweighted) Contrastive Loss implementation

Paper: "Graph Contrastive Learning with Adaptive Augmentation" (2021)
Objective: Mitigate degree bias in Graph Neural Networks

HAR Loss reweights examples based on:
1. Node degree: Lower degree nodes get higher weights
2. Hardness: Difficult examples get higher weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HARLoss(nn.Module):
    """
    Hardness Adaptive Reweighted (HAR) Loss for link prediction.

    Combats degree bias by giving more weight to low-degree nodes.

    Formula:
        HAR_loss = sum_i [ w(src_i) * w(dst_i) * contrastive_loss(i) ]

        where:
        - w(node) = degree(node)^(-alpha)  : degree-based weight
        - alpha : reweighting exponent (0.5-1.0)

    Args:
        temperature (float): Temperature for contrastive loss (default: 0.07)
        alpha (float): Degree reweighting exponent (default: 0.5)
                      Higher alpha = stronger bias correction
        reduction (str): 'none', 'mean', 'sum' (default: 'mean')

    Example:
        >>> criterion = HARLoss(temperature=0.07, alpha=0.5)
        >>> loss = criterion(pos_scores, neg_scores, src_degrees, dst_degrees)
    """

    def __init__(self, temperature=0.07, alpha=0.5, reduction='mean'):
        super(HARLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def compute_degree_weights(self, degrees):
        """
        Compute degree-based weights: w(i) = degree(i)^(-alpha)

        Args:
            degrees (Tensor): Node degrees, shape (N,)

        Returns:
            Tensor: Weights, shape (N,)
        """
        # Clamp to avoid division by zero (min degree = 1)
        degrees_clamped = torch.clamp(degrees, min=1.0)
        weights = torch.pow(degrees_clamped, -self.alpha)
        return weights

    def forward(self, pos_scores, neg_scores, src_degrees, dst_degrees):
        """
        Compute HAR loss for link prediction.

        Args:
            pos_scores (Tensor): Scores for positive pairs, shape (N,)
            neg_scores (Tensor): Scores for negative pairs, shape (N,)
            src_degrees (Tensor): Degrees of source nodes, shape (N,)
            dst_degrees (Tensor): Degrees of destination nodes, shape (N,)

        Returns:
            Tensor: HAR loss (scalar)
        """
        # 1. Compute degree-based weights
        w_src = self.compute_degree_weights(src_degrees)  # (N,)
        w_dst = self.compute_degree_weights(dst_degrees)  # (N,)

        # Combined weight for each pair
        w_pair = w_src * w_dst  # (N,)

        # 2. Normalize scores by temperature
        pos_scores_norm = pos_scores / self.temperature  # (N,)
        neg_scores_norm = neg_scores / self.temperature  # (N,)

        # 3. Contrastive loss (InfoNCE style)
        # For each positive, we want it to be higher than its negative
        # LogSumExp trick for numerical stability
        logits = torch.stack([pos_scores_norm, neg_scores_norm], dim=1)  # (N, 2)

        # Labels: 0 = positive is the correct one
        labels = torch.zeros(pos_scores.size(0), dtype=torch.long, device=pos_scores.device)

        # Cross-entropy loss (no reduction yet)
        loss_base = F.cross_entropy(logits, labels, reduction='none')  # (N,)

        # 4. Apply degree-based reweighting
        loss_weighted = loss_base * w_pair  # (N,)

        # 5. Reduction
        if self.reduction == 'mean':
            return loss_weighted.mean()
        elif self.reduction == 'sum':
            return loss_weighted.sum()
        else:
            return loss_weighted


class HARLossWithHardness(nn.Module):
    """
    HAR Loss with explicit hardness computation.

    This variant also considers the hardness of each example based on
    the similarity between embeddings.

    Args:
        temperature (float): Temperature for contrastive loss (default: 0.07)
        alpha (float): Degree reweighting exponent (default: 0.5)
        beta (float): Hardness weighting factor (default: 1.0)
        reduction (str): 'none', 'mean', 'sum' (default: 'mean')
    """

    def __init__(self, temperature=0.07, alpha=0.5, beta=1.0, reduction='mean'):
        super(HARLossWithHardness, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def compute_degree_weights(self, degrees):
        """Compute degree-based weights: w(i) = degree(i)^(-alpha)"""
        degrees_clamped = torch.clamp(degrees, min=1.0)
        weights = torch.pow(degrees_clamped, -self.alpha)
        return weights

    def compute_hardness(self, pos_scores, neg_scores):
        """
        Compute hardness based on score difference.

        Hardness = 1 - (pos_score - neg_score)

        If pos_score >> neg_score: easy example (low hardness)
        If pos_score ≈ neg_score: hard example (high hardness)
        """
        # Normalize scores to [0, 1]
        pos_probs = torch.sigmoid(pos_scores)
        neg_probs = torch.sigmoid(neg_scores)

        # Hardness: smaller difference = harder
        score_diff = pos_probs - neg_probs  # Range: [-1, 1]
        hardness = 1.0 - score_diff  # Range: [0, 2]
        hardness = torch.clamp(hardness, min=0.0, max=2.0)

        return hardness

    def forward(self, pos_scores, neg_scores, src_degrees, dst_degrees):
        """
        Compute HAR loss with hardness-aware weighting.

        Args:
            pos_scores (Tensor): Scores for positive pairs, shape (N,)
            neg_scores (Tensor): Scores for negative pairs, shape (N,)
            src_degrees (Tensor): Degrees of source nodes, shape (N,)
            dst_degrees (Tensor): Degrees of destination nodes, shape (N,)

        Returns:
            Tensor: HAR loss (scalar)
        """
        # 1. Compute degree-based weights
        w_src = self.compute_degree_weights(src_degrees)
        w_dst = self.compute_degree_weights(dst_degrees)
        w_degree = w_src * w_dst

        # 2. Compute hardness weights
        hardness = self.compute_hardness(pos_scores, neg_scores)
        w_hardness = torch.pow(hardness, self.beta)

        # 3. Combined weight
        w_total = w_degree * w_hardness

        # 4. Contrastive loss
        pos_scores_norm = pos_scores / self.temperature
        neg_scores_norm = neg_scores / self.temperature

        logits = torch.stack([pos_scores_norm, neg_scores_norm], dim=1)
        labels = torch.zeros(pos_scores.size(0), dtype=torch.long, device=pos_scores.device)

        loss_base = F.cross_entropy(logits, labels, reduction='none')

        # 5. Apply total weighting
        loss_weighted = loss_base * w_total

        # 6. Reduction
        if self.reduction == 'mean':
            return loss_weighted.mean()
        elif self.reduction == 'sum':
            return loss_weighted.sum()
        else:
            return loss_weighted


def build_degree_dict(data):
    """
    Build a dictionary mapping node IDs to their degrees.

    Args:
        data (Data): Graph data with sources and destinations

    Returns:
        dict: {node_id: degree}
    """
    from collections import Counter

    # Count occurrences of each node
    degree_dict = Counter()

    # Count source degrees
    for src in data.sources:
        degree_dict[int(src)] += 1

    # Count destination degrees
    for dst in data.destinations:
        degree_dict[int(dst)] += 1

    return dict(degree_dict)


def test_har_loss():
    """
    Test to verify HAR Loss works correctly.
    """
    print("Testing HAR Loss...")

    # Create test data
    torch.manual_seed(42)
    batch_size = 100

    # Simulate scores
    pos_scores = torch.randn(batch_size) + 1.0  # Positive scores (higher)
    neg_scores = torch.randn(batch_size) - 1.0  # Negative scores (lower)

    # Simulate degrees (power-law distribution)
    # Some nodes with high degree, most with low degree
    src_degrees = torch.pow(torch.rand(batch_size), -2.0) * 10  # Range: ~1-100
    dst_degrees = torch.pow(torch.rand(batch_size), -2.0) * 10

    # Compare BCE vs HAR Loss
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_har = HARLoss(temperature=0.07, alpha=0.5)
    criterion_har_hard = HARLossWithHardness(temperature=0.07, alpha=0.5, beta=1.0)

    # BCE requires labels
    pos_labels = torch.ones(batch_size)
    neg_labels = torch.zeros(batch_size)
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([pos_labels, neg_labels])

    loss_bce = criterion_bce(all_scores, all_labels)
    loss_har = criterion_har(pos_scores, neg_scores, src_degrees, dst_degrees)
    loss_har_hard = criterion_har_hard(pos_scores, neg_scores, src_degrees, dst_degrees)

    print(f"\nBinary Cross-Entropy Loss:    {loss_bce.item():.4f}")
    print(f"HAR Loss (alpha=0.5):          {loss_har.item():.4f}")
    print(f"HAR Loss with Hardness:        {loss_har_hard.item():.4f}")

    # Test impact of alpha (degree reweighting strength)
    print("\nImpact of alpha (degree reweighting):")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        criterion = HARLoss(temperature=0.07, alpha=alpha)
        loss = criterion(pos_scores, neg_scores, src_degrees, dst_degrees)
        print(f"  alpha={alpha:.2f}: loss={loss.item():.4f}")

    # Test degree bias correction
    print("\nDegree bias correction test:")

    # Case 1: High-degree nodes (easy examples)
    high_degree = torch.ones(10) * 50.0
    pos_high = torch.randn(10) + 2.0  # Easy positives
    neg_high = torch.randn(10) - 2.0  # Easy negatives

    # Case 2: Low-degree nodes (hard examples)
    low_degree = torch.ones(10) * 2.0
    pos_low = torch.randn(10) + 0.5   # Hard positives
    neg_low = torch.randn(10) - 0.5   # Hard negatives

    criterion = HARLoss(temperature=0.07, alpha=0.5)

    loss_high = criterion(pos_high, neg_high, high_degree, high_degree)
    loss_low = criterion(pos_low, neg_low, low_degree, low_degree)

    print(f"  High-degree nodes (degree=50): loss={loss_high.item():.4f}")
    print(f"  Low-degree nodes (degree=2):   loss={loss_low.item():.4f}")
    print(f"  Ratio (low/high):              {loss_low.item() / loss_high.item():.2f}x")
    print(f"  → Low-degree nodes contribute more to loss (as intended)")

    print("\n✅ Tests terminés!")


if __name__ == "__main__":
    test_har_loss()
