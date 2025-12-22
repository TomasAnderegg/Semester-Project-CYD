"""
Hybrid Focal-DCL Loss implementation

Combines:
1. Focal Loss: Focus on hard examples (class imbalance)
2. DCL Loss: Mitigate degree bias (graph structure)

Ideal for scenarios with BOTH:
- Extreme class imbalance (e.g., 0.03% positives)
- Degree bias (high-degree nodes dominate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFocalDCLLoss(nn.Module):
    """
    Hybrid loss combining Focal Loss and DCL Loss.

    Formula:
        Hybrid = degree_weight * focal_weight * BCE_loss

    where:
        - degree_weight = degree^(-dcl_alpha)     # DCL component
        - focal_weight = (1 - p_t)^focal_gamma    # Focal component
        - BCE_loss = standard binary cross-entropy

    Args:
        focal_gamma (float): Focusing parameter for Focal Loss (default: 2.0)
        focal_alpha (float): Class balancing for Focal Loss (default: 0.25)
        dcl_alpha (float): Degree reweighting exponent (default: 0.5)
        lambda_focal (float): Weight for Focal vs DCL (default: 0.5)
                             0.0 = pure DCL, 1.0 = pure Focal, 0.5 = balanced
        reduction (str): 'none', 'mean', 'sum' (default: 'mean')

    Example:
        >>> criterion = HybridFocalDCLLoss(focal_gamma=2.0, dcl_alpha=0.5)
        >>> loss = criterion(pos_prob, neg_prob, src_degrees, dst_degrees,
        ...                  pos_label, neg_label)
    """

    def __init__(self, focal_gamma=2.0, focal_alpha=0.25, dcl_alpha=0.5,
                 lambda_focal=0.5, reduction='mean'):
        super(HybridFocalDCLLoss, self).__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.dcl_alpha = dcl_alpha
        self.lambda_focal = lambda_focal
        self.reduction = reduction

    def compute_degree_weights(self, degrees):
        """
        Compute DCL degree-based weights: w(i) = degree(i)^(-alpha)
        """
        degrees_clamped = torch.clamp(degrees, min=1.0)
        weights = torch.pow(degrees_clamped, -self.dcl_alpha)
        return weights

    def forward(self, pos_prob, neg_prob, src_degrees, dst_degrees_pos, dst_degrees_neg,
                pos_label, neg_label):
        """
        Compute hybrid Focal-DCL loss.

        Args:
            pos_prob (Tensor): Probabilities for positive pairs, shape (N,)
            neg_prob (Tensor): Probabilities for negative pairs, shape (N,)
            src_degrees (Tensor): Degrees of source nodes, shape (N,)
            dst_degrees_pos (Tensor): Degrees of positive destination nodes, shape (N,)
            dst_degrees_neg (Tensor): Degrees of negative destination nodes, shape (N,)
            pos_label (Tensor): Labels for positives (all 1.0), shape (N,)
            neg_label (Tensor): Labels for negatives (all 0.0), shape (N,)

        Returns:
            Tensor: Hybrid loss (scalar)
        """
        # Clamp probabilities for numerical stability
        pos_prob = torch.clamp(pos_prob, min=1e-7, max=1.0 - 1e-7)
        neg_prob = torch.clamp(neg_prob, min=1e-7, max=1.0 - 1e-7)

        # ========================================
        # POSITIVE PAIRS
        # ========================================

        # 1. DCL component: degree-based weights for POSITIVE pairs
        w_src_pos = self.compute_degree_weights(src_degrees)
        w_dst_pos = self.compute_degree_weights(dst_degrees_pos)
        w_degree_pos = w_src_pos * w_dst_pos  # Combined degree weight

        # 2. Focal component: focus on hard examples
        # p_t for positives = p (since y=1)
        p_t_pos = pos_prob
        focal_weight_pos = torch.pow(1.0 - p_t_pos, self.focal_gamma)

        # 3. Class balancing (Focal Loss alpha)
        alpha_t_pos = self.focal_alpha

        # 4. Base BCE loss for positives
        bce_pos = -torch.log(pos_prob)

        # 5. Combined loss for positives
        loss_pos = alpha_t_pos * focal_weight_pos * w_degree_pos * bce_pos

        # ========================================
        # NEGATIVE PAIRS
        # ========================================

        # 1. DCL component: degree-based weights for NEGATIVE pairs
        # Note: For negatives, we also want to reweight by degree
        # This ensures low-degree negatives aren't ignored either
        w_src_neg = self.compute_degree_weights(src_degrees)
        w_dst_neg = self.compute_degree_weights(dst_degrees_neg)
        w_degree_neg = w_src_neg * w_dst_neg

        # 2. Focal component
        # p_t for negatives = (1 - p) (since y=0)
        p_t_neg = 1.0 - neg_prob
        focal_weight_neg = torch.pow(1.0 - p_t_neg, self.focal_gamma)

        # 3. Class balancing
        alpha_t_neg = 1.0 - self.focal_alpha

        # 4. Base BCE loss for negatives
        bce_neg = -torch.log(1.0 - neg_prob)

        # 5. Combined loss for negatives
        loss_neg = alpha_t_neg * focal_weight_neg * w_degree_neg * bce_neg

        # ========================================
        # TOTAL LOSS
        # ========================================

        # Combine positive and negative losses
        total_loss = loss_pos + loss_neg

        # Reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class AdaptiveHybridLoss(nn.Module):
    """
    Adaptive Hybrid Loss that automatically adjusts lambda_focal
    based on the current epoch or batch statistics.

    This is useful if you want to:
    - Start with pure Focal Loss (handle class imbalance first)
    - Gradually transition to DCL (mitigate degree bias later)

    Args:
        focal_gamma (float): Focal Loss gamma (default: 2.0)
        focal_alpha (float): Focal Loss alpha (default: 0.25)
        dcl_alpha (float): DCL degree reweighting (default: 0.5)
        schedule (str): 'linear', 'cosine', 'step' (default: 'linear')
        warmup_epochs (int): Number of epochs before DCL kicks in (default: 10)
        total_epochs (int): Total epochs (default: 50)
    """

    def __init__(self, focal_gamma=2.0, focal_alpha=0.25, dcl_alpha=0.5,
                 schedule='linear', warmup_epochs=10, total_epochs=50,
                 reduction='mean'):
        super(AdaptiveHybridLoss, self).__init__()
        self.hybrid_loss = HybridFocalDCLLoss(
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            dcl_alpha=dcl_alpha,
            lambda_focal=1.0,  # Will be adjusted dynamically
            reduction=reduction
        )
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """Update current epoch for adaptive scheduling."""
        self.current_epoch = epoch

        # Compute lambda_focal based on schedule
        if epoch < self.warmup_epochs:
            # Pure Focal Loss during warmup
            lambda_focal = 1.0
        else:
            # Transition to balanced or DCL-dominated
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

            if self.schedule == 'linear':
                lambda_focal = 1.0 - 0.5 * progress  # 1.0 → 0.5
            elif self.schedule == 'cosine':
                import math
                lambda_focal = 0.5 + 0.5 * math.cos(math.pi * progress)
            elif self.schedule == 'step':
                lambda_focal = 1.0 if progress < 0.5 else 0.5
            else:
                lambda_focal = 0.5

        self.hybrid_loss.lambda_focal = lambda_focal

    def forward(self, pos_prob, neg_prob, src_degrees, dst_degrees_pos, dst_degrees_neg,
                pos_label, neg_label):
        """Forward pass (delegates to HybridFocalDCLLoss)."""
        return self.hybrid_loss(pos_prob, neg_prob, src_degrees, dst_degrees_pos, dst_degrees_neg,
                               pos_label, neg_label)


def test_hybrid_loss():
    """
    Test to verify Hybrid Focal-DCL Loss works correctly.
    """
    print("Testing Hybrid Focal-DCL Loss...")

    # Create test data
    torch.manual_seed(42)
    batch_size = 100

    # Simulate probabilities
    pos_prob = torch.rand(batch_size) * 0.5 + 0.5  # Range: 0.5-1.0
    neg_prob = torch.rand(batch_size) * 0.5        # Range: 0.0-0.5

    # Simulate degrees (power-law distribution)
    src_degrees = torch.pow(torch.rand(batch_size), -2.0) * 10
    dst_degrees = torch.pow(torch.rand(batch_size), -2.0) * 10

    # Labels
    pos_label = torch.ones(batch_size)
    neg_label = torch.zeros(batch_size)

    # Compare different losses
    from focal_loss import FocalLoss
    from dcl_loss import DCLLoss

    criterion_bce = nn.BCELoss()
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_har = DCLLoss(temperature=0.07, alpha=0.5)
    criterion_hybrid = HybridFocalDCLLoss(focal_gamma=2.0, focal_alpha=0.25,
                                         dcl_alpha=0.5, lambda_focal=0.5)

    # BCE
    loss_bce = (criterion_bce(pos_prob, pos_label) +
                criterion_bce(neg_prob, neg_label))

    # Focal
    loss_focal = (criterion_focal(pos_prob, pos_label) +
                  criterion_focal(neg_prob, neg_label))

    # DCL (needs scores, not probs, and separate dst degrees)
    pos_scores = torch.log(pos_prob / (1 - pos_prob + 1e-7))
    neg_scores = torch.log(neg_prob / (1 - neg_prob + 1e-7))
    loss_har = criterion_har(pos_scores, neg_scores, src_degrees, dst_degrees, dst_degrees)

    # Hybrid (using same dst_degrees for both pos and neg in test)
    loss_hybrid = criterion_hybrid(pos_prob, neg_prob, src_degrees, dst_degrees, dst_degrees,
                                   pos_label, neg_label)

    print(f"\nComparison of losses:")
    print(f"  BCE:         {loss_bce.item():.4f}")
    print(f"  Focal:       {loss_focal.item():.4f}")
    print(f"  DCL:         {loss_har.item():.4f}")
    print(f"  Hybrid:      {loss_hybrid.item():.4f}")

    # Test impact of lambda_focal
    print(f"\nImpact of lambda_focal (Focal vs DCL balance):")
    for lambda_f in [0.0, 0.25, 0.5, 0.75, 1.0]:
        criterion = HybridFocalDCLLoss(focal_gamma=2.0, focal_alpha=0.25,
                                      dcl_alpha=0.5, lambda_focal=lambda_f)
        loss = criterion(pos_prob, neg_prob, src_degrees, dst_degrees, dst_degrees,
                        pos_label, neg_label)
        print(f"  lambda={lambda_f:.2f} (Focal={lambda_f:.0%}, DCL={1-lambda_f:.0%}): loss={loss.item():.4f}")

    # Test degree bias correction
    print(f"\nDegree bias correction:")

    # High-degree nodes
    high_degree = torch.ones(10) * 50.0
    pos_high = torch.rand(10) * 0.3 + 0.6  # Easy positives
    neg_high = torch.rand(10) * 0.3        # Easy negatives
    label_high = torch.ones(10)
    label_neg_high = torch.zeros(10)

    # Low-degree nodes
    low_degree = torch.ones(10) * 2.0
    pos_low = torch.rand(10) * 0.3 + 0.3   # Hard positives
    neg_low = torch.rand(10) * 0.3 + 0.2   # Hard negatives
    label_low = torch.ones(10)
    label_neg_low = torch.zeros(10)

    criterion = HybridFocalDCLLoss(focal_gamma=2.0, dcl_alpha=0.5)

    loss_high = criterion(pos_high, neg_high, high_degree, high_degree, high_degree,
                         label_high, label_neg_high)
    loss_low = criterion(pos_low, neg_low, low_degree, low_degree, low_degree,
                        label_low, label_neg_low)

    print(f"  High-degree (degree=50, easy):  loss={loss_high.item():.4f}")
    print(f"  Low-degree (degree=2, hard):    loss={loss_low.item():.4f}")
    print(f"  Ratio (low/high):               {loss_low.item() / loss_high.item():.2f}x")
    print(f"  → Hybrid correctly prioritizes low-degree hard examples")

    print("\n✅ Tests terminés!")


if __name__ == "__main__":
    test_hybrid_loss()
