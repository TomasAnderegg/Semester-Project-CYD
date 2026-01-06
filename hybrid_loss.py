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
    Hybrid loss combining Focal Loss and DCL Loss as a weighted sum.

    Formula:
        Hybrid = alpha_focal * FocalLoss + alpha_dcl * DCLLoss

    where:
        FocalLoss = alpha_t * (1 - p_t)^gamma * log(p_t)
        DCLLoss = degree_weight * ContrastiveLoss
            ContrastiveLoss = CrossEntropy([pos_score/T, neg_score/T], target=0)
            degree_weight = (degree_src^(-α) * degree_dst_pos^(-α) +
                           degree_src^(-α) * degree_dst_neg^(-α)) / 2

    Args:
        focal_gamma (float): Focusing parameter for Focal Loss (default: 2.0)
        focal_alpha (float): Class balancing for Focal Loss (default: 0.25)
        dcl_alpha (float): Degree reweighting exponent (default: 0.5)
        dcl_temperature (float): Temperature for DCL contrastive loss (default: 0.07)
        alpha_focal (float): Weight for Focal Loss component (default: 0.5)
        alpha_dcl (float): Weight for DCL Loss component (default: 0.5)
        reduction (str): 'none', 'mean', 'sum' (default: 'mean')

    Example:
        >>> criterion = HybridFocalDCLLoss(focal_gamma=2.0, dcl_alpha=0.5,
        ...                                alpha_focal=0.5, alpha_dcl=0.5)
        >>> loss = criterion(pos_prob, neg_prob, src_degrees, dst_degrees_pos, dst_degrees_neg,
        ...                  pos_label, neg_label)
    """

    def __init__(self, focal_gamma=2.0, focal_alpha=0.25, dcl_alpha=0.5,
                 dcl_temperature=0.07, alpha_focal=0.5, alpha_dcl=0.5, reduction='mean'):
        super(HybridFocalDCLLoss, self).__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.dcl_alpha = dcl_alpha
        self.dcl_temperature = dcl_temperature
        self.alpha_focal = alpha_focal
        self.alpha_dcl = alpha_dcl
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
        Compute hybrid Focal-DCL loss as a weighted sum.

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
        # FOCAL LOSS COMPONENT (BCE-based)
        # ========================================

        # Base BCE loss
        bce_pos = -torch.log(pos_prob)
        bce_neg = -torch.log(1.0 - neg_prob)

        # Positive pairs
        p_t_pos = pos_prob
        focal_weight_pos = torch.pow(1.0 - p_t_pos, self.focal_gamma)
        alpha_t_pos = self.focal_alpha
        focal_loss_pos = alpha_t_pos * focal_weight_pos * bce_pos

        # Negative pairs
        p_t_neg = 1.0 - neg_prob
        focal_weight_neg = torch.pow(1.0 - p_t_neg, self.focal_gamma)
        alpha_t_neg = 1.0 - self.focal_alpha
        focal_loss_neg = alpha_t_neg * focal_weight_neg * bce_neg

        focal_loss = focal_loss_pos + focal_loss_neg

        # ========================================
        # DCL LOSS COMPONENT (Contrastive)
        # ========================================

        # Convert probabilities to scores (logits) for contrastive loss
        pos_scores = torch.log(pos_prob / (1.0 - pos_prob))
        neg_scores = torch.log(neg_prob / (1.0 - neg_prob))

        # Normalize by temperature
        pos_scores_norm = pos_scores / self.dcl_temperature
        neg_scores_norm = neg_scores / self.dcl_temperature

        # Contrastive loss (InfoNCE style)
        # Stack [pos_score, neg_score] and want pos_score to be higher
        logits = torch.stack([pos_scores_norm, neg_scores_norm], dim=1)  # (N, 2)
        labels = torch.zeros(pos_prob.size(0), dtype=torch.long, device=pos_prob.device)  # Want pos (index 0)

        # Cross-entropy contrastive loss
        dcl_loss_base = F.cross_entropy(logits, labels, reduction='none')  # (N,)

        # Apply degree-based reweighting
        w_src = self.compute_degree_weights(src_degrees)
        w_dst_pos = self.compute_degree_weights(dst_degrees_pos)
        w_dst_neg = self.compute_degree_weights(dst_degrees_neg)

        w_pair_pos = w_src * w_dst_pos
        w_pair_neg = w_src * w_dst_neg
        w_avg = (w_pair_pos + w_pair_neg) / 2.0

        dcl_loss = dcl_loss_base * w_avg

        # ========================================
        # HYBRID COMBINATION
        # ========================================

        hybrid_loss = self.alpha_focal * focal_loss + self.alpha_dcl * dcl_loss

        # Reduction
        if self.reduction == 'mean':
            return hybrid_loss.mean()
        elif self.reduction == 'sum':
            return hybrid_loss.sum()
        else:
            return hybrid_loss





