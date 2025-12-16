"""
Focal Loss implementation for imbalanced classification

Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
https://arxiv.org/abs/1708.02002

Focal Loss est conçu pour gérer le déséquilibre de classes en réduisant
l'importance des exemples faciles et en se concentrant sur les exemples difficiles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss pour la classification binaire.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    où:
    - p_t est la probabilité prédite pour la vraie classe
    - alpha_t est le poids de la classe (pour gérer le déséquilibre)
    - gamma est le facteur de focalisation (focusing parameter)

    Args:
        alpha (float): Poids pour la classe positive (défaut: 0.25)
                      Si alpha=0.25, alors les positifs ont un poids de 0.25
                      et les négatifs ont un poids de 0.75
        gamma (float): Exposant de focalisation (défaut: 2.0)
                      gamma=0 → équivalent à BCE
                      gamma=2 → réduction forte des exemples faciles
        reduction (str): 'none', 'mean', 'sum' (défaut: 'mean')

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> predictions = torch.sigmoid(logits)  # probabilités
        >>> loss = criterion(predictions, targets)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Probabilités prédites (après sigmoid), shape (N,)
            targets (Tensor): Labels binaires (0 ou 1), shape (N,)

        Returns:
            Tensor: Focal loss
        """
        # Assurer que inputs est bien entre 0 et 1
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)

        # Binary Cross Entropy de base
        BCE = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))

        # Calcul du modulating factor: (1 - p_t)^gamma
        # p_t = p si y=1, sinon (1-p)
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        modulating_factor = (1 - p_t) ** self.gamma

        # Calcul de alpha_t
        # alpha_t = alpha si y=1, sinon (1-alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal Loss
        focal_loss = alpha_t * modulating_factor * BCE

        # Réduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Variante adaptative de Focal Loss qui ajuste automatiquement alpha
    en fonction du ratio de classes dans le batch.

    Utile quand le déséquilibre varie d'un batch à l'autre.
    """

    def __init__(self, gamma=2.0, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Probabilités prédites (après sigmoid), shape (N,)
            targets (Tensor): Labels binaires (0 ou 1), shape (N,)

        Returns:
            Tensor: Adaptive Focal loss
        """
        # Clamp pour stabilité numérique
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)

        # Calculer alpha adaptatif basé sur le ratio dans le batch
        num_positives = targets.sum()
        num_total = targets.numel()

        if num_positives > 0:
            # alpha = proportion de négatifs (pour donner plus de poids aux positifs rares)
            alpha = 1.0 - (num_positives / num_total)
        else:
            alpha = 0.5

        # BCE de base
        BCE = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))

        # Modulating factor
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        modulating_factor = (1 - p_t) ** self.gamma

        # Alpha adaptatif
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

        # Focal Loss
        focal_loss = alpha_t * modulating_factor * BCE

        # Réduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_focal_loss():
    """
    Test pour vérifier que Focal Loss fonctionne correctement.
    """
    print("Testing Focal Loss...")

    # Créer des données de test
    torch.manual_seed(42)
    batch_size = 100

    # Simuler des prédictions
    predictions = torch.rand(batch_size)

    # Créer des targets très déséquilibrés (5% positifs)
    targets = torch.zeros(batch_size)
    targets[:5] = 1.0

    # Comparer BCE vs Focal Loss
    criterion_bce = nn.BCELoss()
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)

    loss_bce = criterion_bce(predictions, targets)
    loss_focal = criterion_focal(predictions, targets)

    print(f"\nDataset: {int(targets.sum())}/{batch_size} positifs ({targets.mean()*100:.1f}%)")
    print(f"Binary Cross-Entropy Loss: {loss_bce.item():.4f}")
    print(f"Focal Loss (gamma=2.0):     {loss_focal.item():.4f}")

    # Test avec différents gammas
    print("\nImpact de gamma:")
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
        criterion = FocalLoss(alpha=0.25, gamma=gamma)
        loss = criterion(predictions, targets)
        print(f"  gamma={gamma:.1f}: loss={loss.item():.4f}")

    # Test avec exemples faciles vs difficiles
    print("\nComparaison facile vs difficile:")

    # Exemple facile: modèle très confiant et correct
    easy_pred = torch.tensor([0.95, 0.05, 0.98, 0.02])
    easy_target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # Exemple difficile: modèle pas sûr
    hard_pred = torch.tensor([0.55, 0.45, 0.58, 0.42])
    hard_target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    criterion_bce = nn.BCELoss()
    criterion_focal = FocalLoss(alpha=0.5, gamma=2.0)

    print("\n  Exemples faciles (modèle confiant):")
    print(f"    BCE:   {criterion_bce(easy_pred, easy_target).item():.4f}")
    print(f"    Focal: {criterion_focal(easy_pred, easy_target).item():.4f}")

    print("\n  Exemples difficiles (modèle incertain):")
    print(f"    BCE:   {criterion_bce(hard_pred, hard_target).item():.4f}")
    print(f"    Focal: {criterion_focal(hard_pred, hard_target).item():.4f}")

    print("\n✅ Tests terminés!")


if __name__ == "__main__":
    test_focal_loss()
