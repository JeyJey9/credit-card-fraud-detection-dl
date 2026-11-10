"""
Loss functions for imbalanced binary classification:
1. Standard Binary Cross-Entropy (BCEWithLogitsLoss)
2. Cost-Sensitive Weighted Binary Cross-Entropy (Custom Class Weights)
3. Binary Focal Loss (Lin et al., ICCV 2017 [31])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Numerically stable implementation of Binary Focal Loss from:
    Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár,
    "Focal Loss for Dense Object Detection", ICCV 2017 [31].

    Formula:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where:
        p = sigmoid(logits)
        p_t = p if y == 1 else (1 - p)
        alpha_t = alpha if y == 1 else (1 - alpha)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable BCE with logits: -log(p_t)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow((1.0 - p_t).clamp(min=1e-8), self.gamma)

        # Alpha weighting factor
        if self.alpha is not None and self.alpha >= 0.0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_loss = alpha_t * modulating_factor * bce_loss
        else:
            focal_loss = modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
