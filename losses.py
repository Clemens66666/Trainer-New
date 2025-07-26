# -*- coding: utf-8 -*-
"""
Loss-Helfer für Klassengewichtung + Label-Smoothing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSmoothBCE(nn.Module):
    """
    BCEWithLogits +
        • pos_weight  gegen Klassen-Imbalance
        • Label-Smoothing ε (0 → klassisch BCE)
    """
    def __init__(self, pos_weight: float = 1.0, smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("pos_weight",
                             torch.tensor(float(pos_weight)))
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """targets ∈ {0,1}  oder bereits weich ∈ [0,1]"""
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="mean"
        )
        return loss
