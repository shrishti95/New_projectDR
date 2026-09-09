"""Imbalance-aware loss used for paper-faithful benchmark training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WeightedFocalCrossEntropy(nn.Module):
    """Multiclass focal cross-entropy with smoothing and class weights."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        classes = logits.shape[1]
        one_hot = F.one_hot(targets, num_classes=classes).to(logits.dtype)
        smoothed = (
            one_hot * (1.0 - self.label_smoothing)
            + self.label_smoothing / classes
        )
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        focal_factor = (1.0 - probabilities).pow(self.gamma)
        weighted_loss = (
            -smoothed
            * focal_factor
            * log_probabilities
            * self.class_weights.unsqueeze(0)
        )
        return weighted_loss.sum(dim=1).mean()
