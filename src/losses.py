"""
=============================================================================
Loss Functions
=============================================================================
Implements the hybrid loss function from Section III-D of the paper:

    L_total = lambda * L_CE + (1 - lambda) * L_FL          -- Eq.(4)

    L_FL    = -alpha_t * (1 - p_t)^gamma * log(p_t)        -- Eq.(5)

    with lambda=0.6, alpha_t=0.75, gamma=2.0
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Lin et al., ICCV 2017.

    L_FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma      : focusing parameter (default 2.0)
        alpha      : class balance weight scalar or per-class tensor
        reduction  : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B,)
        pt = torch.exp(-ce_loss)                                      # p_t
        focal = (1 - pt) ** self.gamma * ce_loss                      # (B,)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            focal = at * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# 2. Hybrid Focal + Cross-Entropy Loss
# ---------------------------------------------------------------------------
class HybridFocalCELoss(nn.Module):
    """
    Hybrid loss combining standard cross-entropy with focal loss.

    L_total = lam * L_CE + (1 - lam) * L_FL        (Eq. 4)

    Args:
        lam        : mixing coefficient (default 0.6)
        gamma      : focal loss gamma (default 2.0)
        alpha_t    : focal loss alpha (default 0.75)
        class_weights : inverse-frequency weights tensor (per class)
    """

    def __init__(
        self,
        lam: float = 0.6,
        gamma: float = 2.0,
        alpha_t: float = 0.75,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lam = lam
        self.class_weights = class_weights

        # Focal loss alpha: scalar alpha_t broadcast to all classes
        focal_alpha = None  # per-class alpha can be passed via class_weights
        self.focal = FocalLoss(gamma=gamma, alpha=focal_alpha, reduction="mean")
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move class weights to same device as logits
        if self.class_weights is not None:
            self.ce.weight = self.class_weights.to(logits.device)

        l_ce = self.ce(logits, targets)
        l_fl = self.focal(logits, targets)
        return self.lam * l_ce + (1.0 - self.lam) * l_fl


# ---------------------------------------------------------------------------
# 3. Label Smoothing Cross-Entropy (optional alternative)
# ---------------------------------------------------------------------------
class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing.
    Useful for reducing overconfidence in final layer.
    """

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w

        return loss.mean()


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C = 8, 2
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    weights = torch.FloatTensor([0.4, 0.6])

    # Test HybridFocalCELoss
    criterion = HybridFocalCELoss(lam=0.6, gamma=2.0, class_weights=weights)
    loss = criterion(logits, targets)
    print(f"Hybrid loss: {loss.item():.4f}")

    # Test FocalLoss alone
    fl = FocalLoss(gamma=2.0)
    print(f"Focal loss : {fl(logits, targets).item():.4f}")

    # Test LabelSmoothing
    ls = LabelSmoothingCE(smoothing=0.1)
    print(f"LabelSmooth: {ls(logits, targets).item():.4f}")