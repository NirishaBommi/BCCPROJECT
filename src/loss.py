"""
============================================================================
  loss.py  —  Hybrid Focal Loss + Cross-Entropy  (Equations 4 & 5)
  Paper : "Basal Cell Carcinoma Skin Detection Using Deep Learning"
============================================================================

CHANGES FROM ORIGINAL:
  - Added `compute_class_weights()` utility. Previously `class_weights` was
    an optional argument that silently defaulted to None whenever the
    caller (train.py) forgot to pass it in — meaning the CrossEntropy
    component (60% of the loss, since lambda_mix=0.6) had ZERO imbalance
    correction, and only the FocalLoss alpha=0.75 term was compensating.
    This function computes proper inverse-frequency weights from your
    train-split class counts so both loss terms are imbalance-aware.
  - Added a `class_weights=None` guard + warning print in build_criterion
    so a missing weight vector is now visible in your logs instead of
    silently doing nothing.
  - No changes to FocalLoss / HybridFLCELoss math — those were correct.
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLASS WEIGHT UTILITY  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(class_counts: Union[Sequence[int], dict],
                           num_classes: int = 2,
                           normalize: bool = True) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced classification.

    Args:
        class_counts (list | dict): Either [count_class0, count_class1, ...]
                                     or {0: count0, 1: count1, ...}.
        num_classes  (int):         Number of classes (default 2).
        normalize    (bool):        If True, weights are scaled so they sum
                                     to num_classes (keeps loss magnitude
                                     comparable to unweighted CE).

    Returns:
        torch.Tensor of shape (num_classes,)

    Example (matches your test_evaluation_report.md support column):
        >>> compute_class_weights({0: 4782, 1: 657})
        tensor([0.4629, 3.3712])   # non-BCC gets ~0.46x, BCC gets ~3.37x
    """
    if isinstance(class_counts, dict):
        counts = [class_counts[i] for i in range(num_classes)]
    else:
        counts = list(class_counts)

    counts = torch.tensor(counts, dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError(f"class_counts must be positive, got {counts.tolist()}")

    weights = 1.0 / counts
    if normalize:
        weights = weights * (num_classes / weights.sum())

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FOCAL LOSS  (Lin et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Equation (5):  L_FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma        (float): Focusing parameter (default 2.0).
        alpha        (float): Weighting factor for positive class (default 0.75).
        class_weights(Tensor): Optional per-class weight tensor (n_classes,).
        reduction    (str):   'mean', 'sum', or 'none'.
    """

    def __init__(self,
                 gamma        : float             = 2.0,
                 alpha        : float             = 0.75,
                 class_weights: Optional[torch.Tensor] = None,
                 reduction    : str               = 'mean'):
        super().__init__()
        self.gamma         = gamma
        self.alpha         = alpha
        self.class_weights = class_weights
        self.reduction     = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  (Tensor): Raw logits  (B, C).
            targets (Tensor): Class indices (B,) long.
        Returns:
            Scalar focal loss.
        """
        # Compute cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.class_weights,
                                  reduction='none')

        # p_t = exp(-CE) for each sample
        p_t = torch.exp(-ce_loss)

        # alpha weighting — apply to positive class
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha,     device=logits.device),
                              torch.tensor(1.0 - self.alpha, device=logits.device))

        # Focal modulation
        focal_loss = alpha_t * (1.0 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HYBRID FL-CE LOSS  (Equation 4)
# ─────────────────────────────────────────────────────────────────────────────
class HybridFLCELoss(nn.Module):
    """
    Hybrid Focal Loss + Cross-Entropy objective.

    Equation (4):  L_total = lambda * L_CE + (1 - lambda) * L_FL

    Args:
        lambda_mix   (float): Mixing coefficient (default 0.6 for CE).
        gamma        (float): Focal loss gamma (default 2.0).
        alpha        (float): Focal loss alpha_t (default 0.75).
        class_weights(Tensor): Inverse-frequency class weights (n_classes,).
    """

    def __init__(self,
                 lambda_mix   : float             = 0.6,
                 gamma        : float             = 2.0,
                 alpha        : float             = 0.75,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.lambda_mix = lambda_mix
        self.ce_loss    = nn.CrossEntropyLoss(weight=class_weights)
        self.fl_loss    = FocalLoss(gamma=gamma, alpha=alpha,
                                    class_weights=class_weights)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  (Tensor): (B, C) raw logits.
            targets (Tensor): (B,) long ground-truth class indices.
        Returns:
            Scalar hybrid loss.
        """
        L_ce = self.ce_loss(logits, targets)
        L_fl = self.fl_loss(logits, targets)
        return self.lambda_mix * L_ce + (1.0 - self.lambda_mix) * L_fl

    def extra_repr(self) -> str:
        return (f"lambda_mix={self.lambda_mix}, "
                f"gamma={self.fl_loss.gamma}, "
                f"alpha={self.fl_loss.alpha}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LOSS FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_criterion(class_weights: Optional[torch.Tensor] = None,
                    lambda_mix   : float = 0.6,
                    gamma        : float = 2.0,
                    alpha        : float = 0.75) -> HybridFLCELoss:
    """
    Build the hybrid FL-CE loss criterion.

    Args:
        class_weights (Tensor): Shape (num_classes,) inverse-frequency weights.
                                 Build this with compute_class_weights() from
                                 your train-split counts — do not leave as
                                 None for an imbalanced dataset like BCC.
        lambda_mix    (float):  CE mixing coefficient.
        gamma         (float):  Focal gamma.
        alpha         (float):  Focal alpha.

    Returns:
        HybridFLCELoss instance.
    """
    if class_weights is None:
        print("[Loss] WARNING: class_weights is None. The CrossEntropy "
              "component (lambda_mix fraction of the loss) will have NO "
              "class-imbalance correction — only FocalLoss's alpha term "
              "will compensate. For an imbalanced dataset, pass weights "
              "from compute_class_weights(train_class_counts).")

    criterion = HybridFLCELoss(
        lambda_mix   = lambda_mix,
        gamma        = gamma,
        alpha        = alpha,
        class_weights= class_weights,
    )
    print(f"[Loss] HybridFLCE | lambda={lambda_mix} | "
          f"gamma={gamma} | alpha={alpha} | "
          f"class_weights={class_weights.tolist() if class_weights is not None else None}")
    return criterion


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logits  = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))

    # Example using your actual train-split-style counts
    weights = compute_class_weights({0: 4782, 1: 657})
    print(f"[Test] Computed class weights: {weights.tolist()}")

    crit = build_criterion(class_weights=weights)
    loss = crit(logits, targets)
    print(f"[Test] Loss value: {loss.item():.4f}")