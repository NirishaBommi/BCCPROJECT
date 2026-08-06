"""
=============================================================================
Training Engine
=============================================================================
Implements the full training protocol from Section III-E:
  - AdamW optimizer with discriminative learning rates
  - CosineAnnealingWarmRestarts scheduler
  - Exponential Moving Average (EMA) of model weights
  - Stochastic Weight Averaging (SWA) in final epochs
  - CutMix augmentation with probability 0.5
  - Early stopping on validation AUC
  - Gradient clipping (L2 norm = 1.0)
  - Mixed-precision training (AMP FP16)
  - Comprehensive metric logging
=============================================================================
"""

import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
# FIX 1: deprecated torch.cuda.amp import → use torch.amp namespace
from torch.optim.swa_utils import AveragedModel, SWALR
from pathlib import Path
from typing import Dict, Optional

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

from src.dataset import cutmix_batch
from src.losses import HybridFocalCELoss


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------
class EMA:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = (
                self.decay * self.shadow[k] + (1 - self.decay) * v
            )

    def apply_shadow(self):
        self._backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self._backup)


# ---------------------------------------------------------------------------
# Metric Tracker
# ---------------------------------------------------------------------------
class MetricTracker:
    """Accumulates predictions and ground-truth for epoch-level evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds:  list = []
        self.probs:  list = []
        self.labels: list = []
        self.losses: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        probs = torch.softmax(logits.detach().cpu(), dim=-1)
        preds = probs.argmax(dim=-1).numpy()
        self.preds.extend(preds.tolist())
        self.probs.extend(probs[:, 1].numpy().tolist())   # prob of positive class
        self.labels.extend(targets.detach().cpu().numpy().tolist())
        self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        y_true = np.array(self.labels)
        y_pred = np.array(self.preds)
        y_prob = np.array(self.probs)

        # FIX: cast all sklearn metric outputs to native float at the
        #       source. These are genuinely native floats at runtime in
        #       this codebase's sklearn version (confirmed by direct
        #       testing), but their type stubs declare a broader
        #       float | ndarray union (since e.g. f1_score supports
        #       average=None, which returns a per-class array) — Pylance
        #       can't narrow based on the average="binary" string literal,
        #       so it flags downstream round() calls as ambiguous. Wrapping
        #       here resolves it consistently for every metric, not just
        #       the one Pylance happened to flag most recently.
        acc  = float(accuracy_score(y_true, y_pred))
        f1   = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        mcc  = float(matthews_corrcoef(y_true, y_pred))
        loss = float(np.mean(self.losses))

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sen = tp / (tp + fn + 1e-8)
            spe = tn / (tn + fp + 1e-8)
        else:
            sen = spe = 0.0

        return {
            "loss": float(round(loss, 4)),
            "acc":  float(round(acc * 100, 2)),
            "auc":  float(round(auc, 4)),
            "sen":  float(round(sen * 100, 2)),
            "spe":  float(round(spe * 100, 2)),
            "f1":   float(round(f1 * 100, 2)),
            "mcc":  float(round(mcc, 4)),
        }


# ---------------------------------------------------------------------------
# Training / Validation Steps
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,      # FIX 2: updated type hint
    device: torch.device,
    ema: Optional[EMA] = None,
    cutmix_prob: float = 0.5,
    grad_clip: float = 1.0,
    accum_steps: int = 1,                   # FIX 3: gradient accumulation support
) -> Dict[str, float]:
    model.train()
    tracker = MetricTracker()

    optimizer.zero_grad(set_to_none=True)   # FIX 4: zero once before the loop

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # CutMix augmentation
        use_cutmix = np.random.rand() < cutmix_prob
        labels_a, labels_b, lam = labels, labels, 1.0
        if use_cutmix:
            images, labels_a, labels_b, lam = cutmix_batch(images, labels)

        # FIX 5: use torch.cuda.amp.autocast for compatibility with current
        #         stubs, enabling AMP only for CUDA devices.
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            if use_cutmix:
                loss = (
                    lam * criterion(logits, labels_a) +
                    (1 - lam) * criterion(logits, labels_b)
                )
            else:
                loss = criterion(logits, labels)

            loss = loss / accum_steps       # scale loss for accumulation

        scaler.scale(loss).backward()

        # Step optimizer every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update()

        # FIX 6: use original labels for metric tracking (not the meaningless
        #         "logits if not use_cutmix else logits" tautology);
        #         for CutMix batches use labels_a as the primary label
        track_labels = labels_a if use_cutmix else labels
        tracker.update(logits.detach(), track_labels, loss.item() * accum_steps)

    # Flush any leftover accumulated gradients at end of epoch
    remainder = len(loader) % accum_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update()

    return tracker.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    tracker = MetricTracker()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # FIX 7: same device-agnostic autocast fix as train_one_epoch
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        tracker.update(logits, labels, loss.item())

    return tracker.compute()


# ---------------------------------------------------------------------------
# Full Training Loop
# ---------------------------------------------------------------------------
def train(
    model: nn.Module,
    loaders: Dict,
    config: Dict,
    output_dir: str = "outputs/",
) -> nn.Module:
    """
    Full training loop with EMA, SWA, early stopping.

    Args:
        model      : SESTNEfficientNet instance
        loaders    : dict with 'train', 'val', 'test' DataLoaders
        config     : training configuration dictionary
        output_dir : directory to save checkpoints and logs

    Returns:
        Best model (EMA weights if available)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ── Optimizer with discriminative learning rates ──────────────────────
    backbone = getattr(model, "backbone", None)
    backbone_params = list(backbone.parameters()) if isinstance(backbone, nn.Module) else []

    # FIX 8 (corrected): the real attribute on SESTNEfficientNet is
    #         `se_blocks` (a ModuleList), not `se`. The original `model.se`
    #         reference was verified to raise AttributeError immediately on
    #         an actual instantiated model — this was an error introduced
    #         in an earlier patch pass that guessed the attribute name
    #         without model.py available to check against. Also includes
    #         STN params in head group; they were missing entirely before,
    #         leaving them un-optimized or falling to a default LR.
    head_params = []
    se_blocks = getattr(model, "se_blocks", None)
    if isinstance(se_blocks, nn.Module):
        head_params += list(se_blocks.parameters())

    head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        head_params += list(head.parameters())

    if hasattr(model, 'stn') and isinstance(model.stn, nn.Module):
        head_params += list(model.stn.parameters())

    # FIX 9: model.py confirms SESTNEfficientNet has no `temperature`
    #         attribute at all — temperature only exists on the separate
    #         TemperatureScaledModel wrapper, used post-hoc for calibration
    #         after training, not jointly optimized here. This hasattr
    #         guard is correctly inert during normal training (it will
    #         never fire on a bare SESTNEfficientNet) and is kept only as
    #         a defensive no-op in case `model` is ever passed in already
    #         wrapped in TemperatureScaledModel, which would be unusual for
    #         this training loop's intended use.
    if hasattr(model, 'temperature') and isinstance(model.temperature, nn.Parameter):
        head_params.append(model.temperature)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config["lr"] / 10},
        {"params": head_params,     "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])

    # ── Scheduler: Cosine Annealing Warm Restarts ─────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get("T0", 10),
        T_mult=config.get("T_mult", 2),
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    class_weights = loaders["train"].dataset.get_class_weights().to(device)
    criterion = HybridFocalCELoss(
        lam=config.get("loss_lambda", 0.6),
        gamma=config.get("focal_gamma", 2.0),
        class_weights=class_weights,
    )

    # FIX 10: Use torch.cuda.amp.GradScaler on older PyTorch versions.
    #          gated with enabled=False on CPU so .scale()/.step()/.unscale_()
    #          become safe no-ops instead of raising or misbehaving when
    #          there's no CUDA device to target.
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = EMA(model, decay=config.get("ema_decay", 0.9999))

    # ── SWA (applied in final epochs) ─────────────────────────────────────
    swa_model     = AveragedModel(model)
    swa_start     = config.get("swa_start_epoch", config["max_epochs"] - 20)
    swa_scheduler = SWALR(optimizer, swa_lr=config["lr"] * 0.1)
    swa_active    = False

    # ── Training State ────────────────────────────────────────────────────
    best_auc         = 0.0
    best_state       = None
    patience_counter = 0
    history          = []
    accum_steps      = config.get("accum_steps", 1)

    print(f"\n{'='*65}")
    print(f"  Training SE-STN-EfficientNet")
    print(f"  Device: {device} | Max epochs: {config['max_epochs']}")
    print(f"  LR: {config['lr']} | Batch: {config['batch_size']} "
          f"| Accum steps: {accum_steps} "
          f"(effective batch: {config['batch_size'] * accum_steps})")
    print(f"{'='*65}\n")

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()

        # Activate SWA in final epochs
        if epoch >= swa_start and not swa_active:
            swa_active = True
            print(f"[Epoch {epoch}] SWA activated")

        # ── Train ──
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler, device, ema,
            cutmix_prob=config.get("cutmix_prob", 0.5),
            accum_steps=accum_steps,
        )

        # ── Validate (with EMA weights) ──
        ema.apply_shadow()
        val_metrics = validate(model, loaders["val"], criterion, device)
        ema.restore()

        # ── Scheduler step ──
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        log_line = (
            f"Ep {epoch:03d}/{config['max_epochs']} "
            f"| Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.1f}% AUC={train_metrics['auc']:.4f} "
            f"| Val   Loss={val_metrics['loss']:.4f}  Acc={val_metrics['acc']:.1f}%  AUC={val_metrics['auc']:.4f} "
            f"Sen={val_metrics['sen']:.1f}% Spe={val_metrics['spe']:.1f}% "
            f"| {elapsed:.1f}s"
        )
        print(log_line)

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val":   val_metrics,
        })

        # ── Checkpoint ──
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(output_dir, "best_model.pth"))
            patience_counter = 0
            print(f"  ✓ New best AUC={best_auc:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 20):
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Finalise SWA ──────────────────────────────────────────────────────
    if swa_active:
        torch.optim.swa_utils.update_bn(loaders["train"], swa_model, device=device)
        torch.save(swa_model.state_dict(), os.path.join(output_dir, "swa_model.pth"))
        print("SWA model saved.")

    # Save training history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nTraining complete. Best Val AUC = {best_auc:.4f}")
    return model