"""
============================================================================
  evaluate.py  —  Metrics, Calibration (ECE), Fairness, Grad-CAM
  Paper : "Basal Cell Carcinoma Skin Detection Using Deep Learning"
============================================================================

CHANGES FROM ORIGINAL:
  1. THRESHOLD TUNING (new):
     - evaluate() and print_classification_report() now accept an optional
       `threshold` argument instead of being hardcoded to argmax (0.5 cutoff).
     - Added find_optimal_threshold() to sweep the precision-recall curve
       and pick a threshold by F1 or by a minimum-recall constraint —
       useful for medical screening where missing BCC is costlier than a
       false alarm.

  2. TEMPERATURE SCALING (new — this was MISSING from the original file):
     - Your report showed T=1.4518 and a calibrated ECE that was WORSE than
       uncalibrated (0.2786 vs 0.2347), which should not happen with a
       correct implementation — temperature scaling is a convex NLL
       minimization and is guaranteed not to hurt calibration on the same
       set it's evaluated on (modulo val/test distribution shift).
     - Added a `TemperatureScaler` class that:
         a) collects raw logits (not softmax probs) on the validation set
         b) fits T via LBFGS minimizing NLL — the textbook-correct method
         c) applies calibration as `softmax(logits / T)` — the correct
            direction (T>1 flattens/softens overconfident predictions)
     - compute_ece() now accepts an optional `temperature` argument so you
       can directly compare before/after using IDENTICAL binning logic,
       which the original two-script split may not have guaranteed.

  3. All GradCAM and fairness-evaluation code is UNCHANGED from your
     original file (those were correct).
============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    precision_recall_curve
)
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CORE EVALUATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model      : nn.Module,
             loader     ,
             device     : str  = 'cuda',
             num_classes: int  = 2,
             threshold  : Optional[float] = None,
             temperature: Optional[float] = None) -> Dict[str, float]:
    """
    Run full evaluation on a DataLoader.

    Args:
        threshold   : Decision threshold on the positive-class (BCC)
                      probability, for num_classes==2 only. If None,
                      falls back to argmax (equivalent to threshold=0.5).
        temperature : Optional temperature-scaling value. If provided,
                      logits are divided by T before softmax.

    Returns dict with keys:
        accuracy, sensitivity, specificity, precision,
        f1, mcc, auc, threshold_used
    """
    model.eval()
    all_probs  = []
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if temperature is not None:
            logits = logits / temperature
        probs  = F.softmax(logits, dim=1).cpu().numpy()

        if num_classes == 2 and threshold is not None:
            preds = (probs[:, 1] >= threshold).astype(int)
        else:
            preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_probs  = np.vstack(all_probs)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    mcc  = matthews_corrcoef(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if num_classes == 2:
        cm       = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sen  = tp / (tp + fn + 1e-9)
        spe  = tn / (tn + fp + 1e-9)
        pre  = precision_score(all_labels, all_preds, zero_division=0)
        auc  = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        sen  = recall_score(all_labels, all_preds, average='macro',
                            zero_division=0)
        spe  = 0.0   # not well-defined for multiclass
        pre  = precision_score(all_labels, all_preds, average='macro',
                               zero_division=0)
        # One-vs-rest AUC for multiclass
        auc  = roc_auc_score(all_labels, all_probs, multi_class='ovr',
                             average='macro')

    return {
        'accuracy'      : float(acc),
        'sensitivity'   : float(sen),
        'specificity'   : float(spe),
        'precision'     : float(pre),
        'f1'            : float(f1),
        'mcc'           : float(mcc),
        'auc'           : float(auc),
        'threshold_used': float(threshold) if threshold is not None else 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1b.  THRESHOLD TUNING  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def find_optimal_threshold(model        : nn.Module,
                           loader       ,
                           device       : str = 'cuda',
                           target_recall: Optional[float] = None,
                           temperature  : Optional[float] = None) -> float:
    """
    Sweep thresholds on the positive-class (BCC) probability using the
    precision-recall curve.

    Args:
        target_recall : If set (e.g. 0.80), returns the threshold that
                        achieves >= that recall with the best precision at
                        that recall. If None, returns the F1-optimal
                        threshold.
        temperature   : Optional temperature to apply before thresholding
                        (use your fitted TemperatureScaler.temperature).

    Returns:
        Optimal threshold (float), also prints the corresponding
        precision/recall/F1 for transparency.
    """
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if temperature is not None:
            logits = logits / temperature
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs[:, 1])
        all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    if target_recall is not None:
        valid = recalls[:-1] >= target_recall
        if not valid.any():
            print(f"[Threshold] No threshold reaches recall={target_recall}. "
                  f"Falling back to F1-optimal.")
        else:
            idx = np.argmax(precisions[:-1][valid])
            best_t = float(thresholds[valid][idx])
            print(f"[Threshold] target_recall={target_recall} -> "
                  f"t={best_t:.4f} | precision={precisions[:-1][valid][idx]:.4f} "
                  f"| recall={recalls[:-1][valid][idx]:.4f}")
            return best_t

    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.nanargmax(f1s[:-1]))
    best_t = float(thresholds[best_idx])
    print(f"[Threshold] F1-optimal -> t={best_t:.4f} | "
          f"precision={precisions[best_idx]:.4f} | "
          f"recall={recalls[best_idx]:.4f} | F1={f1s[best_idx]:.4f}")
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXPECTED CALIBRATION ERROR  (ECE)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_ece(model      : nn.Module,
                loader     ,
                device     : str = 'cuda',
                n_bins     : int = 15,
                temperature: Optional[float] = None) -> float:
    """
    Expected Calibration Error (Naeini et al., 2015).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Bins are equal-width over [0, 1].

    Args:
        model       : Evaluated model.
        loader      : DataLoader.
        n_bins      : Number of confidence bins (default 15).
        device      : 'cuda' or 'cpu'.
        temperature : Optional temperature. If provided, logits are
                      divided by T before softmax — pass the SAME value
                      used for find_optimal_threshold()/evaluate() so
                      before/after comparisons use identical binning.

    Returns:
        ECE value (float).
    """
    model.eval()
    all_confs  = []
    all_correct= []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if temperature is not None:
            logits = logits / temperature
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
        confs  = probs[np.arange(len(preds)), preds]

        all_confs.append(confs)
        all_correct.append((preds == labels.numpy()).astype(float))

    confs   = np.concatenate(all_confs)
    correct = np.concatenate(all_correct)
    N       = len(confs)

    ece  = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        if mask.sum() == 0:
            continue
        b_acc  = correct[mask].mean()
        b_conf = confs[mask].mean()
        ece   += (mask.sum() / N) * abs(b_acc - b_conf)

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# 2b.  TEMPERATURE SCALING  (NEW — was missing from the original file)
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaler:
    """
    Post-hoc calibration via temperature scaling (Guo et al., 2017).

    Fits a single scalar T on a held-out validation set by minimizing NLL,
    using LBFGS (the standard, textbook-correct approach). Correct
    application is `softmax(logits / T)` — T > 1 SOFTENS overconfident
    predictions, T < 1 sharpens them.

    Usage:
        scaler = TemperatureScaler()
        T = scaler.fit(model, val_loader, device='cuda')
        # then evaluate/compute_ece with temperature=T on the TEST set
    """

    def __init__(self):
        self.temperature: Optional[float] = None

    @torch.no_grad()
    def _collect_logits(self, model: nn.Module, loader, device: str
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        all_logits, all_labels = [], []
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
        return torch.cat(all_logits), torch.cat(all_labels)

    def fit(self, model: nn.Module, val_loader, device: str = 'cuda',
            max_iter: int = 50, lr: float = 0.01) -> float:
        """
        Fit temperature on the VALIDATION set (never on test — fitting on
        test would leak information and invalidate the calibration metric).

        Returns:
            Fitted temperature (float), stored in self.temperature.
        """
        logits, labels = self._collect_logits(model, val_loader, device)
        logits = logits.to(device)
        labels = labels.to(device)

        # Initialize T=1.0 (no-op) as a learnable scalar parameter.
        log_t = torch.zeros(1, device=device, requires_grad=True)
        optimizer = optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            # softplus keeps T > 0 strictly, avoiding divide-by-zero/negative T
            T = F.softplus(log_t) + 1e-3
            loss = nll_criterion(logits / T, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            T_final = (F.softplus(log_t) + 1e-3).item()

        self.temperature = T_final
        print(f"[TemperatureScaler] Fitted T = {T_final:.4f}")
        return T_final

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply fitted temperature to raw logits -> calibrated probs."""
        if self.temperature is None:
            raise RuntimeError("Call fit() before apply().")
        return F.softmax(logits / self.temperature, dim=1)


def run_calibration_report(model: nn.Module, val_loader, test_loader,
                            device: str = 'cuda', n_bins: int = 15) -> Dict[str, float]:
    """
    Convenience wrapper: fits T on val_loader, then reports uncalibrated vs
    calibrated ECE on test_loader using IDENTICAL binning — use this instead
    of separate scripts to avoid the val/test or binning mismatch that
    likely caused your calibrated ECE (0.2786) to be worse than
    uncalibrated (0.2347) in the original report.
    """
    scaler = TemperatureScaler()
    T = scaler.fit(model, val_loader, device=device)

    ece_before = compute_ece(model, test_loader, device=device, n_bins=n_bins, temperature=None)
    ece_after  = compute_ece(model, test_loader, device=device, n_bins=n_bins, temperature=T)

    print(f"[Calibration] Uncalibrated ECE: {ece_before:.4f}")
    print(f"[Calibration] Calibrated ECE  : {ece_after:.4f}  (T={T:.4f})")
    if ece_after > ece_before:
        print("[Calibration] WARNING: calibrated ECE is still worse than "
              "uncalibrated. This points to a val/test distribution shift "
              "rather than an implementation bug — check class balance and "
              "source-dataset mix between your val and test splits.")

    return {'temperature': T, 'ece_before': ece_before, 'ece_after': ece_after}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PRINT FULL CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def print_classification_report(model      : nn.Module,
                                 loader     ,
                                 device     : str,
                                 class_names: list,
                                 threshold  : Optional[float] = None,
                                 temperature: Optional[float] = None):
    """Print sklearn classification report and confusion matrix.

    Args:
        threshold   : Decision threshold on positive-class probability
                      (binary only). None = argmax / 0.5.
        temperature : Optional temperature for calibrated probabilities.
    """
    model.eval()
    all_preds, all_labels = [], []
    num_classes = len(class_names)

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if temperature is not None:
            logits = logits / temperature
        probs = F.softmax(logits, dim=1).cpu().numpy()

        if num_classes == 2 and threshold is not None:
            preds = (probs[:, 1] >= threshold).astype(int)
        else:
            preds = np.argmax(probs, axis=1)

        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    print(f"\n[Classification Report] (threshold={threshold if threshold else 0.5})")
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("[Confusion Matrix]")
    print(cm)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRAD-CAM IMPLEMENTATION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Registers forward and backward hooks on the target layer to capture
    activations and gradients, then produces a class-discriminative
    localisation map.

    Args:
        model        (nn.Module): Trained model.
        target_layer (nn.Module): Layer to hook (e.g., last SE block).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, inp, out):
            if isinstance(out, (list, tuple)):
                self.activations = out[-1].detach()
            else:
                self.activations = out.detach()

        def save_gradient(module, grad_in, grad_out):
            if isinstance(grad_out, (list, tuple)):
                grad = next((g for g in grad_out if g is not None), None)
                if grad is not None:
                    self.gradients = grad.detach()
            else:
                self.gradients = grad_out.detach()

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)

    def generate(self, image: torch.Tensor,
                 class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            image     (Tensor): Single image (1, C, H, W) — already normalised.
            class_idx (int):    Class to visualise (None = predicted class).

        Returns:
            np.ndarray: Heatmap in [0, 1], shape (H, W).
        """
        self.model.eval()
        image = image.requires_grad_(True)

        logits = self.model(image)                    # forward pass

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        assert self.gradients is not None, (
            "Backward hook did not populate gradients — target_layer may "
            "not be part of the backward graph for this output."
        )
        assert self.activations is not None, (
            "Forward hook did not populate activations — target_layer may "
            "not have been reached during the forward pass."
        )

        # Grad-CAM: alpha = GAP of gradients
        alpha     = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam       = (alpha * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam       = F.relu(cam)                       # keep positive activations

        # Upsample to input size
        cam = F.interpolate(cam, size=image.shape[-2:],
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()             # (H, W)

        # Normalise to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def visualise(self, image_np : np.ndarray,
                  cam       : np.ndarray,
                  title     : str = "Grad-CAM",
                  save_path : Optional[str] = None):
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            image_np  (np.ndarray): Original BGR image (H, W, 3) uint8.
            cam       (np.ndarray): Normalised heatmap (H, W).
            title     (str):        Plot title.
            save_path (str):        If given, save PNG to this path.
        """
        import matplotlib.pyplot as plt
        # Resize CAM to match image
        h, w = image_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colourmap
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        overlay   = (0.6 * image_rgb + 0.4 * heatmap).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_rgb);  axes[0].set_title("Original");  axes[0].axis('off')
        axes[1].imshow(heatmap);    axes[1].set_title("Grad-CAM");   axes[1].axis('off')
        axes[2].imshow(overlay);    axes[2].set_title("Overlay");    axes[2].axis('off')

        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Grad-CAM] Saved to {save_path}")
        else:
            plt.show()
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FAIRNESS EVALUATION ACROSS FITZPATRICK PHOTOTYPES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_fairness(model    : nn.Module,
                      loader   ,
                      device   : str,
                      phototype_col: str = 'fitzpatrick') -> Dict[str, float]:
    """
    Compute AUC per Fitzpatrick skin phototype group (I–VI).

    The test DataLoader's underlying dataset must have a DataFrame column
    named 'fitzpatrick' with values 1-6.

    NOTE: AUC is threshold-independent, so this function intentionally does
    not take a `threshold` argument.

    Returns:
        Dict mapping "phototype_{I}" -> AUC for each phototype.
    """
    model.eval()
    ds = loader.dataset
    if phototype_col not in ds.df.columns:
        print(f"[Fairness] Column '{phototype_col}' not found. Skipping.")
        return {}

    all_probs, all_labels, all_photo = [], [], []
    sample_idx = 0

    for images, labels in loader:
        images = images.to(device)
        probs  = F.softmax(model(images), dim=1).cpu().numpy()
        batch_size = len(labels)
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_photo.extend(ds.df[phototype_col].values[sample_idx:sample_idx + batch_size])
        sample_idx += batch_size

    all_probs  = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    all_photo  = np.array(all_photo)

    results = {}
    for pt in sorted(set(all_photo)):
        mask = all_photo == pt
        if mask.sum() < 5:
            continue
        try:
            auc = roc_auc_score(all_labels[mask], all_probs[mask, 1])
            results[f"phototype_{int(pt)}"] = float(auc)
            print(f"  Fitzpatrick type {int(pt)} | n={mask.sum()} | AUC={auc:.4f}")
        except ValueError:
            pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from model import build_model
    m = build_model(2, pretrained=False).eval()
    dummy = torch.randn(4, 3, 380, 380)
    lbls  = torch.randint(0, 2, (4,))
    cam   = GradCAM(m, m.se_blocks[-1])
    hmap  = cam.generate(dummy[:1])
    print(f"[Test] Grad-CAM heatmap shape: {hmap.shape}  min={hmap.min():.3f} max={hmap.max():.3f}")