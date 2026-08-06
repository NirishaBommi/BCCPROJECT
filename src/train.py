"""
============================================================================
  train.py  —  Training Loop with EMA, SWA, CutMix, AMP
  Paper : "Basal Cell Carcinoma Skin Detection Using Deep Learning"
============================================================================
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
# FIX 1: keep using torch.cuda.amp here for compatibility with current
#         PyTorch type stubs. The mixed-precision context is enabled only
#         for CUDA devices below, so CPU runs stay safe.
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset  import build_dataloaders, cutmix_batch
from model    import build_model
from loss     import build_criterion
from evaluate import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# 1.  OPTIMISER + LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def build_optimiser(model, lr: float = 1e-4, weight_decay: float = 1e-5):
    """
    AdamW with discriminative learning rates:
        backbone layers → lr / 10
        SE, STN, head   → lr

    Args:
        model        : SESTNEfficientNet instance.
        lr           : Base learning rate (default 1e-4).
        weight_decay : AdamW weight decay (default 1e-5).

    Returns:
        torch.optim.AdamW
    """
    backbone_params = list(model.backbone.parameters())
    other_params    = (list(model.se_blocks.parameters()) +
                       list(model.stn.parameters()) +
                       list(model.head.parameters()))

    param_groups = [
        {'params': backbone_params, 'lr': lr / 10},
        {'params': other_params,    'lr': lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=weight_decay
    )
    print(f"[Optim] AdamW | lr={lr} | backbone_lr={lr/10} | wd={weight_decay}")
    return optimizer


def build_scheduler(optimizer, T_0: int = 10, T_mult: int = 2):
    """
    CosineAnnealingWarmRestarts schedule.

    Args:
        T_0    (int): Initial restart period in epochs.
        T_mult (int): Period multiplier after each restart.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-7
    )
    print(f"[LR]   CosineAnnealingWarmRestarts | T_0={T_0} | T_mult={T_mult}")
    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXPONENTIAL MOVING AVERAGE  (EMA)
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    """
    Maintains an exponential moving average of model weights.

    shadow = decay * shadow + (1 - decay) * param
    Stored in a separate copy — does not modify the original model.

    Args:
        model (nn.Module): Model to track.
        decay (float):     EMA decay coefficient (default 0.9999).

    FIX 11: `model` passed in here must be the *unwrapped* module, not a
             DataParallel wrapper. If a DataParallel-wrapped model is deep-
             copied directly, the shadow's state_dict keys keep the
             'module.' prefix while the main model's checkpoint (saved via
             model.module.state_dict()) does not — so ema_state and
             model_state end up with mismatched key names, and loading
             ema_state into a bare model later raises a key-mismatch error.
             Call sites in train() below unwrap before constructing EMA and
             before every .update() call.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for (name, s_param), (_, m_param) in zip(
            self.shadow.named_parameters(), model.named_parameters()
        ):
            s_param.data.mul_(self.decay).add_(
                m_param.data, alpha=1.0 - self.decay
            )

    def get_model(self) -> nn.Module:
        return self.shadow


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ONE EPOCH OF TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model      : nn.Module,
                    loader     ,
                    optimizer  : torch.optim.Optimizer,
                    criterion  : nn.Module,
                    scaler     : GradScaler,
                    device     : str,
                    ema        : EMA,
                    accum_steps: int   = 2,
                    cutmix_prob: float = 0.2) -> float:
    """
    Run one training epoch with:
        - FP16 mixed-precision (AMP)
        - Gradient accumulation (accum_steps)
        - CutMix augmentation (cutmix_prob)
        - Gradient clipping (max norm 1.0)
        - EMA weight update

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply CutMix with probability cutmix_prob
        use_cutmix = (np.random.rand() < cutmix_prob)
        lbl_a, lbl_b, lam = labels, labels, 1.0
        if use_cutmix:
            images, lbl_a, lbl_b, lam = cutmix_batch(images, labels)

        # FIX 2: enable AMP only on CUDA; CPU runs use regular FP32.
        with autocast(enabled=(device == 'cuda')):
            logits = model(images)
            if use_cutmix:
                loss = (lam * criterion(logits, lbl_a) +
                        (1.0 - lam) * criterion(logits, lbl_b))
            else:
                loss = criterion(logits, labels)
            loss = loss / accum_steps                  # scale for accumulation

        scaler.scale(loss).backward()

        # Gradient accumulation — update every accum_steps steps
        # FIX 3: ema.update() now unwraps DataParallel before updating,
        #         matching the EMA object itself being constructed from an
        #         unwrapped model in train() below.
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(getattr(model, 'module', model))

        total_loss += loss.item() * accum_steps

    # FIX 4: flush any leftover accumulated gradients at end of epoch.
    #         Without this, whenever len(loader) isn't a clean multiple of
    #         accum_steps, the final 1..accum_steps-1 batches each epoch
    #         compute and accumulate gradients that never get applied via
    #         optimizer.step(), and optimizer.zero_grad() never fires for
    #         them either — so those gradients silently persist and keep
    #         accumulating on top of the *next* epoch's gradients too. This
    #         doesn't crash or show up in loss curves; it just quietly
    #         degrades convergence over time.
    remainder = len(loader) % accum_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(getattr(model, 'module', model))

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(config: dict):
    """
    Full training pipeline.

    Config keys:
        train_csv, val_csv, test_csv  : CSV file paths
        root_dir       : Image root directory
        mode           : 'binary' or 'subtype'
        num_classes    : 2 or 5
        batch_size     : Per-GPU batch size (default 4, tuned for 4GB VRAM)
        num_workers    : DataLoader workers
        epochs         : Max training epochs (default 120)
        lr             : Initial learning rate (default 1e-4)
        weight_decay   : AdamW weight decay (default 1e-5)
        T_0, T_mult    : LR scheduler params
        accum_steps    : Gradient accumulation steps (default 2)
        cutmix_prob    : CutMix probability (default 0.2)
        ema_decay      : EMA decay (default 0.9999)
        swa_start      : Epoch to start SWA (default 100)
        patience       : Early stopping patience (default 20)
        output_dir     : Directory to save checkpoints
        device         : 'cuda' or 'cpu'
        preprocess     : Hair removal + colour normalisation
    """
    device     = config.get('device', 'cuda' if torch.cuda.is_available()
                             else 'cpu')

    # FIX 8: cuDNN autotuner. Input resolution is fixed every epoch (same
    #         image size, same batch shape), which is exactly the case
    #         cudnn.benchmark is designed for — it profiles convolution
    #         algorithms on the first batch and reuses the fastest one for
    #         the rest of the run instead of picking a default each time.
    #         No effect on CPU; only enabled when actually training on CUDA.
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    output_dir = config.get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BCC Training  |  device={device}  |  mode={config['mode']}")
    print(f"{'='*60}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    loaders = build_dataloaders(
        train_csv  = config['train_csv'],
        val_csv    = config['val_csv'],
        test_csv   = config['test_csv'],
        root_dir   = config.get('root_dir', ''),
        mode       = config['mode'],
        batch_size = config.get('batch_size', 4),
        num_workers= config.get('num_workers', 2),
        preprocess = config.get('preprocess', True),
    )
    class_weights = getattr(loaders['train'].dataset,
                            'get_class_weights')().to(device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes = config['num_classes'],
        pretrained  = True,
        dropout     = config.get('dropout', 0.5),
    ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[GPU]  Using {torch.cuda.device_count()} GPUs")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = build_criterion(
        class_weights= class_weights,
        lambda_mix   = config.get('lambda_mix', 0.6),
        gamma        = config.get('gamma', 2.0),
        alpha        = config.get('alpha', 0.75),
    )
    optimizer = build_optimiser(
        model,
        lr           = config.get('lr', 1e-4),
        weight_decay = config.get('weight_decay', 1e-5),
    )
    scheduler = build_scheduler(
        optimizer,
        T_0  = config.get('T_0', 10),
        T_mult= config.get('T_mult', 2),
    )

    # ── EMA + SWA + AMP ──────────────────────────────────────────────────────
    # FIX 5: EMA must be built from the unwrapped module, not the
    #         DataParallel wrapper — see the EMA class docstring above for
    #         why a wrapped model causes a checkpoint key mismatch later.
    ema_target: nn.Module = model.module if isinstance(model, nn.DataParallel) else model
    ema    = EMA(ema_target, decay=config.get('ema_decay', 0.9999))
    # FIX 6: GradScaler is disabled on CPU when enabled=False, so .scale()/
    #         .step()/.unscale_() become safe no-ops instead of raising.
    scaler = GradScaler(enabled=(device == 'cuda'))

    swa_model    = AveragedModel(model)
    swa_start    = config.get('swa_start', 100)
    swa_scheduler= SWALR(optimizer, swa_lr=1e-5)

    # ── Training state ────────────────────────────────────────────────────────
    best_auc      = 0.0
    best_ckpt     = os.path.join(output_dir, 'best_model.pth')
    patience      = config.get('patience', 20)
    patience_cnt  = 0
    history       = []

    # ── Epoch loop ────────────────────────────────────────────────────────────
    # FIX 7: epoch initialized before the loop as a defensive guard — if
    #         config['epochs'] were ever 0, the for-loop body never runs and
    #         `epoch` would otherwise be referenced after the loop (in the
    #         SWA finalization check below) while undefined.
    epoch = 0
    for epoch in range(1, config.get('epochs', 120) + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model      = model,
            loader     = loaders['train'],
            optimizer  = optimizer,
            criterion  = criterion,
            scaler     = scaler,
            device     = device,
            ema        = ema,
            accum_steps= config.get('accum_steps', 2),
            cutmix_prob= config.get('cutmix_prob', 0.2),
        )

        # Validate with EMA weights
        val_metrics = evaluate(
            ema.get_model(), loaders['val'], device,
            num_classes=config['num_classes']
        )
        val_auc = val_metrics['auc']
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{config.get('epochs',120)} "
              f"| train_loss={train_loss:.4f} "
              f"| val_auc={val_auc:.4f} "
              f"| val_acc={val_metrics['accuracy']:.4f} "
              f"| {elapsed:.1f}s")

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        **val_metrics})

        # LR schedule step (or SWA after swa_start)
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc     = val_auc
            patience_cnt = 0
            ckpt = {
                'epoch'      : epoch,
                'model_state': (model.module.state_dict()
                                if isinstance(model, nn.DataParallel)
                                else model.state_dict()),
                'ema_state'  : ema.get_model().state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_auc'    : val_auc,
                'val_metrics': val_metrics,
                'config'     : config,
            }
            torch.save(ckpt, best_ckpt)
            print(f"  --> Best model saved (AUC={best_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[Early Stop] No AUC improvement for {patience} epochs.")
                break

    # ── Update SWA BN stats ──────────────────────────────────────────────────
    if epoch >= swa_start:
        print("[SWA] Updating batch normalisation statistics...")
        torch.optim.swa_utils.update_bn(
            loaders['train'], swa_model, device=torch.device(device)
        )
        swa_ckpt = os.path.join(output_dir, 'swa_model.pth')
        torch.save({'model_state': swa_model.state_dict()}, swa_ckpt)
        print(f"[SWA] Saved to {swa_ckpt}")

    # ── Final test evaluation ────────────────────────────────────────────────
    print("\n[Test] Loading best checkpoint...")
    best_state = torch.load(best_ckpt, map_location=device)
    m = build_model(config['num_classes'], pretrained=False).to(device)
    m.load_state_dict(best_state['ema_state'])

    test_metrics = evaluate(m, loaders['test'], device,
                             num_classes=config['num_classes'])
    print("\n[Final Test Results]")
    for k, v in test_metrics.items():
        print(f"  {k:<15} : {v:.4f}")

    return history, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DEFAULT CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'train_csv'  : './dataset/data/train.csv',
    'val_csv'    : './dataset/data/val.csv',
    'test_csv'   : './dataset/data/test.csv',
    'root_dir'   : './data',
    'mode'       : 'binary',      # 'binary' or 'subtype'
    'num_classes': 2,
    'preprocess' : True,

    # --- DataLoader ---
    # NOTE: batch_size=4 + accum_steps=2 gives the same effective batch (8)
    # as batch_size=8 + accum_steps=1, but with half the peak VRAM usage —
    # safer starting point for a 4GB card (RTX 2050). If a couple of epochs
    # run cleanly with memory headroom (check nvidia-smi / Task Manager),
    # you can try batch_size=8, accum_steps=1 for slightly fewer Python/CUDA
    # round-trips per epoch.
    'batch_size' : 4,
    'num_workers': 2,

    # --- Model ---
    'dropout'    : 0.5,

    # --- Loss ---
    'lambda_mix' : 0.6,
    'gamma'      : 2.0,
    'alpha'      : 0.75,

    # --- Optimiser ---
    'lr'         : 1e-4,
    'weight_decay': 1e-5,
    'T_0'        : 10,
    'T_mult'     : 2,
    'accum_steps': 2,
    'cutmix_prob': 0.2,

    # --- EMA & SWA ---
    'ema_decay'  : 0.9999,
    'swa_start'  : 100,

    # --- Loop ---
    'epochs'     : 150,
    'patience'   : 30,

    # --- Output ---
    'output_dir' : './outputs',
}


if __name__ == "__main__":
    train(DEFAULT_CONFIG)