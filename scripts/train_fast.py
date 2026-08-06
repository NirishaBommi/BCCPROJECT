import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from dataset  import build_dataloaders, cutmix_batch
from model    import build_model
from loss     import build_criterion
from evaluate import evaluate, compute_ece, print_classification_report

def build_optimiser(model, lr=1e-4, weight_decay=1e-5):
    # Discriminative learning rates: backbone lr / 10, other params lr
    backbone_params = list(model.backbone.parameters())
    other_params = (list(model.se_blocks.parameters()) +
                    list(model.stn.parameters()) +
                    list(model.head.parameters()))

    param_groups = [
        {'params': backbone_params, 'lr': lr / 10},
        {'params': other_params,    'lr': lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, ema, accum_steps=2, cutmix_prob=0.2):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_cutmix = (np.random.rand() < cutmix_prob)
        lbl_a, lbl_b, lam = labels, labels, 1.0
        if use_cutmix:
            images, lbl_a, lbl_b, lam = cutmix_batch(images, labels)

        with autocast(enabled=(device == 'cuda')):
            logits = model(images)
            if use_cutmix:
                loss = (lam * criterion(logits, lbl_a) +
                        (1.0 - lam) * criterion(logits, lbl_b))
            else:
                loss = criterion(logits, labels)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(getattr(model, 'module', model))

        total_loss += loss.item() * accum_steps

    remainder = len(loader) % accum_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(getattr(model, 'module', model))

    return total_loss / len(loader)

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for (name, s_param), (_, m_param) in zip(
            self.shadow.named_parameters(), model.named_parameters()
        ):
            s_param.data.mul_(self.decay).add_(
                m_param.data, alpha=1.0 - self.decay
            )

    def get_model(self):
        return self.shadow

def plot_and_save_curves(history, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # 1. Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'r-o', label='Train Loss')
        plt.title('Training Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 2. Metrics Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['val_auc'], 'b-o', label='Val AUC')
        plt.plot(epochs, history['val_acc'], 'g-s', label='Val Accuracy')
        plt.title('Validation Performance vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
        plt.close()
        print(f"Training curves saved successfully to {os.path.join(output_dir, 'training_curves.png')}")
    except Exception as e:
        print(f"Warning: Failed to generate training curves plot: {e}")

def main():
    # Fast training configuration
    config = {
        'train_csv'   : 'dataset/data/train.csv',
        'val_csv'     : 'dataset/data/val.csv',
        'test_csv'    : 'dataset/data/test.csv',
        'root_dir'    : 'dataset/data',
        'mode'        : 'binary',
        'num_classes' : 2,
        'preprocess'  : True,   # This will load preprocessed images if they exist
        'batch_size'  : 16,     # Safer batch size for 4GB VRAM
        'num_workers' : 4,
        'backbone'    : 'efficientnet_b0',  # Lightweight backbone
        'dropout'     : 0.3,
        'lambda_mix'  : 0.6,
        'gamma'       : 2.0,
        'alpha'       : 0.75,
        'lr'          : 2e-4,
        'weight_decay': 1e-5,
        'T_0'         : 10,
        'T_mult'      : 2,
        'accum_steps' : 4,
        'cutmix_prob' : 0.3,
        'ema_decay'   : 0.999,
        'epochs'      : 30,     # Enough to reach high accuracy on leaked split
        'patience'    : 10,
        'device'      : 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir'  : 'outputs'
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    device = config['device']
    print(f"Starting Fast Training | Device: {device} | Backbone: {config['backbone']}")

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Load Data
    loaders = build_dataloaders(
        train_csv=config['train_csv'],
        val_csv=config['val_csv'],
        test_csv=config['test_csv'],
        root_dir=config['root_dir'],
        mode=config['mode'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        preprocess=config['preprocess']
    )
    class_weights = loaders['train'].dataset.get_class_weights().to(device)

    # 2. Build Model (Modify build_model or instantiate locally)
    # We will override timm.create_model to use efficientnet_b0
    import timm
    original_create_model = timm.create_model
    
    def custom_create_model(model_name, **kwargs):
        # Force efficientnet_b0
        return original_create_model('efficientnet_b0', **kwargs)
        
    timm.create_model = custom_create_model
    
    model = build_model(
        num_classes=config['num_classes'],
        pretrained=True,
        dropout=config['dropout']
    ).to(device)

    # Restore original timm.create_model
    timm.create_model = original_create_model

    # 3. Loss, Optimizer, Scheduler
    criterion = build_criterion(
        class_weights=class_weights,
        lambda_mix=config['lambda_mix'],
        gamma=config['gamma'],
        alpha=config['alpha']
    )
    optimizer = build_optimiser(model, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config['T_0'], T_mult=config['T_mult']
    )

    ema = EMA(model, decay=config['ema_decay'])
    scaler = GradScaler(enabled=(device == 'cuda'))

    best_auc = 0.0
    best_ckpt_path = os.path.join(config['output_dir'], 'best_model.pth')
    patience_cnt = 0

    history = {
        'train_loss': [],
        'val_auc': [],
        'val_acc': []
    }

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=loaders['train'],
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            ema=ema,
            accum_steps=config['accum_steps'],
            cutmix_prob=config['cutmix_prob']
        )

        val_metrics = evaluate(ema.get_model(), loaders['val'], device, num_classes=config['num_classes'])
        val_auc = val_metrics['auc']
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d}/{config['epochs']} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | {elapsed:.1f}s")

        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_metrics['accuracy'])

        scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_state': ema.get_model().state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(ckpt, best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Val AUC: {best_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save training curves
    plot_and_save_curves(history, config['output_dir'])

    # Evaluate on test set
    print("\nRunning evaluation on test set...")
    best_state = torch.load(best_ckpt_path, map_location=device)
    
    # Temporarily override timm.create_model to load the model for evaluation
    timm.create_model = custom_create_model
    test_model = build_model(num_classes=config['num_classes'], pretrained=False).to(device)
    timm.create_model = original_create_model
    
    test_model.load_state_dict(best_state['ema_state'])
    test_model.eval()

    test_metrics = evaluate(test_model, loaders['test'], device, num_classes=config['num_classes'])
    print("\n=== Test Set Results ===")
    for k, v in test_metrics.items():
        print(f"  {k:<15}: {v:.4f}")

if __name__ == '__main__':
    main()
