import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, 'c:/Users/Hemanth/OneDrive/Desktop/BCCPROJECT/src')
from model import build_model
from dataset import build_dataloaders
from evaluate import evaluate

def test_ckpt(ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device)
    
    backbone = ckpt.get('config', {}).get('backbone', 'efficientnet_b5')
    print(f"Loaded config backbone: {backbone}")

    # Override timm.create_model
    import timm
    original_create_model = timm.create_model
    timm.create_model = lambda name, **kwargs: original_create_model(backbone, **kwargs)
    
    loaders = build_dataloaders(
        train_csv='dataset/data/train.csv',
        val_csv='dataset/data/val.csv',
        test_csv='dataset/data/test.csv',
        root_dir='dataset/data',
        mode='binary',
        batch_size=32,
        num_workers=0,  # 0 workers for safety
        preprocess=True
    )

    for state_key in ['ema_state', 'model_state']:
        if state_key not in ckpt:
            print(f"Key {state_key} not in checkpoint.")
            continue
            
        model = build_model(num_classes=2, pretrained=False).to(device)
        model.load_state_dict(ckpt[state_key])
        model.eval()
        
        metrics = evaluate(model, loaders['test'], device, num_classes=2)
        print(f"--- State: {state_key} ---")
        for k, v in metrics.items():
            print(f"  {k:<15}: {v:.4f}")
            
    timm.create_model = original_create_model

if __name__ == '__main__':
    test_ckpt('outputs/best_model.pth')
