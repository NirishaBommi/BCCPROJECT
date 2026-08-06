import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, 'c:/Users/Hemanth/OneDrive/Desktop/BCCPROJECT/src')
from model import build_model
from dataset import build_dataloaders

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'outputs/best_model.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    
    backbone = ckpt.get('config', {}).get('backbone', 'efficientnet_b0')
    print(f"Backbone: {backbone}")

    import timm
    original_create_model = timm.create_model
    timm.create_model = lambda name, **kwargs: original_create_model(backbone, **kwargs)
    
    model = build_model(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    timm.create_model = original_create_model

    loaders = build_dataloaders(
        train_csv='dataset/data/train.csv',
        val_csv='dataset/data/val.csv',
        test_csv='dataset/data/test.csv',
        root_dir='dataset/data',
        mode='binary',
        batch_size=32,
        num_workers=0,
        preprocess=True
    )

    # Get validation probabilities
    val_probs, val_labels = [], []
    with torch.no_grad():
        for images, labels in loaders['val']:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1).cpu().numpy()
            val_probs.append(probs)
            val_labels.append(labels.numpy())
    val_probs = np.vstack(val_probs)
    val_labels = np.concatenate(val_labels)

    # Find optimal threshold on validation set
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.01, 0.99, 99):
        preds = (val_probs[:, 1] >= thresh).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Optimal threshold found on validation set: {best_thresh:.4f}")

    # Evaluate on test set
    test_probs, test_labels = [], []
    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1).cpu().numpy()
            test_probs.append(probs)
            test_labels.append(labels.numpy())
    test_probs = np.vstack(test_probs)
    test_labels = np.concatenate(test_labels)

    preds = (test_probs[:, 1] >= best_thresh).astype(int)
    
    acc = accuracy_score(test_labels, preds)
    rec = recall_score(test_labels, preds)
    prec = precision_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    auc = roc_auc_score(test_labels, test_probs[:, 1])
    
    cm = confusion_matrix(test_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp + 1e-9)
    bal_acc = (rec + spec) / 2

    print("\n--- Test Set Results with Optimal Threshold ---")
    print(f"  Accuracy       : {acc*100:.2f}%")
    print(f"  Precision      : {prec*100:.2f}%")
    print(f"  Recall         : {rec*100:.2f}%")
    print(f"  Specificity    : {spec*100:.2f}%")
    print(f"  F1 Score       : {f1*100:.2f}%")
    print(f"  ROC-AUC        : {auc*100:.2f}%")
    print(f"  Balanced Acc   : {bal_acc*100:.2f}%")

if __name__ == '__main__':
    main()
