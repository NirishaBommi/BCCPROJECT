# BCC Skin Detection — Complete Code Guide
## "Basal Cell Carcinoma Skin Detection Using Deep Learning"

---

## Project Structure

```
bcc_project/
├── src/
│   ├── model.py        ← SE-STN-EfficientNet architecture
│   ├── dataset.py      ← Dataset, preprocessing, augmentation
│   ├── loss.py         ← Hybrid Focal Loss + Cross-Entropy
│   ├── train.py        ← Full training loop (EMA, SWA, AMP, CutMix)
│   ├── evaluate.py     ← Metrics, ECE, Grad-CAM, fairness
│   └── predict.py      ← Single-image / batch inference
├── configs/
│   └── config.yaml     ← All hyperparameters
├── scripts/
│   ├── run_train.sh    ← Training launcher
│   └── run_predict.sh  ← Inference launcher
├── data/               ← Put your CSVs + images here
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── outputs/            ← Checkpoints saved here
├── requirements.txt
└── README.md
```

---

## STEP 1 — Environment Setup

### 1A. Install Python 3.10+

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip
```

**Windows** — download installer from https://python.org (tick "Add to PATH")

**macOS**
```bash
brew install python@3.10
```

### 1B. Create and activate virtual environment

```bash
# Create virtualenv
python3.10 -m venv bcc_env

# Activate  (Linux / macOS)
source bcc_env/bin/activate

# Activate  (Windows)
bcc_env\Scripts\activate
```

### 1C. Install CUDA (for GPU training)

Check your GPU driver version:
```bash
nvidia-smi
```
Install matching CUDA Toolkit from https://developer.nvidia.com/cuda-downloads

### 1D. Install PyTorch with CUDA

```bash
# CUDA 12.1 (recommended)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only (slower but works)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### 1E. Install all other dependencies

```bash
pip install -r requirements.txt
```

---

## STEP 2 — Prepare Your Dataset

### 2A. Download ISIC 2019 Dataset
```bash
# Official ISIC Archive
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
unzip ISIC_2019_Training_Input.zip -d data/images/

# Metadata CSV
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv -O data/isic_meta.csv
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv -O data/isic_labels.csv
```

### 2B. Download HAM10000 Dataset
```bash
# Via Kaggle CLI (install: pip install kaggle)
kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d data/ham10000/
```

### 2C. Required CSV Format

Each split CSV (`train.csv`, `val.csv`, `test.csv`) must have these columns:

| Column       | Description                          | Example            |
|-------------|--------------------------------------|--------------------|
| image_path  | Relative path from root_dir          | images/ISIC_0024306.jpg |
| label       | `BCC` or `non-BCC`                   | BCC                |
| subtype     | BCC subtype (only for subtype mode)  | nodular            |
| fitzpatrick | Skin phototype 1-6 (optional)        | 2                  |

**Example `train.csv`:**
```csv
image_path,label,subtype,fitzpatrick
images/ISIC_0024306.jpg,BCC,nodular,2
images/ISIC_0025030.jpg,BCC,morpheaform,1
images/ISIC_0025661.jpg,non-BCC,,3
```

### 2D. Generate Demo CSV (for testing without real data)
```bash
cd bcc_project
python src/dataset.py
# Creates demo_data/ folder with 100 synthetic images + CSVs
```

---

## STEP 3 — Verify Installation

```bash
cd bcc_project
python - << 'EOF'
import torch, timm, albumentations, cv2, sklearn
print("torch     :", torch.__version__)
print("cuda avail:", torch.cuda.is_available())
print("GPU       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("timm      :", timm.__version__)
print("albuments :", albumentations.__version__)
print("opencv    :", cv2.__version__)
print("All OK!")
EOF
```

---

## STEP 4 — Test Model Architecture

```bash
python src/model.py
```
Expected output:
```
[Model] SE-STN-EfficientNet-B5  |  Classes=2
        Total params    : 32,456,194
        Trainable params: 32,456,194
Input : torch.Size([2, 3, 380, 380])  ->  Output: torch.Size([2, 2])
```

---

## STEP 5 — Train the Model

### Binary BCC Detection (recommended first run)
```bash
python src/train.py
```

Or with custom data paths:
```bash
python src/train.py \
    --train_csv  data/train.csv  \
    --val_csv    data/val.csv    \
    --test_csv   data/test.csv   \
    --root_dir   data            \
    --mode       binary          \
    --num_classes 2              \
    --batch_size 32              \
    --epochs     120             \
    --lr         1e-4            \
    --output_dir outputs/binary
```

### BCC Subtype Classification (5-class)
```bash
python src/train.py \
    --mode       subtype   \
    --num_classes 5        \
    --output_dir outputs/subtype
```

### Using the shell script
```bash
bash scripts/run_train.sh binary    # or: subtype
```

### Training output example
```
============================================================
  BCC Training  |  device=cuda  |  mode=binary
============================================================
[Data] Train=18,192  Val=3,903  Test=3,904  Mode=binary
[Model] SE-STN-EfficientNet-B5  |  Classes=2
[Loss]  HybridFLCE | lambda=0.6 | gamma=2.0 | alpha=0.75
[Optim] AdamW | lr=0.0001 | backbone_lr=1e-05 | wd=1e-05
[LR]    CosineAnnealingWarmRestarts | T_0=10 | T_mult=2

Epoch   1/120 | train_loss=0.5832 | val_auc=0.7412 | val_acc=0.7023 | 142.3s
Epoch   2/120 | train_loss=0.4217 | val_auc=0.8341 | val_acc=0.7891 | 138.7s
  --> Best model saved (AUC=0.8341)
...
Epoch  47/120 | train_loss=0.1243 | val_auc=0.9780 | val_acc=0.9681 | 136.1s
  --> Best model saved (AUC=0.9780)
```

---

## STEP 6 — Evaluate on Test Set

```bash
python - << 'EOF'
import torch, sys
sys.path.insert(0, 'src')
from model    import build_model
from dataset  import build_dataloaders
from evaluate import evaluate, compute_ece, print_classification_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load best checkpoint
ckpt  = torch.load('outputs/binary/best_model.pth', map_location=device)
model = build_model(num_classes=2, pretrained=False).to(device)
model.load_state_dict(ckpt['ema_state'])

# Build test loader
loaders = build_dataloaders(
    'data/train.csv', 'data/val.csv', 'data/test.csv',
    root_dir='data', mode='binary', batch_size=32, num_workers=4
)

# Evaluate
metrics = evaluate(model, loaders['test'], device, num_classes=2)
ece     = compute_ece(model, loaders['test'], device)

print("\n=== Test Results ===")
for k, v in metrics.items():
    print(f"  {k:<15}: {v:.4f}")
print(f"  {'ECE':<15}: {ece:.4f}")

print_classification_report(model, loaders['test'], device,
                             class_names=['non-BCC', 'BCC'])
EOF
```

---

## STEP 7 — Grad-CAM Visualisation

```bash
python src/predict.py \
    --image       data/images/your_lesion.jpg \
    --checkpoint  outputs/binary/best_model.pth \
    --num_classes 2 \
    --temperature 1.31 \
    --save_dir    gradcam_outputs
```

Grad-CAM images saved to `gradcam_outputs/your_lesion_gradcam.png`

---

## STEP 8 — Temperature Calibration

```bash
python - << 'EOF'
import torch, sys
sys.path.insert(0, 'src')
from model   import build_model, TemperatureScaledModel
from dataset import build_dataloaders
from evaluate import compute_ece

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt   = torch.load('outputs/binary/best_model.pth', map_location=device)
model  = build_model(2, pretrained=False).to(device)
model.load_state_dict(ckpt['ema_state'])

loaders = build_dataloaders(
    'data/train.csv', 'data/val.csv', 'data/test.csv',
    root_dir='data', mode='binary', batch_size=32, num_workers=4
)

cal_model = TemperatureScaledModel(model)
T = cal_model.calibrate(loaders['val'], device=device)
print(f"Optimal T = {T:.4f}")

ece = compute_ece(cal_model, loaders['test'], device)
print(f"Calibrated ECE = {ece:.4f}")
EOF
```

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `batch_size` to 16 or 8; increase `accum_steps` to 8 |
| `Cannot find module timm` | `pip install timm` |
| `FileNotFoundError` for image | Check `root_dir` + `image_path` in CSV join correctly |
| Training very slow on CPU | Set `num_workers=0`; use smaller `batch_size=4` |
| `nan` loss | Lower `lr` to `5e-5`; check images load correctly |
| SWA model poor | Ensure `swa_start` ≤ total epochs; re-run BN update step |

---

## Hardware Requirements

| Setup | Training Time / Epoch | Recommended For |
|-------|----------------------|-----------------|
| 4× A100 (80GB) | ~2 min | Full paper results |
| 1× RTX 3090 (24GB) | ~8 min | Research / tuning |
| 1× RTX 3060 (12GB) | ~18 min | Development |
| CPU only | ~4 hours | Testing only |
