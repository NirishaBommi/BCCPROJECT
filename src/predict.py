"""
============================================================================
  predict.py  —  Single-Image Inference + Grad-CAM Visualisation
  Paper : "Basal Cell Carcinoma Skin Detection Using Deep Learning"
============================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model    import build_model, TemperatureScaledModel
from evaluate import GradCAM
from dataset  import remove_hair, reinhard_normalise, IMG_SIZE, \
                     IMAGENET_MEAN, IMAGENET_STD


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────────────────────────────────────
BINARY_NAMES  = {0: 'non-BCC', 1: 'BCC'}
SUBTYPE_NAMES = {
    0: 'Nodular BCC',
    1: 'Superficial BCC',
    2: 'Infiltrative BCC',
    3: 'Morpheaform BCC',
    4: 'Basosquamous BCC',
}


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING FOR A SINGLE IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_single(image_path: str, preprocess: bool = True) -> torch.Tensor:
    """
    Load and preprocess a single dermoscopic image.

    Args:
        image_path (str):  Path to the image file.
        preprocess (bool): Apply hair removal + colour normalisation.

    Returns:
        Tensor of shape (1, 3, 380, 380), ready for model.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if preprocess:
        image = remove_hair(image)
        image = reinhard_normalise(image)

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    tensor = transform(image=image)['image']
    return tensor.unsqueeze(0)          # (1, 3, 380, 380)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(image_path  : str,
            ckpt_path   : str,
            num_classes : int  = 2,
            device      : str  = 'cuda',
            temperature : float = 1.0,
            preprocess  : bool = True,
            gradcam     : bool = True,
            save_dir    : str  = './gradcam_outputs') -> dict:
    """
    Run inference on a single dermoscopic image.

    Args:
        image_path  (str):   Path to input dermoscopic image.
        ckpt_path   (str):   Path to saved checkpoint (.pth).
        num_classes (int):   2 for binary, 5 for subtype.
        device      (str):   'cuda' or 'cpu'.
        temperature (float): Temperature for calibrated probabilities.
        preprocess  (bool):  Apply hair removal + colour normalisation.
        gradcam     (bool):  Generate and save Grad-CAM heatmap.
        save_dir    (str):   Directory to save Grad-CAM visualisation.

    Returns:
        dict: {
            'predicted_class' : int,
            'predicted_label' : str,
            'confidence'      : float,
            'probabilities'   : list,
            'gradcam_path'    : str or None,
        }
    """
    device = device if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('ema_state', ckpt.get('model_state', ckpt))

    # Rebuild model dynamically based on the backbone used in checkpoint
    backbone = 'efficientnet_b5'
    if isinstance(ckpt, dict) and 'config' in ckpt:
        backbone = ckpt['config'].get('backbone', 'efficientnet_b5')
    
    if backbone != 'efficientnet_b5':
        import timm
        original_create_model = timm.create_model
        timm.create_model = lambda name, **kwargs: original_create_model(backbone, **kwargs)
        model = build_model(num_classes=num_classes, pretrained=False)
        timm.create_model = original_create_model
    else:
        model = build_model(num_classes=num_classes, pretrained=False)

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # FIX: removed dead `model.temperature_val = temperature` assignment.
    #       Nothing downstream ever read that attribute — actual temperature
    #       scaling already happens correctly below via `logits / temperature`,
    #       which divides by this function's local `temperature` parameter
    #       directly. The removed lines created a dangling, misleading
    #       attribute that suggested the model was being configured when it
    #       wasn't actually being used for anything.

    # Preprocess image
    tensor = preprocess_single(image_path, preprocess).to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits / temperature, dim=1)
        # FIX: explicit int() cast. argmax().item() always returns a native
        #       int at runtime (confirmed by direct testing) since argmax
        #       produces integer indices, but .item()'s type stub is the
        #       broader `Number` union covering every possible tensor dtype
        #       .item() could be called on. This single cast resolves every
        #       downstream Pylance complaint that stemmed from `pred` —
        #       dict key access, tensor indexing, and the class_idx
        #       parameter — since they all trace back to this one variable.
        pred   = int(probs.argmax(dim=1).item())
        conf   = probs[0, pred].item()

    label_map = BINARY_NAMES if num_classes == 2 else SUBTYPE_NAMES

    print(f"\n[Prediction] {os.path.basename(image_path)}")
    print(f"  Predicted : {label_map[pred]}  (class {pred})")
    print(f"  Confidence: {conf:.4f}")
    for c in range(num_classes):
        print(f"  P(class {c} = {label_map[c]}): {probs[0,c].item():.4f}")

    gradcam_path = None
    if gradcam:
        # Grad-CAM requires gradients
        model.train()        # enable grads in BatchNorm etc.
        target_layer = model.se_blocks[-1]
        cam_gen      = GradCAM(model, target_layer)

        tensor_grad = tensor.requires_grad_(True)
        heatmap     = cam_gen.generate(tensor_grad, class_idx=pred)

        # Reload original image for overlay
        # FIX: cv2.imread genuinely can return None (invalid path, corrupted
        #       file, unsupported format) — Pylance correctly flags this as
        #       a real possibility, not a stub-narrowing artifact like the
        #       earlier `pred` issue. preprocess_single already read this
        #       same path successfully moments earlier in this function, so
        #       a failure here would be unusual, but guarding explicitly
        #       means a real failure surfaces here with a clear message
        #       rather than as a confusing cv2.cvtColor error on None three
        #       lines later inside visualise().
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(
                f"Cannot re-read image for Grad-CAM overlay: {image_path}"
            )
        fname     = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{fname}_gradcam.png")

        cam_gen.visualise(
            orig_img, heatmap,
            title    = f"{label_map[pred]} (conf={conf:.3f})",
            save_path= save_path
        )
        gradcam_path = save_path

    return {
        'predicted_class': pred,
        'predicted_label': label_map[pred],
        'confidence'     : conf,
        'probabilities'  : probs[0].tolist(),
        'gradcam_path'   : gradcam_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTION  (folder or list of paths)
# ─────────────────────────────────────────────────────────────────────────────
def predict_batch(image_paths : list,
                  ckpt_path   : str,
                  num_classes : int  = 2,
                  device      : str  = 'cuda',
                  temperature : float = 1.0,
                  preprocess  : bool = True) -> list:
    """
    Run inference on multiple images efficiently.

    Returns:
        List of prediction dicts (same keys as predict()).
    """
    device = device if torch.cuda.is_available() else 'cpu'

    # Load checkpoint once
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('ema_state', ckpt.get('model_state', ckpt))

    # Rebuild model dynamically based on the backbone used in checkpoint
    backbone = 'efficientnet_b5'
    if isinstance(ckpt, dict) and 'config' in ckpt:
        backbone = ckpt['config'].get('backbone', 'efficientnet_b5')
    
    if backbone != 'efficientnet_b5':
        import timm
        original_create_model = timm.create_model
        timm.create_model = lambda name, **kwargs: original_create_model(backbone, **kwargs)
        model = build_model(num_classes=num_classes, pretrained=False)
        timm.create_model = original_create_model
    else:
        model = build_model(num_classes=num_classes, pretrained=False)

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    label_map = BINARY_NAMES if num_classes == 2 else SUBTYPE_NAMES
    results   = []

    with torch.no_grad():
        for path in image_paths:
            try:
                tensor = preprocess_single(path, preprocess).to(device)
                logits = model(tensor)
                probs  = F.softmax(logits / temperature, dim=1)
                pred   = int(probs.argmax(dim=1).item())
                conf   = probs[0, pred].item()

                results.append({
                    'image_path'     : path,
                    'predicted_class': pred,
                    'predicted_label': label_map[pred],
                    'confidence'     : conf,
                    'probabilities'  : probs[0].tolist(),
                })
            except Exception as e:
                results.append({'image_path': path, 'error': str(e)})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BCC Skin Lesion Predictor"
    )
    parser.add_argument('--image',       required=True,
                        help='Path to dermoscopic image (JPG/PNG)')
    parser.add_argument('--checkpoint',  required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='2=binary, 5=subtype (default: 2)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for calibration (default: 1.0)')
    parser.add_argument('--no_preprocess', action='store_true',
                        help='Skip hair removal / colour normalisation')
    parser.add_argument('--no_gradcam', action='store_true',
                        help='Skip Grad-CAM generation')
    parser.add_argument('--save_dir', default='./gradcam_outputs',
                        help='Directory to save Grad-CAM images')
    parser.add_argument('--device', default='cuda',
                        help='cuda or cpu (default: cuda)')
    args = parser.parse_args()

    result = predict(
        image_path  = args.image,
        ckpt_path   = args.checkpoint,
        num_classes = args.num_classes,
        device      = args.device,
        temperature = args.temperature,
        preprocess  = not args.no_preprocess,
        gradcam     = not args.no_gradcam,
        save_dir    = args.save_dir,
    )