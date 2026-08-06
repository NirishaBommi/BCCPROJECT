import os
import sys
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory

# Add src directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from model import build_model
from dataset import remove_hair, reinhard_normalise, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from evaluate import GradCAM
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model checkpoint
ckpt_path = os.path.join('outputs', 'best_model.pth')
model = None
temperature = 1.4799  # Optimal temperature from report

if os.path.exists(ckpt_path):
    print(f"[Backend] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Rebuild model backbone dynamically
    backbone = 'efficientnet_b5'
    if isinstance(ckpt, dict) and 'config' in ckpt:
        backbone = ckpt['config'].get('backbone', 'efficientnet_b5')
        print(f"[Backend] Detected backbone from config: {backbone}")
    
    if backbone != 'efficientnet_b5':
        import timm
        original_create_model = timm.create_model
        timm.create_model = lambda name, **kwargs: original_create_model(backbone, **kwargs)
        model = build_model(num_classes=2, pretrained=False).to(device)
        timm.create_model = original_create_model
    else:
        model = build_model(num_classes=2, pretrained=False).to(device)
        
    state = ckpt.get('ema_state', ckpt.get('model_state', ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    print("[Backend] Model loaded successfully.")
else:
    print(f"[Backend] WARNING: Checkpoint not found at {ckpt_path}. Inference will not be available.")

# Load backup model checkpoint (if exists, usually efficientnet_b5)
ckpt_backup_path = os.path.join('outputs', 'best_model_backup.pth')
model_backup = None

if os.path.exists(ckpt_backup_path):
    print(f"[Backend] Loading backup checkpoint from {ckpt_backup_path}...")
    try:
        ckpt_b = torch.load(ckpt_backup_path, map_location=device)
        backbone_b = 'efficientnet_b5'
        if isinstance(ckpt_b, dict) and 'config' in ckpt_b:
            backbone_b = ckpt_b['config'].get('backbone', 'efficientnet_b5')
            print(f"[Backend] Detected backup backbone: {backbone_b}")
        
        if backbone_b != 'efficientnet_b5':
            import timm
            original_create_model = timm.create_model
            timm.create_model = lambda name, **kwargs: original_create_model(backbone_b, **kwargs)
            model_backup = build_model(num_classes=2, pretrained=False).to(device)
            timm.create_model = original_create_model
        else:
            model_backup = build_model(num_classes=2, pretrained=False).to(device)
            
        state_b = ckpt_b.get('ema_state', ckpt_b.get('model_state', ckpt_b))
        model_backup.load_state_dict(state_b, strict=True)
        model_backup.eval()
        print("[Backend] Backup model loaded successfully.")
    except Exception as e:
        print(f"[Backend] Error loading backup checkpoint: {e}")

# Label map
BINARY_NAMES = {0: 'non-BCC (Benign / Other)', 1: 'BCC (Basal Cell Carcinoma)'}

def preprocess_image_pipeline(image_path):
    """
    Load image, apply hair removal and color normalization,
    and convert to PyTorch tensor.
    """
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Cannot read image: {image_path}")
        
    # Preprocess
    prep_img = remove_hair(orig_img)
    prep_img = reinhard_normalise(prep_img)
    
    # BGR -> RGB
    prep_img_rgb = cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB)
    
    # Transform
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    tensor = transform(image=prep_img_rgb)['image']
    return orig_img, prep_img, tensor.unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if model is None:
        return jsonify({'error': 'Model not loaded on server. Please check best_model.pth'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Read settings parameters
    model_type = request.form.get('model_type', 'standard')  # 'standard' or 'ensemble'
    use_tta = request.form.get('use_tta', 'false').lower() == 'true'
    use_cal_threshold = request.form.get('use_threshold', 'true').lower() == 'true'
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename or '')[1]
    if not ext:
        ext = '.jpg'
    
    filename = f"upload_{unique_id}{ext}"
    orig_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(orig_path)
    
    try:
        # Preprocess
        orig_img, prep_img, tensor = preprocess_image_pipeline(orig_path)
        
        # Save preprocessed image to outputs
        prep_filename = f"prep_{unique_id}.png"
        prep_path = os.path.join(OUTPUT_FOLDER, prep_filename)
        cv2.imwrite(prep_path, prep_img)
        
        # Run inference
        tensor_device = tensor.to(device)
        
        # Select active models
        active_models = [model]
        if model_type == 'ensemble' and model_backup is not None:
            active_models.append(model_backup)
            print(f"[Backend] Running ensembled prediction with {len(active_models)} models...")
            
        all_probs = []
        for m in active_models:
            m.eval()
            with torch.no_grad():
                if use_tta:
                    # 1. Original
                    logits_orig = m(tensor_device)
                    probs_orig = F.softmax(logits_orig / temperature, dim=1)
                    
                    # 2. Horizontal Flip TTA
                    tensor_h = torch.flip(tensor_device, dims=[3])
                    logits_h = m(tensor_h)
                    probs_h = F.softmax(logits_h / temperature, dim=1)
                    
                    # 3. Vertical Flip TTA
                    tensor_v = torch.flip(tensor_device, dims=[2])
                    logits_v = m(tensor_v)
                    probs_v = F.softmax(logits_v / temperature, dim=1)
                    
                    probs_m = (probs_orig + probs_h + probs_v) / 3.0
                else:
                    logits = m(tensor_device)
                    probs_m = F.softmax(logits / temperature, dim=1)
                    
                all_probs.append(probs_m.cpu().numpy()[0])
        
        # Average probability values across ensembled backbones
        probs = np.mean(all_probs, axis=0)
        
        # Calibrated Decision Thresholding (optimal t=0.40 vs default argmax t=0.50)
        threshold = 0.40 if use_cal_threshold else 0.50
        if probs[1] >= threshold:
            pred_class = 1
        else:
            pred_class = 0
            
        confidence = float(probs[pred_class])
        
        # Generate Grad-CAM Heatmap (hooks registered on primary model)
        model.train() # enable gradients in BatchNorm/Dropout for hook
        target_layer = model.se_blocks[-1]
        cam_gen = GradCAM(model, target_layer)
        
        tensor_grad = tensor_device.requires_grad_(True)
        heatmap = cam_gen.generate(tensor_grad, class_idx=pred_class)
        
        # Overlay Heatmap
        h, w = orig_img.shape[:2]
        cam_resized = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * orig_img_rgb + 0.4 * heatmap_color).astype(np.uint8)
        
        # Save Grad-CAM image
        cam_filename = f"gradcam_{unique_id}.png"
        cam_path = os.path.join(OUTPUT_FOLDER, cam_filename)
        # Convert back to BGR for saving
        cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Reset model to eval
        model.eval()
        
        return jsonify({
            'predicted_class': pred_class,
            'predicted_label': BINARY_NAMES[pred_class],
            'confidence': confidence,
            'probability_non_bcc': float(probs[0]),
            'probability_bcc': float(probs[1]),
            'orig_url': f'/static/uploads/{filename}',
            'prep_url': f'/static/outputs/{prep_filename}',
            'cam_url': f'/static/outputs/{cam_filename}',
            'ensemble_active': len(active_models) > 1,
            'tta_active': use_tta,
            'threshold_used': threshold
        })
        
    except Exception as e:
        # Reset model state just in case
        if model is not None:
            model.eval()
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Error during processing: {str(e)}"}), 500

@app.route('/static/images/<path:filename>')
def serve_static_images(filename):
    # Fallback to output folder for pre-generated images
    if filename in ['confusion_matrix.png', 'roc_curve.png', 'ece_plot.png']:
        return send_from_directory('outputs', filename)
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    print("[Backend] Starting Flask server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
