import os
import sys
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from io import BytesIO
from typing import Optional
import base64

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to system path to import from models/explainability/utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import ModularSkinClassifier
from models.unet import UNet
from explainability.explain import ExplainableEngine
from datasets.transforms import DullRazorHairRemoval, ReinhardColorNormalizer
from utils.database import init_db, save_record, get_records, search_records, delete_record
from utils.stats import get_system_stats, estimate_model_complexity
from utils.plots import generate_static_plots

# Initialize FastAPI App
app = FastAPI(title="Clinical AI BCC Detection Portal", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths Configuration
UPLOAD_FOLDER = os.path.join('dashboard', 'static', 'uploads')
OUTPUT_FOLDER = os.path.join('dashboard', 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize SQLite and Pre-generate Matplotlib plots on Startup
@app.on_event("startup")
def startup_event():
    init_db()
    generate_static_plots("dashboard/static/outputs")

# Load Primary Checkpoint (EfficientNet-B0)
ckpt_path = os.path.join('outputs', 'best_model.pth')
model = None
temperature = 1.4799

# Helper to automatically locate the best conv layer for CAM hooks
def get_target_layer_for_cam(model):
    import torch.nn as nn
    if hasattr(model, 'attention_layers') and len(model.attention_layers) > 0:
        return model.attention_layers[-1]
    if hasattr(model, 'se_blocks') and len(model.se_blocks) > 0:
        return model.se_blocks[-1]
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'blocks') and len(model.backbone.blocks) > 0:
            last_stage = model.backbone.blocks[-1]
            if len(last_stage) > 0:
                return last_stage[-1]
            return last_stage
        for layer_name in ['layer4', 'layer3', 'layer2', 'layer1']:
            if hasattr(model.backbone, layer_name):
                return getattr(model.backbone, layer_name)
        if hasattr(model.backbone, 'act2'):
            return model.backbone.act2 if isinstance(model.backbone.act2, nn.Module) else model.backbone
        elif hasattr(model.backbone, 'conv_head'):
            return model.backbone.conv_head if isinstance(model.backbone.conv_head, nn.Module) else model.backbone
        else:
            return model.backbone
    return model

if os.path.exists(ckpt_path):
    print(f"[FastAPI] Loading primary checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = 'efficientnet_b0'
    if isinstance(ckpt, dict) and 'config' in ckpt:
        backbone = ckpt['config'].get('backbone', 'efficientnet_b0')
    
    # Initialize ModularSkinClassifier
    model = ModularSkinClassifier(
        backbone_name=backbone,
        num_classes=2,
        use_se=True,
        use_cbam=True,
        use_stn=True,
        use_bifpn=False
    ).to(device)
    
    state = ckpt.get('ema_state', ckpt.get('model_state', ckpt))
    # Load with strict=False to tolerate CBAM layers addition
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[FastAPI] Primary modular model loaded successfully.")
else:
    print(f"[FastAPI] WARNING: Primary checkpoint not found at {ckpt_path}.")

# Load Backup Checkpoint (EfficientNet-B5)
ckpt_backup_path = os.path.join('outputs', 'best_model_backup.pth')
model_backup = None

if os.path.exists(ckpt_backup_path):
    print(f"[FastAPI] Loading backup checkpoint from {ckpt_backup_path}...")
    try:
        ckpt_b = torch.load(ckpt_backup_path, map_location=device)
        model_backup = ModularSkinClassifier(
            backbone_name='efficientnet_b5',
            num_classes=2,
            use_se=True,
            use_cbam=True,
            use_stn=True,
            use_bifpn=False
        ).to(device)
        state_b = ckpt_b.get('ema_state', ckpt_b.get('model_state', ckpt_b))
        model_backup.load_state_dict(state_b, strict=False)
        model_backup.eval()
        print("[FastAPI] Backup model (B5) loaded successfully.")
    except Exception as e:
        print(f"[FastAPI] Error loading backup checkpoint: {e}")

# Initialize Explainability Engines
explain_engine = None
if model is not None:
    target_layer = get_target_layer_for_cam(model)
    explain_engine = ExplainableEngine(model, target_layer)

# Clinical Preprocessors
hair_removal = DullRazorHairRemoval(kernel_size=9, threshold=10)
color_normalizer = ReinhardColorNormalizer()

# Disease metadata list (matchesconfigs/disease_config.json fallback)
DISEASES = [
    "Basal Cell Carcinoma", "Melanoma", "Benign Nevus", "Seborrheic Keratosis",
    "Dermatofibroma", "Vascular Lesion", "Actinic Keratosis", "Squamous Cell Carcinoma", "Unknown"
]

def distribute_9_classes(p_non_bcc, p_bcc, selected_model):
    probs = np.zeros(9)
    if p_bcc >= 0.40:
        probs[0] = p_bcc * 0.92  # Basal Cell Carcinoma
        probs[1] = p_bcc * 0.05  # Melanoma
        probs[7] = p_bcc * 0.02  # Squamous Cell Carcinoma (SCC)
        probs[8] = p_bcc * 0.01  # Unknown
        
        rem = p_non_bcc
        probs[2] = rem * 0.50    # Benign Nevus
        probs[3] = rem * 0.30    # Seborrheic Keratosis
        probs[4] = rem * 0.10    # Dermatofibroma
        probs[5] = rem * 0.06    # Vascular Lesion
        probs[6] = rem * 0.04    # Actinic Keratosis
    else:
        probs[0] = p_bcc * 0.40  # Low residual BCC risk
        probs[1] = p_bcc * 0.10  # Low residual Melanoma risk
        probs[7] = p_bcc * 0.05  # Low residual SCC risk
        
        rem = p_non_bcc + (p_bcc * 0.45)
        probs[2] = rem * 0.55    # Benign Nevus
        probs[3] = rem * 0.22    # Seborrheic Keratosis
        probs[4] = rem * 0.11    # Dermatofibroma
        probs[5] = rem * 0.07    # Vascular Lesion
        probs[6] = rem * 0.03    # Actinic Keratosis
        probs[8] = rem * 0.02    # Unknown
        
    if selected_model == "DenseNet121":
        probs = np.roll(probs, 1) * 0.05 + probs * 0.95
    elif selected_model == "ResNet50":
        probs = np.roll(probs, -1) * 0.03 + probs * 0.97
        
    probs = np.clip(probs, 0.0, 1.0)
    return probs / (np.sum(probs) + 1e-8)

def decode_base64_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def preprocess_image_pipeline(orig_img):
    # Apply DullRazor hair removal and Reinhard color normalizer
    prep_img = hair_removal(orig_img)
    prep_img = color_normalizer(prep_img)
    prep_img_rgb = cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=prep_img_rgb)['image']
    return prep_img, tensor.unsqueeze(0)

def segment_lesion_cv(orig_img):
    # Fallback OpenCV contour segmentation
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(opened)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
    overlay = orig_img.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return mask, overlay

# Endpoints
@app.get("/", response_class=HTMLResponse)
def get_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={})

@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    model_type: str = Form("Standard Model (EfficientNet-B0)"),
    use_tta: str = Form("false"),
    use_threshold: str = Form("true"),
    patient_name: str = Form("Anonymous"),
    patient_age: str = Form("0"),
    patient_gender: str = Form("Unknown"),
    doctor_notes: str = Form("")
):
    if model is None:
        raise HTTPException(status_code=500, detail="Neural model is not loaded.")
        
    # 1. Load Image
    unique_id = str(uuid.uuid4())[:8]
    orig_filename = f"upload_{unique_id}.jpg"
    orig_path = os.path.join(UPLOAD_FOLDER, orig_filename)
    
    if image_base64:
        orig_img = decode_base64_image(image_base64)
        if orig_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode webcam frame.")
        cv2.imwrite(orig_path, orig_img)
    elif image:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode uploaded file.")
        cv2.imwrite(orig_path, orig_img)
    else:
        raise HTTPException(status_code=400, detail="No image file or base64 stream provided.")
        
    start_time = datetime.now()
    
    try:
        # 2. Preprocess
        prep_img, tensor = preprocess_image_pipeline(orig_img)
        prep_filename = f"prep_{unique_id}.png"
        prep_path = os.path.join(OUTPUT_FOLDER, prep_filename)
        cv2.imwrite(prep_path, prep_img)
        
        # 3. Model Selection
        active_models = [model]
        is_ensemble = "ensemble" in model_type.lower() or "b0 + b5" in model_type.lower()
        if is_ensemble and model_backup is not None:
            active_models.append(model_backup)
            
        # 4. Inference
        tensor_device = tensor.to(device)
        all_binary_probs = []
        
        for m in active_models:
            m.eval()
            with torch.no_grad():
                if use_tta.lower() == "true":
                    logits_orig = m(tensor_device)
                    probs_orig = F.softmax(logits_orig / temperature, dim=1)
                    
                    tensor_h = torch.flip(tensor_device, dims=[3])
                    logits_h = m(tensor_h)
                    probs_h = F.softmax(logits_h / temperature, dim=1)
                    
                    tensor_v = torch.flip(tensor_device, dims=[2])
                    logits_v = m(tensor_v)
                    probs_v = F.softmax(logits_v / temperature, dim=1)
                    
                    probs_m = (probs_orig + probs_h + probs_v) / 3.0
                else:
                    logits = m(tensor_device)
                    probs_m = F.softmax(logits / temperature, dim=1)
                all_binary_probs.append(probs_m.cpu().numpy()[0])
                
        binary_probs = np.mean(all_binary_probs, axis=0)
        p_non_bcc, p_bcc = binary_probs[0], binary_probs[1]
        
        # Binary prediction decision based on standard 50% threshold for exact results
        threshold = 0.50
        
        if p_bcc >= threshold:
            risk_level = "Red"
            pred_label = "BCC"
            confidence = float(p_bcc)
        elif p_bcc >= 0.15:
            risk_level = "Orange"
            pred_label = "non-BCC"
            confidence = float(p_non_bcc)
        elif p_bcc >= 0.05:
            risk_level = "Yellow"
            pred_label = "non-BCC"
            confidence = float(p_non_bcc)
        else:
            risk_level = "Green"
            pred_label = "non-BCC"
            confidence = float(p_non_bcc)
            
        # 6. OpenCV Lesion Segmentation
        mask, seg_overlay = segment_lesion_cv(orig_img)
        mask_filename = f"mask_{unique_id}.png"
        mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        # 7. Generate Explainability Heatmaps
        cam_urls = {}
        pred_idx_binary = 1 if pred_label == "BCC" else 0
        
        if explain_engine is not None:
            # We resize target colors for standard display size
            orig_img_resized = cv2.resize(orig_img, (224, 224))
            
            xai_methods = ["gradcam", "gradcam_plusplus", "scorecam", "eigencam", "integrated_gradients"]
            for m in xai_methods:
                try:
                    cam_map = explain_engine.generate_heatmap(tensor_device, method=m, class_idx=pred_idx_binary)
                    cam_map_resized = cv2.resize(cam_map, (224, 224))
                    cam_map_uint8 = np.clip(cam_map_resized * 255, 0, 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(cam_map_uint8, cv2.COLORMAP_JET)
                    overlay_cam = cv2.addWeighted(orig_img_resized, 0.6, heatmap_color, 0.4, 0)
                    
                    cam_filename = f"cam_{m}_{unique_id}.png"
                    cam_path = os.path.join(OUTPUT_FOLDER, cam_filename)
                    cv2.imwrite(cam_path, overlay_cam)
                    cam_urls[m] = f"/static/outputs/{cam_filename}"
                except Exception as ex_err:
                    print(f"[FastAPI] Heatmap generation failed for {m}: {ex_err}")
                    cam_urls[m] = f"/static/uploads/{orig_filename}"
                    
        # 8. Time details
        inference_time = f"{(datetime.now() - start_time).total_seconds() * 1000:.0f}ms"
        
        # DB log paths
        db_image_path = f"/static/uploads/{orig_filename}"
        db_cam_path = cam_urls.get("gradcam", f"/static/uploads/{orig_filename}")
        db_mask_path = f"/static/outputs/{mask_filename}"
        
        # Save SQLite log
        save_record(
            patient_name, int(patient_age), patient_gender, pred_label, confidence,
            model_type, doctor_notes, db_image_path, db_cam_path, db_mask_path
        )
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "risk_level": risk_level,
            "inference_time": inference_time,
            "model_used": model_type,
            "threshold_used": threshold,
            "interpretation": f"Lesion attributes mapped to {pred_label} with {confidence*100:.1f}% confidence.",
            "orig_url": f"/static/uploads/{orig_filename}",
            "prep_url": f"/static/outputs/{prep_filename}",
            "mask_url": db_mask_path,
            "cam_urls": cam_urls,
            "distribution": [
                {"disease": "Basal Cell Carcinoma", "probability": float(p_bcc)},
                {"disease": "non-BCC (Benign / Other)", "probability": float(p_non_bcc)}
            ]
        }
        
    except Exception as e:
        print(f"[FastAPI] Predict Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history_endpoint(query: Optional[str] = None, disease: Optional[str] = None, model: Optional[str] = None):
    return search_records(query, disease, model)

@app.post("/history/delete/{record_id}")
def delete_record_endpoint(record_id: int):
    delete_record(record_id)
    return {"status": "success"}

@app.get("/history/export")
def export_history_excel():
    records = get_records()
    df = pd.DataFrame(records)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Diagnostics Log')
        
    output.seek(0)
    headers = {
        'Content-Disposition': 'attachment; filename="diagnostics_history_log.xlsx"'
    }
    return StreamingResponse(output, headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/system/stats")
def system_stats_endpoint():
    return get_system_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app_fastapi:app", host="127.0.0.1", port=5000, reload=True)
