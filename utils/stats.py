import psutil
import torch
import numpy as np

def get_system_stats():
    """
    Returns live system metric ratios.
    """
    stats = {
        "cpu_usage": f"{psutil.cpu_percent()}%",
        "ram_usage": f"{psutil.virtual_memory().percent}%",
        "gpu_usage": "N/A (CPU Mode)",
        "inference_speed": "N/A",
        "queue_status": "Idle"
    }
    
    if torch.cuda.is_available():
        # Get active device memory
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2) # MB
        cached = torch.cuda.memory_reserved(device) / (1024 ** 2) # MB
        stats["gpu_usage"] = f"{allocated:.1f} MB / {cached:.1f} MB (CUDA Active)"
        
    return stats

def estimate_model_complexity(model):
    """
    Calculates total trainable parameter counts and sizes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": f"{model_size_mb:.2f} MB"
    }
