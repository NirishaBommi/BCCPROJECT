import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ExplainableEngine:
    """
    Unified Explainable AI (XAI) Suite.
    Supported attribution methods:
    - Grad-CAM (Selvaraju et al.)
    - Grad-CAM++ (Chattopadhyay et al.)
    - Score-CAM (Wang et al., optimized top-16 channels)
    - Eigen-CAM (PCA projection)
    - Integrated Gradients (Sundararajan et al., 30 steps path-integral)
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Setup hooks
        def forward_hook(module, input, output):
            if isinstance(output, (list, tuple)):
                self.activations = output[-1].detach()
            else:
                self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, (list, tuple)):
                grad = next((g for g in grad_output if g is not None), None)
                if grad is not None:
                    self.gradients = grad.detach()
            else:
                self.gradients = grad_output.detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate_heatmap(self, input_tensor, method="gradcam", class_idx=None):
        self.model.eval() # Keep in eval to prevent batchnorm batch size 1 errors
        
        # 1. Forward pass
        input_grad = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_grad)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # 2. Backward pass for gradient-based methods
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        
        method = method.lower().replace("++", "_plusplus")
        
        # 3. Compute Heatmap
        if method == "gradcam":
            heatmap = self._compute_gradcam()
        elif method == "gradcam_plusplus":
            heatmap = self._compute_gradcam_plusplus()
        elif method == "scorecam":
            heatmap = self._compute_scorecam(input_tensor, class_idx)
        elif method == "eigencam":
            heatmap = self._compute_eigencam()
        elif method == "integrated_gradients":
            heatmap = self._compute_integrated_gradients(input_tensor, class_idx)
        else:
            heatmap = self._compute_gradcam() # Default fallback
            
        return heatmap
        
    def _compute_gradcam(self):
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        return self._normalize(cam)
        
    def _compute_gradcam_plusplus(self):
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        grads_power2 = gradients ** 2
        grads_power3 = gradients ** 3
        
        # Equation coefficients alpha
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 1e-8
        
        aij = grads_power2 / (2 * grads_power2 + sum_activations[:, np.newaxis, np.newaxis] * grads_power3 + eps)
        weights = np.sum(aij * np.maximum(gradients, 0), axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        return self._normalize(cam)
        
    def _compute_scorecam(self, input_tensor, class_idx):
        if self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        activations = self.activations.cpu()
        mean_acts = torch.mean(activations, dim=(2, 3))[0]
        
        top_k = min(16, activations.shape[1])
        top_channels = torch.topk(mean_acts, top_k).indices.numpy()
        
        cam = torch.zeros((activations.shape[2], activations.shape[3]), dtype=torch.float32)
        
        with torch.no_grad():
            baseline_logits = self.model(input_tensor)
            baseline_score = F.softmax(baseline_logits, dim=1)[0, class_idx].item()
            
        for c in top_channels:
            activation_map = activations[0, c, :, :].unsqueeze(0).unsqueeze(0)
            map_upsampled = F.interpolate(activation_map, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
            map_min, map_max = map_upsampled.min(), map_upsampled.max()
            map_norm = (map_upsampled - map_min) / (map_max - map_min + 1e-8)
            
            # Send map_norm to correct device to prevent mismatch errors
            masked_input = input_tensor * map_norm.to(input_tensor.device)
            
            with torch.no_grad():
                masked_logits = self.model(masked_input)
                masked_score = F.softmax(masked_logits, dim=1)[0, class_idx].item()
                
            weight = max(0, masked_score - baseline_score)
            cam += weight * activations[0, c, :, :]
            
        cam = F.relu(cam)
        return self._normalize(cam.numpy())
        
    def _compute_eigencam(self):
        if self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        activations = self.activations.cpu().numpy()[0]
        
        # Reshape to (C, H*W)
        c, h, w = activations.shape
        flat = activations.reshape(c, h * w).T # (H*W, C)
        
        # PCA projection using SVD
        flat_centered = flat - np.mean(flat, axis=0)
        _, _, vh = np.linalg.svd(flat_centered, full_matrices=False)
        
        # First principal component
        projection = flat_centered @ vh[0, :]
        cam = projection.reshape(h, w)
        
        cam = np.maximum(cam, 0)
        return self._normalize(cam)
        
    def _compute_integrated_gradients(self, input_tensor, class_idx, steps=30):
        # Baseline is a black image
        baseline = torch.zeros_like(input_tensor).to(input_tensor.device)
        
        # Create steps path interpolation
        scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
        
        grads = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            logits = self.model(scaled_input)
            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward()
            grads.append(scaled_input.grad.cpu().numpy()[0])
            
        avg_grads = np.mean(np.array(grads), axis=0) # (C, H, W)
        delta = (input_tensor - baseline).cpu().numpy()[0] # (C, H, W)
        
        ig = delta * avg_grads
        ig_collapsed = np.mean(ig, axis=0)
        
        ig_collapsed = np.maximum(ig_collapsed, 0)
        return self._normalize(ig_collapsed)
        
    def _normalize(self, x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min > 1e-8:
            return (x - x_min) / (x_max - x_min)
        else:
            return np.zeros_like(x)
