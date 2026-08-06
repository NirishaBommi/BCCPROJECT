import torch
import torch.nn.functional as F
import numpy as np
import cv2

class ClinicalExplainEngine:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
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
        """
        Generates a 2D explainability heatmap using the specified XAI method.
        Supported methods: 'gradcam', 'gradcam_plusplus', 'scorecam', 'eigencam'
        """
        self.model.eval()  # keep in eval mode to prevent batchnorm batch size 1 errors
        
        # 1. Forward pass
        input_grad = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_grad)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # 2. Backward pass for gradient-based methods
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        
        # 3. Compute Heatmap based on method
        method = method.lower().replace("++", "_plusplus")
        
        if method == "gradcam":
            heatmap = self._compute_gradcam()
        elif method == "gradcam_plusplus":
            heatmap = self._compute_gradcam_plusplus()
        elif method == "scorecam":
            heatmap = self._compute_scorecam(input_tensor, class_idx)
        elif method == "eigencam":
            heatmap = self._compute_eigencam()
        else:
            heatmap = self._compute_gradcam()  # default fallback
            
        return heatmap

    def _compute_gradcam(self):
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        # Global average pool of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1)
        
        cam = F.relu(cam)
        cam = cam.cpu().numpy()[0]
        return self._normalize(cam)

    def _compute_gradcam_plusplus(self):
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        # Positive gradients
        pos_gradients = torch.clamp(self.gradients, min=0)
        
        # Squared and cubed gradients for weighting coeff alpha
        grads_power_2 = self.gradients ** 2
        grads_power_3 = grads_power_2 * self.gradients
        
        sum_activations = torch.sum(self.activations, dim=(2, 3), keepdim=True)
        eps = 1e-7
        
        # Alpha coefficients
        alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        weights = torch.sum(alpha * pos_gradients, dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()[0]
        return self._normalize(cam)

    def _compute_eigencam(self):
        if self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        # PCA on activations (doesn't require gradients)
        activations = self.activations.cpu().numpy()[0]  # Shape: (C, H, W)
        C, H, W = activations.shape
        
        reshaped = activations.reshape(C, H * W).T  # Shape: (H*W, C)
        # Mean center
        reshaped = reshaped - np.mean(reshaped, axis=0)
        
        # SVD
        try:
            U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)
            projection = U[:, 0].reshape(H, W)
        except Exception:
            # Fallback to mean activation if SVD fails to converge
            projection = np.mean(activations, axis=0)
            
        return self._normalize(projection)

    def _compute_scorecam(self, input_tensor, class_idx):
        if self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        # Score-CAM weights activations based on logit change.
        # Since full Score-CAM requires forward pass per channel (e.g. 1280 passes),
        # we run an optimized version (Fast Score-CAM) using the top 16 most active channels.
        activations = self.activations.cpu()
        mean_acts = torch.mean(activations, dim=(2, 3))[0]
        
        # Get indices of top 16 active channels
        top_k = min(16, activations.shape[1])
        top_channels = torch.topk(mean_acts, top_k).indices.numpy()
        
        cam = torch.zeros((activations.shape[2], activations.shape[3]), dtype=torch.float32)
        
        # Baseline prediction
        with torch.no_grad():
            baseline_logits = self.model(input_tensor)
            baseline_score = F.softmax(baseline_logits, dim=1)[0, class_idx].item()
            
        for c in top_channels:
            activation_map = activations[0, c, :, :].unsqueeze(0).unsqueeze(0)
            # Upsample map to match input resolution
            map_upsampled = F.interpolate(activation_map, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
            # Min-max normalization of map
            map_min, map_max = map_upsampled.min(), map_upsampled.max()
            map_norm = (map_upsampled - map_min) / (map_max - map_min + 1e-8)
            
            # Mask the input image with the normalized map (sending it to same device)
            masked_input = input_tensor * map_norm.to(input_tensor.device)
            
            with torch.no_grad():
                masked_logits = self.model(masked_input)
                masked_score = F.softmax(masked_logits, dim=1)[0, class_idx].item()
                
            # Weight is the increase/change in target class score
            weight = max(0, masked_score - baseline_score)
            cam += weight * activations[0, c, :, :]
            
        cam = F.relu(cam)
        return self._normalize(cam.numpy())

    def _normalize(self, x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x)
        return x

def overlay_heatmap(orig_img, heatmap, opacity=0.4):
    """
    Overlays a heat map on top of the original BGR image.
    """
    h, w = orig_img.shape[:2]
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Apply JET colormap
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend images
    blended = cv2.addWeighted(heatmap_color, opacity, orig_img, 1.0 - opacity, 0)
    return blended
