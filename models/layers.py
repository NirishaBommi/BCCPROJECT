import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHANNEL ATTENTION: SQUEEZE-AND-EXCITATION (SE) BLOCK
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        z = self.squeeze(x).view(b, c)
        s = self.excitation(z).view(b, c, 1, 1)
        return x * s

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM)
# ─────────────────────────────────────────────────────────────────────────────
class CBAMBlock(nn.Module):
    """
    Woo et al., 2018. Dual Channel and Spatial Attention Module.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        
        # Channel Attention
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Channel Attention
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)
        
        channel_out = self.fc(avg_pool) + self.fc(max_pool)
        channel_scale = torch.sigmoid(channel_out).view(b, c, 1, 1)
        x_channel = x * channel_scale
        
        # 2. Spatial Attention
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out = torch.max(x_channel, dim=1, keepdim=True).values
        spatial_concat = torch.cat([avg_out, max_out], dim=1)
        
        spatial_scale = torch.sigmoid(self.spatial_conv(spatial_concat))
        return x_channel * spatial_scale

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPATIAL TRANSFORMER NETWORK (STN)
# ─────────────────────────────────────────────────────────────────────────────
class STNModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.localisation = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.fc_loc = nn.Linear(256, 6)
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.fc_loc(self.localisation(x)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-SCALE FEATURE PYRAMID (FPN / BiFPN)
# ─────────────────────────────────────────────────────────────────────────────
class BiFPNNeck(nn.Module):
    """
    Combines feature maps from intermediate resolutions (e.g. multi-scale layers)
    using learnable weights and fast normalized fusion.
    """
    def __init__(self, feature_channels: list, out_channels: int = 256):
        super().__init__()
        # Project inputs of different channels to target out_channels
        self.project_layers = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) for ch in feature_channels
        ])
        
        # Learnable fusion weights
        self.w1 = nn.Parameter(torch.ones(len(feature_channels), dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(len(feature_channels), dtype=torch.float32))
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: list):
        # features list contains tensors of increasing channels & decreasing sizes
        projected = [self.project_layers[i](f) for i, f in enumerate(features)]
        
        target_size = projected[0].shape[2:] # target largest spatial size
        
        # Fast Normalized Fusion
        w1 = F.relu(self.w1)
        w1_norm = w1 / (torch.sum(w1) + 1e-4)
        
        fused = torch.zeros_like(projected[0])
        for i, f in enumerate(projected):
            if f.shape[2:] != target_size:
                f_resized = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            else:
                f_resized = f
            fused += w1_norm[i] * f_resized
            
        return self.conv_out(fused)
