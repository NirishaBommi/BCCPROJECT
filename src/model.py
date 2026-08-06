"""
============================================================================
  model.py  —  SE-STN-EfficientNet Architecture
  Paper : "Basal Cell Carcinoma Skin Detection Using Deep Learning"
  Arch  : EfficientNet-B5 + Squeeze-and-Excitation + Spatial Transformer
          Network + Classification Head
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SQUEEZE-AND-EXCITATION BLOCK  (Hu et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation recalibration.

    Equation (1): z_c = (1/HxW) * sum_{i,j} u_c(i,j)   [Squeeze]
    Equation (2): s   = sigmoid(W2 * relu(W1 * z))       [Excitation]
    Output      : x_hat_c = s_c * u_c                    [Scale]

    Args:
        channels  (int): Number of input channels C.
        reduction (int): Reduction ratio r (default 16).
    """

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
# 2.  SPATIAL TRANSFORMER NETWORK  (Jaderberg et al., 2015)
# ─────────────────────────────────────────────────────────────────────────────
class STNModule(nn.Module):
    """
    Spatial Transformer Network — learns affine transformation theta in R^6
    to geometrically normalise lesion regions.

    Equation (3):
        [x_s; y_s] = [t11 t12 t13; t21 t22 t23] * [x_t; y_t; 1]

    Args:
        in_channels (int): Channels of input feature map.
    """

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
        # Initialise as identity transform
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        theta = self.fc_loc(self.localisation(x)).view(-1, 2, 3)
        grid  = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='border', align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    """
    Three-layer FC head:  in_features -> 1024 -> 512 -> num_classes
    Each layer: Linear -> BatchNorm -> GELU -> Dropout.
    """

    def __init__(self, in_features: int = 2048,
                 num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL SE-STN-EfficientNet MODEL
# ─────────────────────────────────────────────────────────────────────────────
class SESTNEfficientNet(nn.Module):
    """
    Full SE-STN-EfficientNet for BCC detection.

    Pipeline:
        Input (B,3,380,380)
          -> EfficientNet-B5 backbone (7 MBConv stages, pretrained)
          -> SE blocks after each stage
          -> STN module (geometric normalisation)
          -> Global Average Pooling -> (B, 2048)
          -> Classification Head    -> (B, num_classes)

    Args:
        num_classes  (int):  2=binary BCC, 5=BCC subtypes.
        pretrained   (bool): ImageNet-21k pretrained weights.
        se_reduction (int):  SE reduction ratio r.
        dropout      (float): Dropout probability.
    """

    def __init__(self, num_classes=2, pretrained=True,
                 se_reduction=16, dropout=0.5):
        super().__init__()

        # EfficientNet-B5 backbone
        self.backbone = timm.create_model(
            'efficientnet_b5',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )

        feature_info = self.backbone.feature_info
        channels = getattr(feature_info, "channels", None)

        def _parse_channels(source):
            if isinstance(source, torch.Tensor):
                if source.dim() == 0:
                    return [int(source.item())]
                return [int(item.item()) for item in source]
            if isinstance(source, (list, tuple)):
                parsed = []
                for item in source:
                    if isinstance(item, torch.Tensor):
                        parsed.append(int(item.item()))
                    elif isinstance(item, dict):
                        if "num_chs" in item:
                            parsed.append(int(item["num_chs"]))
                        elif "channels" in item:
                            parsed.append(int(item["channels"]))
                        elif "out_channels" in item:
                            parsed.append(int(item["out_channels"]))
                        else:
                            raise RuntimeError(
                                f"Unsupported feature entry type: {type(item)}"
                            )
                    elif hasattr(item, "num_chs"):
                        parsed.append(int(item.num_chs))
                    elif hasattr(item, "channels"):
                        parsed.append(int(item.channels))
                    else:
                        parsed.append(int(item))
                return parsed
            if isinstance(source, dict):
                return _parse_channels([source])
            if source is None:
                raise RuntimeError(
                    f"Unsupported feature_info type: {type(feature_info)}"
                )
            raise RuntimeError(
                f"Unsupported channel source type: {type(source)}"
            )

        if callable(channels):
            stage_channels = _parse_channels(channels())
        elif channels is not None:
            stage_channels = _parse_channels(channels)
        else:
            info = getattr(feature_info, "info", None)
            if info is None and hasattr(feature_info, "__iter__"):
                info = feature_info
            stage_channels = _parse_channels(info)

        # SE block per stage
        self.se_blocks = nn.ModuleList([
            SEBlock(c, reduction=se_reduction)
            for c in stage_channels
        ])

        # STN on deepest feature map
        self.stn = STNModule(in_channels=stage_channels[-1])

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.head = ClassificationHead(
            in_features=stage_channels[-1],
            num_classes=num_classes,
            dropout=dropout,
        )

        self.num_classes    = num_classes
        self.stage_channels = stage_channels

    def forward(self, x):
        features   = self.backbone(x)
        se_feats   = [se(f) for se, f in zip(self.se_blocks, features)]
        deep_feat  = self.stn(se_feats[-1])
        feat_vec   = self.gap(deep_feat).flatten(1)
        return self.head(feat_vec)

    def get_gradcam_target_layer(self):
        """Returns the target conv layer for Grad-CAM visualisation."""
        return self.se_blocks[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TEMPERATURE SCALING WRAPPER  (post-training calibration)
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaledModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        return self.model(x) / self.temperature

    def calibrate(self, val_loader, device='cuda'):
        self.to(device); self.model.eval()
        optimizer = torch.optim.LBFGS([self.temperature],
                                      lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device); labels = labels.to(device)
                all_logits.append(self.model(imgs))
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def eval_fn():
            optimizer.zero_grad()
            loss = criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        print(f"[Calibration] T = {self.temperature.item():.4f}")
        return self.temperature.item()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes=2, pretrained=True,
                se_reduction=16, dropout=0.5):
    model = SESTNEfficientNet(num_classes=num_classes,
                              pretrained=pretrained,
                              se_reduction=se_reduction,
                              dropout=dropout)
    total    = sum(p.numel() for p in model.parameters())
    trainable= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] SE-STN-EfficientNet-B5  |  Classes={num_classes}")
    print(f"        Total params    : {total:,}")
    print(f"        Trainable params: {trainable:,}")
    return model


if __name__ == "__main__":
    model = build_model(num_classes=2, pretrained=False)
    dummy = torch.randn(2, 3, 380, 380)
    out   = model(dummy)
    print(f"Input : {dummy.shape}  ->  Output: {out.shape}")