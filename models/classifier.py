import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.layers import SEBlock, CBAMBlock, STNModule, BiFPNNeck

class ClassificationHead(nn.Module):
    """
    Three-layer FC head matching the checkpoint structure.
    """
    def __init__(self, in_features: int = 320, num_classes: int = 2, dropout: float = 0.5):
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

class ModularSkinClassifier(nn.Module):
    """
    Refactored modular skin lesion classifier.
    Supports backwards compatible weights loading.
    """
    def __init__(
        self, 
        backbone_name: str = "efficientnet_b0", 
        num_classes: int = 2, 
        use_se: bool = True,
        use_cbam: bool = True,
        use_stn: bool = True,
        use_bifpn: bool = False,
        pretrained: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_se = use_se
        self.use_cbam = use_cbam
        self.use_stn = use_stn
        self.use_bifpn = use_bifpn
        
        is_transformer = "swin" in backbone_name.lower() or "vit" in backbone_name.lower()
        
        if is_transformer:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
            self.use_bifpn = False
            self.use_stn = False
        else:
            # Load CNN backbone with intermediate features to match checkpoint
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
            feature_channels = self.backbone.feature_info.channels()
            in_features = feature_channels[-1]
            
            # Setup BiFPN Neck if enabled
            if self.use_bifpn:
                self.bifpn = BiFPNNeck(feature_channels, out_channels=256)
                in_features = 256
                
        # 2. Attention Layers (CNN paths only)
        self.attention_layers = nn.Sequential()
        if not is_transformer:
            if self.use_se:
                self.attention_layers.add_module("se_block", SEBlock(in_features))
            if self.use_cbam:
                self.attention_layers.add_module("cbam_block", CBAMBlock(in_features))
                
        # 3. Spatial Transformer Network (STN)
        self.stn = None
        if self.use_stn and not is_transformer:
            self.stn = STNModule(in_channels=in_features)
            
        # 4. Global Average Pooling & Head
        self.gap = nn.AdaptiveAvgPool2d(1) if not is_transformer else nn.Identity()
        self.head = ClassificationHead(in_features=in_features, num_classes=num_classes, dropout=0.5)
        
    def forward(self, x):
        is_transformer = "swin" in self.backbone_name.lower() or "vit" in self.backbone_name.lower()
        
        if is_transformer:
            features = self.backbone(x)
        else:
            features_list = self.backbone(x)
            if self.use_bifpn:
                features = self.bifpn(features_list)
            else:
                features = features_list[-1]
                
            # Apply STN alignment
            if self.stn is not None:
                features = self.stn(features)
                
            # Apply attention blocks
            features = self.attention_layers(features)
            
            # Pool to vector
            features = self.gap(features).flatten(1)
            
        return self.head(features)
