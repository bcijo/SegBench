"""
PSPNet architecture with a configurable backbone (e.g., ResNet).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module used in PSPNet decoder."""
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        # Each pooled output has in_channels // len(pool_sizes) channels
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [stage(x) for stage in self.stages]
        pyramids = [F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False) for p in pyramids]
        # Concatenate original feature with pooled representations
        output = torch.cat([x] + pyramids, dim=1)
        return output


class PSPNet(BaseModel):
    """PSPNet architecture with a configurable backbone (e.g., ResNet)."""
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(PSPNet, self).__init__(backbone_name=backbone_name, pretrained=pretrained)
        # Pyramid Pooling Module on top of backbone features
        self.ppm = PyramidPoolingModule(self.backbone_channels)
        # Reduce concatenated features (original + pooled)
        concat_channels = self.backbone_channels * 2
        self.reducer = nn.Sequential(
            nn.Conv2d(concat_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Final conv block before upsampling
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        # Channels output by the decoder for head attachment
        self.dec_channels = 512

    def forward(self, x):
        # Record original spatial dimensions
        h, w = x.size(2), x.size(3)
        # Backbone feature extraction
        feat = self.backbone(x)
        # Pyramid pooling and feature aggregation
        x = self.ppm(feat)
        x = self.reducer(x)
        x = self.final_conv(x)
        # Upsample to original resolution
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # Package features and delegate to head
        features = {'backbone_features': x}
        return self.head(features) 