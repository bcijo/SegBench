"""
UNet architecture with a configurable ResNet50 encoder backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base_model import BaseModel


class UpConvBlock(nn.Module):
    """Upsampling block used in UNet decoder."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpConvBlock, self).__init__()
        # Transposed convolution for upsampling (input channels from layer below, output channels to match skip connection for simpler concatenation handling)
        # Let's keep output channels of upconv as out_channels
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Double convolution block after concatenation
        # Input channels = channels from upconv (out_channels) + channels from skip connection (skip_channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(skip_channels + out_channels, out_channels, kernel_size=3, padding=1), # Corrected input channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        # Concatenate upsampled features with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(BaseModel):
    """UNet architecture with a ResNet50 encoder backbone."""
    def __init__(self, backbone_name='resnet50', pretrained=True):
        # Initialize BaseModel to register attach_head
        super(UNet, self).__init__(backbone_name=backbone_name, pretrained=pretrained)
        # Currently only ResNet50 backbone is supported
        if backbone_name != 'resnet50':
            raise ValueError(f"UNet currently supports only 'resnet50' backbone, got {backbone_name}")
        # Load a fresh ResNet50 for encoder feature extraction
        backbone = models.resnet50(pretrained=pretrained)
        # Encoder blocks
        self.enc0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.pool0 = backbone.maxpool
        self.enc1 = backbone.layer1  # 256 channels
        self.enc2 = backbone.layer2  # 512 channels
        self.enc3 = backbone.layer3  # 1024 channels
        self.enc4 = backbone.layer4  # 2048 channels

        # Decoder with skip connections
        self.up4 = UpConvBlock(2048, 1024, 512)  # Input from enc4 (2048), Skip from enc3 (1024), Output 512 channels
        self.up3 = UpConvBlock(512, 512, 256)   # Input from d3 (512), Skip from enc2 (512), Output 256 channels
        self.up2 = UpConvBlock(256, 256, 128)   # Input from d2 (256), Skip from enc1 (256), Output 128 channels
        self.up1 = UpConvBlock(128, 64, 64)    # Input from d1 (128), Skip from enc0 (64), Output 64 channels

        # Number of channels output by decoder (for head attachment)
        self.dec_channels = 64

    def forward(self, x):
        # Encoder path
        x0 = self.enc0(x)          # [B,64,H/2,W/2]
        x1 = self.pool0(x0)        # [B,64,H/4,W/4]
        x1 = self.enc1(x1)         # [B,256,H/4,W/4]
        x2 = self.enc2(x1)         # [B,512,H/8,W/8]
        x3 = self.enc3(x2)         # [B,1024,H/16,W/16]
        x4 = self.enc4(x3)         # [B,2048,H/32,W/32]

        # Decoder path with skip connections
        d3 = self.up4(x4, x3)      # [B,512,H/16,W/16]
        d2 = self.up3(d3, x2)      # [B,256,H/8,W/8]
        d1 = self.up2(d2, x1)      # [B,128,H/4,W/4]
        d0 = self.up1(d1, x0)      # [B,64,H/2,W/2]

        # Upsample to original resolution
        decoded = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)
        # Package features for the attached head
        features = {'backbone_features': decoded}
        # Delegate to the task-specific head
        return self.head(features) 