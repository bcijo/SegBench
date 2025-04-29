"""
HRNet architecture adapted for segmentation tasks using a timm backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HRNet(nn.Module):
    """HRNet backbone for segmentation tasks using timm."""
    def __init__(self, backbone_name='hrnet_w18_small_v2', pretrained=True):
        super(HRNet, self).__init__()
        # Create HRNet model that returns intermediate features
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        # Determine the number of channels in the final feature map
        self.dec_channels = self.encoder.feature_info.channels()[-1]

    def attach_head(self, head):
        """Attach a task-specific head to the HRNet model."""
        self.head = head

    def forward(self, x):
        # Extract features at multiple resolutions
        feats = self.encoder(x)
        # Use the highest-resolution feature (last one)
        feat = feats[-1]
        # Upsample to original image spatial size
        feat = F.interpolate(feat, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # Package features for the attached head
        features = {'backbone_features': feat}
        return self.head(features) 