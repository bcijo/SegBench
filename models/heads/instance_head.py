"""
Head for instance segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_head import BaseHead


class InstanceHead(BaseHead):
    """Head for instance segmentation."""
    def __init__(self, config: dict):
        super(InstanceHead, self).__init__(config)
        # Two convolutional layers for feature refinement
        self.conv1 = nn.Conv2d(self.in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # Pixel-wise classifier predicting instance ID per pixel
        self.classifier = nn.Conv2d(256, self.num_classes, kernel_size=1)
        # Dropout for regularization
        dropout_p = config.get('dropout', 0.1)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, features: dict) -> dict:
        """
        Forward pass of the instance segmentation head.

        Args:
            features (dict): Contains 'backbone_features' tensor of shape (B, C, H, W)

        Returns:
            dict: Contains 'logits' (B, num_instances, H, W) and 'probs' (B, num_instances, H, W)
        """
        x = features['backbone_features']
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return {'logits': logits, 'probs': probs}

    def get_loss(self, outputs: dict, targets: dict) -> dict:
        """
        Calculate instance segmentation loss as pixel-wise cross-entropy.

        Args:
            outputs (dict): Contains 'logits' tensor
            targets (dict): Contains 'masks' tensor of instance IDs

        Returns:
            dict: Contains 'loss'
        """
        logits = outputs['logits']
        instance_masks = targets['masks']
        # Allow configuring ignore index, default to 255
        ignore_idx = self.config.get('ignore_index', 255)
        loss = F.cross_entropy(logits, instance_masks, ignore_index=ignore_idx)
        return {'loss': loss}

    def get_predictions(self, outputs: dict) -> dict:
        """
        Get instance mask predictions by taking argmax over instance ID logits.

        Args:
            outputs (dict): Contains 'logits' and 'probs'

        Returns:
            dict: Contains 'masks' tensor (B, H, W) and 'probs'
        """
        logits = outputs['logits']
        probs = outputs['probs']
        masks = torch.argmax(logits, dim=1)
        return {'masks': masks, 'probs': probs} 