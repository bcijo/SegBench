"""
Head for image classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_head import BaseHead


class ClassificationHead(BaseHead):
    """Head for image classification."""
    def __init__(self, config):
        super(ClassificationHead, self).__init__(config)
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Fully connected classifier
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, features):
        """
        Forward pass of classification head.

        Args:
            features (dict): Contains:
                - 'backbone_features': Tensor of shape (B, C, H, W)

        Returns:
            dict: Contains:
                - 'logits': Tensor of shape (B, num_classes)
                - 'probs': Tensor of shape (B, num_classes)
        """
        x = features['backbone_features']
        # Global average pool to (B, C, 1, 1)
        x = self.gap(x)
        # Flatten to (B, C)
        x = torch.flatten(x, 1)
        # Class logits
        logits = self.fc(x)
        # Probabilities
        probs = F.softmax(logits, dim=1)
        return {'logits': logits, 'probs': probs}

    def get_loss(self, outputs, targets):
        """
        Calculate classification loss.

        Args:
            outputs (dict): Contains 'logits'
            targets (dict): Contains 'labels'
        """
        logits = outputs['logits']
        labels = targets['labels']
        loss = F.cross_entropy(logits, labels)
        return {'loss': loss}

    def get_predictions(self, outputs):
        """
        Get predicted class indices.

        Args:
            outputs (dict): Contains 'logits' and 'probs'

        Returns:
            dict: Contains 'labels' and 'probs'
        """
        logits = outputs['logits']
        probs = outputs['probs']
        preds = torch.argmax(logits, dim=1)
        return {'labels': preds, 'probs': probs} 