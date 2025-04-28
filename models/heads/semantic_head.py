from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_head import BaseHead

class SemanticHead(BaseHead):
    """Head for semantic segmentation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Convolutional layers for feature processing
        self.conv1 = nn.Conv2d(self.in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # Final classification layer
        self.classifier = nn.Conv2d(256, self.num_classes, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the semantic segmentation head.
        
        Args:
            features (dict): Dictionary containing:
                - 'backbone_features': Feature maps from backbone
                
        Returns:
            dict: Dictionary containing:
                - 'logits': Raw logits of shape (B, C, H, W)
                - 'probs': Class probabilities after softmax
        """
        x = features['backbone_features']
        
        # Feature processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Classification
        logits = self.classifier(x)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probs': probs
        }
    
    def get_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate semantic segmentation loss.
        
        Args:
            outputs (dict): Dictionary containing:
                - 'logits': Raw logits from the model
            targets (dict): Dictionary containing:
                - 'masks': Ground truth segmentation masks
                
        Returns:
            dict: Dictionary containing:
                - 'loss': Cross entropy loss
                - 'aux_loss': Auxiliary loss (if used)
        """
        logits = outputs['logits']
        target_masks = targets['masks']
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, target_masks, ignore_index=255)
        
        return {'loss': loss}
    
    def get_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get predictions from outputs.
        
        Args:
            outputs (dict): Dictionary containing model outputs
            
        Returns:
            dict: Dictionary containing:
                - 'masks': Predicted segmentation masks
                - 'probs': Class probabilities
        """
        logits = outputs['logits']
        probs = outputs['probs']
        
        # Get predicted class indices
        masks = torch.argmax(logits, dim=1)
        
        return {
            'masks': masks,
            'probs': probs
        } 