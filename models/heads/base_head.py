from typing import Dict, Any
import torch
import torch.nn as nn

class BaseHead(nn.Module):
    """Base class for all task-specific heads."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the head.
        
        Args:
            config (dict): Head configuration containing:
                - in_channels: Number of input channels
                - num_classes: Number of output classes
                Additional task-specific parameters
        """
        super().__init__()
        self.config = config
        self.in_channels = config['in_channels']
        self.num_classes = config['num_classes']
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the head.
        
        Args:
            features (dict): Dictionary of feature maps from the backbone
            
        Returns:
            dict: Dictionary containing task-specific outputs
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def get_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the task-specific loss.
        
        Args:
            outputs (dict): Head outputs
            targets (dict): Ground truth targets
            
        Returns:
            dict: Dictionary containing loss values
        """
        raise NotImplementedError("Subclasses must implement get_loss")
    
    def get_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get predictions from outputs.
        
        Args:
            outputs (dict): Head outputs
            
        Returns:
            dict: Dictionary containing predictions in appropriate format
        """
        raise NotImplementedError("Subclasses must implement get_predictions") 