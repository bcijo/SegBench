from typing import Dict, Any, Optional
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config (dict): Model configuration containing:
                - backbone: Name of the backbone network
                - num_classes: Number of output classes
                - in_channels: Number of input channels
                Additional backbone-specific parameters
        """
        super().__init__()
        self.config = config
        self.backbone = None
        self.head = None
        
    def init_backbone(self):
        """Initialize the backbone network."""
        raise NotImplementedError("Subclasses must implement init_backbone")
    
    def init_head(self):
        """Initialize the task-specific head."""
        raise NotImplementedError("Subclasses must implement init_head")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            dict: Dictionary containing:
                - 'out': Main output tensor
                - Additional task-specific outputs
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def get_backbone_output_channels(self) -> int:
        """Get the number of output channels from the backbone."""
        raise NotImplementedError("Subclasses must implement get_backbone_output_channels")
    
    def freeze_backbone(self):
        """Freeze the backbone parameters."""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters."""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def get_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss.
        
        Args:
            outputs (dict): Model outputs
            targets (dict): Ground truth targets
            
        Returns:
            dict: Dictionary containing loss values
        """
        raise NotImplementedError("Subclasses must implement get_loss")
    
    def load_pretrained(self, path: Optional[str] = None):
        """
        Load pretrained weights.
        
        Args:
            path (str, optional): Path to pretrained weights
        """
        if path is not None:
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}") 