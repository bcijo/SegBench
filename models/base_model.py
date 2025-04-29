import torch
import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(BaseModel, self).__init__()
        # Load the backbone model
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone_channels = 2048  # Output channels of ResNet50
        elif backbone_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained).features
            self.backbone_channels = 512  # Output channels of VGG16
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove the fully connected layers from the backbone
        if 'resnet' in backbone_name:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc layers
        # For VGG, the features module is already convolutional

    def forward(self, x):
        raise NotImplementedError("Forward method should be implemented in subclass")

    def attach_head(self, head):
        """Attach a task-specific head to the model."""
        self.head = head