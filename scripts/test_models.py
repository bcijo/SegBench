"""
Test script to verify model architectures and heads.
"""
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import UNet
from models.pspnet import PSPNet
from models.hrnet import HRNet
from models.heads.semantic_head import SemanticHead
from models.heads.instance_head import InstanceHead
from models.heads.panoptic_head import PanopticHead
from models.heads.classification_head import ClassificationHead

def test_model(model, input_size=(2, 3, 512, 512), name=""):
    """Test a model with random input."""
    print(f"\nTesting {name}...")
    try:
        # Create random input
        x = torch.randn(input_size)
        
        # Test forward pass
        out = model(x)
        print(f"✓ Forward pass successful")
        print(f"✓ Output shapes:")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  - {k}: {tuple(v.shape)}")
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def create_head_config(in_channels, num_classes):
    """Create a basic config for heads."""
    return {
        'in_channels': in_channels,
        'num_classes': num_classes,
        'dropout': 0.1,
        'ignore_index': 255,
        # For panoptic head
        'num_semantic_classes': num_classes,
        'num_instance_classes': num_classes,
    }

def main():
    print("Starting model tests...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 2
    height, width = 512, 512
    in_channels = 3
    num_classes = 19
    input_size = (batch_size, in_channels, height, width)
    
    # Instantiate backbones
    backbones = {
        'UNet': UNet(backbone_name='resnet50', pretrained=False),
        'PSPNet': PSPNet(backbone_name='resnet50', pretrained=False),
        'HRNet': HRNet(backbone_name='hrnet_w18_small_v2', pretrained=False)
    }
    
    # Base head classes (will be instantiated dynamically)
    head_classes = {
        'SemanticHead': SemanticHead,
        'InstanceHead': InstanceHead,
        'PanopticHead': PanopticHead,
        'ClassificationHead': ClassificationHead
    }
    
    # Test backbones with dynamically created semantic head
    print("\n=== Testing Backbones ===")
    SemanticHeadClass = head_classes['SemanticHead']
    for name, backbone in backbones.items():
        head_config = create_head_config(backbone.dec_channels, num_classes)
        head = SemanticHeadClass(head_config)
        backbone.attach_head(head)
        backbone = backbone.to(device)
        test_model(backbone, input_size, name)
    
    # Test all heads with a fixed backbone (UNet)
    print("\n=== Testing Heads ===")
    backbone = UNet(backbone_name='resnet50', pretrained=False)
    unet_dec_channels = backbone.dec_channels # Get UNet's decoder channels
    for name, HeadClass in head_classes.items():
        head_config = create_head_config(unet_dec_channels, num_classes)
        head = HeadClass(head_config)
        backbone.attach_head(head)
        backbone = backbone.to(device)
        test_model(backbone, input_size, f"UNet + {name}")
    
    print("\nAll tests completed!")

if __name__ == '__main__':
    main() 