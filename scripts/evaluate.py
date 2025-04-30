import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import torch.nn as nn

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cityscapes import CityscapesDataset
from datasets.pascal_voc import PascalVOCDataset
from models.unet import UNet
from utils.transforms import get_transform
from utils.metrics import SegmentationMetrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                      choices=['train', 'val', 'test'],
                      help='Dataset split to evaluate on')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(config, checkpoint_path):
    # Create model
    model_config = config['model']
    if model_config['name'].lower() == 'unet':
        model = UNet(
            backbone_name=model_config['backbone'],
            pretrained=False
        )
        # Create and attach the segmentation head (consistent with train.py)
        head = nn.Sequential(
            nn.Conv2d(model.dec_channels, model_config['num_classes'], kernel_size=1)
        )
        model.attach_head(head)
    else:
        raise ValueError(f"Model {model_config['name']} not supported")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Handle checkpoints saved with DataParallel
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if saved with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        # Handle checkpoints saved directly as state_dict
        state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    return model

def evaluate(model, loader, device, num_classes):
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluation')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(images)
            pred = outputs.argmax(1)
            
            # Update metrics
            metrics.update(pred, targets)
    
    # Compute metrics
    scores = metrics.get_scores()
    class_ious = metrics.get_class_ious()
    
    return scores, class_ious

def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and load checkpoint
    model = get_model(config, args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    transform = get_transform(config['dataset'], is_train=False)
    if config['dataset']['name'].lower() == 'cityscapes':
        dataset = CityscapesDataset(
            root_dir=config['dataset']['root_dir'],
            split=args.split,
            transform=transform
        )
    elif config['dataset']['name'].lower() == 'pascal_voc':
        dataset = PascalVOCDataset(
            root=config['dataset']['root_dir'],
            split=args.split,
            task='semantic',
            transforms=transform
        )
    else:
        raise ValueError(f"Dataset {config['dataset']['name']} not supported")
    
    loader = DataLoader(
        dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['validation']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    scores, class_ious = evaluate(model, loader, device, config['model']['num_classes'])
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Split: {args.split}")
    print(f"Mean IoU: {scores['miou']:.4f}")
    print(f"Pixel Accuracy: {scores['pixel_acc']:.4f}")
    print(f"Mean Class Accuracy: {scores['class_acc']:.4f}")
    
    # Per-class IoU requires a get_classes method in the dataset (implement if needed)
    # print("\nPer-class IoU:")
    # try:
    #     class_names = dataset.get_classes()[0]  # Assuming get_classes returns (names, mapping)
    #     for i, iou in enumerate(class_ious):
    #         print(f"{class_names[i]}: {iou:.4f}")
    # except AttributeError:
    #     print("Dataset does not have a get_classes method for detailed IoU report.")
    #     for i, iou in enumerate(class_ious):
    #         print(f"Class {i}: {iou:.4f}")

if __name__ == '__main__':
    main() 