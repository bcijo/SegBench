import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cityscapes import CityscapesDataset
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
    if config['model']['name'].lower() == 'unet':
        model = UNet(config['model'])
    else:
        raise ValueError(f"Model {config['model']['name']} not supported")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
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
            pred = outputs['logits'].argmax(1)
            
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
    
    # Create model and load checkpoint
    model = get_model(config, args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    transform = get_transform(config['dataset'], is_train=False)
    dataset = CityscapesDataset(
        root_dir=config['dataset']['root_dir'],
        split=args.split,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['validation']['num_workers'],
        pin_memory=True
    )
    
    # Evaluate
    scores, class_ious = evaluate(model, loader, device, config['dataset']['num_classes'])
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Split: {args.split}")
    print(f"Mean IoU: {scores['miou']:.4f}")
    print(f"Pixel Accuracy: {scores['pixel_acc']:.4f}")
    print(f"Mean Class Accuracy: {scores['class_acc']:.4f}")
    
    print("\nPer-class IoU:")
    class_names = dataset.get_classes()[0]  # Assuming get_classes returns (names, mapping)
    for i, iou in enumerate(class_ious):
        print(f"{class_names[i]}: {iou:.4f}")

if __name__ == '__main__':
    main() 