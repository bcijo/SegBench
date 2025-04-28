import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cityscapes import CityscapesDataset
from models.unet import UNet
from utils.transforms import get_transform
from utils.metrics import SegmentationMetrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(config):
    if config['model']['name'].lower() == 'unet':
        model = UNet(config['model'])
    else:
        raise ValueError(f"Model {config['model']['name']} not supported")
    return model

def get_dataset(config, split, transform=None):
    if config['dataset']['name'].lower() == 'cityscapes':
        dataset = CityscapesDataset(
            root_dir=config['dataset']['root_dir'],
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {config['dataset']['name']} not supported")
    return dataset

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics = SegmentationMetrics(num_classes=19)  # Assuming Cityscapes
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs['logits'], targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        pred = outputs['logits'].argmax(1)
        metrics.update(pred, targets)
        
        # Update progress bar
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    # Compute epoch metrics
    scores = metrics.get_scores()
    return total_loss / len(loader), scores

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = SegmentationMetrics(num_classes=19)  # Assuming Cityscapes
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs['logits'], targets)
            
            # Update metrics
            pred = outputs['logits'].argmax(1)
            metrics.update(pred, targets)
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    # Compute epoch metrics
    scores = metrics.get_scores()
    return total_loss / len(loader), scores

def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(config)
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_transform = get_transform(config['dataset'], is_train=True)
    val_transform = get_transform(config['dataset'], is_train=False)
    
    train_dataset = get_dataset(config, 'train', train_transform)
    val_dataset = get_dataset(config, 'val', val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['validation']['num_workers'],
        pin_memory=True
    )
    
    # Create optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler']['T_max'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # Create criterion
    criterion = nn.CrossEntropyLoss(ignore_index=config['dataset']['ignore_index'])
    
    # Create directories for saving
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Training loop
    best_miou = 0
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_scores = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_scores['miou']:.4f}")
        
        # Validate
        if (epoch + 1) % config['logging']['val_interval'] == 0:
            val_loss, val_scores = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_scores['miou']:.4f}")
            
            # Save best model
            if val_scores['miou'] > best_miou:
                best_miou = val_scores['miou']
                save_path = os.path.join(config['logging']['save_dir'], 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model with mIoU: {best_miou:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            save_path = os.path.join(config['logging']['save_dir'], f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, save_path)

if __name__ == '__main__':
    main() 