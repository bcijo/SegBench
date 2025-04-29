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
    model_config = config['model']
    if model_config['name'].lower() == 'unet':
        model = UNet(
            backbone_name=model_config['backbone'],
            pretrained=model_config.get('pretrained', True)
        )
        # Create and attach the segmentation head
        head = nn.Sequential(
            nn.Conv2d(model.dec_channels, model_config['num_classes'], kernel_size=1)
        )
        model.attach_head(head)
    else:
        raise ValueError(f"Model {model_config['name']} not supported")
    return model

def get_dataset(config, split, transform=None):
    if config['dataset']['name'].lower() == 'cityscapes':
        dataset = CityscapesDataset(
            root_dir=config['dataset']['root_dir'],
            split=split,
            transform=transform
        )
    elif config['dataset']['name'].lower() == 'pascal_voc':
        from datasets.pascal_voc import PascalVOCDataset  # Import PascalVOCDataset
        dataset = PascalVOCDataset(
            root=config['dataset']['root_dir'],
            split=split,
            task='semantic',  # We're doing semantic segmentation
            transforms=transform
        )
    else:
        raise ValueError(f"Dataset {config['dataset']['name']} not supported")
    return dataset

def train_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0
    metrics = SegmentationMetrics(num_classes=config['model']['num_classes'])
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        pred = outputs.argmax(1)
        metrics.update(pred, targets)
        
        # Update progress bar
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    # Compute epoch metrics
    scores = metrics.get_scores()
    return total_loss / len(loader), scores

def validate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0
    metrics = SegmentationMetrics(num_classes=config['model']['num_classes'])
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Update metrics
            pred = outputs.argmax(1)
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
    
    # Set device and print info
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # May provide speedup when using fixed input sizes
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA is not available. Using CPU. Training may be very slow!")
    
    # Create model and move to device
    model = get_model(config)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
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
        pin_memory=torch.cuda.is_available(),  # Only pin memory if CUDA is available
        drop_last=True  # Drop last incomplete batch to avoid batch norm issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['validation']['num_workers'],
        pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA is available
    )
    
    # Create optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer'].get('weight_decay', 0.0001)
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler']['T_max'],
        eta_min=config['training']['scheduler'].get('eta_min', 0.00001)
    )
    
    # Create criterion
    criterion = nn.CrossEntropyLoss(ignore_index=config['dataset']['ignore_index'])
    criterion = criterion.to(device)  # Move criterion to device
    
    # Create directories for saving
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Training loop
    best_miou = 0
    print(f"\nStarting training on device: {device}")
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        model.train()
        train_loss, train_scores = train_epoch(model, train_loader, criterion, optimizer, device, config)
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_scores['miou']:.4f}")
        
        # Validate
        if (epoch + 1) % config['logging']['val_interval'] == 0:
            model.eval()
            val_loss, val_scores = validate(model, val_loader, criterion, device, config)
            print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_scores['miou']:.4f}")
            
            # Save best model
            if val_scores['miou'] > best_miou:
                best_miou = val_scores['miou']
                save_path = os.path.join(config['logging']['save_dir'], 'best_model.pth')
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"Saved best model with mIoU: {best_miou:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            save_path = os.path.join(config['logging']['save_dir'], f'checkpoint_epoch{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }
            torch.save(checkpoint, save_path)

if __name__ == '__main__':
    main() 