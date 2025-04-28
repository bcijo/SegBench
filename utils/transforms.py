from typing import Dict, Tuple, Optional, List
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

class Compose:
    """Composes several transforms together."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

class Resize:
    """Resize the image and target to a given size."""
    
    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size (tuple): Desired output size (H, W)
        """
        self.size = size
    
    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        target = sample['target']
        
        # Resize image
        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        
        # Resize target (nearest neighbor interpolation for masks)
        if isinstance(target, torch.Tensor):
            target = target.unsqueeze(0)  # Add channel dimension
            target = TF.resize(target, self.size, interpolation=Image.NEAREST)
            target = target.squeeze(0)  # Remove channel dimension
        else:
            target = Image.fromarray(target)
            target = target.resize(self.size[::-1], Image.NEAREST)
            target = np.array(target)
        
        return {'image': image, 'target': target}

class ToTensor:
    """Convert image and target to PyTorch tensors."""
    
    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        target = sample['target']
        
        # Convert image to tensor
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        
        # Convert target to tensor
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).long()
        
        return {'image': image, 'target': target}

class Normalize:
    """Normalize the image with given mean and standard deviation."""
    
    def __init__(self, mean: List[float], std: List[float]):
        """
        Args:
            mean (list): Mean values for each channel
            std (list): Standard deviation values for each channel
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict) -> Dict:
        image = sample['image']
        target = sample['target']
        
        # Normalize image
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        return {'image': image, 'target': target}

class RandomHorizontalFlip:
    """Randomly flip the image and target horizontally."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p (float): Probability of flipping
        """
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        if torch.rand(1) < self.p:
            image = sample['image']
            target = sample['target']
            
            # Flip image
            if isinstance(image, torch.Tensor):
                image = TF.hflip(image)
            else:
                image = Image.fromarray(image)
                image = TF.hflip(image)
            
            # Flip target
            if isinstance(target, torch.Tensor):
                target = TF.hflip(target)
            else:
                target = Image.fromarray(target)
                target = TF.hflip(target)
                target = np.array(target)
            
            return {'image': image, 'target': target}
        return sample

def get_transform(config: Dict, is_train: bool = True) -> Compose:
    """
    Get the transformation pipeline based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        is_train (bool): Whether to include training-specific augmentations
        
    Returns:
        Compose: Composition of transforms
    """
    transforms = []
    
    # Resize
    if 'image_size' in config:
        transforms.append(Resize(config['image_size']))
    
    # Data augmentation for training
    if is_train:
        transforms.append(RandomHorizontalFlip(p=0.5))
    
    # Convert to tensor
    transforms.append(ToTensor())
    
    # Normalize
    if 'mean' in config and 'std' in config:
        transforms.append(Normalize(config['mean'], config['std']))
    
    return Compose(transforms) 