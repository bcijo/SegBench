import random
import torch
import torchvision.transforms.functional as F
import numpy as np

class Resize:
    def __init__(self, size):
        """Resize image and target to the given size.
        Args:
            size: Can be tuple (h,w) or list [h,w] or single int
        """
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)  # Convert list to tuple if needed
        else:
            self.size = (size, size)

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        # Add channel dimension for resize, then remove it
        target = target.unsqueeze(0)  # Add channel dimension: [H, W] -> [1, H, W]
        target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        target = target.squeeze(0) # Remove channel dimension: [1, H, W] -> [H, W]
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        """Randomly flip image and target horizontally with probability p."""
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        """Convert image and target to PyTorch tensors.
           Also checks target mask values for validity.
        """
        image = F.to_tensor(image)
        target_np = np.array(target)
        target = torch.as_tensor(target_np, dtype=torch.int64)

        # Check for invalid mask values (should be 0-20 or 255 for PASCAL VOC)
        # Adjust the upper valid limit (20) if using a different dataset/num_classes
        valid_values = (target_np >= 0) & (target_np <= 20) | (target_np == 255)
        if not np.all(valid_values):
            unique_values = np.unique(target_np)
            invalid_values = unique_values[~((unique_values >= 0) & (unique_values <= 20) | (unique_values == 255))]
            print(f"ERROR: Invalid values found in target mask: {invalid_values}")
            print(f"       All unique values in this mask: {unique_values}")
            # Optionally, you might want to raise an error here to stop execution
            # raise ValueError(f"Invalid values found in target mask: {invalid_values}")
            # Or replace invalid values with ignore_index (use with caution)
            # target[~((target >= 0) & (target <= 20) | (target == 255))] = 255

        return image, target

class Normalize:
    def __init__(self, mean, std):
        """Normalize image with given mean and std; target remains unchanged."""
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std)
        return image, target

class Compose:
    def __init__(self, transforms):
        """Chain multiple transforms together."""
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def get_transform(dataset_config, is_train=True):
    """Create a composition of transforms based on dataset config.
    
    Args:
        dataset_config (dict): Dataset configuration containing transform parameters
        is_train (bool): Whether to use training or validation transforms
    
    Returns:
        Compose: A composition of transforms
    """
    transforms = []
    
    # Add resize transform
    if 'input_size' in dataset_config:
        transforms.append(Resize(dataset_config['input_size']))
    
    # Add data augmentation transforms for training
    if is_train:
        if dataset_config.get('use_horizontal_flip', True):
            transforms.append(RandomHorizontalFlip(p=0.5))
    
    # Add basic transforms
    transforms.extend([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    return Compose(transforms)