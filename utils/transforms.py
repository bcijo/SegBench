import random
import torch
import torchvision.transforms.functional as F
import numpy as np

class Resize:
    def __init__(self, size):
        """Resize image and target to the given size."""
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
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
        """Convert image and target to PyTorch tensors."""
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
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