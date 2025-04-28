from torch.utils.data import Dataset
from typing import Dict, Any, Tuple
import torch
import numpy as np
from PIL import Image

class BaseDataset(Dataset):
    """Base class for all datasets."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the dataset
            split (str): Which split to use ('train', 'val', 'test')
            transform: Optional transform to be applied to samples
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []  # List to store image-target pairs
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            dict: A dictionary containing:
                - 'image': The input image as a tensor
                - 'target': The target tensor
                - 'metadata': Additional information (optional)
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from file."""
        return Image.open(image_path).convert('RGB')
    
    def _load_target(self, target_path: str) -> np.ndarray:
        """Load a target from file."""
        raise NotImplementedError("Subclasses must implement _load_target")
    
    def get_classes(self) -> Tuple[list, dict]:
        """
        Get the list of classes and their mapping.
        
        Returns:
            tuple: (list of class names, dict mapping class names to indices)
        """
        raise NotImplementedError("Subclasses must implement get_classes")
    
    def get_colormap(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Get the colormap for visualization.
        
        Returns:
            dict: Mapping from class indices to RGB colors
        """
        raise NotImplementedError("Subclasses must implement get_colormap") 