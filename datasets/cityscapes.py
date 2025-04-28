import os
import numpy as np
from PIL import Image
import torch
from .base_dataset import BaseDataset

class CityscapesDataset(BaseDataset):
    def __init__(self, root, split='train', transforms=None):
        """Initialize the Cityscapes dataset loader.
        
        Args:
            root (str): Root directory of the dataset (e.g., 'datasets/cityscapes').
            split (str): Dataset split ('train' or 'val').
            transforms (callable, optional): Transformations to apply to image and target.
        """
        super().__init__(transforms)
        self.root = root
        self.split = split

        # Define paths to image and label directories
        img_dir = os.path.join(root, split, 'img')
        label_dir = os.path.join(root, split, 'label')
        
        # List and sort image files
        self.image_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.endswith('.png')
        ])
        # Construct corresponding label paths (same filenames)
        self.target_paths = [
            os.path.join(label_dir, os.path.basename(p)) 
            for p in self.image_paths
        ]

        # Define label mapping (example: simplified mapping for 19 classes)
        # In practice, use Cityscapes official mapping
        self.label_mapping = np.full(256, 255, dtype=np.uint8)  # Default to ignore label
        class_ids = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        for i, cid in enumerate(class_ids):
            self.label_mapping[cid] = i  # Map to 0-18

        # Set dataset-specific attributes
        self.num_classes = 19
        self.ignore_label = 255

    def _load_image(self, index):
        """Load the image at the given index as an RGB PIL Image."""
        return Image.open(self.image_paths[index]).convert('RGB')

    def _load_target(self, index):
        """Load and process the label at the given index."""
        target = Image.open(self.target_paths[index]).convert('L')
        target = np.array(target, dtype=np.uint8)
        target = self.label_mapping[target]
        return Image.fromarray(target, mode='L')

    def __len__(self):
        """Return the number of image-label pairs."""
        return len(self.image_paths)