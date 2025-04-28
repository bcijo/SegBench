import os
from PIL import Image
import torch
import numpy as np
from .base_dataset import BaseDataset

class PascalVOCDataset(BaseDataset):
    def __init__(self, root, split='train', task='semantic', transforms=None):
        """Initialize the Pascal VOC dataset loader.
        
        Args:
            root (str): Root directory of the dataset (e.g., 'pascal-voc').
            split (str): Dataset split ('train' or 'val').
            task (str): Task type ('semantic' or 'instance').
            transforms (callable, optional): Transformations to apply to image and target.
        """
        super().__init__(transforms)
        self.root = root
        self.split = split
        self.task = task

        # Define paths to image and mask directories
        img_dir = os.path.join(root, split, 'img')
        mask_dir = os.path.join(root, split, 'mask_semantic' if task == 'semantic' else 'mask_instance')
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Directory not found: {img_dir} or {mask_dir}")

        # List and sort image files
        self.image_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        # Construct corresponding mask paths
        self.mask_paths = [
            os.path.join(mask_dir, os.path.basename(p).replace('.jpg', '.png').replace('.jpeg', '.png'))
            for p in self.image_paths
        ]

        # Verify all files exist
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                raise FileNotFoundError(f"Missing file: {img_path} or {mask_path}")

        # Dataset-specific attributes
        self.num_classes = 21 if task == 'semantic' else None  # 20 classes + background for semantic
        self.ignore_label = 255  # Void label for both tasks

    def _load_image(self, index):
        """Load the image at the given index as an RGB PIL Image."""
        return Image.open(self.image_paths[index]).convert('RGB')

    def _load_target(self, index):
        """Load and process the mask at the given index."""
        mask = Image.open(self.mask_paths[index]).convert('L')
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return mask

    def __len__(self):
        """Return the number of image-mask pairs."""
        return len(self.image_paths)