import os
from PIL import Image
import torch
from .base_dataset import BaseDataset

class DUTOmronDataset(BaseDataset):
    def __init__(self, root, split='train', transforms=None):
        """Initialize the DUT-OMRON dataset loader.
        
        Args:
            root (str): Root directory of the dataset (e.g., 'dut-omron').
            split (str): Dataset split ('train' or 'val').
            transforms (callable, optional): Transformations to apply to image and target.
        """
        super().__init__(transforms)
        self.root = root
        self.split = split

        # Define paths to image and mask directories
        img_dir = os.path.join(root, split, 'img')
        mask_dir = os.path.join(root, split, 'mask')
        
        # List and sort image files
        self.image_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        # Construct corresponding mask paths (same base filenames)
        self.mask_paths = [
            os.path.join(mask_dir, os.path.basename(p).replace('.jpg', '.png').replace('.jpeg', '.png')) 
            for p in self.image_paths
        ]

        # Set dataset-specific attributes
        self.num_classes = 2  # Background (0) and object (1)
        self.ignore_label = None  # Binary masks typically don't need ignore label

    def _load_image(self, index):
        """Load the image at the given index as an RGB PIL Image."""
        return Image.open(self.image_paths[index]).convert('RGB')

    def _load_target(self, index):
        """Load and process the mask at the given index as a binary tensor."""
        mask = Image.open(self.mask_paths[index]).convert('L')
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        # Convert 255 to 1 for binary classification
        mask = (mask > 0).to(torch.int64)
        return mask

    def __len__(self):
        """Return the number of image-mask pairs."""
        return len(self.image_paths)