import torch
from torch.utils.data import Dataset
from abc import abstractmethod

class BaseDataset(Dataset):
    def __init__(self, transforms=None):
        """Initialize the base dataset with optional transforms."""
        self.transforms = transforms
        self.num_classes = None  # To be set by subclass
        self.ignore_label = None  # To be set by subclass

    def __getitem__(self, index):
        """Load and process the image and target at the given index."""
        image = self._load_image(index)
        target = self._load_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return {'image': image, 'target': target}

    @abstractmethod
    def _load_image(self, index):
        """Load the image at the given index."""
        pass

    @abstractmethod
    def _load_target(self, index):
        """Load the target (label) at the given index."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the total number of samples."""
        pass