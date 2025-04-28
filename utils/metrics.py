from typing import Dict, List
import torch
import numpy as np

class SegmentationMetrics:
    """Class to compute segmentation metrics."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes (int): Number of classes
            ignore_index (int): Index to ignore in metrics computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update confusion matrix.
        
        Args:
            pred (torch.Tensor): Model predictions (B, H, W)
            target (torch.Tensor): Ground truth labels (B, H, W)
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        if pred.shape != target.shape:
            raise ValueError('Shape mismatch: prediction and target should have the same shape.')
        
        # Compute confusion matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i][j] += np.sum((pred == i) & (target == j))
    
    def get_scores(self) -> Dict[str, float]:
        """
        Compute various segmentation metrics.
        
        Returns:
            dict: Dictionary containing:
                - miou: Mean IoU
                - pixel_acc: Pixel accuracy
                - class_acc: Mean class accuracy
        """
        # Compute IoU for each class
        ious = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            # True positives
            tp = self.confusion_matrix[i, i]
            # False positives + False negatives
            fp_fn = (self.confusion_matrix[i, :].sum() + 
                    self.confusion_matrix[:, i].sum() - tp)
            
            if tp + fp_fn == 0:
                ious[i] = 0
            else:
                ious[i] = tp / (tp + fp_fn)
        
        # Mean IoU
        valid_classes = ~np.isnan(ious)
        miou = np.mean(ious[valid_classes])
        
        # Pixel accuracy
        pixel_acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        
        # Class accuracy
        class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        class_acc = np.mean(class_acc[~np.isnan(class_acc)])
        
        return {
            'miou': float(miou),
            'pixel_acc': float(pixel_acc),
            'class_acc': float(class_acc)
        }
    
    def get_class_ious(self) -> List[float]:
        """
        Get IoU for each class.
        
        Returns:
            list: List of IoU values for each class
        """
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp_fn = (self.confusion_matrix[i, :].sum() + 
                    self.confusion_matrix[:, i].sum() - tp)
            
            if tp + fp_fn == 0:
                ious.append(0.0)
            else:
                ious.append(float(tp / (tp + fp_fn)))
        
        return ious 