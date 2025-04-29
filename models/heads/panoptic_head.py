"""
Head for panoptic segmentation combining semantic and instance branches.
"""
import torch
import torch.nn as nn
from .base_head import BaseHead
from .semantic_head import SemanticHead
from .instance_head import InstanceHead

class PanopticHead(BaseHead):
    """Head for panoptic segmentation combining semantic and instance segmentation."""
    def __init__(self, config: dict):
        """
        Initialize the panoptic head.

        Args:
            config (dict): Head configuration containing:
                - in_channels: Number of input channels from decoder
                - num_semantic_classes: Number of semantic classes
                - num_instance_classes: Number of instance classes
                - dropout: Dropout probability for both branches (optional)
                - semantic_weight: Weight for semantic loss (optional)
                - instance_weight: Weight for instance loss (optional)
                - ignore_index: Ignore index for instance loss (optional, default 255)
        """
        super(PanopticHead, self).__init__(config)
        in_ch = config['in_channels']
        # Setup semantic branch
        sem_cfg = {
            'in_channels': in_ch,
            'num_classes': config['num_semantic_classes'],
            'dropout': config.get('dropout', 0.1)
        }
        self.semantic_branch = SemanticHead(sem_cfg)
        # Setup instance branch
        inst_cfg = {
            'in_channels': in_ch,
            'num_classes': config['num_instance_classes'],
            'dropout': config.get('dropout', 0.1),
            'ignore_index': config.get('ignore_index', 255)
        }
        self.instance_branch = InstanceHead(inst_cfg)
        # Loss weights
        self.semantic_weight = config.get('semantic_weight', 1.0)
        self.instance_weight = config.get('instance_weight', 1.0)

    def forward(self, features: dict) -> dict:
        """
        Forward pass for panoptic segmentation.

        Args:
            features (dict): Contains 'backbone_features'

        Returns:
            dict: Contains:
                - 'semantic_logits', 'semantic_probs'
                - 'instance_logits', 'instance_probs'
        """
        semantic_out = self.semantic_branch(features)
        instance_out = self.instance_branch(features)
        return {
            'semantic_logits': semantic_out['logits'],
            'semantic_probs': semantic_out['probs'],
            'instance_logits': instance_out['logits'],
            'instance_probs': instance_out['probs']
        }

    def get_loss(self, outputs: dict, targets: dict) -> dict:
        """
        Compute combined panoptic loss.

        Args:
            outputs (dict): Contains semantic and instance logits/probs
            targets (dict): Contains:
                - 'semantic_masks': Ground truth semantic masks
                - 'instance_masks': Ground truth instance masks

        Returns:
            dict: Contains:
                - 'loss': Combined loss
                - 'semantic_loss': Semantic segmentation loss
                - 'instance_loss': Instance segmentation loss
        """
        # Semantic loss
        sem_targets = {'masks': targets['semantic_masks']}
        sem_outputs = {'logits': outputs['semantic_logits'], 'probs': outputs['semantic_probs']}
        sem_loss_dict = self.semantic_branch.get_loss(sem_outputs, sem_targets)
        sem_loss = sem_loss_dict['loss'] * self.semantic_weight
        # Instance loss
        inst_targets = {'masks': targets['instance_masks']}
        inst_outputs = {'logits': outputs['instance_logits'], 'probs': outputs['instance_probs']}
        inst_loss_dict = self.instance_branch.get_loss(inst_outputs, inst_targets)
        inst_loss = inst_loss_dict['loss'] * self.instance_weight
        # Total loss
        total_loss = sem_loss + inst_loss
        return {
            'loss': total_loss,
            'semantic_loss': sem_loss,
            'instance_loss': inst_loss
        }

    def get_predictions(self, outputs: dict) -> dict:
        """
        Get panoptic predictions for both tasks.

        Args:
            outputs (dict): Contains semantic and instance logits/probs

        Returns:
            dict: Contains:
                - 'semantic_masks', 'semantic_probs'
                - 'instance_masks', 'instance_probs'
        """
        sem_preds = self.semantic_branch.get_predictions({
            'logits': outputs['semantic_logits'],
            'probs': outputs['semantic_probs']
        })
        inst_preds = self.instance_branch.get_predictions({
            'logits': outputs['instance_logits'],
            'probs': outputs['instance_probs']
        })
        return {
            'semantic_masks': sem_preds['masks'],
            'semantic_probs': sem_preds['probs'],
            'instance_masks': inst_preds['masks'],
            'instance_probs': inst_preds['probs']
        } 