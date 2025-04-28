from .cityscapes import CityscapesDataset
from .dut_omron import DUTOmronDataset
from .pascal_voc import PascalVOCDataset

def get_dataset(dataset_name, root, split='train', task='semantic', transforms=None):
    """Return the dataset instance based on the dataset name.
    
    Args:
        dataset_name (str): Name of the dataset ('cityscapes', 'dut_omron', 'pascal_voc').
        root (str): Root directory of the dataset.
        split (str): Dataset split ('train' or 'val').
        task (str): Task type ('semantic' or 'instance').
        transforms (callable, optional): Transformations to apply to image and target.
    
    Returns:
        Dataset: An instance of the specified dataset.
    """
    datasets = {
        'cityscapes': CityscapesDataset,
        'dut_omron': DUTOmronDataset,
        'pascal_voc': PascalVOCDataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {list(datasets.keys())}")
    
    return datasets[dataset_name](root=root, split=split, task=task, transforms=transforms)