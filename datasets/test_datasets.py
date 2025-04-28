import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

# Add the parent directory to the path to import from our project modules
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import dataset classes using absolute imports to avoid name conflicts
try:
    # Import our custom dataset classes directly with full paths to avoid namespace conflicts
    # This avoids collision with the system-installed HF datasets package
    sys.path.insert(0, project_root)  # Prioritize our project root
    from_import_successful = False
    
    # Try alternative import approaches
    try:
        # Option 1: Direct absolute imports
        from datasets.dut_omron import DUTOmronDataset
        from datasets.pascal_voc import PascalVOCDataset
        from datasets.cityscapes import CityscapesDataset
        from_import_successful = True
    except ImportError as e:
        print(f"Direct import failed: {e}")
        
    if not from_import_successful:
        try:
            # Option 2: Import using explicit file path
            import importlib.util
            
            def load_from_file(module_name, file_path):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            
            # Get full file paths
            dut_omron_path = os.path.join(Path(__file__).parent, "dut_omron.py")
            pascal_voc_path = os.path.join(Path(__file__).parent, "pascal_voc.py")
            cityscapes_path = os.path.join(Path(__file__).parent, "cityscapes.py")
            
            # Load modules
            dut_omron_module = load_from_file("dut_omron", dut_omron_path)
            pascal_voc_module = load_from_file("pascal_voc", pascal_voc_path)
            cityscapes_module = load_from_file("cityscapes", cityscapes_path)
            
            # Get classes
            DUTOmronDataset = dut_omron_module.DUTOmronDataset
            PascalVOCDataset = pascal_voc_module.PascalVOCDataset
            CityscapesDataset = cityscapes_module.CityscapesDataset
            from_import_successful = True
        except Exception as e:
            print(f"File-based import failed: {e}")
    
    datasets_imported = from_import_successful
except ImportError as e:
    print(f"Warning: Could not import dataset classes: {e}")
    datasets_imported = False

def print_separator(title):
    """Print a separator with a title for better readability."""
    width = 60
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width)

def check_directory_structure(dataset_name, expected_dirs):
    """Check if the expected directory structure exists."""
    missing_dirs = []
    for dir_path in expected_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"Warning: The following directories are missing for {dataset_name}:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    
    return True

def test_dataset_class(dataset_class, root, **kwargs):
    """Test loading a dataset using its class."""
    try:
        # Create dataset instance
        dataset = dataset_class(root=root, **kwargs)
        
        # Try loading a few samples
        num_samples = min(5, len(dataset))
        for i in range(num_samples):
            sample = dataset[i]
            image, target = sample['image'], sample['target']
            
            # Check image and target types and shapes
            assert isinstance(image, torch.Tensor), f"Image should be a tensor, got {type(image)}"
            assert isinstance(target, torch.Tensor), f"Target should be a tensor, got {type(target)}"
            
            # Print image and target info
            print(f"Sample {i}: Image shape: {image.shape}, Target shape: {target.shape}")
            print(f"  Image range: [{image.min():.2f}, {image.max():.2f}], Target unique values: {torch.unique(target).tolist()}")
        
        print(f"Successfully loaded and tested {num_samples} samples")
        return True
    
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dut_omron():
    """Test the DUT-OMRON dataset."""
    print_separator("Testing DUT-OMRON Dataset")
    
    # Define expected directory structure
    root = os.path.join("data", "dut-omron")
    expected_dirs = [
        os.path.join(root, "train", "img"),
        os.path.join(root, "train", "mask"),
        os.path.join(root, "val", "img"),
        os.path.join(root, "val", "mask")
    ]
    
    # Check directory structure
    if not check_directory_structure("DUT-OMRON", expected_dirs):
        return False
    
    # Check file counts
    train_img_dir = os.path.join(root, "train", "img")
    train_mask_dir = os.path.join(root, "train", "mask")
    val_img_dir = os.path.join(root, "val", "img")
    val_mask_dir = os.path.join(root, "val", "mask")
    
    train_img_count = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
    train_mask_count = len(os.listdir(train_mask_dir)) if os.path.exists(train_mask_dir) else 0
    val_img_count = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0
    val_mask_count = len(os.listdir(val_mask_dir)) if os.path.exists(val_mask_dir) else 0
    
    print(f"Train split: {train_img_count} images, {train_mask_count} masks")
    print(f"Val split: {val_img_count} images, {val_mask_count} masks")
    
    if train_img_count != train_mask_count:
        print(f"Warning: Train image count ({train_img_count}) doesn't match mask count ({train_mask_count})")
    
    if val_img_count != val_mask_count:
        print(f"Warning: Val image count ({val_img_count}) doesn't match mask count ({val_mask_count})")
    
    # Test dataset class if available
    if datasets_imported:
        print("\nTesting DUTOmronDataset class:")
        train_success = test_dataset_class(DUTOmronDataset, root, split='train')
        val_success = test_dataset_class(DUTOmronDataset, root, split='val')
        
        if train_success and val_success:
            print("DUTOmronDataset class tested successfully for both splits")
        else:
            print("Issues detected when testing DUTOmronDataset class")
    
    return True

def test_pascal_voc():
    """Test the Pascal VOC dataset."""
    print_separator("Testing Pascal VOC Dataset")
    
    # Define expected directory structure
    root = os.path.join("data", "pascal-voc")
    expected_dirs = [
        os.path.join(root, "train", "img"),
        os.path.join(root, "train", "mask_semantic"),
        os.path.join(root, "train", "mask_instance"),
        os.path.join(root, "val", "img"),
        os.path.join(root, "val", "mask_semantic"),
        os.path.join(root, "val", "mask_instance")
    ]
    
    # Check directory structure
    if not check_directory_structure("Pascal VOC", expected_dirs):
        return False
    
    # Check file counts
    train_img_dir = os.path.join(root, "train", "img")
    train_sem_dir = os.path.join(root, "train", "mask_semantic")
    train_inst_dir = os.path.join(root, "train", "mask_instance")
    val_img_dir = os.path.join(root, "val", "img")
    val_sem_dir = os.path.join(root, "val", "mask_semantic")
    val_inst_dir = os.path.join(root, "val", "mask_instance")
    
    train_img_count = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
    train_sem_count = len(os.listdir(train_sem_dir)) if os.path.exists(train_sem_dir) else 0
    train_inst_count = len(os.listdir(train_inst_dir)) if os.path.exists(train_inst_dir) else 0
    val_img_count = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0
    val_sem_count = len(os.listdir(val_sem_dir)) if os.path.exists(val_sem_dir) else 0
    val_inst_count = len(os.listdir(val_inst_dir)) if os.path.exists(val_inst_dir) else 0
    
    print(f"Train split: {train_img_count} images, {train_sem_count} semantic masks, {train_inst_count} instance masks")
    print(f"Val split: {val_img_count} images, {val_sem_count} semantic masks, {val_inst_count} instance masks")
    
    # Check for mismatches
    if train_img_count != train_sem_count or train_img_count != train_inst_count:
        print(f"Warning: Train image count ({train_img_count}) doesn't match semantic ({train_sem_count}) or instance ({train_inst_count}) mask count")
    
    if val_img_count != val_sem_count or val_img_count != val_inst_count:
        print(f"Warning: Val image count ({val_img_count}) doesn't match semantic ({val_sem_count}) or instance ({val_inst_count}) mask count")
    
    # Test dataset class if available
    if datasets_imported:
        print("\nTesting PascalVOCDataset class (semantic):")
        train_sem_success = test_dataset_class(PascalVOCDataset, root, split='train', task='semantic')
        val_sem_success = test_dataset_class(PascalVOCDataset, root, split='val', task='semantic')
        
        print("\nTesting PascalVOCDataset class (instance):")
        train_inst_success = test_dataset_class(PascalVOCDataset, root, split='train', task='instance')
        val_inst_success = test_dataset_class(PascalVOCDataset, root, split='val', task='instance')
        
        if all([train_sem_success, val_sem_success, train_inst_success, val_inst_success]):
            print("PascalVOCDataset class tested successfully for all splits and tasks")
        else:
            print("Issues detected when testing PascalVOCDataset class")
    
    return True

def test_cityscapes():
    """Test the Cityscapes dataset."""
    print_separator("Testing Cityscapes Dataset")
    
    # Define expected directory structure
    root = os.path.join("data", "cityscapes")
    expected_dirs = [
        os.path.join(root, "train", "img"),
        os.path.join(root, "train", "label"),
        os.path.join(root, "val", "img"),
        os.path.join(root, "val", "label")
    ]
    
    # Check directory structure
    if not check_directory_structure("Cityscapes", expected_dirs):
        print("Cityscapes dataset structure not found or incomplete")
        return False
    
    # Check file counts
    train_img_dir = os.path.join(root, "train", "img")
    train_label_dir = os.path.join(root, "train", "label")
    val_img_dir = os.path.join(root, "val", "img")
    val_label_dir = os.path.join(root, "val", "label")
    
    train_img_count = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
    train_label_count = len(os.listdir(train_label_dir)) if os.path.exists(train_label_dir) else 0
    val_img_count = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0
    val_label_count = len(os.listdir(val_label_dir)) if os.path.exists(val_label_dir) else 0
    
    print(f"Train split: {train_img_count} images, {train_label_count} labels")
    print(f"Val split: {val_img_count} images, {val_label_count} labels")
    
    if train_img_count != train_label_count:
        print(f"Warning: Train image count ({train_img_count}) doesn't match label count ({train_label_count})")
    
    if val_img_count != val_label_count:
        print(f"Warning: Val image count ({val_img_count}) doesn't match label count ({val_label_count})")
    
    # Test dataset class if available
    if datasets_imported:
        print("\nTesting CityscapesDataset class:")
        train_success = test_dataset_class(CityscapesDataset, root, split='train')
        val_success = test_dataset_class(CityscapesDataset, root, split='val')
        
        if train_success and val_success:
            print("CityscapesDataset class tested successfully for both splits")
        else:
            print("Issues detected when testing CityscapesDataset class")
    
    return True

def main():
    print("Starting dataset verification...")
    start_time = time.time()
    
    # Check base_dataset.py existence
    base_path = Path(__file__).parent / "base_dataset.py"
    if not base_path.exists():
        print(f"Warning: base_dataset.py not found at {base_path}")
        print("Some tests may fail due to missing dependencies")
    
    # Test all datasets
    test_dut_omron()
    test_pascal_voc()
    test_cityscapes()
    
    elapsed_time = time.time() - start_time
    print_separator("Verification Complete")
    print(f"All dataset verifications completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main() 