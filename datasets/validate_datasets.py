import os
import sys
import time
from pathlib import Path

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

def test_dut_omron():
    """Test the DUT-OMRON dataset file structure."""
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
        print("DUT-OMRON dataset structure not found or incomplete")
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
    
    if train_img_count == 0 and val_img_count == 0:
        print("Error: DUT-OMRON dataset is empty")
        return False
    
    # Check for matching filenames (except extensions)
    if train_img_count > 0 and train_mask_count > 0:
        img_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_img_dir)}
        mask_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_mask_dir)}
        unmatched = img_basenames.symmetric_difference(mask_basenames)
        if unmatched:
            print(f"Warning: Found {len(unmatched)} unmatched files in train split (showing first 5):")
            for f in list(unmatched)[:5]:
                print(f"  - {f}")

    if val_img_count > 0 and val_mask_count > 0:
        img_basenames = {os.path.splitext(f)[0] for f in os.listdir(val_img_dir)}
        mask_basenames = {os.path.splitext(f)[0] for f in os.listdir(val_mask_dir)}
        unmatched = img_basenames.symmetric_difference(mask_basenames)
        if unmatched:
            print(f"Warning: Found {len(unmatched)} unmatched files in val split (showing first 5):")
            for f in list(unmatched)[:5]:
                print(f"  - {f}")
            
    return True

def test_pascal_voc():
    """Test the Pascal VOC dataset file structure."""
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
        print("Pascal VOC dataset structure not found or incomplete")
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
    
    if train_img_count == 0 and val_img_count == 0:
        print("Error: Pascal VOC dataset is empty")
        return False
    
    # Check for matching filenames (except extensions)
    if train_img_count > 0 and train_sem_count > 0 and train_inst_count > 0:
        img_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_img_dir)}
        sem_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_sem_dir)}
        inst_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_inst_dir)}
        
        unmatched_sem = img_basenames.symmetric_difference(sem_basenames)
        unmatched_inst = img_basenames.symmetric_difference(inst_basenames)
        
        if unmatched_sem:
            print(f"Warning: Found {len(unmatched_sem)} unmatched files between images and semantic masks in train split")
        if unmatched_inst:
            print(f"Warning: Found {len(unmatched_inst)} unmatched files between images and instance masks in train split")
    
    if val_img_count > 0 and val_sem_count > 0 and val_inst_count > 0:
        img_basenames = {os.path.splitext(f)[0] for f in os.listdir(val_img_dir)}
        sem_basenames = {os.path.splitext(f)[0] for f in os.listdir(val_sem_dir)}
        inst_basenames = {os.path.splitext(f)[0] for f in os.listdir(val_inst_dir)}
        
        unmatched_sem = img_basenames.symmetric_difference(sem_basenames)
        unmatched_inst = img_basenames.symmetric_difference(inst_basenames)
        
        if unmatched_sem:
            print(f"Warning: Found {len(unmatched_sem)} unmatched files between images and semantic masks in val split")
        if unmatched_inst:
            print(f"Warning: Found {len(unmatched_inst)} unmatched files between images and instance masks in val split")
    
    return True

def test_cityscapes():
    """Test the Cityscapes dataset file structure."""
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
    
    if train_img_count == 0 and val_img_count == 0:
        print("Error: Cityscapes dataset is empty")
        return False
    
    # Check for matching filenames (except extensions)
    if train_img_count > 0 and train_label_count > 0:
        img_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_img_dir)}
        label_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_label_dir)}
        unmatched = img_basenames.symmetric_difference(label_basenames)
        if unmatched:
            print(f"Warning: Found {len(unmatched)} unmatched files in train split")

    if val_img_count > 0 and val_label_count > 0:
        img_basenames = {os.path.splitext(f)[0] for f in os.listdir(val_img_dir)}
        label_basenames = {os.path.splitext(f)[0] for f in os.listdir(val_label_dir)}
        unmatched = img_basenames.symmetric_difference(label_basenames)
        if unmatched:
            print(f"Warning: Found {len(unmatched)} unmatched files in val split")
    
    return True

def main():
    print("Starting dataset file structure verification...")
    start_time = time.time()
    
    # Test all datasets
    test_dut_omron()
    test_pascal_voc()
    test_cityscapes()
    
    elapsed_time = time.time() - start_time
    print_separator("Verification Complete")
    print(f"All dataset verifications completed in {elapsed_time:.2f} seconds")
    print("\nNOTE: This script only verifies the file structure and counts.")
    print("To test the dataset classes, please use a separate testing script")
    print("that directly imports the dataset classes.")

if __name__ == "__main__":
    main() 