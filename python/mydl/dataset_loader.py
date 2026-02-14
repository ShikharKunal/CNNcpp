"""
Python-based dataset loader using opencv-python (pip package).
Replaces C++ OpenCV dependency with Python opencv-python from pip.
"""
import os
import cv2
import numpy as np
import time
import random
from pathlib import Path


def load_dataset_from_directory_python(root_path, target_size=(32, 32), seed=12345):
    """
    Load dataset from directory structure: root/class_0/, root/class_1/, ...
    
    Args:
        root_path: Path to dataset root directory
        target_size: Tuple of (height, width) to resize images to
        seed: Random seed for shuffling
    
    Returns:
        dict with keys: images (numpy array), labels (list), num_classes (int), load_time_seconds (float)
    """
    t0 = time.time()
    
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Dataset path does not exist or is not a directory: {root_path}")
    
    # Get class directories (sorted alphabetically)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    if not class_dirs:
        raise RuntimeError(f"No class directories found in {root_path}")
    
    class_names = [d.name for d in class_dirs]
    num_classes = len(class_names)
    
    # Collect all image files with their labels
    image_extensions = {'.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG'}
    files_with_labels = []
    
    for class_idx, class_dir in enumerate(class_dirs):
        for img_file in class_dir.iterdir():
            if img_file.suffix in image_extensions:
                files_with_labels.append((str(img_file), class_idx))
    
    if not files_with_labels:
        raise RuntimeError(f"No images found in {root_path}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(files_with_labels)
    
    # Load and process images
    N = len(files_with_labels)
    C, H, W = 3, target_size[0], target_size[1]
    
    # Pre-allocate numpy array (N, C, H, W) format for C++ compatibility
    images = np.zeros((N, C, H, W), dtype=np.float32)
    labels = []
    
    for i, (img_path, label) in enumerate(files_with_labels):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}, skipping")
            continue
        
        # Resize to target size
        img = cv2.resize(img, (W, H))
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # BGR -> RGB and normalize to [0, 1]
        # OpenCV loads as BGR, convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose to (C, H, W)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        
        images[i] = img_chw
        labels.append(label)
    
    load_time = time.time() - t0
    
    return {
        'images': images,
        'labels': labels,
        'num_classes': num_classes,
        'load_time_seconds': load_time,
        'class_names': class_names
    }
