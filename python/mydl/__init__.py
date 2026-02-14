"""
Python wrapper for dataset loading that replaces C++ OpenCV dependency.
Now uses opencv-python (pip package) for image loading.
"""
from . import mydl_cpp
from .dataset_loader import load_dataset_from_directory_python
import numpy as np


def load_dataset_from_directory(root_path, target_size=(32, 32), seed=12345):
    """
    Load dataset using Python opencv-python (pip package).
    Returns DatasetResult object compatible with C++ backend.
    
    Args:
        root_path: Path to dataset root directory (class subdirectories)
        target_size: Tuple of (height, width) to resize images to
        seed: Random seed for shuffling
    
    Returns:
        DatasetResult object with: images, labels, num_classes, load_time_seconds
    """
    # Load images in Python using opencv-python
    result = load_dataset_from_directory_python(root_path, target_size, seed)
    
    # Convert numpy array to C++ format
    images_np = result['images']  # Shape: (N, C, H, W)
    labels = result['labels']
    num_classes = result['num_classes']
    load_time = result['load_time_seconds']
    
    # Flatten numpy array for C++
    data_flat = images_np.flatten().tolist()
    shape = list(images_np.shape)
    
    # Create C++ DatasetResult using the new function
    dataset_result = mydl_cpp.create_dataset_from_numpy(
        data_flat, shape, labels, num_classes, load_time
    )
    
    return dataset_result


# Re-export everything from mydl_cpp
from .mydl_cpp import *

# Override the load function to use Python implementation
__all__ = ['load_dataset_from_directory'] + [name for name in dir(mydl_cpp) if not name.startswith('_')]
