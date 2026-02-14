#!/usr/bin/env python3
"""
Split dataset into train and test sets while maintaining class structure.
Usage: python split_dataset.py --input <dataset_path> --output <output_dir> --split 0.8
"""
import os
import shutil
import argparse
from pathlib import Path
import random

def split_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Split dataset into train/test maintaining directory structure.
    
    Args:
        input_dir: Path to original dataset (e.g., "Assignment 1 Datasets/data_1")
        output_dir: Path to output directory (will create train/ and test/ subdirs)
        train_ratio: Fraction of data for training (default 0.8 = 80%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Splitting dataset from: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Train ratio: {train_ratio:.1%}, Test ratio: {1-train_ratio:.1%}")
    print()
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    class_dirs.sort()
    
    total_train = 0
    total_test = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")
        
        # Create class directories in train and test
        (train_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = {'.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG'}
        image_files = [f for f in class_dir.iterdir() 
                      if f.is_file() and f.suffix in image_extensions]
        
        # Shuffle and split
        random.shuffle(image_files)
        n_train = int(len(image_files) * train_ratio)
        
        train_files = image_files[:n_train]
        test_files = image_files[n_train:]
        
        # Copy files to train directory
        for img_file in train_files:
            shutil.copy2(img_file, train_dir / class_name / img_file.name)
        
        # Copy files to test directory
        for img_file in test_files:
            shutil.copy2(img_file, test_dir / class_name / img_file.name)
        
        total_train += len(train_files)
        total_test += len(test_files)
        
        print(f"  {class_name}: {len(train_files)} train, {len(test_files)} test")
    
    print()
    print("=" * 50)
    print(f"Split complete!")
    print(f"Total train images: {total_train}")
    print(f"Total test images: {total_test}")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/test")
    parser.add_argument("--input", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output", type=str, default="data_split", help="Output directory")
    parser.add_argument("--split", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_dataset(args.input, args.output, args.split, args.seed)

if __name__ == "__main__":
    main()
