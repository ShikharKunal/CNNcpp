#!/usr/bin/env python3
"""
Cross-platform evaluation/test script.
Works on Linux, macOS, and Windows.
"""
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN with mydl")
    parser.add_argument("--dataset", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights")
    args = parser.parse_args()
    
    # Add python directory to path
    script_dir = Path(__file__).parent.resolve()
    python_dir = script_dir / "python"
    sys.path.insert(0, str(python_dir))
    
    # Import mydl
    try:
        from mydl import (
            SimpleCNN, load_dataset_from_directory, DataLoader,
            cross_entropy_loss, accuracy, loss_value
        )
    except ImportError as e:
        print(f"Error importing mydl: {e}")
        print("Make sure to run build.py first!")
        sys.exit(1)
    
    print("=" * 50)
    print("Evaluation Configuration")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Weights: {args.weights}")
    print("=" * 50)
    print()
    
    # Load dataset
    print("Loading dataset...", flush=True)
    dataset_path = Path(args.dataset).resolve()
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    result = load_dataset_from_directory(str(dataset_path))
    images = result.images
    labels = result.labels
    num_classes = result.num_classes
    load_time = result.load_time_seconds
    
    print(f"Dataset load time: {load_time:.4f} s")
    print(f"Samples: {len(labels)}, Classes: {num_classes}")
    
    # Create model and load weights
    model = SimpleCNN(num_classes)
    params = model.count_parameters()
    macs = model.count_macs()
    flops = model.count_flops()
    
    print(f"Trainable parameters: {params}")
    print(f"MACs per forward pass: {macs}")
    print(f"FLOPs per forward pass: {flops}")
    
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        sys.exit(1)
    
    model.load_weights(str(weights_path))
    print(f"Loaded weights from {weights_path}")
    
    # Evaluate
    loader = DataLoader(images, labels, num_classes, batch_size=32)
    
    # Cache method references
    loader_has_next = loader.has_next
    loader_next = loader.next
    model_forward = model.forward
    
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    n_batches = 0
    
    print("\nEvaluating...", flush=True)
    while loader_has_next():
        batch_x, batch_labels = loader_next()
        logits = model_forward(batch_x)
        loss = cross_entropy_loss(logits, batch_labels)
        total_loss += loss_value(loss)
        total_correct += int(accuracy(logits, batch_labels) * len(batch_labels))
        total_n += len(batch_labels)
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches else 0.0
    acc = total_correct / total_n if total_n else 0.0
    
    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Loss:     {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Correct:  {total_correct}/{total_n}")
    print("=" * 50)


if __name__ == "__main__":
    main()

