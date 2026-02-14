#!/usr/bin/env python3
"""
Cross-platform training script with metrics collection and plot generation.
Works on Linux, macOS, and Windows.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Train CNN with mydl")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--weights", type=str, default="weights/model.bin", help="Path to save weights")
    args = parser.parse_args()
    
    # Add python directory to path
    script_dir = Path(__file__).parent.resolve()
    python_dir = script_dir / "python"
    sys.path.insert(0, str(python_dir))
    
    # Import mydl
    try:
        from mydl import (
            SimpleCNN, load_dataset_from_directory, DataLoader,
            cross_entropy_loss, SGD, accuracy, loss_value
        )
    except ImportError as e:
        print(f"Error importing mydl: {e}")
        print("Make sure to run build.py first!")
        sys.exit(1)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = script_dir / "plots" / timestamp
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create weights directory
    weights_path = script_dir / args.weights
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Dataset:    {args.dataset}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learn Rate: {args.lr}")
    print(f"Weights:    {args.weights}")
    print(f"Plots:      {plots_dir}")
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
    
    # Create model
    model = SimpleCNN(num_classes)
    params = model.count_parameters()
    macs = model.count_macs()
    flops = model.count_flops()
    
    print(f"Trainable parameters: {params}")
    print(f"MACs per forward pass: {macs}")
    print(f"FLOPs per forward pass: {flops}")
    
    # Setup training
    optimizer = SGD(model.parameters(), args.lr)
    loader = DataLoader(images, labels, num_classes, args.batch_size)
    
    # Cache method references
    loader_has_next = loader.has_next
    loader_next = loader.next
    optimizer_zero_grad = optimizer.zero_grad
    optimizer_step = optimizer.step
    model_forward = model.forward
    
    # Collect metrics
    metrics = {
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "num_samples": len(labels),
        "num_classes": num_classes,
        "dataset_load_time": load_time,
        "parameters": params,
        "macs": macs,
        "flops": flops,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "epoch_metrics": []
    }
    
    training_start = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        loader.reset()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        n_batches = 0
        
        epoch_start = time.time()
        print(f"Starting epoch {epoch + 1}/{args.epochs}...", flush=True)
        
        while loader_has_next():
            batch_x, batch_labels = loader_next()
            optimizer_zero_grad()
            logits = model_forward(batch_x)
            loss = cross_entropy_loss(logits, batch_labels)
            loss.backward()
            optimizer_step()
            epoch_loss += loss_value(loss)
            epoch_correct += int(accuracy(logits, batch_labels) * len(batch_labels))
            epoch_n += len(batch_labels)
            n_batches += 1
            
            if n_batches % 200 == 0:
                print(f"  Batch {n_batches}, loss: {epoch_loss/n_batches:.4f}", flush=True)
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches if n_batches else 0.0
        acc = epoch_correct / epoch_n if epoch_n else 0.0
        
        print(f"Epoch {epoch + 1}/{args.epochs}  Loss: {avg_loss:.4f}  Accuracy: {acc:.4f}  Time: {epoch_time:.2f}s", flush=True)
        
        metrics["epoch_metrics"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": acc,
            "time_seconds": epoch_time,
            "batches": n_batches
        })
    
    training_time = time.time() - training_start
    metrics["total_training_time"] = training_time
    
    # Save model
    model.save_weights(str(weights_path))
    print(f"Weights saved to {weights_path}", flush=True)
    
    # Save metrics
    metrics_path = plots_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}", flush=True)
    
    # Generate plots
    print("\nGenerating plots...", flush=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs_list = [m['epoch'] for m in metrics['epoch_metrics']]
        losses = [m['loss'] for m in metrics['epoch_metrics']]
        accuracies = [m['accuracy'] for m in metrics['epoch_metrics']]
        times = [m['time_seconds'] for m in metrics['epoch_metrics']]
        
        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, losses, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Cross-Entropy Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, [a * 100 for a in accuracies], 'g-o', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "accuracy_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Time per epoch
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, times, 'r-o', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "time_per_epoch.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(epochs_list, losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_list, [a * 100 for a in accuracies], 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to {plots_dir}/", flush=True)
        
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plots.", flush=True)
        print("Install with: pip install matplotlib", flush=True)
    
    print("\n" + "=" * 50)
    print("✓ Training complete!")
    print(f"  Weights: {weights_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Plots:   {plots_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
