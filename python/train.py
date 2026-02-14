#!/usr/bin/env python3 -u
"""
Train CNN using mydl framework only.
Usage: python train.py --dataset <path> [--config <config.json>] [--epochs 10] [--batch_size 32] [--lr 0.01] [--weights weights.bin]
"""
import os
import sys

# Run from mydl/python so that 'mydl' package is found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mydl import (
    SimpleCNN,
    load_dataset_from_directory,
    DataLoader,
    cross_entropy_loss,
    SGD,
    accuracy,
    loss_value,
)

from config import parse_train_args, load_config


def main():
    args = parse_train_args()
    if args.config:
        cfg = load_config(args.config)
        epochs = cfg.get("epochs", args.epochs)
        batch_size = cfg.get("batch_size", args.batch_size)
        lr = cfg.get("lr", args.lr)
        weights_path = cfg.get("weights", args.weights)
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        weights_path = args.weights

    dataset_path = os.path.abspath(args.dataset)
    if not os.path.isdir(dataset_path):
        print(f"Error: dataset path is not a directory: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset...", flush=True)
    result = load_dataset_from_directory(dataset_path)
    images = result.images
    labels = result.labels
    num_classes = result.num_classes
    load_time = result.load_time_seconds
    print(f"Dataset load time: {load_time:.4f} s")
    print(f"Samples: {len(labels)}, Classes: {num_classes}")

    model = SimpleCNN(num_classes)
    params = model.count_parameters()
    macs = model.count_macs()
    flops = model.count_flops()
    print(f"Trainable parameters: {params}")
    print(f"MACs per forward pass: {macs}")
    print(f"FLOPs per forward pass: {flops}")

    optimizer = SGD(model.parameters(), lr)
    loader = DataLoader(images, labels, num_classes, batch_size)
    
    # ✅ Cache functions to avoid repeated lookups in hot loop
    loader_has_next = loader.has_next
    loader_next = loader.next
    optimizer_zero_grad = optimizer.zero_grad
    optimizer_step = optimizer.step
    model_forward = model.forward

    for epoch in range(epochs):
        loader.reset()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0
        n_batches = 0
        print(f"Starting epoch {epoch + 1}/{epochs}...", flush=True)
        
        # ✅ Tight inner loop with cached method references
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
            # Reduced logging frequency for performance
            if n_batches % 200 == 0:
                print(f"  Batch {n_batches}/~1875, loss: {epoch_loss/n_batches:.4f}", flush=True)

        avg_loss = epoch_loss / n_batches if n_batches else 0.0
        acc = epoch_correct / epoch_n if epoch_n else 0.0
        print(f"Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}  Accuracy: {acc:.4f}", flush=True)

    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}", flush=True)


if __name__ == "__main__":
    main()
