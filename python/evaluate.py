#!/usr/bin/env python3
"""
Evaluate CNN using mydl framework. Loads saved weights and reports accuracy.
Usage: python evaluate.py --dataset <test_data_path> --weights <weights.bin>
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mydl import (
    SimpleCNN,
    load_dataset_from_directory,
    DataLoader,
    accuracy,
)

from config import parse_eval_args


def main():
    args = parse_eval_args()
    dataset_path = os.path.abspath(args.dataset)
    weights_path = os.path.abspath(args.weights)
    if not os.path.isdir(dataset_path):
        print(f"Error: dataset path is not a directory: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(weights_path):
        print(f"Error: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset...")
    result = load_dataset_from_directory(dataset_path)
    load_time = result.load_time_seconds
    print(f"Dataset load time: {load_time:.4f} s")
    print(f"Samples: {len(result.labels)}, Classes: {result.num_classes}")

    model = SimpleCNN(result.num_classes)
    model.load_weights(weights_path)
    print(f"Loaded weights from {weights_path}")

    params = model.count_parameters()
    macs = model.count_macs()
    flops = model.count_flops()
    print(f"Trainable parameters: {params}")
    print(f"MACs per forward pass: {macs}")
    print(f"FLOPs per forward pass: {flops}")

    loader = DataLoader(result.images, result.labels, result.num_classes, batch_size=32)
    
    # ✅ Cache functions to avoid attribute lookups
    loader_has_next = loader.has_next
    loader_next = loader.next
    model_forward = model.forward
    
    total_correct = 0
    total_n = 0
    while loader_has_next():
        batch_x, batch_labels = loader_next()
        logits = model_forward(batch_x)
        total_correct += int(accuracy(logits, batch_labels) * len(batch_labels))
        total_n += len(batch_labels)
    acc = total_correct / total_n if total_n else 0.0
    print(f"Accuracy: {acc:.4f} ({total_correct}/{total_n})")


if __name__ == "__main__":
    main()
