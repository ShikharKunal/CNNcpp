"""
Config for training and evaluation. Uses only Python 3.12 standard library.
"""
import json
import argparse


def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def parse_train_args():
    """Parse command-line args for train.py: dataset path, config path, etc."""
    p = argparse.ArgumentParser(description="Train CNN with mydl")
    p.add_argument("--dataset", type=str, required=True, help="Path to training dataset (root with class_1/, class_2/, ...)")
    p.add_argument("--config", type=str, default=None, help="Path to config JSON (optional)")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--weights", type=str, default="weights.bin", help="Path to save weights")
    p.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
    return p.parse_args()


def parse_eval_args():
    """Parse command-line args for evaluate.py."""
    p = argparse.ArgumentParser(description="Evaluate CNN with mydl")
    p.add_argument("--dataset", type=str, required=True, help="Path to test dataset")
    p.add_argument("--weights", type=str, required=True, help="Path to saved weights")
    p.add_argument("--config", type=str, default=None, help="Path to config JSON (optional)")
    return p.parse_args()
