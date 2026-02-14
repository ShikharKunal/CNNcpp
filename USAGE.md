# Quick Usage Guide

## Three Simple Scripts

### 1. Build
```bash
./build.sh
```
- Compiles C++ extension
- Run once after downloading or after changing C++ code

---

### 2. Train
```bash
./train.sh --dataset <train_dir> [options]
```

**Required:**
- `--dataset` - Path to training data

**Optional:**
- `--epochs` - Number of epochs (default: 10)
- `--batch-size` - Batch size (default: 32)
- `--lr` - Learning rate (default: 0.01)
- `--weights` - Where to save weights (default: weights/model.bin)

**Example:**
```bash
./train.sh --dataset data_1_split/train --epochs 10
```

**Outputs:**
- `weights/model.bin` - Trained model
- `plots/<timestamp>/metrics.json` - All metrics (loss, accuracy per epoch)
- `plots/<timestamp>/` - 4 high-res plots (300 DPI)
  - `loss_curve.png`
  - `accuracy_curve.png`
  - `time_per_epoch.png`
  - `training_curves.png`

---

### 3. Test
```bash
./test.sh --dataset <test_dir> --weights <weights_path>
```

**Example:**
```bash
./test.sh --dataset data_1_split/test --weights weights/model.bin
```

**Output:** Prints test accuracy and efficiency metrics

---

## Complete Workflow Example

```bash
# 1. Build (once)
./build.sh

# 2. Train on data_1
./train.sh --dataset data_1_split/train --epochs 10 --weights weights/data1.bin

# 3. Test on data_1
./test.sh --dataset data_1_split/test --weights weights/data1.bin

# 4. Train on data_2 (100 classes)
./train.sh --dataset data_2_split/train --epochs 10 --weights weights/data2.bin

# 5. Test on data_2
./test.sh --dataset data_2_split/test --weights weights/data2.bin
```

---

## Finding Your Results

After training with timestamp `20240214_153045`:

```
plots/20240214_153045/
├── metrics.json          # All training metrics
├── loss_curve.png        # Loss progression
├── accuracy_curve.png    # Accuracy improvement
├── time_per_epoch.png    # Training time consistency
└── training_curves.png   # Combined view
```

Each training run creates a new timestamped directory, so you never overwrite previous results!

---

## Tips

- **First time setup:** Run `./scripts/setup_venv.sh` to create virtual environment
- **Activate venv:** `source .venv/bin/activate` (scripts do this automatically)
- **Multiple runs:** Each `./train.sh` creates a new timestamped plot directory
- **Custom paths:** Use `--weights` to save different models with different names
- **Help:** Add `--help` to any script for usage information

---

## Troubleshooting

**"No such file or directory"**
→ Make sure you're in the `mydl/` directory

**"Import error: mydl_cpp"**
→ Run `./build.sh` first

**"Dataset not found"**
→ Check your dataset path is correct relative to `mydl/` directory
