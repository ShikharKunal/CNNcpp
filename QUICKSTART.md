# Quick Start Guide

## Cross-Platform Installation & Usage

This framework works on **Linux, macOS, and Windows**.

---

## 1. Install Prerequisites

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-dev python3-pip python3-venv cmake g++ libopencv-dev
```

### macOS
```bash
brew install python cmake opencv
```

### Windows
1. Install [Python 3.8+](https://www.python.org/downloads/)
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
3. Install [CMake](https://cmake.org/download/)
4. Install OpenCV (or use `pip install opencv-python`)

---

## 2. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it (choose your platform)
source .venv/bin/activate           # Linux/macOS
.venv\Scripts\activate.bat          # Windows (cmd)
.venv\Scripts\Activate.ps1          # Windows (PowerShell)

# Install dependencies
pip install pybind11 matplotlib
```

---

## 3. Build

```bash
python3 build.py
```

This compiles the C++ extension. You only need to run this once (or after changing C++ code).

---

## 4. Train

```bash
python3 train_model.py --dataset data_1_split/train --epochs 10
```

**What you get:**
- Trained model: `weights/model.bin`
- Metrics: `plots/YYYYMMDD_HHMMSS/metrics.json`
- 4 plots: `plots/YYYYMMDD_HHMMSS/*.png`

**All options:**
```bash
python3 train_model.py \
  --dataset <path> \
  --epochs 10 \
  --batch-size 32 \
  --lr 0.01 \
  --weights weights/model.bin
```

---

## 5. Test/Evaluate

```bash
python3 test_model.py --dataset data_1_split/test --weights weights/model.bin
```

Prints accuracy and other metrics.

---

## Complete Example Workflow

```bash
# 1. Setup (once)
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate.bat on Windows
pip install pybind11 matplotlib

# 2. Build (once)
python3 build.py

# 3. Train on dataset 1
python3 train_model.py --dataset data_1_split/train --epochs 10

# 4. Test on dataset 1
python3 test_model.py --dataset data_1_split/test --weights weights/model.bin

# 5. Train on dataset 2 (100 classes)
python3 train_model.py --dataset data_2_split/train --epochs 10 --weights weights/data2.bin

# 6. Test on dataset 2
python3 test_model.py --dataset data_2_split/test --weights weights/data2.bin
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'mydl_cpp'"**
→ Run `python3 build.py` first

**"CMake cannot find OpenCV"**
→ Install OpenCV for your platform (see installation section)

**"CMake cannot find pybind11"**
→ Make sure virtual environment is activated and pybind11 is installed

**Windows: "error: command failed"**
→ Make sure Visual Studio Build Tools are installed

---

## What Gets Created

```
mydl/
├── weights/
│   └── model.bin              # Trained model weights
├── plots/
│   └── 20240214_153045/       # Timestamped directory
│       ├── metrics.json       # All training metrics
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       ├── time_per_epoch.png
│       └── training_curves.png
```

Each training run creates a new timestamped directory, so results are never overwritten!

---

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for code + datasets
- **OS**: Linux, macOS, or Windows

---

## Next Steps

See `README.md` for detailed documentation.
See `OPTIMIZATION_JOURNEY.md` for the optimization story.
