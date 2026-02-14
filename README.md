# mydl – Custom Deep Learning Framework

A minimal deep learning framework built from scratch in C++ with Python bindings for **GNR638: Machine Learning for Remote Sensing – II**.

**Key Features:**
- Custom CNN implementation with Conv2D, ReLU, MaxPool2D, and Linear layers
- Automatic differentiation with dynamic computation graphs
- Cross-platform: Works on Linux, macOS, and Windows
- No external ML libraries (PyTorch, TensorFlow, NumPy, Eigen)
- Optimized for speed: ~50s/epoch training time

---

## Quick Start (All Platforms)

### Prerequisites

Install these before getting started:

| Platform | Requirements |
|----------|-------------|
| **Linux** | Python 3.8+, CMake 3.14+, g++ |
| **macOS** | Python 3.8+, CMake 3.14+ (via Homebrew) |
| **Windows** | Python 3.8+, CMake 3.14+, Visual Studio Build Tools |

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/ShikharKunal/CNNcpp.git
cd CNNcpp
```

**2. Install Python dependencies**
```bash
# Linux/macOS
python3 -m pip install -r requirements.txt

# Windows
python -m pip install -r requirements.txt
```

**3. Build the C++ extension**
```bash
# Linux/macOS
python3 build.py

# Windows
python build.py
```

That's it! The framework is ready to use.

---

## Usage

### 1. Build Command

Compiles the C++ backend. Run this once initially, or after modifying C++ code.

```bash
# Linux/macOS
python3 build.py

# Windows
python build.py
```

**Output:** Creates `python/mydl/mydl_cpp.*` extension module

---

### 2. Train Command

Train a CNN model on your dataset.

#### Linux/macOS
```bash
python3 train_model.py --dataset path/to/train_data --epochs 10 --lr 0.01
```

#### Windows (cmd)
```cmd
python train_model.py --dataset path\to\train_data --epochs 10 --lr 0.01
```

#### Windows (PowerShell)
```powershell
python train_model.py --dataset path\to\train_data --epochs 10 --lr 0.01
```

#### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | *required* | Path to training dataset directory |
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 32 | Batch size for training |
| `--lr` | 0.01 | Learning rate |
| `--weights` | weights/model.bin | Path to save trained weights |

#### Example with All Options

```bash
# Linux/macOS
python3 train_model.py \
  --dataset data_1_split/train \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.005 \
  --weights weights/my_model.bin

# Windows
python train_model.py --dataset data_1_split\train --epochs 20 --batch-size 64 --lr 0.005 --weights weights\my_model.bin
```

#### Training Outputs

After training completes, you'll find:

1. **Trained model weights:** `weights/model.bin` (or your specified path)
2. **Metrics and plots:** `plots/<timestamp>/`
   - `metrics.json` - Training history (loss, accuracy, time per epoch)
   - `loss_curve.png` - Training loss over epochs
   - `accuracy_curve.png` - Training accuracy over epochs
   - `time_per_epoch.png` - Time taken per epoch
   - `combined_metrics.png` - All metrics in one figure

---

### 3. Test/Evaluate Command

Evaluate a trained model on test data.

#### Linux/macOS
```bash
python3 test_model.py --dataset path/to/test_data --weights weights/model.bin
```

#### Windows (cmd)
```cmd
python test_model.py --dataset path\to\test_data --weights weights\model.bin
```

#### Windows (PowerShell)
```powershell
python test_model.py --dataset path\to\test_data --weights weights\model.bin
```

#### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | *required* | Path to test dataset directory |
| `--weights` | *required* | Path to trained model weights |

#### Evaluation Output

Prints to console:
- Test accuracy (%)
- Total parameters
- MACs (Multiply-Accumulate operations)
- FLOPs (Floating Point Operations)

**Example output:**
```
Test Accuracy: 92.34%
Model Parameters: 4,056
MACs: 270,720
FLOPs: 541,440
```

---

## Complete Example Workflow

Here's a complete example from build to evaluation:

### Linux/macOS

```bash
# 1. Build
python3 build.py

# 2. Split your dataset (if needed)
python3 split_dataset.py --input data_1 --output data_1_split --split 0.8

# 3. Train on training set
python3 train_model.py --dataset data_1_split/train --epochs 10 --lr 0.01

# 4. Evaluate on test set
python3 test_model.py --dataset data_1_split/test --weights weights/model.bin
```

### Windows

```cmd
REM 1. Build
python build.py

REM 2. Split your dataset (if needed)
python split_dataset.py --input data_1 --output data_1_split --split 0.8

REM 3. Train on training set
python train_model.py --dataset data_1_split\train --epochs 10 --lr 0.01

REM 4. Evaluate on test set
python test_model.py --dataset data_1_split\test --weights weights\model.bin
```

---

## Dataset Format

Your dataset should be organized in the following structure:

```
dataset/
├── class_1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class_2/
│   ├── image1.png
│   └── ...
└── class_n/
    └── ...
```

- Each subdirectory represents a class
- Images can be any size (automatically resized to 32×32)
- Supported formats: PNG, JPEG, JPG

---

## Platform-Specific Setup Details

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3 python3-dev python3-pip cmake g++

# Install Python packages
python3 -m pip install -r requirements.txt

# Build
python3 build.py
```

### Linux (Fedora/RHEL)

```bash
# Install system dependencies
sudo dnf install python3 python3-devel cmake gcc-c++

# Install Python packages
python3 -m pip install -r requirements.txt

# Build
python3 build.py
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python cmake

# Install Python packages
python3 -m pip install -r requirements.txt

# Build
python3 build.py
```

### Windows

1. **Install Python 3.8+** from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Install CMake** from [cmake.org](https://cmake.org/download/)
   - Check "Add CMake to system PATH" during installation

3. **Install Visual Studio Build Tools**
   - Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++" workload

4. **Install Python packages:**
   ```cmd
   python -m pip install -r requirements.txt
   ```

5. **Build:**
   ```cmd
   python build.py
   ```

---

## Model Architecture

The framework implements a SimpleCNN with the following architecture:

```
Input (3×32×32)
    ↓
Conv2D (3→8 channels, 3×3 kernel, stride=2)
    ↓
ReLU
    ↓
MaxPool2D (2×2, stride=2)
    ↓
Flatten
    ↓
Linear (128 → num_classes)
```

**Performance Characteristics:**
- **10 classes:** ~4,000 parameters, ~270K MACs, ~540K FLOPs
- **100 classes:** ~39,000 parameters, ~2.7M MACs, ~5.4M FLOPs
- **Training speed:** ~50 seconds/epoch (on typical hardware)
- **Memory efficient:** Batch tensor reuse, cache-optimized convolution

---

## Performance Optimization

The framework includes several optimizations:

1. **Float32 precision** (instead of double)
2. **Compiler optimizations** (-O3 -ffast-math)
3. **Cache-optimized convolution** loops
4. **Batch tensor reuse** in data loading
5. **Bit operations** for stride=2 calculations
6. **Memory-efficient gradient zeroing** (memset)

**Result:** 6x speedup from initial implementation (~300s → ~50s per epoch)

---

## Continuous Integration

The project uses GitHub Actions for automated testing across:
- **Operating Systems:** Ubuntu, macOS, Windows
- **Python Versions:** 3.8, 3.10, 3.12

[![Build Status](https://github.com/ShikharKunal/CNNcpp/actions/workflows/test.yml/badge.svg)](https://github.com/ShikharKunal/CNNcpp/actions)

---

## Troubleshooting

### Build fails with "pybind11 not found"
```bash
# Make sure pybind11 is installed
python3 -m pip install pybind11
```

### Import error: "cannot import name 'mydl_cpp'"
```bash
# Rebuild the extension
python3 build.py
```

### Windows: "cmake is not recognized"
- Make sure CMake is installed and added to PATH
- Restart your terminal after installation

### macOS: Architecture mismatch error
```bash
# Clean rebuild
rm -rf build python/mydl/mydl_cpp*
python3 build.py
```

### Training appears slow
- Training ~50-60s/epoch is normal
- First epoch may be slower due to data loading
- Check that compiler optimizations are enabled (they are by default)

---

## Project Structure

```
mydl/
├── cpp/                      # C++ backend
│   ├── tensor.cpp           # Tensor class with autograd
│   ├── ops.cpp              # Operations (add, mul, matmul, etc.)
│   ├── layers.cpp           # CNN layers
│   ├── loss.cpp             # Loss functions
│   ├── optimizer.cpp        # SGD optimizer
│   ├── dataloader.cpp       # Dataset loading
│   ├── metrics.cpp          # Accuracy/loss metrics
│   ├── model.cpp            # SimpleCNN model
│   └── bindings.cpp         # Python bindings
├── python/                   # Python frontend
│   └── mydl/                # Main package
│       ├── __init__.py      # Package initialization
│       └── dataset_loader.py # Python-based image loading
├── build.py                 # Build script (all platforms)
├── train_model.py           # Training script
├── test_model.py            # Evaluation script
├── split_dataset.py         # Dataset splitting utility
├── CMakeLists.txt           # CMake configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Dependencies

**Python packages (installed via pip):**
- `pybind11>=2.11.0` - Python-C++ bindings
- `opencv-python>=4.5.0` - Image loading and preprocessing
- `matplotlib>=3.3.0` - Plotting training metrics

**C++ dependencies (header-only, no installation needed):**
- C++17 standard library
- CMake build system

---

## License

This project was created for academic purposes as part of GNR638 coursework at IIT Bombay.

---

## Acknowledgments

- Course: GNR638 – Machine Learning for Remote Sensing – II
- Institution: IIT Bombay
- Semester: Spring 2025-26
