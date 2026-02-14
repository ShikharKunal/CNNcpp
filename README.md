# mydl – Custom Deep Learning Framework (GNR638)

Minimal deep learning framework from scratch for **GNR638: Machine Learning for Remote Sensing – II (Spring 2025–26)**.  
Backend: C++ (no PyTorch/TensorFlow/NumPy/Eigen). Frontend: Python 3.12.  
Uses only: C++ stdlib, Python 3.12 stdlib, OpenCV (image load/resize), pybind11 (bindings).

## Features

- **Autograd**: Dynamic computation graph, DFS topological sort, reverse-mode automatic differentiation with gradient accumulation.
- **Ops**: add, mul, matmul, reshape, flatten, sum, mean (forward + backward).
- **Layers**: Conv2D (manual loops), ReLU, MaxPool2D, Linear.
- **Loss**: Softmax, numerically stable cross-entropy.
- **Optimizer**: SGD with learning rate and zero_grad.
- **Data**: Load images from `root/class_1/`, `root/class_2/`, …; OpenCV only for PNG read and resize to 32×32; batching; dataset load time reported.
- **Metrics**: Accuracy, loss value.
- **CNN**: At least one conv, activation, pooling, fully connected; reports trainable parameters, MACs, and FLOPs per forward pass.

## Setup

### Prerequisites

**All Platforms:**
- Python 3.8+ (with pip)
- CMake 3.14+
- C++17 compiler

**No OpenCV system installation required!** (Uses opencv-python pip package)

**Platform-Specific:**

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-dev python3-pip python3-venv cmake g++
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install python3 python3-devel cmake gcc-c++
```

**macOS:**
```bash
# Install Homebrew first: https://brew.sh
brew install python cmake
```

**Windows:**
1. Install [Python 3.8+](https://www.python.org/downloads/)
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) (for C++ compiler)
3. Install [CMake](https://cmake.org/download/)

### Initial Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows (cmd):
.venv\Scripts\activate.bat
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install pybind11 matplotlib

# Build C++ extension
python3 build.py
```

**Or use the automated setup script (Linux/macOS only):**
```bash
./scripts/setup_venv.sh
```

## Advanced: Python Scripts Directly

If you prefer to use Python scripts directly:

### Train
```bash
cd python
python train.py --dataset /path/to/train --epochs 10 --batch_size 32 --lr 0.01 --weights weights.bin
```

### Evaluate
```bash
cd python
python evaluate.py --dataset /path/to/test --weights weights.bin
```

**Note:** The shell scripts (`train.sh`, `test.sh`) are recommended as they handle paths automatically and generate plots.

## Project layout

```
mydl/
├── cpp/                        # C++ backend
│   ├── tensor.h/cpp           # Tensor + autograd (DFS, backward)
│   ├── ops.h/cpp              # add, mul, matmul, reshape, flatten, sum, mean
│   ├── layers.h/cpp           # Conv2D, ReLU, MaxPool2D, Linear
│   ├── loss.h/cpp             # softmax, cross_entropy_loss
│   ├── optimizer.h/cpp        # SGD
│   ├── dataloader.h/cpp       # directory loader, OpenCV, batching
│   ├── metrics.h/cpp          # accuracy, loss_value
│   ├── model.h/cpp            # SimpleCNN, params/MACs/FLOPs, save/load
│   └── bindings.cpp           # pybind11 → mydl_cpp
├── python/                     # Python frontend
│   ├── mydl/                  # Package
│   │   ├── __init__.py        # re-exports mydl_cpp
│   │   └── mydl_cpp.*.so      # built extension
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── config.py              # Config parser
│   └── sample_config.json     # Example config
├── scripts/
│   ├── setup_venv.sh          # Linux/macOS: create venv and build
│   └── setup_venv.bat         # Windows: create venv and build
├── requirements.txt           # pybind11 (build-time)
├── .gitignore
├── CMakeLists.txt
└── README.md
```

## Quick Start

Three simple Python scripts that work on **all platforms** (Linux, macOS, Windows):

### 1. Build
```bash
python3 build.py
```
Compiles the C++ extension. Run once after downloading or after changing C++ code.

### 2. Train
```bash
python3 train_model.py --dataset <train_dir> [options]
```

**Options:**
- `--dataset <path>` (required) - Path to training dataset
- `--epochs <n>` (default: 10) - Number of epochs
- `--batch-size <n>` (default: 32) - Batch size
- `--lr <float>` (default: 0.01) - Learning rate
- `--weights <path>` (default: weights/model.bin) - Where to save weights

**Example:**
```bash
python3 train_model.py --dataset data_1_split/train --epochs 10 --lr 0.01
```

**Outputs:**
- Trained weights → `weights/model.bin` (or your specified path)
- Metrics JSON → `plots/<timestamp>/metrics.json`
- 4 plots → `plots/<timestamp>/` (loss, accuracy, time, combined)

### 3. Test/Evaluate
```bash
python3 test_model.py --dataset <test_dir> --weights <weights_path>
```

**Example:**
```bash
python3 test_model.py --dataset data_1_split/test --weights weights/model.bin
```

Prints test accuracy and efficiency metrics.

---

## Platform Support

✅ **Works on all platforms:**
- Linux (Ubuntu, Fedora, Debian, etc.)
- macOS (Intel & Apple Silicon)
- Windows (native, WSL2, or Git Bash)

**Requirements:**
- Python 3.8 or higher
- C++17 compiler (GCC/Clang/MSVC)
- CMake 3.14+
- OpenCV 4.x

## Reproducibility

Layer initializations use fixed seeds (e.g. 42, 43) in the C++ code. Data shuffling uses seed 12345. For full reproducibility, these seeds ensure deterministic training with the same dataset.

## Performance Notes

- **Dataset loading**: ~5-10 seconds for 6,000 images, ~20-30 seconds for 10,000 images
- **Training speed**: ~40-60 seconds per epoch (optimized with cache-aware convolution, batch reuse, memset, bit operations)
- **Model complexity**: 
  - 10 classes: ~4,000 parameters, ~270,000 MACs, ~540,000 FLOPs per forward pass
  - 100 classes: ~39,000 parameters, ~2,700,000 MACs, ~5,400,000 FLOPs per forward pass

## Troubleshooting

**Import error: "No module named 'mydl_cpp'"**
- Rebuild the C++ extension: `cd build && make && cd ..`
- Ensure the extension is in `python/mydl/mydl_cpp.cpython-312-darwin.so` (or `.dll` on Windows)

**Architecture mismatch (macOS)**
- If you get "incompatible architecture (have 'x86_64', need 'arm64')", you need arm64 OpenCV
- Install: `arch -arm64 /bin/zsh -c 'eval "$(/opt/homebrew/bin/brew shellenv)" && brew install opencv'`
- Rebuild with: `cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DOpenCV_DIR=/opt/homebrew/opt/opencv/lib/cmake/opencv4 ..`

**Training appears stuck**
- Use `python -u` (unbuffered output) to see real-time progress
- Training is working if you see "Starting epoch..." messages
- Each epoch takes ~90-100 seconds for 60K samples

**CMake cannot find OpenCV**
- Set `OpenCV_DIR` environment variable before running CMake (see setup instructions above)
