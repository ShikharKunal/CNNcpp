# CNN Deep Learning Framework - Setup Guide

A minimal CNN framework built from scratch in C++ with Python bindings.

---

## Prerequisites

| Platform | Requirements |
|----------|-------------|
| **Linux** | Python 3.8+, CMake 3.14+, g++, pip |
| **macOS** | Python 3.8+, CMake 3.14+, Xcode Command Line Tools |
| **Windows** | Python 3.8+, CMake 3.14+, MinGW-w64 GCC |

---

## Setup

### 1. Install Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-dev python3-pip cmake g++
```

**macOS:**
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python cmake
```

**Windows (MinGW):**

**Step 1: Install Python**
1. Download [Python 3.8+](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"**
3. Verify: Open Command Prompt and run `python --version`

**Step 2: Install CMake**

Choose one option:

- **Option A: CMake Installer (Recommended)**
  1. Download from https://cmake.org/download/
  2. Get "Windows x64 Installer" (e.g., `cmake-3.28.1-windows-x86_64.msi`)
  3. Run installer
  4. **Important:** Check "Add CMake to system PATH for all users"
  5. Complete installation
  6. Verify: Open **new** Command Prompt and run `cmake --version`

- **Option B: Portable CMake**
  1. Download "Windows x64 ZIP" from https://cmake.org/download/
  2. Extract to `C:\cmake`
  3. Add to PATH manually:
     - Press `Win + R`, type `sysdm.cpl`, press Enter
     - Advanced → Environment Variables
     - Edit PATH → Add `C:\cmake\bin`
  4. Verify: Open **new** Command Prompt and run `cmake --version`

**Step 3: Install MinGW-w64**

Choose one option:

- **Option A: Via MSYS2 (Recommended)**
  1. Download and install [MSYS2](https://www.msys2.org/)
  2. Open MSYS2 terminal and run:
     ```bash
     pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake
     ```
  3. Add to PATH: `C:\msys64\mingw64\bin`
     - Press `Win + R`, type `sysdm.cpl`, press Enter
     - Advanced → Environment Variables
     - Edit PATH → Add `C:\msys64\mingw64\bin`
  4. Verify: Open **new** Command Prompt and run `gcc --version`

- **Option B: Standalone MinGW-w64**
  1. Download from https://www.mingw-w64.org/downloads/
  2. Follow installation instructions
  3. Add MinGW bin directory to PATH

**Step 4: Configure CMake for MinGW**

Set CMake to use MinGW compiler:

```cmd
# Temporary (current session only)
set CMAKE_GENERATOR=MinGW Makefiles

# Permanent (recommended)
# Add as system environment variable:
# Variable name: CMAKE_GENERATOR
# Variable value: MinGW Makefiles
```

> **Note:** For detailed MinGW setup instructions, see [MINGW_QUICKSTART.md](MINGW_QUICKSTART.md)


### 2. Create Virtual Environment (Recommended)

Using a virtual environment keeps dependencies isolated and avoids conflicts:

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows (cmd)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt when activated.

### 3. Install Python Packages

```bash
# Linux/macOS
python3 -m pip install -r requirements.txt

# Windows
python -m pip install -r requirements.txt
```

**Required packages:** `pybind11`, `opencv-python`, `matplotlib`

### 4. Build the Framework

```bash
# Linux/macOS
python3 build.py

# Windows (set generator first if not set permanently)
set CMAKE_GENERATOR=MinGW Makefiles
python build.py
```

This compiles the C++ backend and creates the Python extension module.

---

## Dataset Preparation

### Dataset Format

Organize your dataset with one folder per class:

```
dataset/
├── class_0/
│   ├── img1.png
│   └── img2.png
├── class_1/
│   └── ...
└── class_n/
    └── ...
```

- Supported formats: PNG, JPEG, JPG
- Images are automatically resized to 32×32

### Split Dataset (Train/Test)

```bash
# Linux/macOS
python3 split_dataset.py --input <dataset_path> --output <output_dir> --split 0.8 --seed 42

# Windows
python split_dataset.py --input <dataset_path> --output <output_dir> --split 0.8 --seed 42
```

**Arguments:**
- `--input`: Path to original dataset
- `--output`: Output directory (creates `train/` and `test/` subdirectories)
- `--split`: Train ratio (default: 0.8 = 80% train, 20% test)
- `--seed`: Random seed for reproducibility (default: 42)

**Example:**
```bash
python3 split_dataset.py --input "Assignment 1 Datasets/data_1" --output data_1_split --split 0.8 --seed 42
```

**Output:**
```
data_1_split/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
```

---

## Training

### Basic Usage

```bash
# Linux/macOS
python3 train_model.py --config config/data_1.json --dataset data_1_split/train

# Windows
python train_model.py --config config/data_1.json --dataset data_1_split\train
```

### Configuration Files

Training uses JSON config files in the `config/` directory:

**`config/data_1.json`** - For data_1 dataset (20 epochs):
```json
{
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.01,
  "weights_dir": "weights",
  "plots_dir": "plots",
  "save_metrics": true,
  "generate_plots": true,
  "print_batch_interval": 200
}
```

**`config/data_2.json`** - For data_2 dataset (30 epochs):
```json
{
  "epochs": 30,
  "batch_size": 32,
  "learning_rate": 0.01,
  ...
}
```

### Training Parameters Used

| Parameter | data_1 | data_2 |
|-----------|--------|--------|
| Epochs | 20 | 30 |
| Batch Size | 32 | 32 |
| Learning Rate | 0.01 | 0.01 |
| Optimizer | SGD | SGD |

### Training Outputs

After training, you'll find:

1. **Model weights:** `weights/<timestamp>.bin`
2. **Metrics and plots:** `plots/<timestamp>/`
   - `metrics.json` - Training history
   - `loss_curve.png` - Loss over epochs
   - `accuracy_curve.png` - Accuracy over epochs
   - `time_per_epoch.png` - Time per epoch
   - `training_curves.png` - Combined metrics

---

## Testing/Evaluation

```bash
# Linux/macOS
python3 test_model.py --dataset data_1_split/test --weights weights/<timestamp>.bin

# Windows
python test_model.py --dataset data_1_split\test --weights weights\<timestamp>.bin
```

**Output:**
- Test loss and accuracy
- Model parameters count
- MACs and FLOPs

---

## Complete Workflow Example

### Linux/macOS

```bash
# 1. Build
python3 build.py

# 2. Split dataset
python3 split_dataset.py --input "Assignment 1 Datasets/data_1" --output data_1_split --split 0.8 --seed 42

# 3. Train
python3 train_model.py --config config/data_1.json --dataset data_1_split/train

# 4. Test
python3 test_model.py --dataset data_1_split/test --weights weights/20260215_214008.bin
```

### Windows

```cmd
REM 1. Build
python build.py

REM 2. Split dataset
python split_dataset.py --input "Assignment 1 Datasets\data_1" --output data_1_split --split 0.8 --seed 42

REM 3. Train
python train_model.py --config config\data_1.json --dataset data_1_split\train

REM 4. Test
python test_model.py --dataset data_1_split\test --weights weights\20260215_214008.bin
```

---

## Platform-Specific Notes

### Path Separators
- **Linux/macOS:** Use `/` (forward slash)
- **Windows:** Use `\` (backslash) or `/` (both work)

### Python Command
- **Linux/macOS:** `python3`
- **Windows:** `python`

### Rebuilding
If you modify C++ code, rebuild:
```bash
# Clean build (if needed)
rm -rf build python/mydl/mydl_cpp.*  # Linux/macOS
rmdir /s build & del python\mydl\mydl_cpp.*  # Windows

# Rebuild
python3 build.py  # Linux/macOS
python build.py   # Windows
```

---

## Troubleshooting

**Build fails:**
```bash
# Ensure pybind11 is installed
python3 -m pip install pybind11
```

**Import error:**
```bash
# Rebuild
python3 build.py
```

**Windows: "cmake not recognized":**
- Restart terminal after installing CMake
- Verify CMake is in PATH

**Slow training:**
- 50-75s/epoch is normal
- First epoch may be slower (data loading)

---

## Random Seed

The dataset splitting uses `--seed 42` by default for reproducibility. This ensures:
- Same train/test split across runs
- Consistent results for validation

To use a different seed:
```bash
python3 split_dataset.py --input data --output data_split --seed 123
```

---

## Project Structure

```
mydl/
├── cpp/                    # C++ backend
├── python/mydl/            # Python package
├── config/                 # Training configs
│   ├── data_1.json        # 20 epochs
│   └── data_2.json        # 30 epochs
├── build.py               # Build script
├── train_model.py         # Training script
├── test_model.py          # Evaluation script
├── split_dataset.py       # Dataset splitter
└── requirements.txt       # Python dependencies
```
