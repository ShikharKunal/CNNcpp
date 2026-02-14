# Platform Compatibility Notes

## Current Status: Unix/macOS Only

This codebase currently works on **Linux** and **macOS** only. Windows support requires additional setup.

---

## For Linux Users

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-dev python3-venv cmake g++ libopencv-dev

# Fedora/RHEL
sudo dnf install python3 python3-devel cmake gcc-c++ opencv-devel
```

### Build & Run
```bash
./build.sh
./train.sh --dataset data_1_split/train
./test.sh --dataset data_1_split/test --weights weights/model.bin
```

**Known Issues:**
- OpenCV paths might differ - set `OpenCV_DIR` if CMake fails:
  ```bash
  export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
  ./build.sh
  ```

---

## For macOS Users

### Prerequisites
```bash
# Install Homebrew first: https://brew.sh
brew install python@3.12 cmake opencv

# For Apple Silicon (M1/M2/M3):
arch -arm64 /bin/zsh -c 'eval "$(/opt/homebrew/bin/brew shellenv)" && brew install opencv'
```

### Build & Run
```bash
./build.sh
./train.sh --dataset data_1_split/train
./test.sh --dataset data_1_split/test --weights weights/model.bin
```

**Known Issues:**
- Architecture mismatch (x86 vs ARM) - make sure OpenCV matches your architecture
- Use `rebuild_arm64.sh` if you're on Apple Silicon

---

## For Windows Users

### Option 1: Use WSL2 (Recommended)

Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install):
```powershell
wsl --install -d Ubuntu
```

Then follow Linux instructions above inside WSL.

### Option 2: Use Git Bash

1. Install [Git for Windows](https://git-scm.com/download/win) (includes Git Bash)
2. Install Python 3.12 from [python.org](https://www.python.org/downloads/)
3. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) (for C++ compiler)
4. Install OpenCV (see below)
5. Run scripts in Git Bash:
   ```bash
   ./build.sh
   ./train.sh --dataset data_1_split/train
   ```

### Option 3: Native Windows (Manual)

**Setup:**
```cmd
REM Create virtual environment
python -m venv .venv
.venv\Scripts\activate.bat

REM Install dependencies
pip install pybind11 matplotlib

REM Configure OpenCV_DIR
set OpenCV_DIR=C:\path\to\opencv\build

REM Build
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
```

**Train:**
```cmd
.venv\Scripts\activate.bat
cd python
python train.py --dataset ..\data_1_split\train --epochs 10 --weights ..\weights\model.bin
```

**Test:**
```cmd
.venv\Scripts\activate.bat
cd python
python evaluate.py --dataset ..\data_1_split\test --weights ..\weights\model.bin
```

**Note:** You won't get automatic plot generation with this method. Use the Python scripts directly.

---

## Python Version Compatibility

**Current requirement:** Python 3.12 (hardcoded in CMakeLists.txt)

**To use a different version:**

Edit `CMakeLists.txt` line 30:
```cmake
# Change this:
find_package(Python3 3.12 COMPONENTS Interpreter Development REQUIRED)

# To this (for any 3.8+):
find_package(Python3 3.8 COMPONENTS Interpreter Development REQUIRED)
```

---

## OpenCV Installation

### Linux
```bash
sudo apt install libopencv-dev  # Ubuntu/Debian
sudo dnf install opencv-devel   # Fedora/RHEL
```

### macOS
```bash
brew install opencv
```

### Windows
1. Download from [opencv.org](https://opencv.org/releases/)
2. Extract to `C:\opencv`
3. Set environment variable:
   ```cmd
   set OpenCV_DIR=C:\opencv\build
   ```

---

## Common Issues

### CMake can't find pybind11
```bash
# Solution: Activate venv before building
source .venv/bin/activate  # Unix/Mac
.venv\Scripts\activate.bat  # Windows
./build.sh
```

### CMake can't find OpenCV
```bash
# Linux - find OpenCV config:
find /usr -name "OpenCVConfig.cmake"

# Set the directory containing it:
export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
./build.sh
```

### "unknown target CPU 'apple-m1'"
- Already fixed: `-march=native` removed from CMakeLists.txt
- If you see this, update CMakeLists.txt (remove `-march=native`)

### Architecture mismatch on macOS
```bash
# Rebuild for correct architecture
rm -rf build
./build.sh
```

---

## Making It Truly Cross-Platform (TODO)

To make this work seamlessly on all platforms, we would need:

1. **Python-based build script** (instead of bash)
2. **Platform detection** in CMakeLists.txt:
   ```cmake
   if(WIN32)
       set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast")
   else()
       set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
   endif()
   ```
3. **Flexible Python version** (3.8+ instead of 3.12 only)
4. **Better OpenCV detection** (check standard paths for all platforms)
5. **Windows batch scripts** (`.bat` files)

---

## Summary

| Platform | Status | Method |
|----------|--------|--------|
| **Linux** | ✅ Works | Use shell scripts |
| **macOS** | ✅ Works | Use shell scripts |
| **Windows** | ⚠️ Partial | Use WSL2, Git Bash, or manual Python scripts |

**Recommendation for cross-platform users:** Use WSL2 on Windows for the smoothest experience.
