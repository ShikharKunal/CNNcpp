# Windows Beta Testing Request

## Background
This framework has been tested on macOS and Linux (via Docker), but not on native Windows yet.

## What We Need
Windows users to test the build and installation process.

## Prerequisites
- Windows 10/11
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- [Python 3.8+](https://www.python.org/downloads/)
- [CMake](https://cmake.org/download/)

## Steps to Test

```cmd
REM 1. Clone repository
git clone <repo-url>
cd mydl

REM 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate.bat

REM 3. Install dependencies
pip install -r requirements.txt

REM 4. Build C++ extension
python build.py

REM 5. Test import
python -c "from python.mydl import SimpleCNN; print('Success!')"
```

## Report Results

Please report:
- ✅ Success or ❌ Failure
- Windows version (10/11)
- Python version
- Any error messages

## Known Windows Issues

### Missing C++ Compiler
**Error:** `cmake: error: ... could not find any compiler`
**Solution:** Install Visual Studio Build Tools with C++ workload

### CMake Not Found
**Error:** `'cmake' is not recognized`
**Solution:** Add CMake to PATH or reinstall with "Add to PATH" option

### Python Extension Suffix
The compiled extension will be `.pyd` instead of `.so` (this is normal).

---

Thank you for helping test! 🙏
