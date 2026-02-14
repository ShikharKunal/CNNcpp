#!/usr/bin/env python3
"""
Cross-platform build script for mydl C++ extension.
Works on Linux, macOS, and Windows.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path


def get_pybind11_path():
    """Get pybind11 cmake directory from venv or system Python."""
    venv_dir = Path(".venv")
    
    # Try venv first
    if venv_dir.exists():
        if platform.system() == "Windows":
            python_exe = venv_dir / "Scripts" / "python.exe"
        else:
            python_exe = venv_dir / "bin" / "python3"
        
        try:
            result = subprocess.run(
                [str(python_exe), "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
                capture_output=True,
                text=True,
                check=True
            )
            pybind11_dir = result.stdout.strip()
            print(f"Found pybind11 in venv at: {pybind11_dir}")
            return pybind11_dir
        except Exception as e:
            print(f"Warning: Could not find pybind11 in venv: {e}")
    
    # Try system Python (for CI environments)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
            capture_output=True,
            text=True,
            check=True
        )
        pybind11_dir = result.stdout.strip()
        print(f"Found pybind11 in system Python at: {pybind11_dir}")
        return pybind11_dir
    except Exception as e:
        print(f"Warning: Could not find pybind11 in system Python: {e}")
        print("Make sure pybind11 is installed: pip install pybind11")
        return None


def get_num_cores():
    """Get number of CPU cores for parallel build."""
    try:
        if platform.system() == "Windows":
            return int(os.environ.get("NUMBER_OF_PROCESSORS", 4))
        else:
            return int(subprocess.check_output(["nproc"]).decode().strip())
    except:
        return 4


def main():
    print("=" * 50)
    print("Building mydl C++ extension")
    print("=" * 50)
    print()
    
    # Change to script directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    # Get pybind11 path
    pybind11_dir = get_pybind11_path()
    
    # Clean previous build
    build_dir = Path("build")
    if build_dir.exists():
        print("Cleaning previous build...")
        import shutil
        shutil.rmtree(build_dir)
    
    build_dir.mkdir()
    os.chdir(build_dir)
    
    # Prepare CMake command
    cmake_args = ["cmake", ".."]
    
    if pybind11_dir:
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={pybind11_dir}")
    
    # Platform-specific configuration
    if platform.system() == "Windows":
        cmake_args.extend(["-DCMAKE_BUILD_TYPE=Release"])
    
    # Configure
    print("Configuring with CMake...")
    try:
        subprocess.run(cmake_args, check=True)
    except subprocess.CalledProcessError:
        print("\nError: CMake configuration failed.")
        print("Make sure OpenCV and pybind11 are installed.")
        sys.exit(1)
    
    # Build
    print("\nBuilding...")
    num_cores = get_num_cores()
    
    if platform.system() == "Windows":
        build_cmd = ["cmake", "--build", ".", "--config", "Release", "--parallel", str(num_cores)]
    else:
        build_cmd = ["make", f"-j{num_cores}"]
    
    try:
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError:
        print("\nError: Build failed.")
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("✓ Build complete!")
    print(f"Extension: python/mydl/mydl_cpp.*")
    print("=" * 50)


if __name__ == "__main__":
    main()
