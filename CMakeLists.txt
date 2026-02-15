cmake_minimum_required(VERSION 3.14)
project(mydl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Compiler optimizations for speed
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific optimization flags
if(MSVC)
  # Windows MSVC
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /fp:fast")
else()
  # Unix/Linux/macOS (GCC/Clang)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math")
endif()

# Find pybind11 (Python bindings)
find_package(pybind11 CONFIG REQUIRED)
# Minimum Python 3.8, supports up to 3.14
find_package(Python3 3.8 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(mydl_cpp
  cpp/tensor.cpp
  cpp/ops.cpp
  cpp/layers.cpp
  cpp/loss.cpp
  cpp/optimizer.cpp
  cpp/dataloader.cpp
  cpp/metrics.cpp
  cpp/model.cpp
  cpp/bindings.cpp
)

target_include_directories(mydl_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)


# pybind11_add_module creates a module that Python can import
# Output directly to python/mydl/ directory
# On Windows, also prevent the Release/Debug subdirectory
set_target_properties(mydl_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/mydl
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/mydl
  OUTPUT_NAME mydl_cpp
)

# For multi-config generators (MSVC), set output directory for all configs
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${CONFIG} CONFIG_UPPER)
  set_target_properties(mydl_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_CURRENT_SOURCE_DIR}/python/mydl
    RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_CURRENT_SOURCE_DIR}/python/mydl
  )
endforeach()
