cmake_minimum_required(VERSION 3.18)

project(disco_toolbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# scikit-build-core defines these for us; keep a helpful fallback for local builds.
if(NOT DEFINED SKBUILD_PLATLIB_DIR)
  set(SKBUILD_PLATLIB_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Prevent MSVC from adding Release/Debug subdirectories
if (MSVC)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
endif()

# Python (needed for extension module build)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Prefer an installed pybind11 (e.g. provided via build-system requires), but allow fallback.
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Build toolbox.orderbook._core
pybind11_add_module(_core MODULE
  src/toolbox/orderbook/_core.cpp
)

# Ensure numpy support works (pybind11 includes NumPy headers via pybind11/numpy.h);
# no explicit NumPy include path needed unless you directly include numpy headers.

target_compile_features(_core PRIVATE cxx_std_17)

# Place the built extension inside the package directory in the wheel:
#   toolbox/orderbook/_core*.so|pyd
set_target_properties(_core PROPERTIES
  OUTPUT_NAME "_core"
  LIBRARY_OUTPUT_DIRECTORY "${SKBUILD_PLATLIB_DIR}/toolbox/orderbook"
  RUNTIME_OUTPUT_DIRECTORY "${SKBUILD_PLATLIB_DIR}/toolbox/orderbook"
  ARCHIVE_OUTPUT_DIRECTORY "${SKBUILD_PLATLIB_DIR}/toolbox/orderbook"
)

# Optional: make builds quieter on macOS about deprecated stdlib warnings, etc.
if(APPLE)
  target_compile_options(_core PRIVATE -Wno-deprecated-declarations)
endif()
