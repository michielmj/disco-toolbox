cmake_minimum_required(VERSION 3.18)
project(disco_toolbox LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

target_compile_features(_core PRIVATE cxx_std_17)

# Optional: make builds quieter on macOS
if(APPLE)
  target_compile_options(_core PRIVATE -Wno-deprecated-declarations)
endif()

# This is what scikit-build-core uses to place the extension in the wheel.
# The DESTINATION path is relative to the wheel's platlib root.
install(TARGETS _core DESTINATION toolbox/orderbook)
