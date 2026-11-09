# SPDX-License-Identifier: MIT
#
# Native (PyTorch-free) build: kernel library, correctness tests and benchmark.
# The PyTorch extension is built separately with setup.py (see README).
cmake_minimum_required(VERSION 3.24)  # 3.24+: CMAKE_CUDA_ARCHITECTURES=native

# Must be decided before project() enables the CUDA language.
# "native" targets the GPU installed in this machine; override for other hardware, e.g.
# -DCMAKE_CUDA_ARCHITECTURES="75;86;89". bf16 conversions are supported on all architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for")
endif()

project(cuda_fused_layernorm
  VERSION 1.0.0
  DESCRIPTION "Fused LayerNorm CUDA kernels with a PyTorch extension"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # Benchmarks are meaningless in Debug (CMake adds -G, i.e. unoptimised device code).
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LN_BUILD_TESTS "Build the native correctness tests" ON)
option(LN_ENABLE_LINEINFO "Embed source line info so Nsight Compute can show source-level metrics" ON)
option(LN_PTXAS_VERBOSE "Print per-kernel register / shared-memory usage at compile time" OFF)

find_package(CUDAToolkit REQUIRED)

function(ln_configure_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  endif()
  if(LN_ENABLE_LINEINFO)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
  if(LN_PTXAS_VERBOSE)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  endif()
endfunction()

# Framework-independent kernel library (the same sources are compiled into the PyTorch extension).
add_library(ln_kernels STATIC
  csrc/kernels/ln_forward.cu
  csrc/kernels/ln_backward.cu)
target_include_directories(ln_kernels
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels)
target_link_libraries(ln_kernels PUBLIC CUDA::cudart)
ln_configure_target(ln_kernels)

add_executable(ln_native_bench native/bench_native.cpp)
target_include_directories(ln_native_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/native)
target_link_libraries(ln_native_bench PRIVATE ln_kernels)
ln_configure_target(ln_native_bench)

if(LN_BUILD_TESTS)
  enable_testing()
  add_executable(ln_native_tests native/test_native.cpp)
  target_include_directories(ln_native_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/native)
  target_link_libraries(ln_native_tests PRIVATE ln_kernels)
  ln_configure_target(ln_native_tests)
  add_test(NAME native_correctness COMMAND ln_native_tests)  # requires a CUDA-capable GPU
endif()

install(TARGETS ln_native_bench RUNTIME DESTINATION bin)
