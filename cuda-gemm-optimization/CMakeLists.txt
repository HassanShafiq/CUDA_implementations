# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.24)  # 3.24+: CMAKE_CUDA_ARCHITECTURES=native

# Must be decided before project() enables the CUDA language.
# "native" targets the GPU(s) installed in this machine. Override for other
# hardware or for redistributable builds, e.g. -DCMAKE_CUDA_ARCHITECTURES="75;86;89".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for")
endif()

project(cuda_gemm_optimization
  VERSION 1.0.0
  DESCRIPTION "Step-by-step CUDA SGEMM optimization, benchmarked against cuBLAS"
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

option(GEMM_BUILD_TESTS "Build the correctness tests" ON)
option(GEMM_ENABLE_LINEINFO "Embed source line info so Nsight Compute can show source-level metrics" ON)
option(GEMM_PTXAS_VERBOSE "Print per-kernel register / shared-memory usage at compile time" OFF)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Warning and device-code options applied to every target of this project.
# ---------------------------------------------------------------------------
function(gemm_configure_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  endif()
  if(GEMM_ENABLE_LINEINFO)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
  if(GEMM_PTXAS_VERBOSE)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
add_library(gemm_kernels STATIC
  src/kernels/01_naive.cu
  src/kernels/02_coalesced.cu
  src/kernels/03_smem_tiled.cu
  src/kernels/04_blocktile_1d.cu
  src/kernels/05_regtile_2d.cu
  src/kernels/06_vectorized.cu
  src/kernel_registry.cpp)
target_include_directories(gemm_kernels
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels)
target_link_libraries(gemm_kernels PUBLIC CUDA::cudart)
gemm_configure_target(gemm_kernels)

add_library(gemm_cublas STATIC src/cublas_baseline.cpp)
target_link_libraries(gemm_cublas PUBLIC gemm_kernels CUDA::cublas)
gemm_configure_target(gemm_cublas)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(gemm_bench src/benchmark.cpp)
target_link_libraries(gemm_bench PRIVATE gemm_cublas)
gemm_configure_target(gemm_bench)

if(GEMM_BUILD_TESTS)
  enable_testing()
  add_executable(gemm_tests tests/test_correctness.cpp)
  target_link_libraries(gemm_tests PRIVATE gemm_kernels)
  gemm_configure_target(gemm_tests)
  add_test(NAME correctness COMMAND gemm_tests)  # requires a CUDA-capable GPU
endif()

install(TARGETS gemm_bench RUNTIME DESTINATION bin)
