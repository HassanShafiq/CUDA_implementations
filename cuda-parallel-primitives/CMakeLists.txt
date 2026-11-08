# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.24)  # 3.24+: CMAKE_CUDA_ARCHITECTURES=native

# Must be decided before project() enables the CUDA language.
# "native" targets the GPU(s) installed in this machine. Override for other hardware,
# e.g. -DCMAKE_CUDA_ARCHITECTURES="75;86;89". Warp shuffles need compute capability >= 3.0.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for")
endif()

project(cuda_parallel_primitives
  VERSION 1.0.0
  DESCRIPTION "Reduction, prefix sum and softmax in CUDA: optimization ladders vs CUB"
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

option(PRIM_BUILD_TESTS "Build the correctness tests" ON)
option(PRIM_ENABLE_LINEINFO "Embed source line info so Nsight Compute can show source-level metrics" ON)
option(PRIM_PTXAS_VERBOSE "Print per-kernel register / shared-memory usage at compile time" OFF)

find_package(CUDAToolkit REQUIRED)

function(prim_configure_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  endif()
  if(PRIM_ENABLE_LINEINFO)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
  if(PRIM_PTXAS_VERBOSE)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  endif()
endfunction()

# Kernel library: reduction, scan and softmax ladders plus their registries.
add_library(prim_kernels STATIC
  src/reduce/reduce.cu
  src/scan/scan.cu
  src/softmax/softmax.cu)
target_include_directories(prim_kernels
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
target_link_libraries(prim_kernels PUBLIC CUDA::cudart)
prim_configure_target(prim_kernels)

# CUB reference implementations (header-only library shipped with the CUDA Toolkit).
add_library(prim_cub STATIC src/common/cub_baseline.cu)
target_link_libraries(prim_cub PUBLIC prim_kernels)
prim_configure_target(prim_cub)

add_executable(prim_bench src/benchmark.cpp)
target_link_libraries(prim_bench PRIVATE prim_kernels prim_cub)
prim_configure_target(prim_bench)

if(PRIM_BUILD_TESTS)
  enable_testing()
  add_executable(prim_tests tests/test_correctness.cpp)
  target_link_libraries(prim_tests PRIVATE prim_kernels)
  prim_configure_target(prim_tests)
  add_test(NAME correctness COMMAND prim_tests)  # requires a CUDA-capable GPU
endif()

install(TARGETS prim_bench RUNTIME DESTINATION bin)
