// SPDX-License-Identifier: MIT
//
// Thin wrappers around CUB (shipped with the CUDA Toolkit) used as the reference
// implementation for reduction and exclusive scan in the benchmark.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace prim {

/// Bytes of temporary device storage CUB needs for DeviceReduce::Sum over n floats.
std::size_t cub_reduce_temp_bytes(int n);
/// out[0] = sum(in[0..n)), using caller-provided temporary storage.
void cub_reduce_sum(const float* in, float* out, int n, void* temp, std::size_t temp_bytes, cudaStream_t stream);

/// Bytes of temporary device storage CUB needs for DeviceScan::ExclusiveSum over n ints.
std::size_t cub_scan_temp_bytes(int n);
/// Exclusive prefix sum of n ints, using caller-provided temporary storage.
void cub_scan_exclusive_sum(const int* in, int* out, int n, void* temp, std::size_t temp_bytes,
                            cudaStream_t stream);

}  // namespace prim
