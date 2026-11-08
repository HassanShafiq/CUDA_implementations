// SPDX-License-Identifier: MIT
#include <cub/cub.cuh>

#include "prim/cub_baseline.hpp"
#include "prim/cuda_utils.hpp"

namespace prim {

std::size_t cub_reduce_temp_bytes(int n) {
  std::size_t bytes = 0;
  // With a null temp pointer CUB only reports the required size.
  PRIM_CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, static_cast<const float*>(nullptr),
                                         static_cast<float*>(nullptr), n));
  return bytes;
}

void cub_reduce_sum(const float* in, float* out, int n, void* temp, std::size_t temp_bytes, cudaStream_t stream) {
  PRIM_CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, in, out, n, stream));
}

std::size_t cub_scan_temp_bytes(int n) {
  std::size_t bytes = 0;
  PRIM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<const int*>(nullptr),
                                                static_cast<int*>(nullptr), n));
  return bytes;
}

void cub_scan_exclusive_sum(const int* in, int* out, int n, void* temp, std::size_t temp_bytes,
                            cudaStream_t stream) {
  PRIM_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, n, stream));
}

}  // namespace prim
