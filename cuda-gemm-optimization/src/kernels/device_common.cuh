// SPDX-License-Identifier: MIT
//
// Small device-side helpers shared by every kernel in the ladder.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace gemm::dev {

/// Row-major element offset. The multiplication is done in size_t so that
/// matrices with more than 2^31 elements (e.g. 50k x 50k) do not overflow int.
__device__ __forceinline__ std::size_t idx(int row, int ld, int col) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(ld) + static_cast<std::size_t>(col);
}

/// alpha * acc + beta * c_old as a single fused multiply-add.
__device__ __forceinline__ float axpby(float alpha, float acc, float beta, float c_old) {
  return fmaf(alpha, acc, beta * c_old);
}

/// Writes alpha * acc + beta * c[offset]. When beta == 0 the old value is not
/// read, matching BLAS semantics (C may be uninitialised or contain NaN).
__device__ __forceinline__ void store_axpby(float* __restrict__ c, std::size_t offset, float alpha,
                                            float acc, float beta) {
  const float c_old = (beta == 0.0f) ? 0.0f : c[offset];
  c[offset] = axpby(alpha, acc, beta, c_old);
}

}  // namespace gemm::dev
