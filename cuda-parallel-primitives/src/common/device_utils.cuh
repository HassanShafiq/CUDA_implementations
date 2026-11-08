// SPDX-License-Identifier: MIT
//
// Device-side helpers shared by the kernels: warp shuffles, alignment checks and
// the numerically careful building blocks of softmax.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace prim::dev {

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarpSize = 32;

/// True when `p` is 16-byte aligned, i.e. safe to access through float4 / int4.
__device__ __forceinline__ bool is_aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

/// Sum across the 32 lanes of a warp; the result is valid in lane 0 only.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(kFullMask, v, offset);
  }
  return v;
}

// ---------------------------------------------------------------------------
// Softmax building blocks
// ---------------------------------------------------------------------------

/// exp(x - m), with exp(-inf) defined as 0 even when m is also -inf (avoids NaN from -inf - -inf).
__device__ __forceinline__ float exp_shifted(float x, float m) {
  return (x == -INFINITY) ? 0.0f : expf(x - m);
}

/// 1 / s, or 0 when the row had no finite entries.
__device__ __forceinline__ float safe_inverse(float s) { return (s > 0.0f) ? 1.0f / s : 0.0f; }

/// Running (max, sum-of-exponentials-relative-to-max) pair of the "online softmax" algorithm.
struct MaxSum {
  float max;
  float sum;
};

/// Factor that rescales a sum accumulated relative to `old_max` to be relative to `new_max`.
__device__ __forceinline__ float rescale(float old_max, float new_max) {
  return (old_max == new_max) ? 1.0f : expf(old_max - new_max);
}

/// Folds one more element into a running (max, sum) pair.
__device__ __forceinline__ MaxSum accumulate(MaxSum a, float x) {
  const float m = fmaxf(a.max, x);
  return MaxSum{m, a.sum * rescale(a.max, m) + exp_shifted(x, m)};
}

/// Merges two partial (max, sum) pairs (associative and commutative).
__device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
  const float m = fmaxf(a.max, b.max);
  return MaxSum{m, a.sum * rescale(a.max, m) + b.sum * rescale(b.max, m)};
}

/// All-lanes max across a warp (butterfly), every lane receives the result.
__device__ __forceinline__ float warp_allreduce_max(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(kFullMask, v, offset));
  }
  return v;
}

/// All-lanes sum across a warp (butterfly), every lane receives the result.
__device__ __forceinline__ float warp_allreduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, offset);
  }
  return v;
}

/// All-lanes combine of (max, sum) pairs across a warp.
__device__ __forceinline__ MaxSum warp_allreduce_max_sum(MaxSum v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    MaxSum other;
    other.max = __shfl_xor_sync(kFullMask, v.max, offset);
    other.sum = __shfl_xor_sync(kFullMask, v.sum, offset);
    v = combine(v, other);
  }
  return v;
}

}  // namespace prim::dev
