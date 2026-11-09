// SPDX-License-Identifier: MIT
//
// Device-side building blocks shared by the forward and backward kernels:
//   * dtype traits (storage type <-> accumulation type, fp32 math for fp16/bf16),
//   * 16-byte vector type for 128-bit memory accesses,
//   * Welford (count, mean, M2) statistics with a numerically stable parallel merge,
//   * warp and block reductions.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ln {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// ---------------------------------------------------------------------------
// dtype traits: storage type T, accumulation type Acc, conversions.
// fp16 / bf16 are stored in 16 bits but ALL arithmetic is done in fp32; fp64 stays fp64.
// ---------------------------------------------------------------------------
template <typename T>
struct Traits;

template <>
struct Traits<float> {
  using Acc = float;
  __device__ static __forceinline__ float to_acc(float v) { return v; }
  __device__ static __forceinline__ float from_acc(float v) { return v; }
};

template <>
struct Traits<double> {
  using Acc = double;
  __device__ static __forceinline__ double to_acc(double v) { return v; }
  __device__ static __forceinline__ double from_acc(double v) { return v; }
};

template <>
struct Traits<__half> {
  using Acc = float;
  __device__ static __forceinline__ float to_acc(__half v) { return __half2float(v); }
  __device__ static __forceinline__ __half from_acc(float v) { return __float2half(v); }
};

template <>
struct Traits<__nv_bfloat16> {
  using Acc = float;
  __device__ static __forceinline__ float to_acc(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __forceinline__ __nv_bfloat16 from_acc(float v) { return __float2bfloat16(v); }
};

template <typename T>
using acc_t = typename Traits<T>::Acc;

__device__ __forceinline__ float rsqrt_acc(float v) { return rsqrtf(v); }
__device__ __forceinline__ double rsqrt_acc(double v) { return rsqrt(v); }

// ---------------------------------------------------------------------------
// 16-byte vector of T: 4 x float, 8 x half/bf16, 2 x double. A struct copy of an aligned
// 16-byte object compiles to a single 128-bit load / store.
// ---------------------------------------------------------------------------
template <typename T>
struct alignas(16) Vec {
  static constexpr int kN = 16 / sizeof(T);
  T v[kN];
};

__device__ __forceinline__ bool is_aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// ---------------------------------------------------------------------------
// Welford statistics.
//
// A (count, mean, M2) triple describes a set of samples, with M2 = sum((x - mean)^2).
// Triples merge exactly (Chan et al.) without ever forming sum(x^2), so the variance does not
// suffer the catastrophic cancellation of E[x^2] - E[x]^2 when |mean| >> std.
// ---------------------------------------------------------------------------
template <typename A>
struct Stats {
  A count;
  A mean;
  A m2;
};

/// Merges two partial statistics. Not bitwise commutative, so callers always pass the operands
/// in a fixed order (see warp_allreduce) to make every lane compute identical results.
template <typename A>
__device__ __forceinline__ Stats<A> merge(const Stats<A>& a, const Stats<A>& b) {
  const A n = a.count + b.count;
  if (n == A(0)) return Stats<A>{A(0), A(0), A(0)};
  const A delta = b.mean - a.mean;
  const A frac = b.count / n;
  Stats<A> r;
  r.count = n;
  r.mean = a.mean + delta * frac;
  r.m2 = a.m2 + b.m2 + delta * delta * a.count * frac;
  return r;
}

/// Streaming Welford update with one sample.
template <typename A>
__device__ __forceinline__ void welford_update(Stats<A>& s, A x) {
  s.count += A(1);
  const A delta = x - s.mean;
  s.mean += delta / s.count;
  s.m2 += delta * (x - s.mean);
}

/// Butterfly all-reduce of statistics across the 32 lanes of a warp. At every step both partner
/// lanes merge (lower lane, upper lane) in that order, so all 32 lanes end with bitwise-identical
/// results; the row's mean / rstd therefore never differ between lanes.
template <typename A>
__device__ __forceinline__ Stats<A> warp_allreduce(Stats<A> s) {
  const int lane = threadIdx.x & (kWarpSize - 1);
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    Stats<A> other;
    other.count = __shfl_xor_sync(kFullMask, s.count, offset);
    other.mean = __shfl_xor_sync(kFullMask, s.mean, offset);
    other.m2 = __shfl_xor_sync(kFullMask, s.m2, offset);
    s = ((lane & offset) == 0) ? merge(s, other) : merge(other, s);
  }
  return s;
}

/// All-reduce of statistics across a block of kWarps warps. `smem` holds kWarps entries.
/// Every thread merges the per-warp results in the same order, so results are identical.
template <typename A, int kWarps>
__device__ __forceinline__ Stats<A> block_allreduce(Stats<A> s, Stats<A>* smem) {
  s = warp_allreduce(s);
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = s;
  __syncthreads();
  Stats<A> result = smem[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) result = merge(result, smem[w]);
  __syncthreads();  // smem may be reused by the caller
  return result;
}

// ---------------------------------------------------------------------------
// Plain sums
// ---------------------------------------------------------------------------

/// Shared-memory tree sum (the textbook version, used by the baseline kernel). `red` holds
/// kBlock elements. All threads receive the result.
template <typename A, int kBlock>
__device__ __forceinline__ A block_sum_tree(A v, A* red) {
  const int tid = threadIdx.x;
  red[tid] = v;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const A result = red[0];
  __syncthreads();  // red may be reused by the caller
  return result;
}

/// Sums two values across a block of kWarps warps with shuffles; `smem` holds 2 * kWarps entries.
/// Addition is commutative, so the butterfly leaves all lanes with bitwise-identical sums.
template <typename A, int kWarps>
__device__ __forceinline__ void block_allreduce_sum2(A& a, A& b, A* smem) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    a += __shfl_xor_sync(kFullMask, a, offset);
    b += __shfl_xor_sync(kFullMask, b, offset);
  }
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    smem[warp] = a;
    smem[kWarps + warp] = b;
  }
  __syncthreads();
  A sa = smem[0];
  A sb = smem[kWarps];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    sa += smem[w];
    sb += smem[kWarps + w];
  }
  __syncthreads();  // smem may be reused by the caller
  a = sa;
  b = sb;
}

}  // namespace ln
