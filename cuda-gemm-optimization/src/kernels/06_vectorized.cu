// SPDX-License-Identifier: MIT
//
// Kernel 6 - Vectorised (128-bit) memory access
// ---------------------------------------------
// Builds on kernel 5 and replaces scalar 32-bit accesses with float4 (128-bit)
// accesses wherever the data layout allows:
//   * global -> shared tile loads of A and B   (LDG.128)
//   * shared memory reads of the register fragments (LDS.128)
//   * global stores of C                       (STG.128)
// Fewer, wider memory instructions reduce instruction-issue pressure and
// shared-memory wavefronts, leaving more issue slots for FMAs.
//
// Column mapping for conflict-free LDS.128 on B
//   A thread owns two groups of 4 adjacent columns of the tile:
//       cols [4*tc, 4*tc+3]  and  [BN/2 + 4*tc, BN/2 + 4*tc + 3]
//   Eight consecutive lanes then read 8 x 16 contiguous bytes = 128 B, which a
//   quarter-warp services without bank conflicts.
//
// Requirements (checked by the registry's `supports` function):
//   n % 4 == 0, k % 4 == 0 and 16-byte aligned A, B, C base pointers, so every
//   float4 access is aligned and entirely in or out of bounds.
#include "device_common.cuh"
#include "gemm/kernels.hpp"

namespace gemm {
namespace {

using dev::idx;

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;

__device__ __forceinline__ float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void store4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_vectorized(int m, int n, int k, float alpha, const float* __restrict__ a,
                     const float* __restrict__ b, float beta, float* __restrict__ c) {
  constexpr int kThreads = (BM * BN) / (TM * TN);
  constexpr int kThreadsN = BN / TN;
  constexpr int kAsPitch = BM + 4;  // keeps every row of the transposed tile 16-byte aligned

  static_assert(TN == 8 && TM % 4 == 0, "fragment loads are written for float4 groups");
  static_assert(BN == 2 * 4 * kThreadsN, "column mapping needs BN == 8 * (threads along N)");
  static_assert(BM * BK == 4 * kThreads, "one float4 of the A tile per thread");
  static_assert(BK * BN == 4 * kThreads, "one float4 of the B tile per thread");
  static_assert(BK % 4 == 0 && BN % 4 == 0, "tile rows must be float4 multiples");

  __shared__ __align__(16) float as[BK * kAsPitch];  // transposed: as[d * kAsPitch + row]
  __shared__ __align__(16) float bs[BK * BN];        // bs[d * BN + col]

  const int tid = threadIdx.x;
  const int thread_col = tid % kThreadsN;
  const int thread_row = tid / kThreadsN;

  const int a_row = tid / (BK / 4);        // one float4 along k per thread
  const int a_col = (tid % (BK / 4)) * 4;
  const int b_row = tid / (BN / 4);        // one float4 along n per thread
  const int b_col = (tid % (BN / 4)) * 4;

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM][TN] = {};
  float reg_m[TM];
  float reg_n[TN];

  for (int bk = 0; bk < k; bk += BK) {
    // ---- global -> shared (128-bit loads) --------------------------------
    float4 va = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int a_g_row = block_row + a_row;
    const int a_g_col = bk + a_col;
    if (a_g_row < m && a_g_col < k) va = load4(&a[idx(a_g_row, k, a_g_col)]);
    as[(a_col + 0) * kAsPitch + a_row] = va.x;  // transpose while storing
    as[(a_col + 1) * kAsPitch + a_row] = va.y;
    as[(a_col + 2) * kAsPitch + a_row] = va.z;
    as[(a_col + 3) * kAsPitch + a_row] = va.w;

    float4 vb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int b_g_row = bk + b_row;
    const int b_g_col = block_col + b_col;
    if (b_g_row < k && b_g_col < n) vb = load4(&b[idx(b_g_row, n, b_g_col)]);
    store4(&bs[b_row * BN + b_col], vb);
    __syncthreads();

    // ---- compute: shared -> registers (128-bit), then outer product ------
#pragma unroll
    for (int d = 0; d < BK; ++d) {
#pragma unroll
      for (int i4 = 0; i4 < TM / 4; ++i4) {
        const float4 v = load4(&as[d * kAsPitch + thread_row * TM + i4 * 4]);
        reg_m[i4 * 4 + 0] = v.x;
        reg_m[i4 * 4 + 1] = v.y;
        reg_m[i4 * 4 + 2] = v.z;
        reg_m[i4 * 4 + 3] = v.w;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {  // two groups of four columns, BN/2 apart
        const float4 v = load4(&bs[d * BN + h * (BN / 2) + thread_col * 4]);
        reg_n[h * 4 + 0] = v.x;
        reg_n[h * 4 + 1] = v.y;
        reg_n[h * 4 + 2] = v.z;
        reg_n[h * 4 + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(reg_m[i], reg_n[j], acc[i][j]);
      }
    }
    __syncthreads();
  }

  // ---- epilogue: C = alpha * acc + beta * C with 128-bit stores ------------
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int out_row = block_row + thread_row * TM + i;
    if (out_row >= m) continue;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int out_col = block_col + h * (BN / 2) + thread_col * 4;
      if (out_col >= n) continue;  // n % 4 == 0 => the whole float4 is in range
      float* c_ptr = &c[idx(out_row, n, out_col)];
      float4 old = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (beta != 0.0f) old = load4(c_ptr);  // never read C when beta == 0
      float4 out;
      out.x = dev::axpby(alpha, acc[i][h * 4 + 0], beta, old.x);
      out.y = dev::axpby(alpha, acc[i][h * 4 + 1], beta, old.y);
      out.z = dev::axpby(alpha, acc[i][h * 4 + 2], beta, old.z);
      out.w = dev::axpby(alpha, acc[i][h * 4 + 3], beta, old.w);
      store4(c_ptr, out);
    }
  }
}

}  // namespace

void launch_sgemm_vectorized(const GemmArgs& g, cudaStream_t stream) {
  const dim3 block((kBM * kBN) / (kTM * kTN));
  const dim3 grid(ceil_div(g.n, kBN), ceil_div(g.m, kBM));
  sgemm_vectorized<kBM, kBN, kBK, kTM, kTN>
      <<<grid, block, 0, stream>>>(g.m, g.n, g.k, g.alpha, g.a, g.b, g.beta, g.c);
}

}  // namespace gemm
