// SPDX-License-Identifier: MIT
//
// Kernel 5 - 2D register tiling (outer-product formulation)
// ---------------------------------------------------------
// Each thread owns a TM x TN patch of C held entirely in registers. For every
// k-step it loads TM values of A and TN values of B from shared memory into
// registers and performs TM*TN FMAs (an outer product):
//     16 shared loads per 64 FMAs  (0.25 loads/FMA; kernels 3 and 4: 2.0 and 1.125)
//
// Tile configuration: BM = BN = 128, BK = 8, TM = TN = 8
//   threads per block = (BM * BN) / (TM * TN) = 256
//
// Bank-conflict avoidance
//  * B columns are assigned with a stride of (BN / TN) = 16 rather than as 8
//    contiguous columns per thread, so lanes read consecutive words
//    (bs[d][thread_col + j * 16]) - one bank each.
//  * The A tile is stored TRANSPOSED (as[d][row]) so a thread's TM values are
//    contiguous and a warp reads only two distinct addresses (broadcast).
//  * The transposed store would otherwise hit 8-way bank conflicts; a pitch of
//    BM + 4 words spreads the 8 k-columns over different banks.
#include "device_common.cuh"
#include "gemm/kernels.hpp"

namespace gemm {
namespace {

using dev::idx;
using dev::store_axpby;

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_regtile_2d(int m, int n, int k, float alpha, const float* __restrict__ a,
                     const float* __restrict__ b, float beta, float* __restrict__ c) {
  constexpr int kThreads = (BM * BN) / (TM * TN);
  constexpr int kThreadsN = BN / TN;       // threads along the N direction of the tile
  constexpr int kAStride = kThreads / BK;  // rows of the A tile covered per load pass
  constexpr int kBStride = kThreads / BN;  // rows of the B tile covered per load pass
  constexpr int kAsPitch = BM + 4;         // padded pitch of the transposed A tile

  static_assert(kThreads % BK == 0 && BM % kAStride == 0, "A tile must divide evenly among threads");
  static_assert(kThreads % BN == 0 && BK % kBStride == 0, "B tile must divide evenly among threads");

  __shared__ float as[BK * kAsPitch];  // as[d * kAsPitch + row]  (transposed)
  __shared__ float bs[BK * BN];        // bs[d * BN + col]

  const int tid = threadIdx.x;
  const int thread_col = tid % kThreadsN;
  const int thread_row = tid / kThreadsN;

  const int a_col = tid % BK;  // k offset inside the A tile
  const int a_row = tid / BK;
  const int b_col = tid % BN;
  const int b_row = tid / BN;

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM][TN] = {};
  float reg_m[TM];
  float reg_n[TN];

  for (int bk = 0; bk < k; bk += BK) {
    // Stage A (transposed) and B; out-of-range elements become zero.
#pragma unroll
    for (int off = 0; off < BM; off += kAStride) {
      const int r = a_row + off;
      const int g_row = block_row + r;
      const int g_col = bk + a_col;
      as[a_col * kAsPitch + r] = (g_row < m && g_col < k) ? a[idx(g_row, k, g_col)] : 0.0f;
    }
#pragma unroll
    for (int off = 0; off < BK; off += kBStride) {
      const int r = b_row + off;
      const int g_row = bk + r;
      const int g_col = block_col + b_col;
      bs[r * BN + b_col] = (g_row < k && g_col < n) ? b[idx(g_row, n, g_col)] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int d = 0; d < BK; ++d) {
#pragma unroll
      for (int i = 0; i < TM; ++i) reg_m[i] = as[d * kAsPitch + thread_row * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) reg_n[j] = bs[d * BN + thread_col + j * kThreadsN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(reg_m[i], reg_n[j], acc[i][j]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int out_row = block_row + thread_row * TM + i;
    if (out_row >= m) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int out_col = block_col + thread_col + j * kThreadsN;
      if (out_col < n) store_axpby(c, idx(out_row, n, out_col), alpha, acc[i][j], beta);
    }
  }
}

}  // namespace

void launch_sgemm_regtile_2d(const GemmArgs& g, cudaStream_t stream) {
  const dim3 block((kBM * kBN) / (kTM * kTN));
  const dim3 grid(ceil_div(g.n, kBN), ceil_div(g.m, kBM));
  sgemm_regtile_2d<kBM, kBN, kBK, kTM, kTN>
      <<<grid, block, 0, stream>>>(g.m, g.n, g.k, g.alpha, g.a, g.b, g.beta, g.c);
}

}  // namespace gemm
