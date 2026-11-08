// SPDX-License-Identifier: MIT
//
// Kernel 4 - 1D register blocking ("block tiling")
// ------------------------------------------------
// Each thread now computes TM vertically adjacent elements of C instead of one.
// Per k-step a thread issues TM loads from A's tile and ONE load from B's tile
// for TM FMAs (9 shared loads / 8 FMAs) versus 2 loads / 1 FMA in kernel 3.
// The B value is kept in a register and reused TM times.
//
// Tile configuration: BM = BN = 64, BK = 8, TM = 8
//   threads per block = BM * BN / TM = 512
//   A tile (BM x BK = 512 floats) and B tile (BK x BN = 512 floats) are each
//   loaded by exactly one element per thread -> no inner load loops needed.
//
// Shared-memory access pattern during compute:
//   bs[d][thread_col]      : consecutive lanes -> consecutive banks
//   as[row][d] (row fixed per warp since BN = 64 > warp size) -> broadcast
#include "device_common.cuh"
#include "gemm/kernels.hpp"

namespace gemm {
namespace {

using dev::idx;
using dev::store_axpby;

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 8;
constexpr int kTM = 8;

template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
    sgemm_blocktile_1d(int m, int n, int k, float alpha, const float* __restrict__ a,
                       const float* __restrict__ b, float beta, float* __restrict__ c) {
  constexpr int kThreads = (BM * BN) / TM;
  static_assert(BM * BK == kThreads, "each thread must load exactly one element of the A tile");
  static_assert(BK * BN == kThreads, "each thread must load exactly one element of the B tile");

  __shared__ float as[BM * BK];
  __shared__ float bs[BK * BN];

  const int tid = threadIdx.x;
  const int thread_col = tid % BN;  // column of C owned by this thread (within the tile)
  const int thread_row = tid / BN;  // index of the TM-high strip owned by this thread

  const int a_col = tid % BK;  // load mapping, A tile
  const int a_row = tid / BK;
  const int b_col = tid % BN;  // load mapping, B tile
  const int b_row = tid / BN;

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM] = {};

  for (int bk = 0; bk < k; bk += BK) {
    const int a_g_row = block_row + a_row;
    const int a_g_col = bk + a_col;
    as[a_row * BK + a_col] = (a_g_row < m && a_g_col < k) ? a[idx(a_g_row, k, a_g_col)] : 0.0f;

    const int b_g_row = bk + b_row;
    const int b_g_col = block_col + b_col;
    bs[b_row * BN + b_col] = (b_g_row < k && b_g_col < n) ? b[idx(b_g_row, n, b_g_col)] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int d = 0; d < BK; ++d) {
      const float b_val = bs[d * BN + thread_col];  // cached in a register, reused TM times
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        acc[i] = fmaf(as[(thread_row * TM + i) * BK + d], b_val, acc[i]);
      }
    }
    __syncthreads();
  }

  const int out_col = block_col + thread_col;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int out_row = block_row + thread_row * TM + i;
    if (out_row < m && out_col < n) {
      store_axpby(c, idx(out_row, n, out_col), alpha, acc[i], beta);
    }
  }
}

}  // namespace

void launch_sgemm_blocktile_1d(const GemmArgs& g, cudaStream_t stream) {
  const dim3 block((kBM * kBN) / kTM);
  const dim3 grid(ceil_div(g.n, kBN), ceil_div(g.m, kBM));
  sgemm_blocktile_1d<kBM, kBN, kBK, kTM>
      <<<grid, block, 0, stream>>>(g.m, g.n, g.k, g.alpha, g.a, g.b, g.beta, g.c);
}

}  // namespace gemm
