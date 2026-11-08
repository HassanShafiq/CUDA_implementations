// SPDX-License-Identifier: MIT
//
// Kernel 3 - Shared-memory cache blocking
// ---------------------------------------
// The block cooperatively stages a BS x BS tile of A and of B in shared memory,
// all threads reuse it BS times, then the block slides to the next k-tile.
//
// Effect: each global-memory value is fetched once per tile instead of once
// per thread, raising arithmetic intensity from 0.25 to BS/4 FLOP per byte
// of global traffic (derivation in docs/OPTIMIZATION_NOTES.md).
//
// Shared-memory access pattern (no bank conflicts):
//   tile_a[ty][i] : all lanes of a warp share ty and i  -> broadcast
//   tile_b[i][tx] : lanes read consecutive words         -> one per bank
//
// Out-of-range tile elements are loaded as zero, so any (m, n, k) is valid.
#include "device_common.cuh"
#include "gemm/kernels.hpp"

namespace gemm {
namespace {

using dev::idx;
using dev::store_axpby;

constexpr int kTile = 32;

template <int BS>
__global__ void sgemm_smem_tiled(int m, int n, int k, float alpha, const float* __restrict__ a,
                                 const float* __restrict__ b, float beta, float* __restrict__ c) {
  __shared__ float tile_a[BS][BS];
  __shared__ float tile_b[BS][BS];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * BS + ty;
  const int col = blockIdx.x * BS + tx;

  float acc = 0.0f;
  for (int t = 0; t < k; t += BS) {
    const int a_col = t + tx;
    const int b_row = t + ty;
    tile_a[ty][tx] = (row < m && a_col < k) ? a[idx(row, k, a_col)] : 0.0f;
    tile_b[ty][tx] = (b_row < k && col < n) ? b[idx(b_row, n, col)] : 0.0f;
    __syncthreads();  // tile fully written before anyone reads it

#pragma unroll
    for (int i = 0; i < BS; ++i) {
      acc = fmaf(tile_a[ty][i], tile_b[i][tx], acc);
    }
    __syncthreads();  // everyone done reading before the tile is overwritten
  }

  if (row < m && col < n) {
    store_axpby(c, idx(row, n, col), alpha, acc, beta);
  }
}

}  // namespace

void launch_sgemm_smem_tiled(const GemmArgs& g, cudaStream_t stream) {
  const dim3 block(kTile, kTile);
  const dim3 grid(ceil_div(g.n, kTile), ceil_div(g.m, kTile));
  sgemm_smem_tiled<kTile><<<grid, block, 0, stream>>>(g.m, g.n, g.k, g.alpha, g.a, g.b, g.beta, g.c);
}

}  // namespace gemm
