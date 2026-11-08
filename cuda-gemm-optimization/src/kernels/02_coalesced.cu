// SPDX-License-Identifier: MIT
//
// Kernel 2 - Global memory coalescing
// -----------------------------------
// Same arithmetic as kernel 1; only the thread->element mapping changes.
// Consecutive threadIdx.x now map to consecutive COLUMNS of C, so within a warp:
//   * B[i][col]   -> 32 consecutive floats  (one coalesced 128-byte request)
//   * C[row][col] -> 32 consecutive floats  (coalesced store)
//   * A[row][i]   -> same address for all lanes (hardware broadcast)
// Result: the memory system moves far fewer sectors for the same useful data.
#include "device_common.cuh"
#include "gemm/kernels.hpp"

namespace gemm {
namespace {

using dev::idx;
using dev::store_axpby;

constexpr int kBlock = 32;

__global__ void sgemm_coalesced(int m, int n, int k, float alpha, const float* __restrict__ a,
                                const float* __restrict__ b, float beta, float* __restrict__ c) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // fast index -> columns (coalesced)
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc = fmaf(a[idx(row, k, i)], b[idx(i, n, col)], acc);
  }
  store_axpby(c, idx(row, n, col), alpha, acc, beta);
}

}  // namespace

void launch_sgemm_coalesced(const GemmArgs& g, cudaStream_t stream) {
  const dim3 block(kBlock, kBlock);
  const dim3 grid(ceil_div(g.n, kBlock), ceil_div(g.m, kBlock));
  sgemm_coalesced<<<grid, block, 0, stream>>>(g.m, g.n, g.k, g.alpha, g.a, g.b, g.beta, g.c);
}

}  // namespace gemm
