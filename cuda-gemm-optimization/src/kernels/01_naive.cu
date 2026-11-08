// SPDX-License-Identifier: MIT
//
// Kernel 1 - Naive SGEMM
// ----------------------
// One thread computes one element of C by walking a row of A and a column of
// B straight from global memory.
//
// Deliberate weakness (this is the baseline of the ladder): the thread->data
// mapping puts consecutive threadIdx.x on consecutive ROWS of C. Threads of a
// warp therefore read A (and write C) with a stride of `k` (resp. `n`)
// elements, so every warp-wide access touches many separate 32-byte sectors
// instead of one contiguous 128-byte line. See docs/OPTIMIZATION_NOTES.md.
#include "device_common.cuh"
#include "gemm/kernels.hpp"

namespace gemm {
namespace {

using dev::idx;
using dev::store_axpby;

constexpr int kBlock = 32;  // 32 x 32 = 1024 threads per block

__global__ void sgemm_naive(int m, int n, int k, float alpha, const float* __restrict__ a,
                            const float* __restrict__ b, float beta, float* __restrict__ c) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;  // fast index -> rows (uncoalesced)
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc = fmaf(a[idx(row, k, i)], b[idx(i, n, col)], acc);
  }
  store_axpby(c, idx(row, n, col), alpha, acc, beta);
}

}  // namespace

void launch_sgemm_naive(const GemmArgs& g, cudaStream_t stream) {
  const dim3 block(kBlock, kBlock);
  const dim3 grid(ceil_div(g.m, kBlock), ceil_div(g.n, kBlock));
  sgemm_naive<<<grid, block, 0, stream>>>(g.m, g.n, g.k, g.alpha, g.a, g.b, g.beta, g.c);
}

}  // namespace gemm
