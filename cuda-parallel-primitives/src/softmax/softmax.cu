// SPDX-License-Identifier: MIT
//
// Row-wise softmax, numerically stable (subtract the row maximum), four variants.
//
//   1 naive           one thread per row, three sequential passes over the row
//   2 block_smem      one block per row, shared-memory tree reductions, three passes
//   3 warp_per_row    one warp per row, register shuffles only (no __syncthreads)
//   4 online_block    one block per row, single-pass online (max, sum), two passes in total
//
// Global-memory view for a row of `cols` floats:
//   variants 1-3: max pass + sum pass + write pass = 3 reads + 1 write
//   variant 4   : (max,sum) pass + write pass      = 2 reads + 1 write
// For short rows the repeated reads are served by L1/L2, so the benefit shows up
// mostly on long rows; measure on your own hardware.
//
// Semantics (also implemented by the CPU reference): -inf entries get probability 0
// and an all--inf row produces zeros instead of NaN, which makes masked attention
// rows safe.
#include <cstddef>
#include <vector>

#include "device_utils.cuh"
#include "prim/cuda_utils.hpp"
#include "prim/primitives.hpp"

namespace prim {
namespace {

constexpr int kBlock = 256;                       // threads per block, block-per-row variants
constexpr int kWarps = kBlock / dev::kWarpSize;
constexpr int kNaiveBlock = 128;                  // rows per block, naive variant
constexpr int kWarpRowsPerBlock = 4;              // rows per block, warp-per-row variant
constexpr int kWarpRowBlock = kWarpRowsPerBlock * dev::kWarpSize;

__device__ __forceinline__ std::size_t row_offset(int row, int cols) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
}

// ---------------------------------------------------------------------------
// Variant 1: one thread per row. Parallelism is limited to `rows` threads, and
// neighbouring threads read addresses `cols` floats apart (uncoalesced).
// ---------------------------------------------------------------------------
__global__ void softmax_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  const int row = blockIdx.x * kNaiveBlock + threadIdx.x;
  if (row >= rows) return;
  const float* x = in + row_offset(row, cols);
  float* y = out + row_offset(row, cols);

  float row_max = -INFINITY;
  for (int c = 0; c < cols; ++c) row_max = fmaxf(row_max, x[c]);
  float sum = 0.0f;
  for (int c = 0; c < cols; ++c) sum += dev::exp_shifted(x[c], row_max);
  const float inv = dev::safe_inverse(sum);
  for (int c = 0; c < cols; ++c) y[c] = dev::exp_shifted(x[c], row_max) * inv;
}

// ---------------------------------------------------------------------------
// Variant 2: one block per row. Threads stride across the row (coalesced), then
// combine partial results with a shared-memory tree reduction (twice: max, sum).
// ---------------------------------------------------------------------------
__global__ void softmax_block_smem(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float red[kBlock];
  const int tid = threadIdx.x;
  const float* x = in + row_offset(blockIdx.x, cols);
  float* y = out + row_offset(blockIdx.x, cols);

  float local_max = -INFINITY;
  for (int c = tid; c < cols; c += kBlock) local_max = fmaxf(local_max, x[c]);
  red[tid] = local_max;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float row_max = red[0];
  __syncthreads();  // everyone has read red[0] before it is reused

  float local_sum = 0.0f;
  for (int c = tid; c < cols; c += kBlock) local_sum += dev::exp_shifted(x[c], row_max);
  red[tid] = local_sum;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float inv = dev::safe_inverse(red[0]);

  for (int c = tid; c < cols; c += kBlock) y[c] = dev::exp_shifted(x[c], row_max) * inv;
}

// ---------------------------------------------------------------------------
// Variant 3: one warp per row. The row is covered by 32 lanes in a strided loop and
// reduced with register shuffles: no shared memory and no __syncthreads at all.
// Best for short and medium rows with many rows in flight.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kWarpRowBlock)
    softmax_warp_per_row(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  const int lane = threadIdx.x % dev::kWarpSize;
  const int warp = threadIdx.x / dev::kWarpSize;
  const int row = blockIdx.x * kWarpRowsPerBlock + warp;
  if (row >= rows) return;  // uniform across the warp, so shuffles below stay convergent
  const float* x = in + row_offset(row, cols);
  float* y = out + row_offset(row, cols);

  float local_max = -INFINITY;
  for (int c = lane; c < cols; c += dev::kWarpSize) local_max = fmaxf(local_max, x[c]);
  const float row_max = dev::warp_allreduce_max(local_max);

  float local_sum = 0.0f;
  for (int c = lane; c < cols; c += dev::kWarpSize) local_sum += dev::exp_shifted(x[c], row_max);
  const float inv = dev::safe_inverse(dev::warp_allreduce_sum(local_sum));

  for (int c = lane; c < cols; c += dev::kWarpSize) y[c] = dev::exp_shifted(x[c], row_max) * inv;
}

// ---------------------------------------------------------------------------
// Variant 4: online softmax (Milakov & Gimelshein, 2018). The maximum and the sum of
// exponentials are computed in ONE pass: whenever the running maximum grows, the sum
// accumulated so far is rescaled by exp(old_max - new_max). Partial (max, sum) pairs
// from threads, warps and the block merge with the same associative rule.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kBlock)
    softmax_online_block(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float warp_max[kWarps];
  __shared__ float warp_sum[kWarps];
  __shared__ float row_stats[2];  // [0] = row max, [1] = 1 / row sum

  const int tid = threadIdx.x;
  const int lane = tid % dev::kWarpSize;
  const int warp = tid / dev::kWarpSize;
  const float* x = in + row_offset(blockIdx.x, cols);
  float* y = out + row_offset(blockIdx.x, cols);

  dev::MaxSum acc{-INFINITY, 0.0f};
  for (int c = tid; c < cols; c += kBlock) acc = dev::accumulate(acc, x[c]);

  acc = dev::warp_allreduce_max_sum(acc);
  if (lane == 0) {
    warp_max[warp] = acc.max;
    warp_sum[warp] = acc.sum;
  }
  __syncthreads();

  if (warp == 0) {
    dev::MaxSum v{-INFINITY, 0.0f};
    if (lane < kWarps) v = dev::MaxSum{warp_max[lane], warp_sum[lane]};
    v = dev::warp_allreduce_max_sum(v);
    if (lane == 0) {
      row_stats[0] = v.max;
      row_stats[1] = dev::safe_inverse(v.sum);
    }
  }
  __syncthreads();

  const float row_max = row_stats[0];
  const float inv = row_stats[1];
  for (int c = tid; c < cols; c += kBlock) y[c] = dev::exp_shifted(x[c], row_max) * inv;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
bool validate(int rows, int cols) {
  check_count(rows, "rows");
  check_count(cols, "cols");
  check_count(static_cast<long long>(rows) * cols, "rows * cols");
  return rows > 0 && cols > 0;  // nothing to do otherwise
}

void run_naive(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
  if (!validate(rows, cols)) return;
  softmax_naive<<<ceil_div(rows, kNaiveBlock), kNaiveBlock, 0, stream>>>(in, out, rows, cols);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void run_block_smem(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
  if (!validate(rows, cols)) return;
  softmax_block_smem<<<rows, kBlock, 0, stream>>>(in, out, rows, cols);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void run_warp_per_row(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
  if (!validate(rows, cols)) return;
  softmax_warp_per_row<<<ceil_div(rows, kWarpRowsPerBlock), kWarpRowBlock, 0, stream>>>(in, out, rows, cols);
  PRIM_CUDA_CHECK(cudaGetLastError());
}
void run_online_block(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
  if (!validate(rows, cols)) return;
  softmax_online_block<<<rows, kBlock, 0, stream>>>(in, out, rows, cols);
  PRIM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

const std::vector<SoftmaxKernel>& softmax_registry() {
  static const std::vector<SoftmaxKernel> registry = {
      {1, "naive", "one thread per row, three passes", run_naive},
      {2, "block_smem", "block per row, shared-memory tree reductions", run_block_smem},
      {3, "warp_per_row", "warp per row, shuffle reductions, no block syncs", run_warp_per_row},
      {4, "online_block", "block per row, single-pass online max/sum", run_online_block},
  };
  return registry;
}

}  // namespace prim
