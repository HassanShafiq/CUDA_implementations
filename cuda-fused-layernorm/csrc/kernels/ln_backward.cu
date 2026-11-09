// SPDX-License-Identifier: MIT
//
// LayerNorm backward (fused, deterministic).
//
// With xhat = (x - mean) * rstd,  y = gamma * xhat + beta  and  dxhat = dy * gamma:
//
//   dbeta  = sum over rows of dy
//   dgamma = sum over rows of dy * xhat
//   dx     = rstd * ( dxhat - mean_c(dxhat) - xhat * mean_c(dxhat * xhat) )
//
// where mean_c is the average over the normalised dimension of one row (derivation in
// docs/OPTIMIZATION_NOTES.md). The statistics `mean` and `rstd` are not recomputed: they are
// the [rows] vectors saved by the forward pass.
//
// Two quantities need very different parallelisation:
//   * dx needs two reductions ALONG a row (sum of dxhat and of dxhat * xhat);
//   * dgamma / dbeta need reductions ACROSS rows, into one value per column.
//
// A single fused kernel handles both. Block b processes rows b, b + G, b + 2G, ... (G = number of
// blocks, at most 256). For each row it computes dx, and every thread accumulates dgamma / dbeta
// for the columns it owns into a block-private accumulator (shared memory when it fits, otherwise
// a private slice of global scratch). Each block ends with one partial [2, cols] result; a second
// tiny kernel sums the G partials per column in a fixed order.
//
// Properties: no atomics, hence results are bitwise reproducible run to run, and G does not depend
// on the GPU, so they are also reproducible across devices.
#include <cstddef>
#include <stdexcept>
#include <string>

#include "layernorm.h"
#include "ln_common.cuh"

namespace ln {
namespace {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / kWarpSize;
constexpr int kMaxBlocks = 256;                       // upper bound on the number of partials
constexpr std::size_t kStaticSmemLimit = 48 * 1024;   // dynamic shared memory without opt-in

int num_blocks(int rows) { return rows < kMaxBlocks ? rows : kMaxBlocks; }

template <typename T>
struct BwdParams {
  const T* dy;
  const T* x;
  const T* gamma;           // may be null
  const acc_t<T>* mean;
  const acc_t<T>* rstd;
  T* dx;
  acc_t<T>* partial;        // [num_blocks, 2, cols] scratch; null when no parameter gradients
  int rows;
  int cols;
};

// ---------------------------------------------------------------------------
// Fused dx + partial dgamma/dbeta. One block cooperates on one row at a time; thread t owns
// columns t, t + kBlock, ... in every row, so its accumulator slots are never shared
// (no atomics, no synchronisation needed for the accumulation).
// ---------------------------------------------------------------------------
template <typename T, bool kUseSmem>
__global__ void __launch_bounds__(kBlock) ln_bwd_fused(BwdParams<T> p) {
  using C = Traits<T>;
  using A = acc_t<T>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ A red[2 * kWarps];

  const int tid = threadIdx.x;
  const bool need_params = (p.partial != nullptr);

  A* acc_g = nullptr;  // block-private dgamma accumulator, [cols]
  A* acc_b = nullptr;  // block-private dbeta accumulator,  [cols]
  if (need_params) {
    if constexpr (kUseSmem) {
      acc_g = reinterpret_cast<A*>(smem_raw);
    } else {
      acc_g = p.partial + static_cast<std::size_t>(blockIdx.x) * 2 * p.cols;
    }
    acc_b = acc_g + p.cols;
    for (int c = tid; c < p.cols; c += kBlock) {
      acc_g[c] = A(0);
      acc_b[c] = A(0);
    }
  }

  const A inv_cols = A(1) / static_cast<A>(p.cols);

  for (int row = blockIdx.x; row < p.rows; row += gridDim.x) {
    const std::size_t base = static_cast<std::size_t>(row) * p.cols;
    const T* x = p.x + base;
    const T* dy = p.dy + base;
    T* dx = p.dx + base;
    const A mean = p.mean[row];
    const A rstd = p.rstd[row];

    // Pass 1: row reductions s1 = sum(dxhat), s2 = sum(dxhat * xhat); accumulate parameter grads.
    A s1 = A(0);
    A s2 = A(0);
    for (int c = tid; c < p.cols; c += kBlock) {
      const A xhat = (C::to_acc(x[c]) - mean) * rstd;
      const A dyv = C::to_acc(dy[c]);
      const A g = (p.gamma != nullptr) ? C::to_acc(p.gamma[c]) : A(1);
      const A dxhat = dyv * g;
      s1 += dxhat;
      s2 += dxhat * xhat;
      if (need_params) {
        acc_g[c] += dyv * xhat;
        acc_b[c] += dyv;
      }
    }
    block_allreduce_sum2<A, kWarps>(s1, s2, red);
    const A c1 = s1 * inv_cols;
    const A c2 = s2 * inv_cols;

    // Pass 2: dx = rstd * (dxhat - c1 - xhat * c2). x and dy are re-read; they were just
    // touched, so the re-read is served by L1 / L2 for typical row lengths.
    for (int c = tid; c < p.cols; c += kBlock) {
      const A xhat = (C::to_acc(x[c]) - mean) * rstd;
      const A g = (p.gamma != nullptr) ? C::to_acc(p.gamma[c]) : A(1);
      const A dxhat = C::to_acc(dy[c]) * g;
      dx[c] = C::from_acc(rstd * (dxhat - c1 - xhat * c2));
    }
  }

  // Publish this block's partial sums (global-scratch accumulators are already in place).
  if constexpr (kUseSmem) {
    if (need_params) {
      A* out = p.partial + static_cast<std::size_t>(blockIdx.x) * 2 * p.cols;
      for (int c = tid; c < p.cols; c += kBlock) {
        out[c] = acc_g[c];
        out[p.cols + c] = acc_b[c];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Second stage: one thread per column sums the per-block partials in a fixed order.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void ln_bwd_reduce_params(const acc_t<T>* __restrict__ partial, T* __restrict__ dgamma,
                                     T* __restrict__ dbeta, int blocks, int cols) {
  using A = acc_t<T>;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  A g = A(0);
  A b = A(0);
  for (int i = 0; i < blocks; ++i) {
    const A* part = partial + static_cast<std::size_t>(i) * 2 * cols;
    g += part[c];
    b += part[cols + c];
  }
  if (dgamma != nullptr) dgamma[c] = Traits<T>::from_acc(g);
  if (dbeta != nullptr) dbeta[c] = Traits<T>::from_acc(b);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
template <typename T>
void launch_backward(const BackwardCall& c, cudaStream_t stream) {
  using A = acc_t<T>;
  const bool need_params = (c.dgamma != nullptr || c.dbeta != nullptr);

  BwdParams<T> p;
  p.dy = static_cast<const T*>(c.dy);
  p.x = static_cast<const T*>(c.x);
  p.gamma = static_cast<const T*>(c.gamma);
  p.mean = static_cast<const A*>(c.mean);
  p.rstd = static_cast<const A*>(c.rstd);
  p.dx = static_cast<T*>(c.dx);
  p.partial = need_params ? static_cast<A*>(c.workspace) : nullptr;
  p.rows = c.rows;
  p.cols = c.cols;

  const int blocks = num_blocks(c.rows);
  const std::size_t smem_bytes = 2 * static_cast<std::size_t>(c.cols) * sizeof(A);
  if (smem_bytes <= kStaticSmemLimit) {
    ln_bwd_fused<T, true><<<blocks, kBlock, need_params ? smem_bytes : 0, stream>>>(p);
  } else {
    ln_bwd_fused<T, false><<<blocks, kBlock, 0, stream>>>(p);
  }
  LN_CUDA_CHECK(cudaGetLastError());

  if (need_params) {
    ln_bwd_reduce_params<T><<<ceil_div(c.cols, kBlock), kBlock, 0, stream>>>(
        p.partial, static_cast<T*>(c.dgamma), static_cast<T*>(c.dbeta), blocks, c.cols);
    LN_CUDA_CHECK(cudaGetLastError());
  }
}

void validate(const BackwardCall& c) {
  if (c.rows < 0) throw std::invalid_argument("rows must be >= 0");
  if (c.cols <= 0) throw std::invalid_argument("cols must be > 0");
  if (c.rows > 0 && (c.dy == nullptr || c.x == nullptr || c.mean == nullptr || c.rstd == nullptr ||
                     c.dx == nullptr)) {
    throw std::invalid_argument("dy, x, mean, rstd and dx must not be null");
  }
  if (c.rows > 0 && (c.dgamma != nullptr || c.dbeta != nullptr) && c.workspace == nullptr) {
    throw std::invalid_argument("a workspace of backward_workspace_bytes() is required for dgamma / dbeta");
  }
}

}  // namespace

std::size_t backward_workspace_bytes(DType dtype, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return 0;
  return 2 * static_cast<std::size_t>(num_blocks(rows)) * static_cast<std::size_t>(cols) * stats_size(dtype);
}

void layernorm_backward(const BackwardCall& call, cudaStream_t stream) {
  validate(call);
  if (call.rows == 0) {
    // No rows: dx is empty and the parameter gradients are sums over nothing, i.e. zero.
    if (call.dgamma != nullptr || call.dbeta != nullptr) {
      const std::size_t bytes = static_cast<std::size_t>(call.cols) * dtype_size(call.dtype);
      if (call.dgamma != nullptr) LN_CUDA_CHECK(cudaMemsetAsync(call.dgamma, 0, bytes, stream));
      if (call.dbeta != nullptr) LN_CUDA_CHECK(cudaMemsetAsync(call.dbeta, 0, bytes, stream));
    }
    return;
  }
  switch (call.dtype) {
    case DType::kFloat32: launch_backward<float>(call, stream); break;
    case DType::kFloat16: launch_backward<__half>(call, stream); break;
    case DType::kBFloat16: launch_backward<__nv_bfloat16>(call, stream); break;
    case DType::kFloat64: launch_backward<double>(call, stream); break;
  }
}

}  // namespace ln
