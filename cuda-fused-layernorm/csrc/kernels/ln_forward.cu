// SPDX-License-Identifier: MIT
//
// LayerNorm forward: an optimisation ladder of four kernels.
//
// LayerNorm is memory-bound: ~10 FLOPs per element against 2 * sizeof(T) bytes of mandatory
// traffic (read x, write y). The ladder therefore reduces how often each row is read from
// global memory and how much synchronisation the row reduction needs:
//
//   1 two_pass     block per row, shared-memory tree sums: mean pass, variance pass, write pass
//                  -> 3 reads + 1 write of the row; two cancellation-safe passes
//   2 welford      block per row, one streaming Welford pass (mean and variance together) with
//                  shuffle-based merging, then a write pass -> 2 reads + 1 write
//   3 warp_row     one warp per row (4 rows per block); the row lives in REGISTERS, loaded once
//                  with 128-bit loads; no __syncthreads -> 1 read + 1 write (the minimum)
//   4 block_fused  same register-cached design with a whole block per row, for rows that do not
//                  fit in one warp's registers (up to 8192 fp32 elements)
//
// Kernels 3 and 4 need cols % (16 / sizeof(T)) == 0, 16-byte aligned pointers and a bounded row
// length; `select_forward_kernel` falls back to kernel 2 otherwise.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "layernorm.h"
#include "ln_common.cuh"

namespace ln {
namespace {

constexpr int kBlock = 256;                  // threads per block in block-per-row kernels
constexpr int kWarps = kBlock / kWarpSize;
constexpr int kChunks = 8;                   // 16-byte chunks per thread held in registers
constexpr int kRowsPerWarpBlock = 4;         // rows per block in the warp-per-row kernel

template <typename T>
struct FwdParams {
  const T* x;
  const T* gamma;  // may be null
  const T* beta;   // may be null
  T* y;
  acc_t<T>* mean;  // may be null
  acc_t<T>* rstd;  // may be null
  int rows;
  int cols;
  acc_t<T> eps;
};

// ---------------------------------------------------------------------------
// Kernel 1: two-pass, shared-memory tree reductions.
// Subtracting the mean before squaring makes the variance numerically safe, at the cost of
// reading the row three times (mean, variance, normalise).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void __launch_bounds__(kBlock) ln_fwd_two_pass(FwdParams<T> p) {
  using C = Traits<T>;
  using A = acc_t<T>;
  __shared__ A red[kBlock];

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const T* x = p.x + static_cast<std::size_t>(row) * p.cols;
  T* y = p.y + static_cast<std::size_t>(row) * p.cols;
  const A inv_cols = A(1) / static_cast<A>(p.cols);

  A sum = A(0);
  for (int c = tid; c < p.cols; c += kBlock) sum += C::to_acc(x[c]);
  const A mean = block_sum_tree<A, kBlock>(sum, red) * inv_cols;

  A sq = A(0);
  for (int c = tid; c < p.cols; c += kBlock) {
    const A d = C::to_acc(x[c]) - mean;
    sq += d * d;
  }
  const A var = block_sum_tree<A, kBlock>(sq, red) * inv_cols;
  const A rstd = rsqrt_acc(var + p.eps);

  for (int c = tid; c < p.cols; c += kBlock) {
    const A g = (p.gamma != nullptr) ? C::to_acc(p.gamma[c]) : A(1);
    const A b = (p.beta != nullptr) ? C::to_acc(p.beta[c]) : A(0);
    y[c] = C::from_acc((C::to_acc(x[c]) - mean) * rstd * g + b);
  }
  if (tid == 0) {
    if (p.mean != nullptr) p.mean[row] = mean;
    if (p.rstd != nullptr) p.rstd[row] = rstd;
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: single-pass Welford statistics.
// Each thread folds its strided elements into a (count, mean, M2) triple; triples are merged
// across lanes and warps without a separate variance pass. 2 reads + 1 write per row.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void __launch_bounds__(kBlock) ln_fwd_welford(FwdParams<T> p) {
  using C = Traits<T>;
  using A = acc_t<T>;
  __shared__ Stats<A> warp_stats[kWarps];

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const T* x = p.x + static_cast<std::size_t>(row) * p.cols;
  T* y = p.y + static_cast<std::size_t>(row) * p.cols;

  Stats<A> local{A(0), A(0), A(0)};
  for (int c = tid; c < p.cols; c += kBlock) welford_update(local, C::to_acc(x[c]));
  const Stats<A> s = block_allreduce<A, kWarps>(local, warp_stats);

  const A mean = s.mean;
  const A rstd = rsqrt_acc(s.m2 / s.count + p.eps);

  for (int c = tid; c < p.cols; c += kBlock) {
    const A g = (p.gamma != nullptr) ? C::to_acc(p.gamma[c]) : A(1);
    const A b = (p.beta != nullptr) ? C::to_acc(p.beta[c]) : A(0);
    y[c] = C::from_acc((C::to_acc(x[c]) - mean) * rstd * g + b);
  }
  if (tid == 0) {
    if (p.mean != nullptr) p.mean[row] = mean;
    if (p.rstd != nullptr) p.rstd[row] = rstd;
  }
}

// ---------------------------------------------------------------------------
// Kernels 3 and 4: register-cached, vectorised, fully fused.
//
// kRowThreads threads cooperate on one row (32 = one warp, 256 = one block). Thread `lane` owns
// 16-byte chunks lane, lane + kRowThreads, ... of the row; at most kChunks of them, so the whole
// row stays in registers:
//   1. load every chunk once (LDG.128) and convert to the accumulation type;
//   2. compute the thread's exact local (count, mean, M2) from the register copy (two-pass, so
//      no cancellation) and merge the triples across the row's threads in ONE reduction;
//   3. normalise from registers and store with 128-bit stores.
// Global traffic is the minimum: one read and one write per element.
// ---------------------------------------------------------------------------
template <typename T, int kRowThreads, int kMaxChunks>
__global__ void __launch_bounds__(kRowThreads == kWarpSize ? kWarpSize * kRowsPerWarpBlock : kRowThreads)
    ln_fwd_fused(FwdParams<T> p) {
  using C = Traits<T>;
  using A = acc_t<T>;
  using V = Vec<T>;
  constexpr int kN = V::kN;
  constexpr int kRowsPerBlock = (kRowThreads == kWarpSize) ? kRowsPerWarpBlock : 1;
  constexpr int kRowWarps = kRowThreads / kWarpSize;

  __shared__ Stats<A> smem_stats[kRowWarps];  // only used when a row spans several warps

  const int lane = threadIdx.x % kRowThreads;
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kRowThreads;
  if (row >= p.rows) return;  // uniform across the warp / block: no barrier is skipped

  const int nvec = p.cols / kN;  // number of 16-byte chunks per row (cols % kN == 0)
  const V* xv = reinterpret_cast<const V*>(p.x + static_cast<std::size_t>(row) * p.cols);

  // (1) single read of the row into registers
  A r[kMaxChunks][kN];
  A count = A(0);
  A sum = A(0);
#pragma unroll
  for (int j = 0; j < kMaxChunks; ++j) {
    const int idx = lane + j * kRowThreads;
    if (idx < nvec) {
      const V v = xv[idx];
#pragma unroll
      for (int k = 0; k < kN; ++k) {
        r[j][k] = C::to_acc(v.v[k]);
        sum += r[j][k];
      }
      count += static_cast<A>(kN);
    } else {
#pragma unroll
      for (int k = 0; k < kN; ++k) r[j][k] = A(0);
    }
  }

  // (2) exact local statistics from registers, then one merge across the row's threads
  const A local_mean = (count > A(0)) ? sum / count : A(0);
  A m2 = A(0);
#pragma unroll
  for (int j = 0; j < kMaxChunks; ++j) {
    if (lane + j * kRowThreads < nvec) {
#pragma unroll
      for (int k = 0; k < kN; ++k) {
        const A d = r[j][k] - local_mean;
        m2 += d * d;
      }
    }
  }
  Stats<A> s{count, local_mean, m2};
  if constexpr (kRowWarps == 1) {
    s = warp_allreduce(s);
  } else {
    s = block_allreduce<A, kRowWarps>(s, smem_stats);
  }
  const A mean = s.mean;
  const A rstd = rsqrt_acc(s.m2 / s.count + p.eps);

  // (3) normalise from registers; gamma / beta are read once as 128-bit vectors
  V* yv = reinterpret_cast<V*>(p.y + static_cast<std::size_t>(row) * p.cols);
  const V* gv = reinterpret_cast<const V*>(p.gamma);
  const V* bv = reinterpret_cast<const V*>(p.beta);
#pragma unroll
  for (int j = 0; j < kMaxChunks; ++j) {
    const int idx = lane + j * kRowThreads;
    if (idx < nvec) {
      V g;
      V b;
      if (gv != nullptr) g = gv[idx];
      if (bv != nullptr) b = bv[idx];
      V o;
#pragma unroll
      for (int k = 0; k < kN; ++k) {
        const A gk = (gv != nullptr) ? C::to_acc(g.v[k]) : A(1);
        const A bk = (bv != nullptr) ? C::to_acc(b.v[k]) : A(0);
        o.v[k] = C::from_acc((r[j][k] - mean) * rstd * gk + bk);
      }
      yv[idx] = o;
    }
  }
  if (lane == 0) {
    if (p.mean != nullptr) p.mean[row] = mean;
    if (p.rstd != nullptr) p.rstd[row] = rstd;
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
bool is_aligned16_host(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

bool supports_any(const ForwardCall&) { return true; }

// Register-cached kernels: whole 16-byte chunks only, bounded row length, aligned pointers.
template <int kRowThreads>
bool supports_fused(const ForwardCall& c) {
  const int n = static_cast<int>(16 / dtype_size(c.dtype));  // elements per 16-byte chunk
  if (c.cols % n != 0) return false;
  if (c.cols > kRowThreads * kChunks * n) return false;
  if (!is_aligned16_host(c.x) || !is_aligned16_host(c.y)) return false;
  if (c.gamma != nullptr && !is_aligned16_host(c.gamma)) return false;
  if (c.beta != nullptr && !is_aligned16_host(c.beta)) return false;
  return true;
}

template <typename T>
FwdParams<T> make_params(const ForwardCall& c) {
  using A = acc_t<T>;
  FwdParams<T> p;
  p.x = static_cast<const T*>(c.x);
  p.gamma = static_cast<const T*>(c.gamma);
  p.beta = static_cast<const T*>(c.beta);
  p.y = static_cast<T*>(c.y);
  p.mean = static_cast<A*>(c.mean);
  p.rstd = static_cast<A*>(c.rstd);
  p.rows = c.rows;
  p.cols = c.cols;
  p.eps = static_cast<A>(c.eps);
  return p;
}

template <typename T>
void launch_forward(int id, const ForwardCall& c, cudaStream_t stream) {
  const FwdParams<T> p = make_params<T>(c);
  switch (id) {
    case 1:
      ln_fwd_two_pass<T><<<c.rows, kBlock, 0, stream>>>(p);
      break;
    case 2:
      ln_fwd_welford<T><<<c.rows, kBlock, 0, stream>>>(p);
      break;
    case 3:
      ln_fwd_fused<T, kWarpSize, kChunks>
          <<<ceil_div(c.rows, kRowsPerWarpBlock), kWarpSize * kRowsPerWarpBlock, 0, stream>>>(p);
      break;
    case 4:
      ln_fwd_fused<T, kBlock, kChunks><<<c.rows, kBlock, 0, stream>>>(p);
      break;
    default:
      throw std::invalid_argument("internal error: unknown forward kernel id");
  }
  LN_CUDA_CHECK(cudaGetLastError());
}

void validate(const ForwardCall& c) {
  if (c.rows < 0) throw std::invalid_argument("rows must be >= 0");
  if (c.cols <= 0) throw std::invalid_argument("cols must be > 0");
  if (c.rows > 0 && (c.x == nullptr || c.y == nullptr)) throw std::invalid_argument("x and y must not be null");
  if (!(c.eps >= 0.0)) throw std::invalid_argument("eps must be >= 0");
}

}  // namespace

const std::vector<ForwardKernelInfo>& forward_registry() {
  static const std::vector<ForwardKernelInfo> registry = {
      {1, "two_pass", "block per row, shared-memory tree sums, 3 reads", supports_any},
      {2, "welford", "block per row, one-pass Welford + shuffles, 2 reads", supports_any},
      {3, "warp_row", "warp per row, row in registers, 128-bit accesses, 1 read",
       supports_fused<kWarpSize>},
      {4, "block_fused", "block per row, row in registers, 128-bit accesses, 1 read",
       supports_fused<kBlock>},
  };
  return registry;
}

int select_forward_kernel(const ForwardCall& call) {
  for (int id : {3, 4}) {
    if (forward_registry()[static_cast<std::size_t>(id - 1)].supports(call)) return id;
  }
  return 2;
}

void layernorm_forward(int kernel_id, const ForwardCall& call, cudaStream_t stream) {
  validate(call);
  if (call.rows == 0) return;

  const int id = (kernel_id == 0) ? select_forward_kernel(call) : kernel_id;
  const auto& registry = forward_registry();
  if (id < 1 || id > static_cast<int>(registry.size())) {
    throw std::invalid_argument("unknown forward kernel id " + std::to_string(id));
  }
  const ForwardKernelInfo& info = registry[static_cast<std::size_t>(id - 1)];
  if (!info.supports(call)) {
    throw std::invalid_argument(std::string("forward kernel '") + info.name + "' does not support this " +
                                "problem (cols=" + std::to_string(call.cols) + ", dtype=" +
                                dtype_name(call.dtype) + "); use kernel_id=0 for automatic selection");
  }

  switch (call.dtype) {
    case DType::kFloat32: launch_forward<float>(id, call, stream); break;
    case DType::kFloat16: launch_forward<__half>(id, call, stream); break;
    case DType::kBFloat16: launch_forward<__nv_bfloat16>(id, call, stream); break;
    case DType::kFloat64: launch_forward<double>(id, call, stream); break;
  }
}

}  // namespace ln
