// SPDX-License-Identifier: MIT
//
// Public interface of the parallel-primitives library.
//
// Each primitive exposes an ordered "ladder" of implementations, from a simple
// baseline to the most optimised variant. The benchmark and the tests iterate
// over the same registries, so they can never drift apart.
//
//   reduce   : sum of n floats                      -> 1 float
//   scan     : EXCLUSIVE prefix sum of n int32      -> n int32   (sums must fit in int32)
//   softmax  : numerically stable row-wise softmax of a rows x cols float matrix
//
// Conventions
//  * All pointers are device pointers; all launches are asynchronous on `stream`.
//  * Input and output buffers must not overlap.
//  * Element counts are limited to kMaxElements (2^30) so 32-bit index math in the
//    kernels cannot overflow.
//  * Any alignment is accepted; vectorised paths check alignment at run time and
//    fall back to scalar accesses.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace prim {

constexpr int kMaxElements = 1 << 30;

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

/// Throws std::invalid_argument when a count is negative or above kMaxElements.
inline void check_count(long long n, const char* what) {
  if (n < 0 || n > kMaxElements) {
    throw std::invalid_argument(std::string(what) + " must be in [0, 2^30], got " + std::to_string(n));
  }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------
struct ReduceKernel {
  int id;                                        ///< Stable 1-based id used on the command line.
  const char* name;                              ///< Short name used in reports.
  const char* summary;                           ///< One-line description of the idea introduced.
  std::size_t (*scratch_elems)(int n);           ///< Number of floats of device scratch memory needed.
  void (*run)(const float* in, float* out, int n, float* scratch, cudaStream_t stream);
};

const std::vector<ReduceKernel>& reduce_registry();

// ---------------------------------------------------------------------------
// Exclusive prefix sum
// ---------------------------------------------------------------------------
struct ScanKernel {
  int id;
  const char* name;
  const char* summary;
  std::size_t (*scratch_elems)(int n);           ///< Number of ints of device scratch memory needed.
  void (*run)(const int* in, int* out, int n, int* scratch, cudaStream_t stream);
};

const std::vector<ScanKernel>& scan_registry();

// ---------------------------------------------------------------------------
// Row-wise softmax
//
// Semantics: out[r][c] = exp(x[r][c] - max_r) / sum_c exp(x[r][c] - max_r).
// Entries equal to -inf (e.g. attention masks) get probability 0; a row that is
// entirely -inf yields a row of zeros (not NaN). +inf / NaN inputs are not supported.
// ---------------------------------------------------------------------------
struct SoftmaxKernel {
  int id;
  const char* name;
  const char* summary;
  void (*run)(const float* in, float* out, int rows, int cols, cudaStream_t stream);
};

const std::vector<SoftmaxKernel>& softmax_registry();

}  // namespace prim
