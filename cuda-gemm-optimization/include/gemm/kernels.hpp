// SPDX-License-Identifier: MIT
//
// Public interface of the SGEMM kernel library.
//
// All matrices are single precision and stored ROW-MAJOR:
//     C (m x n) = alpha * A (m x k) * B (k x n) + beta * C
// When beta == 0, C is never read (so uninitialised C is fine, as in BLAS).
#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace gemm {

/// Problem description plus device pointers. Pointers must stay valid until the
/// launched work has finished.
struct GemmArgs {
  int m;
  int n;
  int k;
  float alpha;
  const float* a;
  const float* b;
  float beta;
  float* c;
};

/// Integer ceiling division for positive values.
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

using LaunchFn = void (*)(const GemmArgs&, cudaStream_t);
using SupportFn = bool (*)(const GemmArgs&);

/// One entry of the optimisation ladder.
struct KernelInfo {
  int id;               ///< Stable 1-based identifier used on the command line.
  const char* name;     ///< Short name used in reports.
  const char* summary;  ///< One-line description of the optimisation introduced.
  LaunchFn launch;      ///< Enqueues the kernel on a stream (asynchronous).
  SupportFn supports;   ///< Whether this kernel can handle the given problem.
};

/// Kernels ordered from the naive baseline to the most optimised variant.
const std::vector<KernelInfo>& kernel_registry();

// Launchers. Each enqueues work asynchronously; use cudaGetLastError() for
// launch-configuration errors and synchronise to observe execution errors.
void launch_sgemm_naive(const GemmArgs& args, cudaStream_t stream);
void launch_sgemm_coalesced(const GemmArgs& args, cudaStream_t stream);
void launch_sgemm_smem_tiled(const GemmArgs& args, cudaStream_t stream);
void launch_sgemm_blocktile_1d(const GemmArgs& args, cudaStream_t stream);
void launch_sgemm_regtile_2d(const GemmArgs& args, cudaStream_t stream);
void launch_sgemm_vectorized(const GemmArgs& args, cudaStream_t stream);

}  // namespace gemm
