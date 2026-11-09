// SPDX-License-Identifier: MIT
//
// Public, framework-independent API of the fused LayerNorm kernels.
//
// The kernels know nothing about PyTorch: they take raw device pointers, a dtype tag and a
// CUDA stream. The PyTorch extension (csrc/torch) and the native tests/benchmarks (native/)
// are thin clients of this header, which keeps the CUDA code testable without a framework.
//
// Layout
//   x, y, dy, dx : [rows, cols] row-major, contiguous; each row is normalised independently.
//   gamma, beta  : [cols], optional (nullptr = no affine transform: gamma = 1, beta = 0).
//   mean, rstd   : [rows] per-row statistics saved by the forward pass for the backward pass.
//                  Stored in float for fp32 / fp16 / bf16 inputs and in double for fp64.
//
// Math (biased variance, as in torch.nn.LayerNorm):
//   mean = (1/C) * sum(x),  var = (1/C) * sum((x - mean)^2),  rstd = 1 / sqrt(var + eps)
//   y    = (x - mean) * rstd * gamma + beta
//
// All arithmetic is carried out in fp32 (fp64 for double inputs), whatever the storage type.
// Errors: invalid arguments throw std::invalid_argument, CUDA failures throw ln::LnError.
// All launches are asynchronous on the given stream.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ln {

/// Thrown when a CUDA runtime call fails.
class LnError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {
inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
  if (status != cudaSuccess) {
    throw LnError(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                  " failed: " + cudaGetErrorString(status));
  }
}
}  // namespace detail

}  // namespace ln

#define LN_CUDA_CHECK(expr) ::ln::detail::check_cuda((expr), #expr, __FILE__, __LINE__)

namespace ln {

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

enum class DType { kFloat32, kFloat16, kBFloat16, kFloat64 };

inline std::size_t dtype_size(DType d) {
  switch (d) {
    case DType::kFloat32: return 4;
    case DType::kFloat16: return 2;
    case DType::kBFloat16: return 2;
    case DType::kFloat64: return 8;
  }
  return 0;
}

inline const char* dtype_name(DType d) {
  switch (d) {
    case DType::kFloat32: return "float32";
    case DType::kFloat16: return "float16";
    case DType::kBFloat16: return "bfloat16";
    case DType::kFloat64: return "float64";
  }
  return "?";
}

/// Size in bytes of one per-row statistic (mean / rstd) for the given storage dtype.
inline std::size_t stats_size(DType d) { return d == DType::kFloat64 ? 8 : 4; }

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
struct ForwardCall {
  DType dtype;
  const void* x;      // [rows, cols]
  const void* gamma;  // [cols] or nullptr
  const void* beta;   // [cols] or nullptr
  void* y;            // [rows, cols]
  void* mean;         // [rows] or nullptr (skip saving statistics)
  void* rstd;         // [rows] or nullptr
  int rows;
  int cols;
  double eps;
};

/// One entry of the forward-kernel ladder (ordered from the simplest to the most optimised).
struct ForwardKernelInfo {
  int id;                                    ///< 1-based id; 0 is reserved for "auto".
  const char* name;
  const char* summary;
  bool (*supports)(const ForwardCall& call);  ///< Shape / dtype / alignment requirements.
};

const std::vector<ForwardKernelInfo>& forward_registry();

/// Id of the fastest kernel that supports `call` (what kernel_id = 0 resolves to).
int select_forward_kernel(const ForwardCall& call);

/// Runs LayerNorm forward. `kernel_id` 0 selects automatically; a specific id that does not
/// support the call throws std::invalid_argument.
void layernorm_forward(int kernel_id, const ForwardCall& call, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
struct BackwardCall {
  DType dtype;
  const void* dy;      // [rows, cols]
  const void* x;       // [rows, cols]
  const void* gamma;   // [cols] or nullptr
  const void* mean;    // [rows], saved by the forward pass
  const void* rstd;    // [rows], saved by the forward pass
  void* dx;            // [rows, cols]
  void* dgamma;        // [cols] or nullptr
  void* dbeta;         // [cols] or nullptr; if both are null no parameter gradients are computed
  void* workspace;     // device scratch of backward_workspace_bytes(); required iff dgamma or dbeta
  int rows;
  int cols;
};

/// Scratch bytes needed to compute dgamma / dbeta (two-stage deterministic reduction).
std::size_t backward_workspace_bytes(DType dtype, int rows, int cols);

/// Runs LayerNorm backward: dx always, dgamma / dbeta when requested. Deterministic: the result
/// does not depend on thread scheduling or on the GPU model.
void layernorm_backward(const BackwardCall& call, cudaStream_t stream);

}  // namespace ln
