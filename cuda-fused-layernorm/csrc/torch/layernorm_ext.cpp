// SPDX-License-Identifier: MIT
//
// PyTorch binding for the fused LayerNorm kernels.
//
// This file is the only place that knows about ATen: it validates tensors, allocates outputs with
// the caching allocator, enters the tensor's device guard, takes the current CUDA stream and calls
// the framework-independent API in layernorm.h. Python-side autograd lives in
// python/fused_layernorm/functional.py.
//
// Exposed to Python as `fused_layernorm._C`:
//   forward(x2d, weight, bias, eps, kernel_id)            -> (y2d, mean, rstd)
//   backward(dy2d, x2d, weight, mean, rstd, need_params)  -> (dx2d, dgamma, dbeta)
//   forward_kernels()                                      -> [(id, name, summary), ...]
//   kernel_supported(kernel_id, x2d, weight, bias)         -> bool
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "layernorm.h"

namespace {

ln::DType to_ln_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return ln::DType::kFloat32;
    case at::kHalf: return ln::DType::kFloat16;
    case at::kBFloat16: return ln::DType::kBFloat16;
    case at::kDouble: return ln::DType::kFloat64;
    default: break;
  }
  TORCH_CHECK(false, "fused_layer_norm supports float32, float16, bfloat16 and float64, got ", t);
  return ln::DType::kFloat32;  // unreachable: TORCH_CHECK(false) throws
}

/// Statistics are kept in fp32 (fp64 for double inputs), whatever the storage dtype.
at::ScalarType stats_dtype(at::ScalarType t) { return t == at::kDouble ? at::kDouble : at::kFloat; }

void check_matrix(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 2, name, " must be 2-D [rows, cols], got ", t.dim(), "-D");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_vector(const std::optional<at::Tensor>& t, const char* name, const at::Tensor& like, int64_t n) {
  if (!t.has_value() || !t->defined()) return;
  TORCH_CHECK(t->is_cuda() && t->device() == like.device(), name, " must be on the same CUDA device as the input");
  TORCH_CHECK(t->dim() == 1 && t->size(0) == n, name, " must have shape [", n, "], got ", t->sizes());
  TORCH_CHECK(t->is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t->scalar_type() == like.scalar_type(), name, " must have the same dtype as the input (", like.scalar_type(),
              "), got ", t->scalar_type());
}

const void* ptr_or_null(const std::optional<at::Tensor>& t) {
  return (t.has_value() && t->defined()) ? t->data_ptr() : nullptr;
}

int to_int(int64_t v, const char* what) {
  TORCH_CHECK(v >= 0 && v <= INT32_MAX, what, " is out of range for the kernels: ", v);
  return static_cast<int>(v);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward(const at::Tensor& x, const std::optional<at::Tensor>& weight,
                                                       const std::optional<at::Tensor>& bias, double eps,
                                                       int64_t kernel_id) {
  check_matrix(x, "x");
  check_vector(weight, "weight", x, x.size(1));
  check_vector(bias, "bias", x, x.size(1));
  TORCH_CHECK(eps >= 0.0, "eps must be >= 0");

  const c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor y = at::empty_like(x);
  const auto stats_opts = x.options().dtype(stats_dtype(x.scalar_type()));
  at::Tensor mean = at::empty({x.size(0)}, stats_opts);
  at::Tensor rstd = at::empty({x.size(0)}, stats_opts);

  ln::ForwardCall call{to_ln_dtype(x.scalar_type()),
                       x.data_ptr(),
                       ptr_or_null(weight),
                       ptr_or_null(bias),
                       y.data_ptr(),
                       mean.data_ptr(),
                       rstd.data_ptr(),
                       to_int(x.size(0), "rows"),
                       to_int(x.size(1), "cols"),
                       eps};
  ln::layernorm_forward(static_cast<int>(kernel_id), call, at::cuda::getCurrentCUDAStream());
  return std::make_tuple(y, mean, rstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward(const at::Tensor& dy, const at::Tensor& x,
                                                        const std::optional<at::Tensor>& weight,
                                                        const at::Tensor& mean, const at::Tensor& rstd,
                                                        bool need_params) {
  check_matrix(dy, "dy");
  check_matrix(x, "x");
  TORCH_CHECK(dy.sizes() == x.sizes(), "dy and x must have the same shape");
  TORCH_CHECK(dy.scalar_type() == x.scalar_type(), "dy and x must have the same dtype");
  TORCH_CHECK(dy.device() == x.device(), "dy and x must be on the same device");
  check_vector(weight, "weight", x, x.size(1));
  const auto stats_type = stats_dtype(x.scalar_type());
  TORCH_CHECK(mean.is_cuda() && rstd.is_cuda() && mean.is_contiguous() && rstd.is_contiguous() &&
                  mean.scalar_type() == stats_type && rstd.scalar_type() == stats_type &&
                  mean.numel() == x.size(0) && rstd.numel() == x.size(0),
              "mean / rstd must be contiguous CUDA tensors of shape [rows] and dtype ", stats_type);

  const c10::cuda::CUDAGuard device_guard(x.device());
  const int rows = to_int(x.size(0), "rows");
  const int cols = to_int(x.size(1), "cols");
  const ln::DType dtype = to_ln_dtype(x.scalar_type());

  at::Tensor dx = at::empty_like(x);
  at::Tensor dgamma = at::empty({need_params ? x.size(1) : 0}, x.options());
  at::Tensor dbeta = at::empty({need_params ? x.size(1) : 0}, x.options());
  at::Tensor workspace;
  if (need_params) {
    const auto bytes = static_cast<int64_t>(ln::backward_workspace_bytes(dtype, rows, cols));
    workspace = at::empty({bytes}, x.options().dtype(at::kByte));
  }

  ln::BackwardCall call{dtype,
                        dy.data_ptr(),
                        x.data_ptr(),
                        ptr_or_null(weight),
                        mean.data_ptr(),
                        rstd.data_ptr(),
                        dx.data_ptr(),
                        need_params ? dgamma.data_ptr() : nullptr,
                        need_params ? dbeta.data_ptr() : nullptr,
                        need_params ? workspace.data_ptr() : nullptr,
                        rows,
                        cols};
  ln::layernorm_backward(call, at::cuda::getCurrentCUDAStream());
  return std::make_tuple(dx, dgamma, dbeta);
}

std::vector<std::tuple<int, std::string, std::string>> forward_kernels() {
  std::vector<std::tuple<int, std::string, std::string>> out;
  for (const ln::ForwardKernelInfo& k : ln::forward_registry()) out.emplace_back(k.id, k.name, k.summary);
  return out;
}

/// True when forward kernel `kernel_id` can run on these tensors (kernel_id 0 = auto = always true).
bool kernel_supported(int64_t kernel_id, const at::Tensor& x, const std::optional<at::Tensor>& weight,
                      const std::optional<at::Tensor>& bias) {
  check_matrix(x, "x");
  if (kernel_id == 0) return true;
  const auto& registry = ln::forward_registry();
  if (kernel_id < 1 || kernel_id > static_cast<int64_t>(registry.size())) return false;
  // Fresh outputs from the caching allocator are at least 512-byte aligned, so x's pointer
  // stands in for y's alignment.
  ln::ForwardCall call{to_ln_dtype(x.scalar_type()), x.data_ptr(), ptr_or_null(weight), ptr_or_null(bias),
                       x.data_ptr(), nullptr, nullptr,
                       to_int(x.size(0), "rows"), to_int(x.size(1), "cols"), 1e-5};
  return registry[static_cast<std::size_t>(kernel_id - 1)].supports(call);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Fused LayerNorm CUDA kernels (forward + backward)";
  m.def("forward", &forward, "LayerNorm forward on a contiguous [rows, cols] CUDA tensor: returns (y, mean, rstd)",
        pybind11::arg("x"), pybind11::arg("weight") = pybind11::none(), pybind11::arg("bias") = pybind11::none(),
        pybind11::arg("eps") = 1e-5, pybind11::arg("kernel_id") = 0);
  m.def("backward", &backward,
        "LayerNorm backward: returns (dx, dgamma, dbeta); dgamma / dbeta are empty unless need_params",
        pybind11::arg("dy"), pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("mean"),
        pybind11::arg("rstd"), pybind11::arg("need_params"));
  m.def("forward_kernels", &forward_kernels, "List of (id, name, summary) of the forward kernel ladder");
  m.def("kernel_supported", &kernel_supported, "Whether a forward kernel supports the given tensors",
        pybind11::arg("kernel_id"), pybind11::arg("x"), pybind11::arg("weight") = pybind11::none(),
        pybind11::arg("bias") = pybind11::none());
}
