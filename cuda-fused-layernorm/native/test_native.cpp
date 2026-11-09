// SPDX-License-Identifier: MIT
//
// ln_native_tests - correctness tests for the CUDA kernels WITHOUT PyTorch.
//
// Every forward kernel x dtype (fp32, fp16, bf16, fp64) x shape x affine mode, and the backward
// kernel with and without gamma / parameter gradients, are compared against a double-precision CPU
// reference computed from the same (already quantised) inputs. Highlights:
//   * shapes around every warp / chunk / block boundary and around the register-cache capacity,
//     including rows > 256 so backward blocks loop over several rows;
//   * odd rows carry a large mean offset (|mean| >> std), which breaks the naive E[x^2] - E[x]^2
//     formula but not Welford / two-pass statistics;
//   * outputs are pre-filled with NaN so unwritten elements are detected;
//   * misaligned pointers (the vectorised kernels must be rejected, kernel_id=0 must fall back);
//   * bitwise determinism of the backward pass; invalid-argument handling.
//
// Usage: ln_native_tests [--quick] [--only fp32|fp16|bf16|fp64|edge]
// Exit status: 0 = all passed, 1 = failures, 2 = environment error.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "layernorm.h"
#include "native_utils.hpp"
#include "reference.hpp"

namespace {

using namespace native;

struct Context {
  bool quick = false;
  std::string only;  // "", fp32, fp16, bf16, fp64 or edge
  std::mt19937_64 rng{987654321};
  int passed = 0;
  int failed = 0;
  int skipped = 0;
};

void report(Context& ctx, bool ok, const std::string& what) {
  if (ok) {
    ++ctx.passed;
  } else {
    ++ctx.failed;
    std::printf("  FAIL %s\n", what.c_str());
  }
}

struct Shape {
  int rows;
  int cols;
};

struct Tol {
  double atol;
  double rtol;
};

template <typename T> Tol forward_tol();
template <> Tol forward_tol<float>() { return {5e-5, 1e-4}; }
template <> Tol forward_tol<double>() { return {1e-8, 1e-8}; }
template <> Tol forward_tol<__half>() { return {4e-3, 2e-3}; }
template <> Tol forward_tol<__nv_bfloat16>() { return {3e-2, 1e-2}; }

template <typename T> Tol backward_tol();
template <> Tol backward_tol<float>() { return {2e-4, 1e-3}; }
template <> Tol backward_tol<double>() { return {1e-8, 1e-8}; }
template <> Tol backward_tol<__half>() { return {1e-2, 1e-2}; }
template <> Tol backward_tol<__nv_bfloat16>() { return {6e-2, 3e-2}; }

/// Mean offset added to odd rows so |mean| >> std; limited by what the storage type can represent.
template <typename T> double mean_offset();
template <> double mean_offset<float>() { return 100.0; }
template <> double mean_offset<double>() { return 1e5; }
template <> double mean_offset<__half>() { return 20.0; }
template <> double mean_offset<__nv_bfloat16>() { return 8.0; }

/// Random data quantised to storage type T; `as_double` holds exactly the values the GPU sees.
template <typename T>
struct HostData {
  std::vector<T> stored;
  std::vector<double> as_double;
};

template <typename T>
HostData<T> make_data(Context& ctx, std::size_t count, double lo, double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  HostData<T> d;
  d.stored.resize(count);
  d.as_double.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    d.stored[i] = from_double<T>(dist(ctx.rng));
    d.as_double[i] = to_double(d.stored[i]);
  }
  return d;
}

template <typename T>
HostData<T> make_rows_with_offset(Context& ctx, int rows, int cols) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  HostData<T> d;
  d.stored.resize(static_cast<std::size_t>(rows) * cols);
  d.as_double.resize(d.stored.size());
  for (int r = 0; r < rows; ++r) {
    const double offset = (r % 2 == 1) ? mean_offset<T>() : 0.0;
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * cols + c;
      d.stored[i] = from_double<T>(dist(ctx.rng) * 2.0 + offset);
      d.as_double[i] = to_double(d.stored[i]);
    }
  }
  return d;
}

template <typename T>
std::vector<double> download_as_double(const DeviceBuffer<T>& buf, std::size_t count) {
  std::vector<T> host(count);
  buf.download(host.data(), count);
  std::vector<double> out(count);
  for (std::size_t i = 0; i < count; ++i) out[i] = to_double(host[i]);
  return out;
}

const char* dtype_label(ln::DType d) { return ln::dtype_name(d); }

/// Scientific notation: std::to_string would print tiny errors as 0.000000.
std::string sci(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3e", v);
  return buf;
}

// ----------------------------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------------------------
template <typename T>
void test_forward(Context& ctx, const std::vector<Shape>& shapes) {
  using S = stat_t<T>;
  const ln::DType dt = DTypeOf<T>::value;
  const Tol tol = forward_tol<T>();
  const double eps = 1e-5;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const int before = ctx.failed;

  for (const Shape& s : shapes) {
    const std::size_t n = static_cast<std::size_t>(s.rows) * s.cols;
    const HostData<T> x = make_rows_with_offset<T>(ctx, s.rows, s.cols);
    const HostData<T> gamma = make_data<T>(ctx, static_cast<std::size_t>(s.cols), 0.5, 1.5);
    const HostData<T> beta = make_data<T>(ctx, static_cast<std::size_t>(s.cols), -1.0, 1.0);

    DeviceBuffer<T> d_x(n), d_gamma(static_cast<std::size_t>(s.cols)), d_beta(static_cast<std::size_t>(s.cols));
    DeviceBuffer<T> d_y(n);
    DeviceBuffer<S> d_mean(static_cast<std::size_t>(s.rows)), d_rstd(static_cast<std::size_t>(s.rows));
    d_x.upload(x.stored.data(), n);
    d_gamma.upload(gamma.stored.data(), gamma.stored.size());
    d_beta.upload(beta.stored.data(), beta.stored.size());

    // affine modes: 0 = gamma + beta, 1 = none, 2 = gamma only
    for (int mode = 0; mode < 3; ++mode) {
      const bool use_gamma = (mode != 1);
      const bool use_beta = (mode == 0);
      const ForwardRef ref = layernorm_forward_ref(x.as_double, use_gamma ? &gamma.as_double : nullptr,
                                                   use_beta ? &beta.as_double : nullptr, s.rows, s.cols, eps);

      ln::ForwardCall call{dt,
                           d_x.get(),
                           use_gamma ? d_gamma.get() : nullptr,
                           use_beta ? d_beta.get() : nullptr,
                           d_y.get(),
                           d_mean.get(),
                           d_rstd.get(),
                           s.rows,
                           s.cols,
                           eps};

      std::vector<int> ids = {0};
      for (const ln::ForwardKernelInfo& k : ln::forward_registry()) ids.push_back(k.id);

      for (int id : ids) {
        if (id != 0 && !ln::forward_registry()[static_cast<std::size_t>(id - 1)].supports(call)) {
          ++ctx.skipped;
          continue;
        }
        // NaN sentinel: every output element must be overwritten.
        const std::vector<T> nan_t(n, from_double<T>(nan));
        const std::vector<S> nan_s(static_cast<std::size_t>(s.rows), static_cast<S>(nan));
        d_y.upload(nan_t.data(), n);
        d_mean.upload(nan_s.data(), nan_s.size());
        d_rstd.upload(nan_s.data(), nan_s.size());

        ln::layernorm_forward(id, call, nullptr);
        LN_CUDA_CHECK(cudaGetLastError());
        LN_CUDA_CHECK(cudaDeviceSynchronize());

        const std::string label = std::string("forward/") + dtype_label(dt) + " kernel=" + std::to_string(id) +
                                  " rows=" + std::to_string(s.rows) + " cols=" + std::to_string(s.cols) +
                                  " affine=" + std::to_string(mode);
        const Mismatch my = compare(download_as_double(d_y, n), ref.y, tol.atol, tol.rtol);
        report(ctx, my.count == 0,
               label + " y: " + std::to_string(my.count) + " mismatches, max abs err " + sci(my.max_abs_err));
        const Mismatch mm = compare(download_as_double(d_mean, static_cast<std::size_t>(s.rows)), ref.mean,
                                    tol.atol, tol.rtol);
        report(ctx, mm.count == 0, label + " mean: " + std::to_string(mm.count) + " mismatches");
        const Mismatch mr = compare(download_as_double(d_rstd, static_cast<std::size_t>(s.rows)), ref.rstd,
                                    tol.atol, tol.rtol);
        report(ctx, mr.count == 0, label + " rstd: " + std::to_string(mr.count) + " mismatches");
      }
    }
  }
  std::printf("[%s] forward  %s\n", ctx.failed == before ? " OK " : "FAIL", dtype_label(dt));
}

// ----------------------------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------------------------
template <typename T>
void test_backward(Context& ctx, const std::vector<Shape>& shapes) {
  using S = stat_t<T>;
  const ln::DType dt = DTypeOf<T>::value;
  const Tol tol = backward_tol<T>();
  const double eps = 1e-5;
  const int before = ctx.failed;

  for (const Shape& s : shapes) {
    const std::size_t n = static_cast<std::size_t>(s.rows) * s.cols;
    const std::size_t cols = static_cast<std::size_t>(s.cols);
    const HostData<T> x = make_rows_with_offset<T>(ctx, s.rows, s.cols);
    const HostData<T> dy = make_data<T>(ctx, n, -1.0, 1.0);
    const HostData<T> gamma = make_data<T>(ctx, cols, 0.5, 1.5);
    const HostData<T> beta = make_data<T>(ctx, cols, -1.0, 1.0);

    for (int use_gamma = 0; use_gamma <= 1; ++use_gamma) {
      for (int want_params = 0; want_params <= 1; ++want_params) {
        const ForwardRef fwd = layernorm_forward_ref(x.as_double, use_gamma ? &gamma.as_double : nullptr,
                                                     nullptr, s.rows, s.cols, eps);
        const BackwardRef ref = layernorm_backward_ref(dy.as_double, x.as_double,
                                                       use_gamma ? &gamma.as_double : nullptr, fwd.mean, fwd.rstd,
                                                       s.rows, s.cols);

        std::vector<S> mean_s(fwd.mean.size()), rstd_s(fwd.rstd.size());
        for (std::size_t i = 0; i < mean_s.size(); ++i) {
          mean_s[i] = static_cast<S>(fwd.mean[i]);
          rstd_s[i] = static_cast<S>(fwd.rstd[i]);
        }

        DeviceBuffer<T> d_x(n), d_dy(n), d_dx(n), d_gamma(cols), d_dgamma(cols), d_dbeta(cols);
        DeviceBuffer<S> d_mean(mean_s.size()), d_rstd(rstd_s.size());
        const std::size_t ws_bytes = ln::backward_workspace_bytes(dt, s.rows, s.cols);
        DeviceBuffer<unsigned char> d_ws(ws_bytes);
        d_x.upload(x.stored.data(), n);
        d_dy.upload(dy.stored.data(), n);
        d_gamma.upload(gamma.stored.data(), cols);
        d_mean.upload(mean_s.data(), mean_s.size());
        d_rstd.upload(rstd_s.data(), rstd_s.size());

        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::vector<T> nan_n(n, from_double<T>(nan));
        const std::vector<T> nan_c(cols, from_double<T>(nan));

        ln::BackwardCall call{dt,
                              d_dy.get(),
                              d_x.get(),
                              use_gamma ? d_gamma.get() : nullptr,
                              d_mean.get(),
                              d_rstd.get(),
                              d_dx.get(),
                              want_params ? d_dgamma.get() : nullptr,
                              want_params ? d_dbeta.get() : nullptr,
                              want_params ? d_ws.get() : nullptr,
                              s.rows,
                              s.cols};

        auto run = [&]() {
          d_dx.upload(nan_n.data(), n);
          d_dgamma.upload(nan_c.data(), cols);
          d_dbeta.upload(nan_c.data(), cols);
          ln::layernorm_backward(call, nullptr);
          LN_CUDA_CHECK(cudaGetLastError());
          LN_CUDA_CHECK(cudaDeviceSynchronize());
        };
        run();

        const std::string label = std::string("backward/") + dtype_label(dt) + " rows=" + std::to_string(s.rows) +
                                  " cols=" + std::to_string(s.cols) + " gamma=" + std::to_string(use_gamma) +
                                  " params=" + std::to_string(want_params);
        const std::vector<double> got_dx = download_as_double(d_dx, n);
        const Mismatch mdx = compare(got_dx, ref.dx, tol.atol, tol.rtol);
        report(ctx, mdx.count == 0,
               label + " dx: " + std::to_string(mdx.count) + " mismatches, max abs err " +
                   sci(mdx.max_abs_err));

        if (want_params) {
          const std::vector<double> got_dg = download_as_double(d_dgamma, cols);
          const std::vector<double> got_db = download_as_double(d_dbeta, cols);
          const Mismatch mg = compare(got_dg, ref.dgamma, tol.atol, tol.rtol);
          const Mismatch mb = compare(got_db, ref.dbeta, tol.atol, tol.rtol);
          report(ctx, mg.count == 0,
                 label + " dgamma: " + std::to_string(mg.count) + " mismatches, max abs err " +
                     sci(mg.max_abs_err));
          report(ctx, mb.count == 0,
                 label + " dbeta: " + std::to_string(mb.count) + " mismatches, max abs err " +
                     sci(mb.max_abs_err));

          // Bitwise determinism: a second run (NaN-refilled outputs) must reproduce every bit.
          run();
          const bool same = download_as_double(d_dx, n) == got_dx && download_as_double(d_dgamma, cols) == got_dg &&
                            download_as_double(d_dbeta, cols) == got_db;
          report(ctx, same, label + " is not bitwise deterministic");
        }
      }
    }
  }
  std::printf("[%s] backward %s\n", ctx.failed == before ? " OK " : "FAIL", dtype_label(dt));
}

// ----------------------------------------------------------------------------------------------
// Alignment, fallback and argument validation
// ----------------------------------------------------------------------------------------------
void test_edge_cases(Context& ctx) {
  const int before = ctx.failed;
  const int rows = 5, cols = 64;
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const HostData<float> x = make_data<float>(ctx, n + 1, -2.0, 2.0);

  DeviceBuffer<float> d_x(n + 1), d_y(n + 1);
  d_x.upload(x.stored.data(), n + 1);

  // Misaligned x / y (one float past a 256-byte boundary).
  const std::vector<double> x_shift(x.as_double.begin() + 1, x.as_double.begin() + 1 + static_cast<long>(n));
  const ForwardRef ref = layernorm_forward_ref(x_shift, nullptr, nullptr, rows, cols, 1e-5);
  ln::ForwardCall call{ln::DType::kFloat32, d_x.get() + 1, nullptr, nullptr, d_y.get() + 1, nullptr, nullptr, rows,
                       cols, 1e-5};

  report(ctx, ln::select_forward_kernel(call) == 2, "misaligned input must fall back to the welford kernel");
  bool threw = false;
  try {
    ln::layernorm_forward(3, call, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  report(ctx, threw, "explicit vectorised kernel on misaligned pointers must throw std::invalid_argument");

  ln::layernorm_forward(0, call, nullptr);  // auto: must still be correct
  LN_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> y_host(n);
  LN_CUDA_CHECK(cudaMemcpy(y_host.data(), d_y.get() + 1, n * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<double> y_d(y_host.begin(), y_host.end());
  report(ctx, compare(y_d, ref.y, 5e-5, 1e-4).count == 0, "misaligned fallback result");

  // Invalid arguments
  threw = false;
  try {
    ln::layernorm_forward(99, call, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  report(ctx, threw, "unknown kernel id must throw");

  ln::ForwardCall bad = call;
  bad.cols = 0;
  threw = false;
  try {
    ln::layernorm_forward(0, bad, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  report(ctx, threw, "cols == 0 must throw");

  // rows == 0 is a no-op for forward and zero-fills parameter gradients in backward.
  ln::ForwardCall empty = call;
  empty.rows = 0;
  ln::layernorm_forward(0, empty, nullptr);
  report(ctx, true, "rows == 0 forward is a no-op");

  DeviceBuffer<float> d_dg(cols), d_db(cols);
  const std::vector<float> junk(static_cast<std::size_t>(cols), 7.0f);
  d_dg.upload(junk.data(), junk.size());
  d_db.upload(junk.data(), junk.size());
  // PyTorch hands out a null pointer for zero-size tensors, so rows == 0 must accept a null workspace.
  ln::BackwardCall eb{ln::DType::kFloat32, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, d_dg.get(),
                      d_db.get(),          nullptr,  0,       cols};
  ln::layernorm_backward(eb, nullptr);
  LN_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<std::size_t>(cols));
  d_dg.download(got.data(), got.size());
  bool all_zero = true;
  for (float v : got) all_zero = all_zero && (v == 0.0f);
  report(ctx, all_zero, "rows == 0 backward must zero dgamma");

  std::printf("[%s] edge cases\n", ctx.failed == before ? " OK " : "FAIL");
}

std::vector<Shape> make_shapes(bool quick) {
  std::vector<Shape> shapes = {{1, 1},   {1, 2},   {2, 3},   {3, 7},   {4, 31},  {5, 32},   {5, 33},  {7, 127},
                               {7, 128}, {4, 255}, {4, 256}, {4, 257}, {3, 768}, {3, 1000}, {2, 1024}, {2, 1025},
                               {2, 2048}, {300, 32}};
  if (!quick) {
    shapes.insert(shapes.end(), {{2, 4096}, {2, 8192}, {2, 8200}, {2, 12000}, {1000, 768}, {2048, 1024}});
  }
  return shapes;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Context ctx;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--quick") == 0) {
        ctx.quick = true;
      } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
        ctx.only = argv[++i];
      } else {
        std::fprintf(stderr, "usage: ln_native_tests [--quick] [--only fp32|fp16|bf16|fp64|edge]\n");
        return 2;
      }
    }
    int devices = 0;
    LN_CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
      std::fputs("No CUDA device found.\n", stderr);
      return 2;
    }

    const std::vector<Shape> shapes = make_shapes(ctx.quick);
    const auto want = [&](const char* tag) { return ctx.only.empty() || ctx.only == tag; };
    if (want("fp32")) test_forward<float>(ctx, shapes);
    if (want("fp16")) test_forward<__half>(ctx, shapes);
    if (want("bf16")) test_forward<__nv_bfloat16>(ctx, shapes);
    if (want("fp64")) test_forward<double>(ctx, shapes);
    if (want("fp32")) test_backward<float>(ctx, shapes);
    if (want("fp16")) test_backward<__half>(ctx, shapes);
    if (want("bf16")) test_backward<__nv_bfloat16>(ctx, shapes);
    if (want("fp64")) test_backward<double>(ctx, shapes);
    if (want("edge")) test_edge_cases(ctx);

    std::printf("\n%d checks passed, %d failed (%d kernel/shape combinations skipped as unsupported)\n", ctx.passed,
                ctx.failed, ctx.skipped);
    return ctx.failed == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
