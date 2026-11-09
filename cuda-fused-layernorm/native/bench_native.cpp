// SPDX-License-Identifier: MIT
//
// ln_native_bench - benchmark the forward kernel ladder and the backward kernel WITHOUT PyTorch.
//
// Methodology
//  * Inputs are random (fixed seed). Before timing, every kernel's output on the first rows of the
//    problem is checked against a double-precision CPU reference, so a fast-but-wrong kernel
//    cannot appear in the results.
//  * CUDA events around each launch after a warm-up; the MEDIAN is reported. Optionally the L2
//    cache is flushed before every timed launch (default) so small problems are measured from DRAM.
//  * LayerNorm is memory-bound; effective bandwidth counts the minimum necessary traffic:
//      forward : read x + write y            = 2 * rows * cols * sizeof(T)
//      backward: read x, dy + write dx       = 3 * rows * cols * sizeof(T)
//    (gamma / beta / statistics are negligible). "% copy BW" compares with a device-to-device
//    cudaMemcpy measured on the same GPU (read + write bytes).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "layernorm.h"
#include "native_utils.hpp"
#include "reference.hpp"

namespace {

using namespace native;

struct Options {
  int rows = 8192;
  std::vector<int> cols{768, 1024, 4096};
  std::string dtype = "fp16";  // fp32 | fp16 | bf16 | fp64
  bool backward = true;
  bool flush_l2 = true;
  int warmup = 5;
  int iters = 30;
  int device = 0;
  unsigned seed = 42;
};

void print_usage() {
  std::puts(
      "Usage: ln_native_bench [options]\n"
      "\n"
      "  --rows R           rows (tokens)                       (default 8192)\n"
      "  --cols a,b,c       normalised sizes                    (default 768,1024,4096)\n"
      "  --dtype T          fp32 | fp16 | bf16 | fp64           (default fp16)\n"
      "  --no-backward      skip the backward benchmark\n"
      "  --no-flush-l2      do not flush L2 between launches\n"
      "  --iters N          timed iterations                    (default 30)\n"
      "  --warmup N         untimed warm-up launches            (default 5)\n"
      "  --device N         CUDA device ordinal                 (default 0)\n"
      "  --seed N           RNG seed                            (default 42)\n"
      "  -h, --help         show this message");
}

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) continue;
    const int v = std::stoi(item);
    if (v <= 0) throw std::invalid_argument("expected positive integers, got '" + item + "'");
    values.push_back(v);
  }
  if (values.empty()) throw std::invalid_argument("empty list '" + text + "'");
  return values;
}

std::optional<Options> parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      return argv[++i];
    };
    if (flag == "-h" || flag == "--help") {
      print_usage();
      return std::nullopt;
    } else if (flag == "--rows") {
      o.rows = std::stoi(value());
    } else if (flag == "--cols") {
      o.cols = parse_int_list(value());
    } else if (flag == "--dtype") {
      o.dtype = value();
      if (o.dtype != "fp32" && o.dtype != "fp16" && o.dtype != "bf16" && o.dtype != "fp64") {
        throw std::invalid_argument("--dtype must be fp32, fp16, bf16 or fp64");
      }
    } else if (flag == "--no-backward") {
      o.backward = false;
    } else if (flag == "--no-flush-l2") {
      o.flush_l2 = false;
    } else if (flag == "--iters") {
      o.iters = std::stoi(value());
    } else if (flag == "--warmup") {
      o.warmup = std::stoi(value());
    } else if (flag == "--device") {
      o.device = std::stoi(value());
    } else if (flag == "--seed") {
      o.seed = static_cast<unsigned>(std::stoul(value()));
    } else {
      throw std::invalid_argument("unknown option '" + flag + "' (see --help)");
    }
  }
  if (o.rows < 1 || o.iters < 1 || o.warmup < 0) {
    throw std::invalid_argument("--rows and --iters must be >= 1, --warmup >= 0");
  }
  return o;
}

// ----------------------------------------------------------------------------------------------
struct Timing {
  double median_ms = 0.0;
  double min_ms = 0.0;
};

/// Overwrites a buffer larger than any GPU L2 so the next launch starts with a cold cache.
class L2Flusher {
 public:
  explicit L2Flusher(bool enabled) : enabled_(enabled), buf_(enabled ? kBytes : 0) {}
  void flush() {
    if (enabled_) LN_CUDA_CHECK(cudaMemset(buf_.get(), 0, kBytes));
  }

 private:
  static constexpr std::size_t kBytes = std::size_t{256} << 20;
  bool enabled_;
  DeviceBuffer<unsigned char> buf_;
};

template <typename Launch>
Timing measure(Launch&& launch, L2Flusher& flusher, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) launch();
  LN_CUDA_CHECK(cudaGetLastError());
  LN_CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    flusher.flush();
    timer.start();
    launch();
    samples.push_back(timer.stop());
  }
  LN_CUDA_CHECK(cudaGetLastError());

  std::sort(samples.begin(), samples.end());
  const std::size_t mid = samples.size() / 2;
  Timing t;
  t.min_ms = samples.front();
  t.median_ms = (samples.size() % 2 == 1) ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
  return t;
}

double copy_bandwidth_gbps(const Options& o, L2Flusher& flusher) {
  constexpr std::size_t kBytes = std::size_t{128} << 20;
  DeviceBuffer<unsigned char> src(kBytes), dst(kBytes);
  const Timing t = measure(
      [&] { LN_CUDA_CHECK(cudaMemcpyAsync(dst.get(), src.get(), kBytes, cudaMemcpyDeviceToDevice, nullptr)); },
      flusher, o.warmup, o.iters);
  return 2.0 * static_cast<double>(kBytes) / (t.median_ms * 1.0e6);
}

void print_row(const char* id, const char* name, const Timing& t, double bytes, double copy_bw,
               const std::string& verdict) {
  const double gbps = bytes / (t.median_ms * 1.0e6);
  std::printf("  %-4s %-14s %12.4f %10.1f %9.1f%%  %s\n", id, name, t.median_ms, gbps, 100.0 * gbps / copy_bw,
              verdict.c_str());
}

// ----------------------------------------------------------------------------------------------
template <typename T>
bool run_dtype(const Options& o, double copy_bw, L2Flusher& flusher) {
  using S = stat_t<T>;
  const ln::DType dt = DTypeOf<T>::value;
  const double eps = 1e-5;
  const std::size_t esize = sizeof(T);
  bool all_ok = true;

  for (int cols : o.cols) {
    const std::size_t n = static_cast<std::size_t>(o.rows) * cols;
    std::mt19937_64 rng(o.seed);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    std::vector<T> hx(n), hdy(n), hg(static_cast<std::size_t>(cols)), hb(static_cast<std::size_t>(cols));
    std::vector<double> dx_(n), ddy(n), dg(hg.size()), db(hb.size());
    for (std::size_t i = 0; i < n; ++i) {
      hx[i] = from_double<T>(dist(rng));
      dx_[i] = to_double(hx[i]);
      hdy[i] = from_double<T>(dist(rng) / 3.0);
      ddy[i] = to_double(hdy[i]);
    }
    for (std::size_t i = 0; i < hg.size(); ++i) {
      hg[i] = from_double<T>(0.5 + std::fabs(dist(rng)) / 3.0);
      dg[i] = to_double(hg[i]);
      hb[i] = from_double<T>(dist(rng) / 3.0);
      db[i] = to_double(hb[i]);
    }

    DeviceBuffer<T> d_x(n), d_y(n), d_dy(n), d_dx(n), d_g(hg.size()), d_b(hb.size()), d_dg(hg.size()),
        d_db(hb.size());
    DeviceBuffer<S> d_mean(static_cast<std::size_t>(o.rows)), d_rstd(static_cast<std::size_t>(o.rows));
    DeviceBuffer<unsigned char> d_ws(ln::backward_workspace_bytes(dt, o.rows, cols));
    d_x.upload(hx.data(), n);
    d_dy.upload(hdy.data(), n);
    d_g.upload(hg.data(), hg.size());
    d_b.upload(hb.data(), hb.size());

    // CPU reference on the first rows only (rows are independent).
    const int check_rows = std::min(o.rows, 64);
    const std::size_t check_n = static_cast<std::size_t>(check_rows) * cols;
    const std::vector<double> cx(dx_.begin(), dx_.begin() + static_cast<long>(check_n));
    const ForwardRef ref = layernorm_forward_ref(cx, &dg, &db, check_rows, cols, eps);

    ln::ForwardCall call{dt, d_x.get(), d_g.get(), d_b.get(), d_y.get(), d_mean.get(), d_rstd.get(), o.rows, cols, eps};
    const double fwd_bytes = 2.0 * static_cast<double>(n) * static_cast<double>(esize);
    std::printf("\n[forward] dtype=%s rows=%d cols=%d\n", ln::dtype_name(dt), o.rows, cols);
    std::printf("  %-4s %-14s %12s %10s %10s  %s\n", "id", "kernel", "median [ms]", "GB/s", "% copy BW", "verification");

    const double atol = (dt == ln::DType::kFloat32) ? 5e-5 : (dt == ln::DType::kFloat64) ? 1e-8
                        : (dt == ln::DType::kFloat16) ? 4e-3 : 3e-2;
    const double rtol = (dt == ln::DType::kFloat32) ? 1e-4 : (dt == ln::DType::kFloat64) ? 1e-8
                        : (dt == ln::DType::kFloat16) ? 2e-3 : 1e-2;

    std::vector<int> ids;
    for (const ln::ForwardKernelInfo& k : ln::forward_registry()) ids.push_back(k.id);
    ids.push_back(0);
    for (int id : ids) {
      const bool is_auto = (id == 0);
      if (!is_auto && !ln::forward_registry()[static_cast<std::size_t>(id - 1)].supports(call)) {
        std::printf("  %-4d %-14s %s\n", id, ln::forward_registry()[static_cast<std::size_t>(id - 1)].name,
                    "skipped (shape/alignment not supported)");
        continue;
      }
      ln::layernorm_forward(id, call, nullptr);
      LN_CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<T> got(check_n);
      d_y.download(got.data(), check_n);
      std::vector<double> got_d(check_n);
      for (std::size_t i = 0; i < check_n; ++i) got_d[i] = to_double(got[i]);
      const Mismatch m = compare(got_d, ref.y, atol, rtol);
      const bool ok = (m.count == 0);
      all_ok = all_ok && ok;
      char buf[64];
      if (ok) {
        std::snprintf(buf, sizeof(buf), "PASS");
      } else {
        std::snprintf(buf, sizeof(buf), "FAIL (%zu mismatches)", m.count);
      }

      const Timing t = measure([&] { ln::layernorm_forward(id, call, nullptr); }, flusher, o.warmup, o.iters);
      char idbuf[16];
      if (is_auto) {
        std::snprintf(idbuf, sizeof(idbuf), "auto");
      } else {
        std::snprintf(idbuf, sizeof(idbuf), "%d", id);
      }
      const char* name = is_auto ? "auto-select"
                                 : ln::forward_registry()[static_cast<std::size_t>(id - 1)].name;
      print_row(idbuf, name, t, fwd_bytes, copy_bw, buf);
    }

    if (o.backward) {
      // The backward pass needs the statistics saved by the forward pass.
      ln::layernorm_forward(0, call, nullptr);
      LN_CUDA_CHECK(cudaDeviceSynchronize());
      ln::BackwardCall bcall{dt, d_dy.get(), d_x.get(), d_g.get(), d_mean.get(), d_rstd.get(), d_dx.get(),
                             d_dg.get(), d_db.get(), d_ws.get(), o.rows, cols};
      const double bwd_bytes = 3.0 * static_cast<double>(n) * static_cast<double>(esize);
      std::printf("\n[backward] dtype=%s rows=%d cols=%d (dx + dgamma + dbeta)\n", ln::dtype_name(dt), o.rows, cols);
      std::printf("  %-4s %-14s %12s %10s %10s  %s\n", "id", "kernel", "median [ms]", "GB/s", "% copy BW", "");
      const Timing t = measure([&] { ln::layernorm_backward(bcall, nullptr); }, flusher, o.warmup, o.iters);
      print_row("-", "fused_2stage", t, bwd_bytes, copy_bw, "");
    }
  }
  return all_ok;
}

int run(const Options& o) {
  LN_CUDA_CHECK(cudaSetDevice(o.device));
  cudaDeviceProp prop{};
  LN_CUDA_CHECK(cudaGetDeviceProperties(&prop, o.device));
  std::printf("Device %d: %s | compute capability %d.%d | %d SMs | %.1f GiB\n", o.device, prop.name, prop.major,
              prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  L2Flusher flusher(o.flush_l2);
  const double copy_bw = copy_bandwidth_gbps(o, flusher);
  std::printf("Device-to-device copy bandwidth (read + write): %.1f GB/s\n", copy_bw);

  bool ok = true;
  if (o.dtype == "fp32") ok = run_dtype<float>(o, copy_bw, flusher);
  else if (o.dtype == "fp16") ok = run_dtype<__half>(o, copy_bw, flusher);
  else if (o.dtype == "bf16") ok = run_dtype<__nv_bfloat16>(o, copy_bw, flusher);
  else ok = run_dtype<double>(o, copy_bw, flusher);

  if (!ok) std::fputs("\nERROR: at least one kernel failed verification.\n", stderr);
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::optional<Options> o = parse_args(argc, argv);
    if (!o) return 0;
    return run(*o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
