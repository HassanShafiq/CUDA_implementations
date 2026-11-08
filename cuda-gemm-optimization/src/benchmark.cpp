// SPDX-License-Identifier: MIT
//
// gemm_bench - benchmark and verify every SGEMM kernel against cuBLAS.
//
// Methodology
//  * Inputs are uniform random in [-1, 1] from a fixed seed (reproducible).
//  * Every kernel is first run once from a pristine copy of C and compared with
//    the cuBLAS result, so a fast-but-wrong kernel can never be reported.
//  * Timing uses CUDA events around each individual launch after a warm-up;
//    the MEDIAN of the samples is reported (robust to clock/OS jitter), the
//    minimum is recorded in the CSV as a best-case reference.
//  * GFLOP/s counts 2*m*n*k floating-point operations.
//  * Use beta = 0 (default) for timing: with beta != 0, C accumulates across
//    repeated launches (harmless for timing, but values grow).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm/cublas_baseline.hpp"
#include "gemm/cuda_utils.hpp"
#include "gemm/kernels.hpp"
#include "gemm/reference.hpp"

namespace {

using gemm::GemmArgs;
using gemm::KernelInfo;

struct Options {
  std::vector<int> sizes{512, 1024, 2048, 4096};
  int m = 0;
  int n = 0;
  int k = 0;  // m, n, k all > 0 selects a single rectangular problem
  std::vector<int> kernel_ids;  // empty selects every kernel
  int warmup = 5;
  int iters = 20;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool verify = true;
  bool baseline = true;
  std::string csv_path;
  int device = 0;
  unsigned seed = 42;
};

struct Problem {
  int m;
  int n;
  int k;
};

struct TimingStats {
  double median_ms = 0.0;
  double min_ms = 0.0;
};

void print_usage() {
  std::puts(
      "Usage: gemm_bench [options]\n"
      "\n"
      "  --sizes a,b,c      square problems M=N=K=a, b, c   (default 512,1024,2048,4096)\n"
      "  --m M --n N --k K  one rectangular problem (overrides --sizes)\n"
      "  --kernels 1,3,6    kernel ids to run               (default: all)\n"
      "  --iters N          timed iterations per kernel     (default 20)\n"
      "  --warmup N         untimed warm-up launches        (default 5)\n"
      "  --alpha X          alpha                           (default 1)\n"
      "  --beta X           beta                            (default 0)\n"
      "  --no-verify        skip comparison with cuBLAS\n"
      "  --no-baseline      do not run cuBLAS at all (implies --no-verify; for profiling)\n"
      "  --csv FILE         also write results as CSV\n"
      "  --device N         CUDA device ordinal             (default 0)\n"
      "  --seed N           RNG seed                        (default 42)\n"
      "  -h, --help         show this message");
}

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) continue;
    const int value = std::stoi(item);
    if (value <= 0) throw std::invalid_argument("expected positive integers, got '" + item + "'");
    values.push_back(value);
  }
  if (values.empty()) throw std::invalid_argument("empty list '" + text + "'");
  return values;
}

std::optional<Options> parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      return argv[++i];
    };
    if (flag == "-h" || flag == "--help") {
      print_usage();
      return std::nullopt;
    } else if (flag == "--sizes") {
      opt.sizes = parse_int_list(value());
    } else if (flag == "--m") {
      opt.m = std::stoi(value());
    } else if (flag == "--n") {
      opt.n = std::stoi(value());
    } else if (flag == "--k") {
      opt.k = std::stoi(value());
    } else if (flag == "--kernels") {
      const std::string v = value();
      if (v != "all") opt.kernel_ids = parse_int_list(v);
    } else if (flag == "--iters") {
      opt.iters = std::stoi(value());
    } else if (flag == "--warmup") {
      opt.warmup = std::stoi(value());
    } else if (flag == "--alpha") {
      opt.alpha = std::stof(value());
    } else if (flag == "--beta") {
      opt.beta = std::stof(value());
    } else if (flag == "--no-verify") {
      opt.verify = false;
    } else if (flag == "--no-baseline") {
      opt.baseline = false;
    } else if (flag == "--csv") {
      opt.csv_path = value();
    } else if (flag == "--device") {
      opt.device = std::stoi(value());
    } else if (flag == "--seed") {
      opt.seed = static_cast<unsigned>(std::stoul(value()));
    } else {
      throw std::invalid_argument("unknown option '" + flag + "' (see --help)");
    }
  }

  const bool any_dim = opt.m > 0 || opt.n > 0 || opt.k > 0;
  if (any_dim && !(opt.m > 0 && opt.n > 0 && opt.k > 0)) {
    throw std::invalid_argument("--m, --n and --k must be given together");
  }
  if (opt.iters < 1 || opt.warmup < 0) throw std::invalid_argument("--iters must be >= 1, --warmup >= 0");
  if (!opt.baseline) opt.verify = false;
  return opt;
}

std::vector<const KernelInfo*> select_kernels(const std::vector<int>& ids) {
  std::vector<const KernelInfo*> selected;
  for (const KernelInfo& kernel : gemm::kernel_registry()) {
    if (ids.empty() || std::find(ids.begin(), ids.end(), kernel.id) != ids.end()) {
      selected.push_back(&kernel);
    }
  }
  for (int id : ids) {
    const auto it = std::find_if(gemm::kernel_registry().begin(), gemm::kernel_registry().end(),
                                 [id](const KernelInfo& k) { return k.id == id; });
    if (it == gemm::kernel_registry().end()) throw std::invalid_argument("unknown kernel id " + std::to_string(id));
  }
  return selected;
}

template <typename Launch>
TimingStats measure(Launch&& launch, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) launch();
  GEMM_CUDA_CHECK(cudaGetLastError());
  GEMM_CUDA_CHECK(cudaDeviceSynchronize());

  gemm::GpuTimer timer;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    timer.start();
    launch();
    samples.push_back(timer.stop());
  }
  GEMM_CUDA_CHECK(cudaGetLastError());

  std::sort(samples.begin(), samples.end());
  const std::size_t mid = samples.size() / 2;
  TimingStats stats;
  stats.min_ms = samples.front();
  stats.median_ms = (samples.size() % 2 == 1) ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
  return stats;
}

double gflops(const Problem& p, double ms) {
  return 2.0 * static_cast<double>(p.m) * p.n * p.k / (ms * 1.0e6);
}

void print_device(const cudaDeviceProp& prop, int ordinal) {
  std::printf("Device %d: %s | compute capability %d.%d | %d SMs | %.1f GiB | %zu KiB smem/block\n",
              ordinal, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
              prop.sharedMemPerBlock / 1024);
}

// Returns false if any kernel failed verification.
bool run_problem(const Options& opt, const Problem& p, const std::vector<const KernelInfo*>& kernels,
                 const gemm::CublasGemm& cublas, std::ofstream* csv) {
  const std::size_t na = static_cast<std::size_t>(p.m) * p.k;
  const std::size_t nb = static_cast<std::size_t>(p.k) * p.n;
  const std::size_t nc = static_cast<std::size_t>(p.m) * p.n;

  std::mt19937 rng(opt.seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> host_a(na), host_b(nb), host_c0(nc);
  for (float& v : host_a) v = dist(rng);
  for (float& v : host_b) v = dist(rng);
  for (float& v : host_c0) v = dist(rng);

  gemm::DeviceBuffer<float> dev_a(na), dev_b(nb), dev_c(nc);
  dev_a.copy_from_host(host_a.data(), na);
  dev_b.copy_from_host(host_b.data(), nb);
  const GemmArgs args{p.m, p.n, p.k, opt.alpha, dev_a.get(), dev_b.get(), opt.beta, dev_c.get()};

  std::printf("\nM=%d N=%d K=%d  (alpha=%g beta=%g, %d warm-up + %d timed iterations)\n", p.m, p.n, p.k,
              opt.alpha, opt.beta, opt.warmup, opt.iters);
  std::printf("  %-2s %-14s %12s %14s %10s  %s\n", "id", "kernel", "median [ms]", "GFLOP/s", "% cuBLAS",
              "verification");

  // ---- cuBLAS baseline (also produces the reference result) ----------------
  std::vector<float> reference(nc);
  double baseline_gflops = 0.0;
  if (opt.baseline) {
    dev_c.copy_from_host(host_c0.data(), nc);
    cublas.run(args, nullptr);
    GEMM_CUDA_CHECK(cudaDeviceSynchronize());
    dev_c.copy_to_host(reference.data(), nc);

    const TimingStats t = measure([&] { cublas.run(args, nullptr); }, opt.warmup, opt.iters);
    baseline_gflops = gflops(p, t.median_ms);
    std::printf("  %-2s %-14s %12.3f %14.1f %9.1f%%  %s\n", "-", "cublas", t.median_ms, baseline_gflops,
                100.0, "reference");
    if (csv != nullptr) {
      *csv << p.m << ',' << p.n << ',' << p.k << ",0,cublas," << t.median_ms << ',' << t.min_ms << ','
           << baseline_gflops << ",100,reference\n";
    }
  }

  // ---- custom kernels -------------------------------------------------------
  bool all_ok = true;
  for (const KernelInfo* kernel : kernels) {
    if (!kernel->supports(args)) {
      std::printf("  %-2d %-14s %12s %14s %10s  %s\n", kernel->id, kernel->name, "-", "-", "-",
                  "skipped (shape/alignment not supported)");
      continue;
    }

    std::string verdict = "not checked";
    bool verified = true;
    if (opt.verify) {
      dev_c.copy_from_host(host_c0.data(), nc);
      kernel->launch(args, nullptr);
      GEMM_CUDA_CHECK(cudaGetLastError());
      GEMM_CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> result(nc);
      dev_c.copy_to_host(result.data(), nc);
      const gemm::CompareResult cmp = gemm::compare_results(result, reference, 1e-3, 1e-3);
      verified = cmp.mismatches == 0;
      char buf[96];
      if (verified) {
        std::snprintf(buf, sizeof(buf), "PASS (max abs err %.2e)", cmp.max_abs_err);
      } else {
        std::snprintf(buf, sizeof(buf), "FAIL (%zu mismatches, max abs err %.2e)", cmp.mismatches,
                      cmp.max_abs_err);
      }
      verdict = buf;
      all_ok = all_ok && verified;
    }

    dev_c.copy_from_host(host_c0.data(), nc);
    const TimingStats t = measure([&] { kernel->launch(args, nullptr); }, opt.warmup, opt.iters);
    const double perf = gflops(p, t.median_ms);
    const double pct = baseline_gflops > 0.0 ? 100.0 * perf / baseline_gflops : 0.0;

    if (baseline_gflops > 0.0) {
      std::printf("  %-2d %-14s %12.3f %14.1f %9.1f%%  %s\n", kernel->id, kernel->name, t.median_ms, perf, pct,
                  verdict.c_str());
    } else {
      std::printf("  %-2d %-14s %12.3f %14.1f %10s  %s\n", kernel->id, kernel->name, t.median_ms, perf, "n/a",
                  verdict.c_str());
    }
    if (csv != nullptr) {
      *csv << p.m << ',' << p.n << ',' << p.k << ',' << kernel->id << ',' << kernel->name << ',' << t.median_ms
           << ',' << t.min_ms << ',' << perf << ',' << pct << ','
           << (opt.verify ? (verified ? "pass" : "fail") : "unchecked") << '\n';
    }
  }
  return all_ok;
}

int run(const Options& opt) {
  GEMM_CUDA_CHECK(cudaSetDevice(opt.device));
  cudaDeviceProp prop{};
  GEMM_CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  print_device(prop, opt.device);

  std::vector<Problem> problems;
  if (opt.m > 0) {
    problems.push_back({opt.m, opt.n, opt.k});
  } else {
    for (int s : opt.sizes) problems.push_back({s, s, s});
  }
  const std::vector<const KernelInfo*> kernels = select_kernels(opt.kernel_ids);

  std::ofstream csv_file;
  std::ofstream* csv = nullptr;
  if (!opt.csv_path.empty()) {
    csv_file.open(opt.csv_path);
    if (!csv_file) throw std::runtime_error("cannot open CSV file '" + opt.csv_path + "'");
    csv_file << "m,n,k,kernel_id,kernel,median_ms,min_ms,gflops,pct_of_cublas,verification\n";
    csv = &csv_file;
  }

  const gemm::CublasGemm cublas;
  bool all_ok = true;
  for (const Problem& p : problems) all_ok = run_problem(opt, p, kernels, cublas, csv) && all_ok;

  if (!all_ok) std::fputs("\nERROR: at least one kernel failed verification.\n", stderr);
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::optional<Options> opt = parse_args(argc, argv);
    if (!opt) return 0;  // --help
    return run(*opt);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
