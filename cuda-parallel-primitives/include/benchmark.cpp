// SPDX-License-Identifier: MIT
//
// prim_bench - benchmark and verify the reduction, scan and softmax kernels.
//
// Methodology
//  * Inputs come from a fixed seed (reproducible). Every kernel is run once and
//    checked against a CPU reference BEFORE it is timed, so a fast-but-wrong kernel can
//    never appear in the results.
//  * Timing uses CUDA events around each individual launch after a warm-up; the MEDIAN is
//    reported (robust to clock/OS jitter) and the minimum goes to the CSV.
//  * All three primitives are memory-bound, so the metric is EFFECTIVE BANDWIDTH counted
//    with the minimum necessary traffic (reduce: n reads; scan: n reads + n writes;
//    softmax: one read + one write of the matrix). Multi-pass algorithms move more data
//    than this, so they necessarily score below 100%.
//  * "% copy BW" compares with a device-to-device cudaMemcpy measured on the same GPU
//    (read + write bytes), an empirical stand-in for the achievable DRAM bandwidth.
//  * "% baseline" is the speed relative to CUB (reduce, scan): 100% = as fast as CUB.
//    There is no library baseline for softmax in this project.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "prim/cub_baseline.hpp"
#include "prim/cuda_utils.hpp"
#include "prim/primitives.hpp"
#include "prim/reference.hpp"

namespace {

using namespace prim;

struct Options {
  std::string op = "all";  // all | reduce | scan | softmax
  std::vector<int> sizes{1 << 20, 1 << 24};  // element counts for reduce and scan
  int rows = 8192;                           // softmax matrix rows
  std::vector<int> cols{256, 1024, 4096};    // softmax matrix columns
  std::vector<int> kernel_ids;               // empty = all
  int warmup = 5;
  int iters = 20;
  bool verify = true;
  bool baseline = true;
  std::string csv_path;
  int device = 0;
  unsigned seed = 42;
};

struct TimingStats {
  double median_ms = 0.0;
  double min_ms = 0.0;
};

void print_usage() {
  std::puts(
      "Usage: prim_bench [options]\n"
      "\n"
      "  --op all|reduce|scan|softmax   what to run              (default all)\n"
      "  --sizes a,b,c                  element counts, reduce/scan (default 1048576,16777216)\n"
      "  --rows R                       softmax rows             (default 8192)\n"
      "  --cols a,b,c                   softmax columns          (default 256,1024,4096)\n"
      "  --kernels 1,3                  kernel ids (per primitive; default all)\n"
      "  --iters N                      timed iterations         (default 20)\n"
      "  --warmup N                     untimed warm-up launches (default 5)\n"
      "  --no-verify                    skip comparison with the CPU reference\n"
      "  --no-baseline                  skip CUB and the copy-bandwidth measurement (for profiling)\n"
      "  --csv FILE                     also write results as CSV\n"
      "  --device N                     CUDA device ordinal      (default 0)\n"
      "  --seed N                       RNG seed                 (default 42)\n"
      "  -h, --help                     show this message");
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
    } else if (flag == "--op") {
      opt.op = value();
      if (opt.op != "all" && opt.op != "reduce" && opt.op != "scan" && opt.op != "softmax") {
        throw std::invalid_argument("--op must be all, reduce, scan or softmax");
      }
    } else if (flag == "--sizes") {
      opt.sizes = parse_int_list(value());
    } else if (flag == "--rows") {
      opt.rows = std::stoi(value());
    } else if (flag == "--cols") {
      opt.cols = parse_int_list(value());
    } else if (flag == "--kernels") {
      const std::string v = value();
      if (v != "all") opt.kernel_ids = parse_int_list(v);
    } else if (flag == "--iters") {
      opt.iters = std::stoi(value());
    } else if (flag == "--warmup") {
      opt.warmup = std::stoi(value());
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
  if (opt.iters < 1 || opt.warmup < 0 || opt.rows < 1) {
    throw std::invalid_argument("--iters and --rows must be >= 1, --warmup >= 0");
  }
  for (int n : opt.sizes) check_count(n, "size");
  for (int c : opt.cols) check_count(static_cast<long long>(c) * opt.rows, "rows * cols");
  return opt;
}

bool selected(const Options& opt, int id) {
  return opt.kernel_ids.empty() ||
         std::find(opt.kernel_ids.begin(), opt.kernel_ids.end(), id) != opt.kernel_ids.end();
}

void validate_kernel_ids(const Options& opt) {
  const int max_id = static_cast<int>(
      std::max({reduce_registry().size(), scan_registry().size(), softmax_registry().size()}));
  for (int id : opt.kernel_ids) {
    if (id > max_id) throw std::invalid_argument("unknown kernel id " + std::to_string(id));
  }
}

// ----------------------------------------------------------------------------
// Timing
// ----------------------------------------------------------------------------
template <typename Launch>
TimingStats measure(Launch&& launch, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) launch();
  PRIM_CUDA_CHECK(cudaGetLastError());
  PRIM_CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    timer.start();
    launch();
    samples.push_back(timer.stop());
  }
  PRIM_CUDA_CHECK(cudaGetLastError());

  std::sort(samples.begin(), samples.end());
  const std::size_t mid = samples.size() / 2;
  TimingStats stats;
  stats.min_ms = samples.front();
  stats.median_ms = (samples.size() % 2 == 1) ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
  return stats;
}

/// Empirical achievable bandwidth: a device-to-device copy counts read + write bytes.
double measure_copy_bandwidth_gbps(const Options& opt) {
  constexpr std::size_t kBytes = std::size_t{128} << 20;
  DeviceBuffer<unsigned char> src(kBytes);
  DeviceBuffer<unsigned char> dst(kBytes);
  const TimingStats t = measure(
      [&] { PRIM_CUDA_CHECK(cudaMemcpyAsync(dst.get(), src.get(), kBytes, cudaMemcpyDeviceToDevice, nullptr)); },
      opt.warmup, opt.iters);
  return 2.0 * static_cast<double>(kBytes) / (t.median_ms * 1.0e6);
}

// ----------------------------------------------------------------------------
// Reporting (console table + CSV)
// ----------------------------------------------------------------------------
class Reporter {
 public:
  Reporter(const std::string& csv_path, double copy_bw_gbps) : copy_bw_(copy_bw_gbps) {
    if (!csv_path.empty()) {
      csv_.open(csv_path);
      if (!csv_) throw std::runtime_error("cannot open CSV file '" + csv_path + "'");
      csv_ << "op,problem,kernel_id,kernel,median_ms,min_ms,gb_per_s,pct_copy_bw,pct_baseline,verification\n";
    }
  }

  void begin_problem(const std::string& op, const std::string& problem) {
    op_ = op;
    problem_ = problem;
    std::printf("\n[%s] %s\n", op.c_str(), problem.c_str());
    std::printf("  %-3s %-22s %12s %10s %11s %11s  %s\n", "id", "kernel", "median [ms]", "GB/s", "% copy BW",
                "% baseline", "verification");
  }

  /// `bytes` = minimum necessary traffic; `baseline_ms` <= 0 when there is no baseline.
  void row(const std::string& id, const std::string& name, const TimingStats& t, double bytes,
           double baseline_ms, const std::string& verdict) {
    const double gbps = bytes / (t.median_ms * 1.0e6);
    const std::string pct_copy = copy_bw_ > 0.0 ? format_pct(100.0 * gbps / copy_bw_) : "n/a";
    const std::string pct_base = baseline_ms > 0.0 ? format_pct(100.0 * baseline_ms / t.median_ms) : "n/a";
    std::printf("  %-3s %-22s %12.4f %10.1f %11s %11s  %s\n", id.c_str(), name.c_str(), t.median_ms, gbps,
                pct_copy.c_str(), pct_base.c_str(), verdict.c_str());
    if (csv_.is_open()) {
      csv_ << op_ << ',' << problem_ << ',' << id << ',' << name << ',' << t.median_ms << ',' << t.min_ms << ','
           << gbps << ',' << pct_copy << ',' << pct_base << ',' << verdict << '\n';
    }
  }

  void skipped(const std::string& id, const std::string& name, const std::string& why) {
    std::printf("  %-3s %-22s %s\n", id.c_str(), name.c_str(), why.c_str());
  }

 private:
  static std::string format_pct(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f%%", v);
    return buf;
  }

  double copy_bw_;
  std::string op_;
  std::string problem_;
  std::ofstream csv_;
};

// ----------------------------------------------------------------------------
// reduce
// ----------------------------------------------------------------------------
bool bench_reduce(const Options& opt, Reporter& rep) {
  bool all_ok = true;
  for (int n : opt.sizes) {
    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> host(static_cast<std::size_t>(n));
    for (float& v : host) v = unit(rng);
    const double expected = opt.verify ? reduce_reference(host) : 0.0;

    DeviceBuffer<float> d_in(host.size());
    DeviceBuffer<float> d_out(1);
    d_in.copy_from_host(host.data(), host.size());
    const double bytes = 4.0 * n;

    rep.begin_problem("reduce", "n = " + std::to_string(n));
    double baseline_ms = 0.0;
    if (opt.baseline) {
      DeviceBuffer<unsigned char> temp(cub_reduce_temp_bytes(n));
      const std::size_t temp_bytes = temp.size();
      const TimingStats t =
          measure([&] { cub_reduce_sum(d_in.get(), d_out.get(), n, temp.get(), temp_bytes, nullptr); },
                  opt.warmup, opt.iters);
      baseline_ms = t.median_ms;
      rep.row("-", "cub", t, bytes, 0.0, "reference");
    }

    for (const ReduceKernel& k : reduce_registry()) {
      if (!selected(opt, k.id)) continue;
      DeviceBuffer<float> scratch(k.scratch_elems(n));
      const auto launch = [&] { k.run(d_in.get(), d_out.get(), n, scratch.get(), nullptr); };

      std::string verdict = "not checked";
      if (opt.verify) {
        launch();
        PRIM_CUDA_CHECK(cudaGetLastError());
        PRIM_CUDA_CHECK(cudaDeviceSynchronize());
        float got = 0.0f;
        d_out.copy_to_host(&got, 1);
        const double rel = std::fabs(got - expected) / std::fabs(expected);
        const bool ok = rel <= 1e-4;
        all_ok = all_ok && ok;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s (rel err %.1e)", ok ? "PASS" : "FAIL", rel);
        verdict = buf;
      }
      const TimingStats t = measure(launch, opt.warmup, opt.iters);
      rep.row(std::to_string(k.id), k.name, t, bytes, baseline_ms, verdict);
    }
  }
  return all_ok;
}

// ----------------------------------------------------------------------------
// scan
// ----------------------------------------------------------------------------
bool bench_scan(const Options& opt, Reporter& rep) {
  bool all_ok = true;
  for (int n : opt.sizes) {
    std::mt19937 rng(opt.seed);
    std::uniform_int_distribution<int> small_int(0, 3);  // keeps prefix sums inside int32
    std::vector<int> host(static_cast<std::size_t>(n));
    for (int& v : host) v = small_int(rng);
    const std::vector<int> expected = opt.verify ? scan_reference(host) : std::vector<int>();

    DeviceBuffer<int> d_in(host.size());
    DeviceBuffer<int> d_out(host.size());
    d_in.copy_from_host(host.data(), host.size());
    const double bytes = 8.0 * n;  // n reads + n writes

    rep.begin_problem("scan", "n = " + std::to_string(n));
    double baseline_ms = 0.0;
    if (opt.baseline) {
      DeviceBuffer<unsigned char> temp(cub_scan_temp_bytes(n));
      const std::size_t temp_bytes = temp.size();
      const TimingStats t = measure(
          [&] { cub_scan_exclusive_sum(d_in.get(), d_out.get(), n, temp.get(), temp_bytes, nullptr); },
          opt.warmup, opt.iters);
      baseline_ms = t.median_ms;
      rep.row("-", "cub", t, bytes, 0.0, "reference");
    }

    for (const ScanKernel& k : scan_registry()) {
      if (!selected(opt, k.id)) continue;
      DeviceBuffer<int> scratch(k.scratch_elems(n));
      const auto launch = [&] { k.run(d_in.get(), d_out.get(), n, scratch.get(), nullptr); };

      std::string verdict = "not checked";
      if (opt.verify) {
        launch();
        PRIM_CUDA_CHECK(cudaGetLastError());
        PRIM_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> got(host.size());
        d_out.copy_to_host(got.data(), got.size());
        const CompareResult r = compare_ints(got, expected);
        const bool ok = r.mismatches == 0;
        all_ok = all_ok && ok;
        char buf[64];
        if (ok) {
          std::snprintf(buf, sizeof(buf), "PASS (exact)");
        } else {
          std::snprintf(buf, sizeof(buf), "FAIL (%zu mismatches)", r.mismatches);
        }
        verdict = buf;
      }
      const TimingStats t = measure(launch, opt.warmup, opt.iters);
      rep.row(std::to_string(k.id), k.name, t, bytes, baseline_ms, verdict);
    }
  }
  return all_ok;
}

// ----------------------------------------------------------------------------
// softmax
// ----------------------------------------------------------------------------
bool bench_softmax(const Options& opt, Reporter& rep) {
  bool all_ok = true;
  for (int cols : opt.cols) {
    const int rows = opt.rows;
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<float> logits(-8.0f, 8.0f);
    std::vector<float> host(count);
    for (float& v : host) v = logits(rng);
    const std::vector<float> expected = opt.verify ? softmax_reference(host, rows, cols) : std::vector<float>();

    DeviceBuffer<float> d_in(count);
    DeviceBuffer<float> d_out(count);
    d_in.copy_from_host(host.data(), count);
    const double bytes = 8.0 * static_cast<double>(count);  // one read + one write of the matrix

    rep.begin_problem("softmax", "rows = " + std::to_string(rows) + ", cols = " + std::to_string(cols));
    for (const SoftmaxKernel& k : softmax_registry()) {
      if (!selected(opt, k.id)) continue;
      const auto launch = [&] { k.run(d_in.get(), d_out.get(), rows, cols, nullptr); };

      std::string verdict = "not checked";
      if (opt.verify) {
        launch();
        PRIM_CUDA_CHECK(cudaGetLastError());
        PRIM_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> got(count);
        d_out.copy_to_host(got.data(), count);
        const CompareResult r = compare_floats(got, expected, 1e-6, 1e-4);
        const bool ok = r.mismatches == 0;
        all_ok = all_ok && ok;
        char buf[80];
        if (ok) {
          std::snprintf(buf, sizeof(buf), "PASS (max abs err %.1e)", r.max_abs_err);
        } else {
          std::snprintf(buf, sizeof(buf), "FAIL (%zu mismatches)", r.mismatches);
        }
        verdict = buf;
      }
      const TimingStats t = measure(launch, opt.warmup, opt.iters);
      rep.row(std::to_string(k.id), k.name, t, bytes, 0.0, verdict);
    }
  }
  return all_ok;
}

// ----------------------------------------------------------------------------
int run(const Options& opt) {
  validate_kernel_ids(opt);
  PRIM_CUDA_CHECK(cudaSetDevice(opt.device));
  cudaDeviceProp prop{};
  PRIM_CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  std::printf("Device %d: %s | compute capability %d.%d | %d SMs | %.1f GiB\n", opt.device, prop.name, prop.major,
              prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  double copy_bw = 0.0;
  if (opt.baseline) {
    copy_bw = measure_copy_bandwidth_gbps(opt);
    std::printf("Device-to-device copy bandwidth (read + write): %.1f GB/s\n", copy_bw);
  }

  Reporter reporter(opt.csv_path, copy_bw);
  bool all_ok = true;
  if (opt.op == "all" || opt.op == "reduce") all_ok = bench_reduce(opt, reporter) && all_ok;
  if (opt.op == "all" || opt.op == "scan") all_ok = bench_scan(opt, reporter) && all_ok;
  if (opt.op == "all" || opt.op == "softmax") all_ok = bench_softmax(opt, reporter) && all_ok;

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
