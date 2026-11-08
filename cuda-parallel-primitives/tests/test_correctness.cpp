// SPDX-License-Identifier: MIT
//
// Correctness tests for every kernel in every registry.
//
//  * reduce : integer-valued floats make the sum exact in ANY summation order, so each
//             variant must match the reference bit-for-bit; random floats are also
//             checked with a relative tolerance. Misaligned input pointers exercise the
//             scalar fallback of the vectorised kernel.
//  * scan   : exact integer comparison on sizes around every tile boundary (up to three
//             recursion levels), with misaligned input/output pointers.
//  * softmax: shapes around warp/block boundaries; large logits (stability), -inf masks
//             and fully masked rows.
//
// Usage: prim_tests [--quick]     (--quick caps problem sizes for slow environments)
// Exit status: 0 = all passed, 1 = at least one failure, 2 = environment error.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "prim/cuda_utils.hpp"
#include "prim/primitives.hpp"
#include "prim/reference.hpp"

namespace {

using namespace prim;

struct Context {
  bool quick = false;
  std::mt19937 rng{20240607};
  int passed = 0;
  int failed = 0;
};

void report(Context& ctx, bool ok, const std::string& what) {
  if (ok) {
    ++ctx.passed;
  } else {
    ++ctx.failed;
    std::printf("  FAIL %s\n", what.c_str());
  }
}

// ----------------------------------------------------------------------------
// reduce
// ----------------------------------------------------------------------------
float run_reduce(const ReduceKernel& k, const std::vector<float>& host, int misalign) {
  const int n = static_cast<int>(host.size());
  DeviceBuffer<float> d_in(host.size() + misalign);
  DeviceBuffer<float> d_out(1);
  DeviceBuffer<float> scratch(k.scratch_elems(n));
  d_in.copy_from_host(host.data(), host.size(), static_cast<std::size_t>(misalign));
  k.run(d_in.get() + misalign, d_out.get(), n, scratch.get(), nullptr);
  PRIM_CUDA_CHECK(cudaGetLastError());
  PRIM_CUDA_CHECK(cudaDeviceSynchronize());
  float result = 0.0f;
  d_out.copy_to_host(&result, 1);
  return result;
}

void test_reduce(Context& ctx) {
  std::vector<int> sizes = {1, 2, 3, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1000, 4095, 4096, 4097,
                            65535, 65537};
  if (!ctx.quick) {
    sizes.insert(sizes.end(), {1 << 20, (1 << 20) + 13, 3000001});
  }
  std::uniform_int_distribution<int> small_int(0, 3);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);

  for (const ReduceKernel& k : reduce_registry()) {
    const int before = ctx.failed;
    for (int n : sizes) {
      for (int misalign = 0; misalign <= 1; ++misalign) {
        std::vector<float> host(static_cast<std::size_t>(n));
        for (float& v : host) v = static_cast<float>(small_int(ctx.rng));
        const double expected = reduce_reference(host);  // < 2^24: exactly representable
        const float got = run_reduce(k, host, misalign);
        report(ctx, static_cast<double>(got) == expected,
               std::string("reduce/") + k.name + " exact n=" + std::to_string(n) + " misalign=" +
                   std::to_string(misalign) + " got=" + std::to_string(got) + " expected=" +
                   std::to_string(expected));
      }
    }
    for (int n : {1000, 65537, ctx.quick ? 70001 : 1 << 22}) {
      std::vector<float> host(static_cast<std::size_t>(n));
      for (float& v : host) v = unit(ctx.rng);
      const double expected = reduce_reference(host);
      const double got = run_reduce(k, host, 0);
      report(ctx, std::fabs(got - expected) <= 1e-4 * std::fabs(expected),
             std::string("reduce/") + k.name + " random n=" + std::to_string(n));
    }
    std::printf("[%s] reduce %d (%s)\n", ctx.failed == before ? " OK " : "FAIL", k.id, k.name);
  }
}

// ----------------------------------------------------------------------------
// scan
// ----------------------------------------------------------------------------
std::vector<int> run_scan(const ScanKernel& k, const std::vector<int>& host, int misalign) {
  const int n = static_cast<int>(host.size());
  DeviceBuffer<int> d_in(host.size() + misalign);
  DeviceBuffer<int> d_out(host.size() + misalign);
  DeviceBuffer<int> scratch(k.scratch_elems(n));
  d_in.copy_from_host(host.data(), host.size(), static_cast<std::size_t>(misalign));
  k.run(d_in.get() + misalign, d_out.get() + misalign, n, scratch.get(), nullptr);
  PRIM_CUDA_CHECK(cudaGetLastError());
  PRIM_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> result(host.size());
  d_out.copy_to_host(result.data(), host.size(), static_cast<std::size_t>(misalign));
  return result;
}

void test_scan(Context& ctx) {
  // Sizes straddle the tile sizes of every variant (256, 512, 1024 elements) and force
  // two and three levels of recursion.
  std::vector<int> sizes = {1,   2,    3,    31,   32,   33,   255,  256,  257,  511,   512,
                            513, 1023, 1024, 1025, 2048, 4097, 65536, 65537, 70001};
  if (!ctx.quick) {
    sizes.insert(sizes.end(), {(1 << 20) + 5, (1 << 22) + 1});
  }
  std::uniform_int_distribution<int> small_int(0, 3);

  for (const ScanKernel& k : scan_registry()) {
    const int before = ctx.failed;
    for (int n : sizes) {
      for (int misalign = 0; misalign <= 1; ++misalign) {
        std::vector<int> host(static_cast<std::size_t>(n));
        for (int& v : host) v = small_int(ctx.rng);
        const CompareResult r = compare_ints(run_scan(k, host, misalign), scan_reference(host));
        report(ctx, r.mismatches == 0,
               std::string("scan/") + k.name + " n=" + std::to_string(n) + " misalign=" +
                   std::to_string(misalign) + " first mismatch at " + std::to_string(r.first_mismatch));
      }
    }
    std::printf("[%s] scan %d (%s)\n", ctx.failed == before ? " OK " : "FAIL", k.id, k.name);
  }
}

// ----------------------------------------------------------------------------
// softmax
// ----------------------------------------------------------------------------
std::vector<float> run_softmax(const SoftmaxKernel& k, const std::vector<float>& host, int rows, int cols) {
  DeviceBuffer<float> d_in(host.size());
  DeviceBuffer<float> d_out(host.size());
  d_in.copy_from_host(host.data(), host.size());
  k.run(d_in.get(), d_out.get(), rows, cols, nullptr);
  PRIM_CUDA_CHECK(cudaGetLastError());
  PRIM_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> result(host.size());
  d_out.copy_to_host(result.data(), host.size());
  return result;
}

void check_softmax(Context& ctx, const SoftmaxKernel& k, const std::vector<float>& host, int rows, int cols,
                   const std::string& label) {
  const std::vector<float> got = run_softmax(k, host, rows, cols);
  const std::vector<float> expected = softmax_reference(host, rows, cols);
  const CompareResult r = compare_floats(got, expected, 1e-6, 1e-4);
  report(ctx, r.mismatches == 0,
         std::string("softmax/") + k.name + " " + label + " rows=" + std::to_string(rows) + " cols=" +
             std::to_string(cols) + " mismatches=" + std::to_string(r.mismatches) + " max_abs_err=" +
             std::to_string(r.max_abs_err));
}

void test_softmax(Context& ctx) {
  struct Shape {
    int rows;
    int cols;
  };
  std::vector<Shape> shapes = {{1, 1},   {1, 2},   {3, 5},    {7, 31},  {7, 32},  {7, 33},
                               {5, 127}, {5, 128}, {5, 129},  {9, 1000}, {17, 257}, {4, 513}};
  if (!ctx.quick) shapes.insert(shapes.end(), {{4, 4096}, {2, 10000}, {1000, 100}});
  std::uniform_real_distribution<float> logits(-5.0f, 5.0f);

  for (const SoftmaxKernel& k : softmax_registry()) {
    const int before = ctx.failed;
    for (const Shape& s : shapes) {
      std::vector<float> host(static_cast<std::size_t>(s.rows) * s.cols);
      for (float& v : host) v = logits(ctx.rng);
      check_softmax(ctx, k, host, s.rows, s.cols, "random");

      // Large offset: exp() would overflow without max subtraction.
      for (float& v : host) v += 1000.0f;
      check_softmax(ctx, k, host, s.rows, s.cols, "large-logits");

      // Masked entries (-inf); the first row is masked completely when there are >= 2 rows.
      for (std::size_t i = 0; i < host.size(); i += 3) host[i] = -INFINITY;
      if (s.rows >= 2) {
        for (int c = 0; c < s.cols; ++c) host[static_cast<std::size_t>(c)] = -INFINITY;
      }
      check_softmax(ctx, k, host, s.rows, s.cols, "masked");
    }
    std::printf("[%s] softmax %d (%s)\n", ctx.failed == before ? " OK " : "FAIL", k.id, k.name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Context ctx;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--quick") == 0) {
        ctx.quick = true;
      } else {
        std::fprintf(stderr, "usage: prim_tests [--quick]\n");
        return 2;
      }
    }

    int device_count = 0;
    PRIM_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      std::fputs("No CUDA device found.\n", stderr);
      return 2;
    }

    test_reduce(ctx);
    test_scan(ctx);
    test_softmax(ctx);

    std::printf("\n%d checks passed, %d failed\n", ctx.passed, ctx.failed);
    return ctx.failed == 0 ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 2;
  }
}
