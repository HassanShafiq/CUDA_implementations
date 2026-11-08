// SPDX-License-Identifier: MIT
//
// Correctness tests: every kernel is checked against a double-precision CPU
// reference on shapes chosen to break tile-size assumptions (sizes of 1, primes,
// off-by-one around tile edges, tall/skinny and wide matrices) with both the
// beta == 0 and beta != 0 epilogues. Kernels that do not support a shape
// (e.g. the float4 kernel when n or k is not a multiple of 4) are skipped.
//
// Exit status: 0 = all executed checks passed, 1 = at least one failure,
// 2 = environment error (e.g. no CUDA device).
#include <cstdio>
#include <exception>
#include <random>
#include <vector>

#include "gemm/cuda_utils.hpp"
#include "gemm/kernels.hpp"
#include "gemm/reference.hpp"

namespace {

struct Shape {
  int m;
  int n;
  int k;
};

struct Epilogue {
  float alpha;
  float beta;
};

bool run_case(const gemm::KernelInfo& kernel, const Shape& s, const Epilogue& e, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const std::size_t na = static_cast<std::size_t>(s.m) * s.k;
  const std::size_t nb = static_cast<std::size_t>(s.k) * s.n;
  const std::size_t nc = static_cast<std::size_t>(s.m) * s.n;
  std::vector<float> a(na), b(nb), c0(nc), expected, got(nc);
  for (float& v : a) v = dist(rng);
  for (float& v : b) v = dist(rng);
  for (float& v : c0) v = dist(rng);

  gemm::reference_gemm(s.m, s.n, s.k, e.alpha, a, b, e.beta, c0, expected);

  gemm::DeviceBuffer<float> da(na), db(nb), dc(nc);
  da.copy_from_host(a.data(), na);
  db.copy_from_host(b.data(), nb);
  dc.copy_from_host(c0.data(), nc);

  const gemm::GemmArgs args{s.m, s.n, s.k, e.alpha, da.get(), db.get(), e.beta, dc.get()};
  kernel.launch(args, nullptr);
  GEMM_CUDA_CHECK(cudaGetLastError());
  GEMM_CUDA_CHECK(cudaDeviceSynchronize());
  dc.copy_to_host(got.data(), nc);

  const gemm::CompareResult r = gemm::compare_results(got, expected, 1e-3, 1e-4);
  if (r.mismatches != 0) {
    std::printf("  FAIL %-13s m=%d n=%d k=%d alpha=%g beta=%g : %zu mismatches, max abs err %.3e\n",
                kernel.name, s.m, s.n, s.k, e.alpha, e.beta, r.mismatches, r.max_abs_err);
    return false;
  }
  return true;
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    GEMM_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      std::fputs("No CUDA device found.\n", stderr);
      return 2;
    }

    const std::vector<Shape> shapes = {
        {1, 1, 1},    {1, 4, 4},     {7, 5, 3},     {8, 8, 8},     {31, 33, 17}, {32, 32, 32},
        {33, 65, 31}, {64, 64, 64},  {64, 128, 32}, {127, 129, 131}, {128, 128, 128}, {131, 260, 36},
        {257, 130, 67}, {300, 300, 300}, {5, 512, 8}, {512, 4, 40}, {129, 129, 129}};
    const std::vector<Epilogue> epilogues = {{1.0f, 0.0f}, {0.7f, 0.3f}, {1.0f, 1.0f}};

    std::mt19937 rng(12345);
    int passed = 0, failed = 0, skipped = 0;
    for (const gemm::KernelInfo& kernel : gemm::kernel_registry()) {
      int kernel_failed = 0;
      for (const Shape& s : shapes) {
        for (const Epilogue& e : epilogues) {
          // `supports` inspects pointers too, so probe with the shape only: the
          // device buffers inside run_case come from cudaMalloc (256-byte aligned).
          const gemm::GemmArgs probe{s.m, s.n, s.k, e.alpha, nullptr, nullptr, e.beta, nullptr};
          if (!kernel.supports(probe)) {
            ++skipped;
            continue;
          }
          if (run_case(kernel, s, e, rng)) {
            ++passed;
          } else {
            ++failed;
            ++kernel_failed;
          }
        }
      }
      std::printf("[%s] kernel %d (%s)\n", kernel_failed == 0 ? " OK " : "FAIL", kernel.id, kernel.name);
    }

    std::printf("\n%d passed, %d failed, %d skipped (unsupported shape)\n", passed, failed, skipped);
    return failed == 0 ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 2;
  }
}
