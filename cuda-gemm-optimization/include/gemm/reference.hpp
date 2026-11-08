// SPDX-License-Identifier: MIT
//
// Host reference implementation and result comparison helpers.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gemm {

/// Straightforward double-precision accumulation; the ground truth for tests.
inline void reference_gemm(int m, int n, int k, float alpha, const std::vector<float>& a,
                           const std::vector<float>& b, float beta, const std::vector<float>& c_in,
                           std::vector<float>& c_out) {
  c_out.assign(static_cast<std::size_t>(m) * n, 0.0f);
  std::vector<double> row(static_cast<std::size_t>(n));
  for (int i = 0; i < m; ++i) {
    std::fill(row.begin(), row.end(), 0.0);
    for (int p = 0; p < k; ++p) {  // i-p-j order keeps B accesses sequential
      const double a_ip = a[static_cast<std::size_t>(i) * k + p];
      for (int j = 0; j < n; ++j) row[j] += a_ip * b[static_cast<std::size_t>(p) * n + j];
    }
    for (int j = 0; j < n; ++j) {
      const std::size_t o = static_cast<std::size_t>(i) * n + j;
      const double old = (beta == 0.0f) ? 0.0 : static_cast<double>(beta) * c_in[o];
      c_out[o] = static_cast<float>(static_cast<double>(alpha) * row[j] + old);
    }
  }
}

struct CompareResult {
  std::size_t mismatches = 0;
  double max_abs_err = 0.0;
  std::size_t first_mismatch = 0;
};

/// Element is accepted when |got - expected| <= atol + rtol * |expected|.
/// NaN in `got` is always a mismatch.
inline CompareResult compare_results(const std::vector<float>& got, const std::vector<float>& expected,
                                     double atol, double rtol) {
  CompareResult r;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double g = got[i];
    const double e = expected[i];
    const double err = std::fabs(g - e);
    const bool bad = std::isnan(g) || err > atol + rtol * std::fabs(e);
    if (bad) {
      if (r.mismatches == 0) r.first_mismatch = i;
      ++r.mismatches;
    }
    if (!std::isnan(err)) r.max_abs_err = std::max(r.max_abs_err, err);
  }
  return r;
}

}  // namespace gemm
