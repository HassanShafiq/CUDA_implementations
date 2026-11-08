// SPDX-License-Identifier: MIT
//
// Host reference implementations (ground truth for tests) and comparison helpers.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace prim {

/// Sum accumulated in double precision.
inline double reduce_reference(const std::vector<float>& v) {
  double sum = 0.0;
  for (float x : v) sum += x;
  return sum;
}

/// Exclusive prefix sum: out[i] = v[0] + ... + v[i-1], out[0] = 0 (64-bit accumulation).
inline std::vector<int> scan_reference(const std::vector<int>& v) {
  std::vector<int> out(v.size());
  long long running = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = static_cast<int>(running);
    running += v[i];
  }
  return out;
}

/// Softmax with the semantics documented in primitives.hpp, in double precision.
inline std::vector<float> softmax_reference(const std::vector<float>& in, int rows, int cols) {
  const double neg_inf = -std::numeric_limits<double>::infinity();
  std::vector<float> out(in.size());
  for (int r = 0; r < rows; ++r) {
    const float* x = &in[static_cast<std::size_t>(r) * cols];
    float* y = &out[static_cast<std::size_t>(r) * cols];
    double row_max = neg_inf;
    for (int c = 0; c < cols; ++c) row_max = std::max(row_max, static_cast<double>(x[c]));
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += (x[c] == -INFINITY) ? 0.0 : std::exp(x[c] - row_max);
    for (int c = 0; c < cols; ++c) {
      const double e = (x[c] == -INFINITY) ? 0.0 : std::exp(x[c] - row_max);
      y[c] = static_cast<float>(sum > 0.0 ? e / sum : 0.0);
    }
  }
  return out;
}

struct CompareResult {
  std::size_t mismatches = 0;
  double max_abs_err = 0.0;
  std::size_t first_mismatch = 0;
};

/// Element is accepted when |got - expected| <= atol + rtol * |expected|; NaN is always a mismatch.
inline CompareResult compare_floats(const std::vector<float>& got, const std::vector<float>& expected,
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

/// Exact comparison for integer results.
inline CompareResult compare_ints(const std::vector<int>& got, const std::vector<int>& expected) {
  CompareResult r;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      if (r.mismatches == 0) r.first_mismatch = i;
      ++r.mismatches;
    }
  }
  return r;
}

}  // namespace prim
