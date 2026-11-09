// SPDX-License-Identifier: MIT
//
// Double-precision CPU reference for LayerNorm forward and backward (ground truth for tests).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace native {

struct ForwardRef {
  std::vector<double> y;
  std::vector<double> mean;
  std::vector<double> rstd;
};

/// gamma / beta may be null (identity affine transform).
inline ForwardRef layernorm_forward_ref(const std::vector<double>& x, const std::vector<double>* gamma,
                                        const std::vector<double>* beta, int rows, int cols, double eps) {
  ForwardRef r;
  r.y.resize(x.size());
  r.mean.resize(static_cast<std::size_t>(rows));
  r.rstd.resize(static_cast<std::size_t>(rows));
  for (int i = 0; i < rows; ++i) {
    const double* xr = &x[static_cast<std::size_t>(i) * cols];
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) mean += xr[c];
    mean /= cols;
    double var = 0.0;
    for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
    var /= cols;
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) {
      const double g = gamma ? (*gamma)[static_cast<std::size_t>(c)] : 1.0;
      const double b = beta ? (*beta)[static_cast<std::size_t>(c)] : 0.0;
      r.y[static_cast<std::size_t>(i) * cols + c] = (xr[c] - mean) * rstd * g + b;
    }
    r.mean[static_cast<std::size_t>(i)] = mean;
    r.rstd[static_cast<std::size_t>(i)] = rstd;
  }
  return r;
}

struct BackwardRef {
  std::vector<double> dx;
  std::vector<double> dgamma;
  std::vector<double> dbeta;
};

inline BackwardRef layernorm_backward_ref(const std::vector<double>& dy, const std::vector<double>& x,
                                          const std::vector<double>* gamma, const std::vector<double>& mean,
                                          const std::vector<double>& rstd, int rows, int cols) {
  BackwardRef r;
  r.dx.resize(x.size());
  r.dgamma.assign(static_cast<std::size_t>(cols), 0.0);
  r.dbeta.assign(static_cast<std::size_t>(cols), 0.0);
  for (int i = 0; i < rows; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * cols;
    double s1 = 0.0, s2 = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double xhat = (x[base + c] - mean[static_cast<std::size_t>(i)]) * rstd[static_cast<std::size_t>(i)];
      const double g = gamma ? (*gamma)[static_cast<std::size_t>(c)] : 1.0;
      const double dxhat = dy[base + c] * g;
      s1 += dxhat;
      s2 += dxhat * xhat;
      r.dgamma[static_cast<std::size_t>(c)] += dy[base + c] * xhat;
      r.dbeta[static_cast<std::size_t>(c)] += dy[base + c];
    }
    for (int c = 0; c < cols; ++c) {
      const double xhat = (x[base + c] - mean[static_cast<std::size_t>(i)]) * rstd[static_cast<std::size_t>(i)];
      const double g = gamma ? (*gamma)[static_cast<std::size_t>(c)] : 1.0;
      const double dxhat = dy[base + c] * g;
      r.dx[base + c] = rstd[static_cast<std::size_t>(i)] * (dxhat - s1 / cols - xhat * s2 / cols);
    }
  }
  return r;
}

struct Mismatch {
  std::size_t count = 0;
  double max_abs_err = 0.0;
  std::size_t first = 0;
};

/// Element passes when |got - ref| <= atol + rtol * |ref|; NaN always fails.
inline Mismatch compare(const std::vector<double>& got, const std::vector<double>& ref, double atol, double rtol) {
  Mismatch m;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double err = std::fabs(got[i] - ref[i]);
    const bool bad = std::isnan(got[i]) || err > atol + rtol * std::fabs(ref[i]);
    if (bad) {
      if (m.count == 0) m.first = i;
      ++m.count;
    }
    if (!std::isnan(err)) m.max_abs_err = std::max(m.max_abs_err, err);
  }
  return m;
}

}  // namespace native
