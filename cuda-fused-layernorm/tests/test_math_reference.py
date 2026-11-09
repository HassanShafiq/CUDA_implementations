# SPDX-License-Identifier: MIT
"""GPU-free verification of the mathematics the kernels implement (NumPy only).

  1. The closed-form LayerNorm backward used by csrc/kernels/ln_backward.cu matches central finite
     differences of the forward function.
  2. The statistics scheme of the fused forward kernels (exact per-thread (count, mean, M2), merged
     with Chan's formula in a fixed order) reproduces the true variance, also when |mean| >> std,
     where E[x^2] - E[x]^2 loses essentially all precision in float32.

Run:  pytest tests/test_math_reference.py        (no GPU, no PyTorch required)
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------------------------
# 1. Backward formulas
# ---------------------------------------------------------------------------------------------
def forward(x, gamma, beta, eps):
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    return (x - mean) * rstd * gamma + beta, mean, rstd


def backward(dy, x, gamma, mean, rstd):
    """Exactly the formulas of ln_backward.cu."""
    xhat = (x - mean) * rstd
    dxhat = dy * gamma
    c1 = dxhat.mean(axis=1, keepdims=True)
    c2 = (dxhat * xhat).mean(axis=1, keepdims=True)
    dx = rstd * (dxhat - c1 - xhat * c2)
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    return dx, dgamma, dbeta


@pytest.mark.parametrize("rows,cols", [(1, 2), (3, 5), (4, 37), (6, 128)])
def test_backward_matches_finite_differences(rows, cols):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(rows, cols)) * 2 + 3
    gamma = rng.uniform(0.5, 1.5, size=cols)
    beta = rng.normal(size=cols)
    dy = rng.normal(size=(rows, cols))
    eps = 1e-5

    y, mean, rstd = forward(x, gamma, beta, eps)
    dx, dgamma, dbeta = backward(dy, x, gamma, mean, rstd)

    def loss(x_, g_, b_):
        return float((forward(x_, g_, b_, eps)[0] * dy).sum())

    h = 1e-6

    def numeric(arr, index, which):
        args = [x.copy(), gamma.copy(), beta.copy()]
        args[which][index] += h
        up = loss(*args)
        args[which][index] -= 2 * h
        return (up - loss(*args)) / (2 * h)

    for idx in np.ndindex(*x.shape):
        assert dx[idx] == pytest.approx(numeric(x, idx, 0), abs=1e-6, rel=1e-5)
    for c in range(cols):
        assert dgamma[c] == pytest.approx(numeric(gamma, (c,), 1), abs=1e-6, rel=1e-5)
        assert dbeta[c] == pytest.approx(numeric(beta, (c,), 2), abs=1e-6, rel=1e-5)


def test_dx_properties():
    """With eps = 0, dx sums to zero over a row and is orthogonal to xhat: LayerNorm's output is
    invariant to shifting and positively scaling a row, which removes exactly those two gradient
    directions. (For eps > 0 the second property holds only up to a residual of O(eps / var).)"""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 64)) * 2 + 1
    dy = rng.normal(size=(5, 64))
    gamma = np.ones(64)
    _, mean, rstd = forward(x, gamma, np.zeros(64), 0.0)
    dx, _, _ = backward(dy, x, gamma, mean, rstd)
    xhat = (x - mean) * rstd
    np.testing.assert_allclose(dx.sum(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose((dx * xhat).sum(axis=1), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------------------------
# 2. Statistics scheme of the fused forward kernels
# ---------------------------------------------------------------------------------------------
def merge(a, b):
    """Chan et al. parallel merge of (count, mean, M2) triples; mirrors ln::merge."""
    n = a[0] + b[0]
    if n == 0:
        return (a[0] * 0, a[1] * 0, a[2] * 0)
    delta = b[1] - a[1]
    frac = b[0] / n
    return (n, a[1] + delta * frac, a[2] + b[2] + delta * delta * a[0] * frac)


def kernel_style_stats(row, threads, dtype):
    """Thread t owns elements t, t+threads, ...; exact local two-pass statistics from the 'registers';
    then merge across threads (tree order like the butterfly reduction)."""
    row = row.astype(dtype)
    parts = []
    for t in range(threads):
        mine = row[t::threads]
        if mine.size == 0:
            parts.append((dtype(0), dtype(0), dtype(0)))
            continue
        count = dtype(mine.size)
        mean = mine.sum(dtype=dtype) / count
        m2 = ((mine - mean) ** 2).sum(dtype=dtype)
        parts.append((count, mean, m2))
    while len(parts) > 1:  # butterfly-like pairwise reduction
        half = len(parts) // 2
        parts = [merge(parts[i], parts[i + half]) for i in range(half)]
    return parts[0]


@pytest.mark.parametrize("cols", [1, 31, 128, 1000, 4096])
@pytest.mark.parametrize("threads", [32, 256])
def test_merge_scheme_reproduces_variance(cols, threads):
    rng = np.random.default_rng(2)
    row = rng.normal(size=cols) * 3 + 2
    count, mean, m2 = kernel_style_stats(row, threads, np.float64)
    assert count == cols
    assert mean == pytest.approx(row.mean(), rel=1e-12)
    assert m2 / count == pytest.approx(row.var(), rel=1e-10)


def test_welford_beats_naive_formula_when_mean_dominates():
    rng = np.random.default_rng(3)
    row = (rng.normal(size=4096) + 1000.0).astype(np.float32)
    true_var = row.astype(np.float64).var()

    naive = float(np.float32(np.mean(row * row, dtype=np.float32) - np.mean(row, dtype=np.float32) ** 2))
    count, mean, m2 = kernel_style_stats(row, 256, np.float32)
    welford = float(m2 / count)

    err_naive = abs(naive - true_var) / true_var
    err_welford = abs(welford - true_var) / true_var
    assert err_welford < 1e-4
    assert err_naive > 100 * err_welford  # the textbook formula is orders of magnitude worse
