# Optimization notes

What every kernel does, why, and what to confirm in Nsight Compute (see [PROFILING.md](PROFILING.md)).
No performance numbers are quoted here; measure on your own GPU and record them in the README.

## Why LayerNorm is a memory problem

Per element, LayerNorm does roughly ten floating-point operations against `2 * sizeof(T)` bytes of
mandatory traffic (read `x`, write `y`). That is far below the FLOP/byte ratio at which any modern GPU
becomes compute-bound, so runtime is set by how many times data crosses the memory interface and by
how much synchronisation the two row reductions (mean and variance) need.

**Why fuse.** Written as separate PyTorch ops (`mean`, `x - mean`, square, `mean`, `sqrt`, divide, scale,
shift), every step launches a kernel and writes a full-size intermediate tensor to global memory only for the
next kernel to read it back. A fused kernel keeps all intermediates in registers and touches global memory
exactly twice per element. `benchmarks/bench_layernorm.py` includes this unfused composition
(`layer_norm_unfused`) as a baseline. PyTorch's own `F.layer_norm` is already a fused kernel, so it is the
honest bar to compare against.

## Forward ladder (`csrc/kernels/ln_forward.cu`)

| # | Kernel | Mapping | Statistics | Global traffic per row |
|---|--------|---------|------------|------------------------|
| 1 | `two_pass` | block per row | shared-memory tree sum; separate mean and variance passes | 3 reads + 1 write |
| 2 | `welford` | block per row | one streaming Welford pass, shuffle merge | 2 reads + 1 write |
| 3 | `warp_row` | warp per row (4 rows per block) | exact local stats from registers, one warp merge | 1 read + 1 write |
| 4 | `block_fused` | block per row | same, merged across 8 warps | 1 read + 1 write |

`kernel_id = 0` (auto) picks 3, then 4, then 2, depending on what the problem supports.

### Step 1 to 2: removing the variance pass with Welford statistics

The variance needs the mean, so a naive implementation needs two passes before normalising. The textbook
shortcut `E[x^2] - E[x]^2` fuses them but is numerically unsafe: with `|mean| >> std` both terms are huge and
nearly equal, and the subtraction cancels almost all significant digits (see
`tests/test_math_reference.py`, which demonstrates an error orders of magnitude larger in fp32, and
`test_large_mean_is_numerically_stable`).

Welford's method keeps `(count, mean, M2)` with `M2 = sum((x - mean)^2)`. Two such summaries merge exactly
(Chan et al.):

```
n     = n_a + n_b
delta = mean_b - mean_a
mean  = mean_a + delta * n_b / n
M2    = M2_a + M2_b + delta^2 * n_a * n_b / n
```

so threads, warps and the block can combine partial results without ever forming `sum(x^2)`.

*Determinism detail.* `merge(a, b)` is not bitwise commutative. The butterfly warp reduction therefore always
merges `(lower lane, upper lane)` in that order, so all 32 lanes finish with bit-identical statistics and a
row's mean / rstd never differ between lanes. Block-level merging walks the warp results in a fixed order.

### Step 2 to 3: keeping the row in registers

If the whole row fits in registers it never needs to be read twice. Thread `lane` owns the 16-byte chunks
`lane, lane + R, lane + 2R, ...` (`R` = threads per row), at most 8 of them:

1. Load every chunk once with 128-bit loads (4 floats, 8 halves, 2 doubles per chunk).
2. Compute the thread's *exact* local `(count, mean, M2)` from the register copy with a two-pass formula
   (cheap, because the data is already on chip), then merge across the row in a single reduction.
3. Normalise from registers and store with 128-bit stores.

This reaches the 1-read + 1-write minimum. Capacity limits follow from the register budget: `R * 8` chunks,
i.e. 1024 fp32 columns (2048 for fp16 / bf16) for the warp kernel and 8192 fp32 columns for the block kernel.
Larger rows, columns that are not a multiple of the chunk, or misaligned pointers fall back to kernel 2; this
is enforced by each kernel's `supports()` function and covered by tests, not by undefined behavior.

### Step 3 to 4: warp versus block per row

A warp per row needs no `__syncthreads` and packs four rows into a block, which suits many short rows
(typical transformer hidden sizes of 512 to 1024). A block per row gives more threads, and therefore more
loads in flight, to wider rows. Check achieved occupancy and register count (`-DLN_PTXAS_VERBOSE=ON`) for
both: register caching trades registers for traffic, and a spill (local-memory traffic in Nsight Compute)
would defeat the purpose.

### Accumulation policy

Storage types fp16 / bf16 are converted to fp32 immediately after loading, all statistics and arithmetic run
in fp32, and results are rounded once on store. fp64 inputs stay in fp64. The per-row `mean` and `rstd` saved
for backward are fp32 (fp64 for fp64 inputs).

## Backward (`csrc/kernels/ln_backward.cu`)

### The formulas

With `xhat = (x - mean) * rstd`, `y = gamma * xhat + beta`, `dxhat = dy * gamma` and `C` columns:

```
dbeta  = sum over rows of dy
dgamma = sum over rows of dy * xhat
dx     = rstd * ( dxhat - mean_C(dxhat) - xhat * mean_C(dxhat * xhat) )
```

Derivation. `rstd = (var + eps)^(-1/2)`, `var = (1/C) sum (x_k - mean)^2`, so
`d rstd / d x_i = -(rstd^2 / C) * xhat_i`, and

```
d xhat_j / d x_i = rstd * ( delta_ij - 1/C - xhat_i * xhat_j / C )
```

Contracting with `dxhat_j` gives the expression for `dx_i` above. The result is exact for any `eps` (it is
already folded into `rstd`). `tests/test_math_reference.py` checks these formulas against central finite
differences.

### Two reductions in two directions

* `dx` needs sums **along** a row (`sum(dxhat)`, `sum(dxhat * xhat)`): a block cooperates on one row.
* `dgamma` / `dbeta` need sums **across** rows into one value per column: with thousands of rows and only
  hundreds or thousands of columns, a single column-wise kernel would have too little parallelism, and
  floating-point atomics would make results depend on scheduling.

The fused kernel handles both. Block `b` of `G <= 256` blocks processes rows `b, b + G, b + 2G, ...`. Per row it
computes `dx`; thread `t` owns columns `t, t + 256, ...` in every row and accumulates `dgamma` / `dbeta` for
them in a **block-private** accumulator (shared memory when `2 * cols * sizeof(acc) <= 48 KiB`, otherwise a
private slice of global scratch), so no atomics and no synchronisation are needed for the accumulation. Each
block ends with one partial `[2, cols]` result; a small second kernel sums the `G` partials per column in a
fixed order.

Consequences: results are **bitwise reproducible** run to run (tested), and because `G` depends only on `rows`
and not on the GPU, also across GPUs.

### Traffic

Backward must read `x` and `dy` and write `dx` (3 tensor-sized transfers) plus the partial sums. The kernel
re-reads `x` and `dy` in its second per-row loop; they were touched moments earlier, so for typical row lengths
this is served by L1 / L2 rather than DRAM. Caching the row in registers as in the forward pass, and
vectorising the backward loads, are the natural next steps.

## What is deliberately not here

1. Vectorised / register-cached backward (see above).
2. fp32 `gamma` / `beta` with fp16 / bf16 activations (mixed-precision parameters, as PyTorch AMP often uses).
3. Fusing residual add, dropout or bias into the LayerNorm kernel.
4. RMSNorm and GroupNorm variants (same structure, simpler statistics).
5. Row lengths beyond the register-cache capacity with a shared-memory staged variant.
