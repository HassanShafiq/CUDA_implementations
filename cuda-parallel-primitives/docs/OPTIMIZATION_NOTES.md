# Optimization notes

For each ladder: the problem every step solves, the arithmetic behind the expectation, and what to
confirm in Nsight Compute (see [PROFILING.md](PROFILING.md)). No performance numbers are quoted;
measure on your own GPU and record them in the README.

## 1. Reduction (`src/reduce/reduce.cu`)

A sum performs one add per 4 bytes read (0.25 FLOP/byte). It is memory-bound by construction, so the whole
ladder is about removing everything that is not "stream the input from DRAM": idle threads, divergence,
bank conflicts, barriers, launch and scheduling overhead. The reference points are CUB's `DeviceReduce::Sum`
and the measured device-to-device copy bandwidth.

**Multi-pass structure.** One launch can only reduce each block's slice to one value, so the input goes through
passes `n -> blocks -> ... -> 1`, reading from and writing to two ping-pong scratch buffers. Variants that
cover more elements per block need fewer passes.

| Step | Change | Why it matters | What to verify |
|------|--------|----------------|----------------|
| 1 `interleaved_divergent` | `if (tid % (2*s) == 0)` | Active threads are scattered across warps, so warps stay scheduled with mostly idle lanes; `%` is costly | `smsp__thread_inst_executed_per_inst_executed.ratio` (average active lanes per warp instruction) is low |
| 2 `interleaved_strided` | `index = 2*s*tid` | Active threads are contiguous, whole warps retire. But thread `t` accesses `sdata[2*s*t]`: stride 2, 4, 8, 16, 32 words gives 2-way up to 32-way bank conflicts | active-lane ratio improves; `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and `_st` rise |
| 3 `sequential` | `sdata[tid] += sdata[tid + s]`, `s` halving | Consecutive threads touch consecutive words: no conflicts, no divergence below warp size. Half the threads still idle in the first step | bank conflicts near zero |
| 4 `first_add_on_load` | each thread loads two elements and adds them | Uses the previously idle half of the block, halves the block count and the number of passes | fewer blocks launched; higher achieved bandwidth |
| 5 `warp_shuffle` | grid-stride loop (<= 1024 blocks), `float4` loads, `__shfl_down_sync` | Each thread accumulates ~16 elements in a register (independent loads in flight, one barrier per block instead of eight); 128-bit loads cut the number of memory instructions; shuffles need no shared memory or barriers inside a warp | DRAM throughput close to the copy bandwidth; low barrier stall share |

**Floating-point note.** Addition is not associative, so each variant returns slightly different last bits. The
tests therefore use integer-valued floats (exact in any order) and a relative tolerance for random data.

## 2. Prefix sum (`src/scan/scan.cu`)

**Exclusive scan:** `out[i] = in[0] + ... + in[i-1]`. Block-level kernels scan a tile on chip and emit its total;
a three-phase driver extends this to any `n`:

1. scan every tile independently, writing `sums[tile]`;
2. exclusive-scan `sums` (recursively, same kernel);
3. add the scanned total of all preceding tiles to every element of each tile.

| Step | Work | Synchronisations per tile | Notes |
|------|------|---------------------------|-------|
| 1 `hillis_steele` | O(n log n): about 7x the additions of a sequential scan for a 256-element tile (`8*256 - 255 = 1793` vs 255) | ~9 | every thread active each step, double-buffered, simple |
| 2 `blelloch` | O(n): `n-1` adds up-sweep + `n-1` down-sweep | ~19 | fewer operations but strided shared access (stride doubles each level): 2-way up to 32-way bank conflicts |
| 3 `blelloch_padded` | same | ~19 | index `i + (i >> 5)` skews words that are 32 apart into different banks, largely eliminating conflicts at a cost of 1/32 extra shared memory |
| 4 `warp_shuffle` | O(n), mostly in registers | 2 | each thread loads 4 consecutive ints (one 128-bit load) and scans them serially; Kogge-Stone shuffle scan across the warp; warp 0 scans the 8 warp totals |

**Traffic.** Phases 1 and 3 each read and write the whole array: about `4n` words versus the `2n` minimum. That
is why every variant here is expected to trail CUB, which uses a *single-pass decoupled look-back* scan: tiles
publish their aggregate / inclusive prefix through flags in global memory and later tiles "look back" over
predecessors, so no separate offset pass is required.

**Scratch layout.** Per recursion level `[sums | scanned sums | next level]`, each array rounded up to 16 ints so
sub-arrays stay 16-byte aligned for the `int4` path. Scratch size is
`2 * round16(blocks_l)` summed over levels, available as `ScanKernel::scratch_elems(n)`.

**Overflow.** Values and prefix sums must fit in int32; the benchmark uses inputs in `[0, 3]`.

## 3. Softmax (`src/softmax/softmax.cu`)

`softmax(x)_c = exp(x_c - m) / sum_j exp(x_j - m)` with `m = max(x)`. Subtracting the row maximum is not an
optimisation but a requirement: `exp(1000)` overflows float, `exp(1000 - 1000)` does not (see the `large-logits`
test). Entries equal to `-inf` get probability 0 and an all-`-inf` row gives zeros, which makes masked attention
rows safe.

| Step | Mapping | Reductions | Global passes |
|------|---------|------------|---------------|
| 1 `naive` | thread per row | sequential loops | 3 reads + 1 write; neighbouring threads read addresses `cols` floats apart (uncoalesced); parallelism limited to `rows` threads |
| 2 `block_smem` | block per row, coalesced strided loops | two shared-memory trees (log2 256 = 8 barriers each) | 3 reads + 1 write |
| 3 `warp_per_row` | warp per row (4 rows per block) | register shuffles (`__shfl_xor_sync`), **no** `__syncthreads` | 3 reads + 1 write; best for short rows with many rows in flight |
| 4 `online_block` | block per row | single pass maintaining `(max, sum)` | 2 reads + 1 write |

**Online softmax (Milakov & Gimelshein, 2018).** Keep a running pair `(m, s)` where `s = sum exp(x - m)`. Folding in
`x`: `m' = max(m, x)`, `s' = s * exp(m - m') + exp(x - m')`. Two partial pairs merge with the same rule, which is
associative and commutative, so it composes across loop iterations, lanes, warps and the block. The maximum and the
normaliser are obtained in one traversal of the row.

**Trade-off.** Online softmax removes one pass over the row but adds a rescale (`expf`) whenever the running maximum
changes. For short rows, repeated reads are served by L1/L2, so differences are small; the advantage grows with row
length. Measure.

**Numerical details.** `rescale(old, new)` returns exactly 1 when `old == new`, which avoids `exp(-inf - -inf) = NaN`
on masked prefixes; `exp_shifted` returns 0 for `-inf`; `safe_inverse` returns 0 for an empty sum.

## What is deliberately not here (next steps)

1. **Decoupled look-back scan** (single pass, ~2n traffic).
2. **Templated reduction/scan** over type and operator (min, max, custom monoids), plus an inclusive scan.
3. **Softmax with the row cached in shared memory or registers**, reaching the 1-read + 1-write minimum when it fits;
   vectorised row loads; split rows across blocks for very long rows.
4. **Auto-tuning** of block size and items per thread per architecture.
5. **Cooperative groups** (`cg::reduce`, `cg::inclusive_scan`) as a higher-level rewrite of the shuffle code.
