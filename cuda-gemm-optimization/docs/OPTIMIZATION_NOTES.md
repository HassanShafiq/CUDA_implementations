# Optimization notes

For each kernel: the problem it solves, what changed, the arithmetic behind the expectation, and what to
confirm in Nsight Compute (see [PROFILING.md](PROFILING.md)). No performance numbers are quoted; measure
on your own GPU and record them in the README.

Notation: a block computes a `BM x BN` tile of C, stepping through `k` in slices of `BK`; a thread owns
a `TM x TN` patch of that tile. SGEMM performs `2*m*n*k` FLOPs (one FMA = 2 FLOPs).

## Why SGEMM is a good optimization exercise

Naively, every FMA needs two operands from memory, so performance is capped by memory traffic, not
by the GPU's FP32 throughput. The whole ladder is about reusing each loaded value as many times as
possible at progressively faster levels of the hierarchy:

```
global memory  ->  shared memory (per block)  ->  registers (per thread)
```

## Kernel 1 - `naive`

*Mapping:* one thread per element of C; thread `(x, y)` computes `C[x][y]` with `x = threadIdx.x`.

*Problem:* consecutive lanes (consecutive `threadIdx.x`) walk down a **column** of the result. A warp's
load of `A[row][i]` touches 32 addresses `k` floats apart, and its store to `C` is `n` floats apart: up to 32
separate 32-byte sectors per request instead of one 128-byte line.

*Arithmetic intensity:* each thread does `2K` FLOPs and reads `2K` floats (`8K` bytes) -> 0.25 FLOP/byte,
before any cache help.

*Verify:* low `Global Load Efficiency` / many sectors per request (`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`),
memory-bound in the Speed Of Light section.

## Kernel 2 - `coalesced`

*Change:* swap the roles of `x` and `y` so consecutive lanes map to consecutive **columns**.
Per warp: `B[i][col]` is one contiguous 128-byte request, `C` stores are contiguous, and `A[row][i]`
is a single address broadcast to all lanes.

*Same arithmetic, far less traffic.* This is a pure memory-access-pattern fix.

*Verify:* sectors per request drops toward the ideal (4 sectors for 128 contiguous bytes);
bytes-per-sector utilization rises; DRAM/L2 traffic for the same problem falls.

## Kernel 3 - `smem_tiled` (32x32)

*Change:* the block loads a `32x32` tile of A and of B into shared memory, synchronizes, computes
the partial product, synchronizes again, then moves to the next `k`-slice.

*Arithmetic intensity (global):* per slice a block loads `2*BS^2` floats (`8*BS^2` bytes) and
performs `2*BS^3` FLOPs -> `BS/4` FLOP/byte = **8** for `BS = 32`, 32x higher than kernel 1.

*Shared-memory pattern:* `tile_a[ty][i]` is a broadcast (all lanes share `ty` and `i`);
`tile_b[i][tx]` reads consecutive words, one per bank. No conflicts.

*Remaining limit:* every FMA still performs two shared-memory loads (2.0 loads/FMA), so the kernel
is now bound by shared-memory bandwidth and load/store issue, not by global memory.

*Verify:* global traffic drops sharply; `Shared Memory` throughput climbs; look at stall reasons
(`barrier` and `short scoreboard`).

## Kernel 4 - `blocktile_1d` (64x64 tile, BK=8, TM=8)

*Change:* a thread computes `TM = 8` vertically adjacent outputs. Per `k`-step it loads one `B` value
into a register and reuses it for 8 FMAs, loading 8 values of `A`.

*Shared loads per FMA:* `(TM + 1) / TM = 1.125` (was 2.0).

*Global intensity for a `BM x BN` tile:* `2*BM*BN*BK` FLOPs per `4*(BM+BN)*BK` bytes =
`BM*BN / (2*(BM+BN))` -> **16** FLOP/byte for 64x64.

*Cost:* each thread now keeps 8 accumulators live; 512 threads per block. Check register use
(`-DGEMM_PTXAS_VERBOSE=ON`) and achieved occupancy.

## Kernel 5 - `regtile_2d` (128x128 tile, BK=8, TM=TN=8)

*Change:* an 8x8 patch per thread, computed as an **outer product**: per `k`-step load 8 values of A
and 8 of B into registers, then do 64 FMAs.

*Shared loads per FMA:* `(TM + TN) / (TM * TN) = 16 / 64 = 0.25` (8x fewer than kernel 3).
*Global intensity:* `128*128 / (2*256)` = **32** FLOP/byte.

*Bank-conflict engineering (the part that is easy to get wrong):*

| Access | Layout choice | Result |
|--------|---------------|--------|
| read `bs[d][thread_col + j*16]` | columns strided by `BN/TN = 16` instead of 8 contiguous per thread | 16 consecutive words per half-warp -> one bank each; the other half-warp broadcasts the same addresses |
| read `as[d][thread_row*8 + i]` | A tile stored **transposed** | a thread's 8 values are contiguous; a warp reads only 2 distinct addresses (different banks) |
| write transposed A | pitch `BM + 4` instead of `BM` | with pitch 128 all lanes with the same `a_row` hit one bank (8-way conflict); with pitch 132 the 32 lanes (`a_col` 0-7, `a_row` 0-3) map to 32 distinct banks |

*Verify:* `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and the `_st` counterpart should
be near zero; FMA pipe utilization should be the dominant limiter.

## Kernel 6 - `vectorized`

*Change:* same tiling as kernel 5, but 128-bit accesses everywhere the layout permits:

* global -> shared: each thread loads one `float4` of the A tile and one of the B tile;
* shared -> registers: A fragment as 2 x `float4`, B fragment as 2 x `float4`
  (4 vector loads instead of 16 scalar loads per `k`-step);
* epilogue: `float4` stores of C.

*Column mapping:* a thread owns columns `[4*tc, 4*tc+3]` and `[BN/2 + 4*tc, ...]`, so eight lanes read
128 contiguous bytes per `float4` load, serviced by a quarter-warp without conflicts.

*Safety:* requires `n % 4 == 0`, `k % 4 == 0` and 16-byte aligned base pointers so every `float4`
is aligned and wholly in or out of bounds. `kernel_registry.cpp` encodes this; the harness skips the
kernel with an explanation otherwise.

*Verify:* in the SASS/source view, global and shared accesses appear as `LDG.E.128` / `LDS.128` /
`STG.E.128`; instruction count and issue-slot utilization improve.

## What is deliberately not here (next steps)

1. **Double buffering / software pipelining** (`cp.async` on Ampere+, or manual register staging) to
   overlap global loads of slice `t+1` with compute on slice `t`.
2. **Warp-level tiling** so each warp owns a larger, explicitly shaped sub-tile (the CUTLASS approach).
3. **Auto-tuning** of `BM, BN, BK, TM, TN` per architecture (they are template parameters already).
4. **Tensor cores** (TF32 / FP16 via WMMA or `mma.sync`), which change the roofline entirely.
5. **Split-K** for skinny problems with too few output tiles to fill the GPU.
