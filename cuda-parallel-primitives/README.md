# Parallel Primitives in CUDA: Reduction, Prefix Sum and Softmax

Three of the most reused GPU building blocks, each implemented as a ladder of kernels where every
step removes one specific bottleneck. All kernels are verified against a CPU reference and benchmarked
against CUB (reduction, scan) and against the GPU's measured copy bandwidth (all three). The goal is to
show *why* each step helps and how to confirm it with a profiler, not just to reach a high number.

| Primitive | Operation | Data type |
|-----------|-----------|-----------|
| `reduce`  | sum of `n` values to one value | `float` |
| `scan`    | **exclusive** prefix sum | `int32` (sums must fit in int32) |
| `softmax` | numerically stable row-wise softmax of a `rows x cols` matrix | `float` |

## Kernel ladders

### Reduction

| # | Kernel | Idea | Bottleneck addressed |
|---|--------|------|----------------------|
| 1 | `interleaved_divergent` | tree reduction, `tid % (2s) == 0` | (baseline) warp divergence, slow modulo |
| 2 | `interleaved_strided` | `index = 2*s*tid` | divergence removed; introduces shared-memory bank conflicts |
| 3 | `sequential` | `tid < s: a[tid] += a[tid+s]` | bank conflicts removed |
| 4 | `first_add_on_load` | two elements per thread added during the load | idle threads in the first step, block count |
| 5 | `warp_shuffle` | grid-stride loop, `float4` loads, warp shuffles | launch/scheduling overhead, `__syncthreads`, load width |

### Prefix sum (exclusive)

| # | Kernel | Idea | Trade-off |
|---|--------|------|-----------|
| 1 | `hillis_steele` | log n steps, all threads active | O(n log n) work |
| 2 | `blelloch` | work-efficient up-sweep / down-sweep | O(n) work, but bank conflicts |
| 3 | `blelloch_padded` | same, with `i + (i >> 5)` index padding | conflict-free, 1/32 extra shared memory |
| 4 | `warp_shuffle` | 4 items/thread (`int4`), register shuffles, 2 block syncs | fewest synchronisations |

Arbitrary `n` is handled by a three-phase scheme (scan tiles, scan the tile totals recursively, add offsets).

### Softmax

| # | Kernel | Idea | Global passes over a row |
|---|--------|------|--------------------------|
| 1 | `naive` | one thread per row | 3 reads + 1 write |
| 2 | `block_smem` | block per row, shared-memory tree reductions | 3 reads + 1 write |
| 3 | `warp_per_row` | warp per row, shuffle reductions, no block syncs | 3 reads + 1 write |
| 4 | `online_block` | single-pass online (max, sum) | 2 reads + 1 write |

Details, derivations and the profiler metrics to check at each step are in
[`docs/OPTIMIZATION_NOTES.md`](docs/OPTIMIZATION_NOTES.md) and [`docs/PROFILING.md`](docs/PROFILING.md).

## Project layout

```
include/prim/
  primitives.hpp      public API: kernel registries for reduce, scan, softmax
  cuda_utils.hpp      error checking, RAII DeviceBuffer, GpuTimer
  cub_baseline.hpp    CUB reference wrappers
  reference.hpp       CPU reference implementations and comparison helpers
src/
  reduce/reduce.cu    five reduction kernels + multi-pass driver
  scan/scan.cu        four block-scan kernels + three-phase driver
  softmax/softmax.cu  four softmax kernels
  common/             device helpers (shuffles, online-softmax math), CUB wrappers
  benchmark.cpp       prim_bench: verification + timing + CSV output
tests/
  test_correctness.cpp  prim_tests
docs/                 optimization notes and profiling guide
scripts/              benchmark helper (PowerShell)
```

## Requirements

* NVIDIA GPU (compute capability 3.0+ for warp shuffles) with a recent driver
* CUDA Toolkit 12.x (includes `nvcc` and CUB)
* CMake 3.24 or newer
* **Visual Studio 2022** (Desktop development with C++) on Windows, or GCC/Clang on Linux
* Ninja (bundled with Visual Studio's CMake tools)

## Build and run

### Visual Studio 2022 (Windows)

1. Install the CUDA Toolkit *after* Visual Studio so the integration is detected.
2. `File > Open > Folder...` and select the repository root. Visual Studio reads `CMakePresets.json`.
3. Select the **Windows x64 Release** configuration, then `Build > Build All`.
4. Run from a terminal:

```powershell
.\out\build\windows-release\prim_tests.exe
.\out\build\windows-release\prim_bench.exe --op all
```

### Command line

From an *x64 Native Tools Command Prompt for VS 2022* (Windows) or any shell (Linux):

```bash
cmake --preset windows-release        # or: linux-release
cmake --build --preset windows-release
ctest --preset windows-release
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`; override with e.g. `-DCMAKE_CUDA_ARCHITECTURES=86`.

> Always benchmark the **Release** configuration. In Debug, CMake compiles device code with `-G`
> (unoptimised), which invalidates every comparison.

## Using the benchmark

```
prim_bench [--op all|reduce|scan|softmax] [--sizes 1048576,16777216]
           [--rows 8192 --cols 256,1024,4096] [--kernels 1,3] [--iters 20] [--warmup 5]
           [--no-verify] [--no-baseline] [--csv results/run.csv] [--device 0]
```

Each row reports median time, effective bandwidth (GB/s), percentage of the measured device-to-device
copy bandwidth, speed relative to CUB (reduce and scan only) and the verification verdict.
For a timestamped CSV on Windows: `.\scripts\run_benchmarks.ps1`.

## Methodology

**Correctness.** `prim_tests` checks every kernel of every ladder:

* *reduce* uses integer-valued floats, whose sums are exact in any evaluation order, so all variants must
  agree with the reference **bit for bit**; random floats are also checked with a relative tolerance.
* *scan* is compared exactly against a 64-bit CPU scan on sizes straddling every tile size
  (up to three recursion levels).
* *softmax* covers shapes around warp and block boundaries, very large logits (which overflow `exp`
  without max-subtraction), `-inf` masks and fully masked rows.
* Reduce and scan are also run on **misaligned** pointers to exercise the scalar fallbacks of the
  vectorised paths. `prim_tests --quick` caps the problem sizes.

`prim_bench` additionally verifies every kernel on the benchmarked problem before timing it.

**Timing.** CUDA events around each launch, untimed warm-up, median of the timed iterations.

**Metric.** All three primitives are memory-bound, so effective bandwidth is computed from the *minimum
necessary* traffic (reduce: `4n` bytes; scan: `8n`; softmax: one read plus one write of the matrix).
Algorithms that make several passes move more data and therefore score below 100% by construction.
"% copy BW" uses a measured device-to-device `cudaMemcpy` (read plus write bytes) as an empirical stand-in
for achievable DRAM bandwidth.

**Caveats.** Clocks are not locked and L2 is not flushed between launches. Small inputs fit in L2 and can
exceed the DRAM-based reference. Compare variants within one machine and repeat runs.

## Results

Results depend on the GPU, driver and clocks, so none are hard-coded. After running `prim_bench`, record
your numbers here together with the hardware they came from.

```
GPU:       <model>, driver <version>, CUDA <version>
Command:   prim_bench --op all --sizes 1048576,16777216 --iters 50
(paste the tables printed by prim_bench here)
```

## Design decisions

* **Registries instead of ad-hoc lists.** Benchmark and tests iterate over the same registries, so a new
  variant is automatically tested and benchmarked.
* **Library plus thin executables; RAII everywhere.** No raw `cudaMalloc/cudaFree`; failures throw
  `CudaError` with file, line and expression.
* **Arbitrary sizes.** Bounds-checked loads for all `n`, recursive multi-level scan, multi-pass reduction
  with ping-pong scratch buffers; element counts are validated (`<= 2^30`) so 32-bit index math cannot overflow.
* **Run-time alignment checks** in vectorised kernels (`float4` / `int4`) with scalar fallback, instead of
  undefined behaviour on misaligned pointers.
* **Defined semantics for edge cases.** `-inf` entries and fully masked rows in softmax yield 0 rather than NaN;
  documented in `primitives.hpp`.
* **Exact-arithmetic tests** for operations that are not associative in floating point.

## Known limitations and next steps

* Scan uses the classic three-phase scheme (~4n words of traffic). A single-pass *decoupled look-back* scan
  (as in CUB) needs ~2n and is the natural next step, as are scans of other element types and an inclusive variant.
* Reduction handles `float` sums only; generalising to templated operators (min/max/custom) is straightforward.
* Softmax kernels re-read the row from global memory; a variant keeping the row in shared memory or registers
  (when it fits) would reduce traffic to the minimum. Vectorised row loads are not implemented.
* Tile sizes are compile-time constants chosen for generality, not auto-tuned per architecture.

## License

MIT, see [`LICENSE`](LICENSE).
