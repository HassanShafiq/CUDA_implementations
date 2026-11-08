# CUDA SGEMM Optimization: from naive to register-tiled, vectorized kernels

A step-by-step study of single-precision matrix multiplication (SGEMM) on NVIDIA GPUs.
Six kernels, each introducing one optimization, are verified for correctness and benchmarked
against cuBLAS. The goal is to show *why* each step helps and how to confirm it with a profiler,
not just to reach a high number.

```
C (m x n) = alpha * A (m x k) * B (k x n) + beta * C        row-major, FP32
```

## Kernel ladder

| # | Kernel | Optimization introduced | Bottleneck it addresses |
|---|--------|-------------------------|-------------------------|
| 1 | `naive` | One thread per output element | (baseline; deliberately uncoalesced) |
| 2 | `coalesced` | Thread-to-data mapping fixed so a warp touches contiguous memory | Wasted global-memory sectors |
| 3 | `smem_tiled` | 32x32 shared-memory tiles, each value loaded once per block | Redundant global loads, low arithmetic intensity |
| 4 | `blocktile_1d` | Each thread computes 8 outputs; B value reused from a register | Shared-memory load pressure (2.0 to 1.125 loads/FMA) |
| 5 | `regtile_2d` | 8x8 outputs per thread (outer product), conflict-free shared-memory layout | Shared-memory bandwidth (1.125 to 0.25 loads/FMA), bank conflicts |
| 6 | `vectorized` | 128-bit `float4` loads/stores for global and shared memory | Instruction-issue pressure from scalar accesses |

Details, derivations and the Nsight Compute metrics to check at each step are in
[`docs/OPTIMIZATION_NOTES.md`](docs/OPTIMIZATION_NOTES.md) and [`docs/PROFILING.md`](docs/PROFILING.md).

## Project layout

```
include/gemm/
  kernels.hpp          public API: GemmArgs, kernel registry, launchers
  cuda_utils.hpp       error checking, RAII DeviceBuffer, GpuTimer
  cublas_baseline.hpp  cuBLAS wrapper (row-major to column-major mapping)
  reference.hpp        double-precision CPU reference and result comparison
src/
  kernels/             one file per kernel + shared device helpers
  kernel_registry.cpp  ordered kernel list with per-kernel shape requirements
  cublas_baseline.cpp
  benchmark.cpp        gemm_bench: verification + timing + CSV output
tests/
  test_correctness.cpp gemm_tests: odd shapes and both epilogues vs CPU reference
docs/                  optimization notes and profiling guide
scripts/               benchmark helper (PowerShell)
```

## Requirements

* NVIDIA GPU with a recent driver
* CUDA Toolkit 12.x (includes `nvcc` and cuBLAS)
* CMake 3.24 or newer
* A host compiler supported by your CUDA version: **Visual Studio 2022** (Desktop development with C++) on
  Windows, or GCC/Clang on Linux
* Ninja (bundled with Visual Studio's CMake tools)

## Build and run

### Visual Studio 2022 (Windows)

1. Install the CUDA Toolkit *after* Visual Studio so the integration is detected.
2. `File > Open > Folder...` and select the repository root. Visual Studio reads `CMakePresets.json`.
3. Choose the **Windows x64 Release** configuration, then `Build > Build All`.
4. Run from a terminal (arguments are easier there than in the debugger):

```powershell
.\out\build\windows-release\gemm_tests.exe
.\out\build\windows-release\gemm_bench.exe --sizes 1024,2048,4096
```

### Command line

From an *x64 Native Tools Command Prompt for VS 2022* (Windows) or any shell (Linux):

```bash
cmake --preset windows-release        # or: linux-release
cmake --build --preset windows-release
ctest --preset windows-release        # runs gemm_tests
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native` (the installed GPU). Override it if needed:
`cmake --preset windows-release -DCMAKE_CUDA_ARCHITECTURES=86`.

> Always benchmark the **Release** configuration. In Debug, CMake compiles device code with `-G`
> (unoptimized), which makes every kernel dramatically slower and the comparison meaningless.

## Using the benchmark

```
gemm_bench [--sizes 512,1024,2048,4096] [--m M --n N --k K] [--kernels 1,3,6]
           [--iters 20] [--warmup 5] [--alpha 1] [--beta 0]
           [--no-verify] [--no-baseline] [--csv results/run.csv] [--device 0]
```

Example output format:

```
M=2048 N=2048 K=2048  (alpha=1 beta=0, 5 warm-up + 20 timed iterations)
  id kernel          median [ms]        GFLOP/s   % cuBLAS  verification
  -  cublas              ...             ...        100.0%   reference
  1  naive               ...             ...          ...    PASS (max abs err ...)
  ...
```

For a timestamped CSV on Windows: `.\scripts\run_benchmarks.ps1`.

## Methodology

**Correctness.** Two layers. `gemm_tests` compares every kernel with a double-precision CPU reference
on 17 shapes chosen to break tile assumptions (1x1x1, primes, tile-edge +/- 1, tall/skinny, wide) with
`beta = 0`, a generic `alpha/beta`, and `beta = 1`. It also covers the BLAS rule that `C` is never read
when `beta == 0`. `gemm_bench` additionally verifies each kernel against cuBLAS on the benchmarked
problem before timing it, so a fast-but-wrong kernel cannot appear in the results.

**Timing.** CUDA events around each launch, untimed warm-up, median of the timed iterations (robust to
clock and OS jitter); the minimum is stored in the CSV. GFLOP/s uses `2*m*n*k` operations. cuBLAS runs
in default math mode (no TF32), so the comparison is FP32 CUDA-core code against FP32 CUDA-core code.

**Fair-comparison caveats.** Inputs are not flushed from L2 between launches, and clocks are not locked;
treat results as comparative within one machine, and re-run a few times.

## Results

Results depend on the GPU, driver and clocks, so none are hard-coded here. After running
`gemm_bench`, record your own numbers below together with the hardware they came from.

```
GPU:            <model>, driver <version>, CUDA <version>
Command:        gemm_bench --sizes 1024,2048,4096 --iters 50
(paste the table printed by gemm_bench here)
```

## Design decisions

* **Library plus thin executables.** Kernels, registry and cuBLAS wrapper are static libraries; the
  benchmark and the tests share exactly the same kernel list, so they cannot drift apart.
* **Host code in `.cpp`, device code in `.cu`.** Keeps nvcc's scope small and compile times short.
* **RAII everywhere.** No raw `cudaMalloc/cudaFree`; failures throw `CudaError` with file, line and
  expression, so tests report failures instead of aborting.
* **Capability checks instead of silent misbehavior.** The float4 kernel declares what it needs
  (n and k divisible by 4, 16-byte aligned pointers); the harness skips it with a message otherwise.
* **64-bit index arithmetic** in kernels (`size_t`) so large matrices cannot overflow `int`.
* **Bounds-checked tile loads** (out-of-range elements load as zero) so all kernels accept arbitrary
  `m, n, k`; only the vectorized kernel has the divisibility requirement above.

## Known limitations and next steps

* Kernels are tuned for generality, not for one architecture; tile sizes are template parameters
  and would be the first thing to auto-tune per GPU.
* Not implemented: double-buffered shared-memory pipelines (`cp.async` on Ampere and newer), warp-level
  tiling, split-K for skinny problems, and tensor-core variants (TF32/FP16 via WMMA or MMA).
* Only FP32 is covered.

## License

MIT, see [`LICENSE`](LICENSE).
