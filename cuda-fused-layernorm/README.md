# Fused LayerNorm in CUDA, with PyTorch Integration

A fused LayerNorm implemented from scratch in CUDA C++ and wired into PyTorch as a custom autograd
operator. It contains a ladder of four forward kernels (each removing one source of memory traffic or
synchronisation), a fused deterministic backward kernel, fp32 / fp16 / bf16 / fp64 support, a drop-in
`nn.Module`, and a test and benchmark stack that checks every kernel against a reference and measures it
against PyTorch's own implementation.

```
y = (x - mean) * rstd * gamma + beta        rstd = 1 / sqrt(var + eps),  statistics over the last dim(s)
```

```python
from fused_layernorm import FusedLayerNorm

ln = FusedLayerNorm(1024).cuda().half()      # same arguments and state dict as torch.nn.LayerNorm
y = ln(x)                                    # autograd works: y.sum().backward()
```

## Highlights

* **Forward ladder** (`csrc/kernels/ln_forward.cu`)

  | # | Kernel | Idea | Global traffic per row |
  |---|--------|------|------------------------|
  | 1 | `two_pass` | block per row, shared-memory tree sums | 3 reads + 1 write |
  | 2 | `welford` | one-pass Welford statistics, shuffle merge | 2 reads + 1 write |
  | 3 | `warp_row` | warp per row, row held in registers, 128-bit accesses | 1 read + 1 write |
  | 4 | `block_fused` | block per row, row held in registers, 128-bit accesses | 1 read + 1 write |

  `kernel_id=0` (default) selects the fastest kernel that supports the input; unsupported shapes or
  misaligned pointers fall back safely (never undefined behavior).
* **Numerically careful.** Welford / Chan statistics instead of `E[x^2] - E[x]^2`; fp32 accumulation for
  fp16 / bf16; all lanes of a row end with bit-identical statistics.
* **Fused backward** (`ln_backward.cu`): `dx`, `dgamma`, `dbeta` in one pass over the rows plus a tiny
  second stage. No atomics, so gradients are **bitwise reproducible** (tested).
* **Layered design.** Framework-independent CUDA library, thin ATen binding, Python autograd layer. The
  kernels build and are tested **without PyTorch** (CMake), so they can be debugged in isolation.
* **Verification at three levels:** GPU-free math checks (NumPy), native CUDA tests against a double-precision
  CPU reference, and PyTorch tests against `F.layer_norm` including `gradcheck`.

Details and derivations: [`docs/OPTIMIZATION_NOTES.md`](docs/OPTIMIZATION_NOTES.md),
[`docs/PYTORCH_INTEGRATION.md`](docs/PYTORCH_INTEGRATION.md), [`docs/PROFILING.md`](docs/PROFILING.md).

## Project layout

```
csrc/
  include/layernorm.h        framework-independent C++ API (raw pointers, dtype tag, stream)
  kernels/ln_common.cuh      dtype traits, 128-bit vectors, Welford statistics, reductions
  kernels/ln_forward.cu      forward kernel ladder + launcher + kernel registry
  kernels/ln_backward.cu     fused backward + parameter-gradient reduction
  torch/layernorm_ext.cpp    PyTorch binding (ATen: allocation, device guard, stream)
python/fused_layernorm/      functional API + autograd.Function, nn.Module, unfused reference
native/                      PyTorch-free tests (test_native.cpp), benchmark, CPU reference
tests/                       pytest suites (test_layernorm.py, test_math_reference.py)
benchmarks/bench_layernorm.py  benchmark vs PyTorch (aten) and an unfused composition
docs/  scripts/  CMakeLists.txt  setup.py  pyproject.toml
```

## Requirements

* NVIDIA GPU with a recent driver; CUDA Toolkit 12.x
* **Visual Studio 2022** (Desktop development with C++) on Windows, or GCC/Clang on Linux
* CMake 3.24+ and Ninja (bundled with Visual Studio) for the native build
* Python 3.9+ and a CUDA build of PyTorch 2.0+ for the extension. The CUDA Toolkit's major.minor version
  must match `torch.version.cuda`.

## Build and run

### A. Native build and tests (no PyTorch needed)

Visual Studio 2022: `File > Open > Folder...`, select the repository root, choose the **Windows x64 Release**
configuration (from `CMakePresets.json`), then `Build > Build All`. Or from an *x64 Native Tools Command
Prompt*:

```bat
cmake --preset windows-release
cmake --build --preset windows-release
out\build\windows-release\ln_native_tests.exe
out\build\windows-release\ln_native_bench.exe --rows 8192 --cols 768,1024,4096 --dtype fp16
```

> Always benchmark **Release**: in Debug, CMake compiles device code with `-G` (unoptimised).

### B. PyTorch extension

From an *x64 Native Tools Command Prompt for VS 2022*, inside the Python environment that has PyTorch:

```bat
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
pip install -e . --no-build-isolation
pytest -v
python benchmarks\bench_layernorm.py --rows 8192 --cols 768 1024 4096 --dtype fp16 fp32
```

(`scripts\build_extension.ps1` wraps the build.) Troubleshooting table in
[`docs/PYTORCH_INTEGRATION.md`](docs/PYTORCH_INTEGRATION.md).

### C. GPU-free math checks

```bash
pip install numpy pytest
pytest tests/test_math_reference.py
```

## Testing strategy

| Level | Command | What it proves |
|-------|---------|----------------|
| Math (NumPy, no GPU) | `pytest tests/test_math_reference.py` | closed-form backward equals finite differences; the Welford-merge scheme reproduces the variance and beats `E[x^2]-E[x]^2` by orders of magnitude when `|mean| >> std` |
| Native CUDA | `ln_native_tests` (`--quick` for a short run) | every kernel x dtype x shape x affine mode vs a double-precision CPU reference; NaN-prefilled outputs catch unwritten elements; misaligned pointers; fallback and error handling; backward bitwise determinism |
| PyTorch | `pytest -v` | parity with `F.layer_norm` forward and backward (fp32 / fp16 / bf16 / fp64), `gradcheck`, multi-dim `normalized_shape`, non-contiguous inputs, empty batch, module / state-dict parity, input validation |

Shapes deliberately straddle every boundary: chunk size, warp, register-cache capacity (1024 / 8192 fp32
columns), and more than 256 rows so backward blocks loop over several rows.

## Benchmark methodology

* Every implementation is verified before it is timed.
* CUDA events around each call, untimed warm-up, **median** reported; L2 is flushed before each timed call
  by default so small problems are measured from DRAM.
* LayerNorm is memory-bound, so the metric is **effective bandwidth on the minimum traffic**
  (forward: read x + write y; forward + backward: 5 tensor-sized transfers). "% copy BW" compares with a
  device-to-device `cudaMemcpy` measured on the same GPU. Implementations that move more data score lower
  by construction.
* Baselines: `aten` (`F.layer_norm`, PyTorch's own fused kernel, the bar to clear) and `unfused` (LayerNorm
  written as separate ops, which shows what fusion buys).

## Results

Results depend on the GPU, driver and clocks, so none are hard-coded. After running the benchmark, record
your numbers below with the hardware they came from.

```
GPU:       <model>, driver <version>, CUDA <version>, torch <version>
Command:   python benchmarks/bench_layernorm.py --rows 8192 --cols 768 1024 4096 --dtype fp16 fp32
(paste the tables printed by the benchmark here)
```

## Design decisions

* **Kernel registry with `supports()`** instead of silent assumptions: vectorised kernels declare their
  requirements (chunk-multiple column count, 16-byte alignment, bounded row length) and the dispatcher falls
  back; an explicitly requested kernel that cannot run raises a clear error.
* **Exact local statistics from registers**, merged once: avoids both a second pass over global memory and
  the cancellation of the textbook variance formula.
* **Determinism by construction**: fixed merge order for statistics; two-stage parameter-gradient reduction
  with block-private accumulators instead of floating-point atomics.
* **RAII and exceptions on the host side**, `TORCH_CHECK` at the framework boundary, and no raw
  `cudaMalloc` in the PyTorch path (outputs and scratch come from the caching allocator).
* **No `--use_fast_math`**: it would trade the accuracy of `rsqrt` and division for speed.

## Limitations and next steps

* No higher-order autograd, no `torch.compile` / `torch.library.custom_op` registration, no autocast awareness.
* `weight` / `bias` must match the input dtype (no fp32 parameters with fp16 activations yet).
* The backward pass re-reads `x` and `dy` in a second loop and uses scalar loads; caching the row in
  registers and vectorising it would mirror the forward optimisations.
* Possible extensions: fused residual-add + LayerNorm, RMSNorm, and a shared-memory staged variant for rows
  beyond the register-cache capacity.

## License

MIT, see [`LICENSE`](LICENSE).
