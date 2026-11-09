# PyTorch integration

How the framework-independent CUDA kernels are exposed to PyTorch, and how to build and troubleshoot the
extension.

## Layering

```
python/fused_layernorm/
  module.py        FusedLayerNorm (nn.Module)      - drop-in for torch.nn.LayerNorm
  functional.py    fused_layer_norm + autograd.Function, argument validation
        |  _C.forward / _C.backward
csrc/torch/layernorm_ext.cpp     ATen glue: tensor checks, allocation, device guard, stream
        |  ln::layernorm_forward / ln::layernorm_backward
csrc/include/layernorm.h         framework-independent API (raw pointers, dtype tag, stream)
csrc/kernels/*.cu                the CUDA kernels
```

Only `layernorm_ext.cpp` includes ATen. The same `.cu` files are built into the extension (via `setup.py`)
and into the native library used by `native/` (via CMake), so the kernels are tested and benchmarked without
PyTorch, and PyTorch cannot change their behavior.

## What the glue layer is responsible for

| Concern | How it is handled |
|---------|-------------------|
| Device | `c10::cuda::CUDAGuard` makes the tensor's device current, so multi-GPU programs work |
| Stream | the kernels launch on `at::cuda::getCurrentCUDAStream()`, so they are ordered correctly with the rest of the PyTorch work on that stream (this has not been validated under CUDA graph capture) |
| Allocation | outputs come from PyTorch's caching allocator (`at::empty_like`), not `cudaMalloc`; backward scratch is an `at::empty` byte tensor that is freed with normal tensor lifetime |
| Layout | the Python layer flattens leading dimensions to `[rows, cols]` and makes the input contiguous; any `normalized_shape` is supported by folding it into `cols` |
| dtypes | float32, float16, bfloat16, float64; weight / bias must match the input dtype (clear `ValueError` otherwise) |
| Errors | `std::invalid_argument` becomes `ValueError`, CUDA failures become `RuntimeError` with file, line and expression; `TORCH_CHECK` messages for tensor problems |
| Empty batches | `rows == 0` returns an empty tensor; parameter gradients are zeros |

## Autograd

`_FusedLayerNormFunction` is a `torch.autograd.Function`:

* **forward** saves only `x`, `weight` and the per-row `mean` / `rstd` vectors: `O(rows)` extra memory instead
  of keeping intermediates, and backward does not recompute the statistics.
* **backward** uses `ctx.needs_input_grad` so `dgamma` / `dbeta` are only produced when something requires
  them (frozen parameters or `elementwise_affine=False` cost nothing), and returns `None` for the
  non-tensor arguments.
* It is decorated with `once_differentiable`: higher-order gradients are not implemented and raise a clear
  error rather than returning wrong values.
* Gradients are bitwise deterministic (no floating-point atomics).

## Building

### Requirements

* A PyTorch build with CUDA, and the **CUDA Toolkit whose major.minor matches** it
  (`python -c "import torch; print(torch.version.cuda)"`). A mismatch is the most common build failure.
* C++17 compiler: Visual Studio 2022 (Desktop development with C++) on Windows, GCC or Clang on Linux.

### Windows

1. Open an **x64 Native Tools Command Prompt for VS 2022** (so `cl.exe` and the SDK are on `PATH`).
2. Activate the Python environment that has PyTorch installed.
3. From the repository root:

```bat
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6          :: your GPU's compute capability, e.g. 7.5, 8.0, 8.6, 8.9
pip install -e . --no-build-isolation
python -c "import fused_layernorm; print(fused_layernorm.forward_kernels())"
```

`--no-build-isolation` is required because the extension must be compiled against the *installed* PyTorch.
`TORCH_CUDA_ARCH_LIST` limits compilation to your GPU and keeps build times short.

### Linux

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
pip install -e . --no-build-isolation
```

### Troubleshooting

| Symptom | Likely cause and fix |
|---------|----------------------|
| `The detected CUDA version (X) mismatches the version that was used to compile PyTorch (Y)` | install the matching CUDA Toolkit, or a PyTorch build for the toolkit you have |
| `Microsoft Visual C++ 14.0 or greater is required` / `cl.exe` not found | build from the x64 Native Tools prompt and set `DISTUTILS_USE_SDK=1` |
| `nvcc fatal : Unsupported gpu architecture` | set `TORCH_CUDA_ARCH_LIST` to values your toolkit supports |
| `ImportError: ... fused_layernorm._C` | the build did not complete, or you are running from a different environment than the one you built in |
| very long first build | normal (templates for 4 dtypes); set `MAX_JOBS` to control parallelism |

## Using it

```python
import torch
from fused_layernorm import FusedLayerNorm, fused_layer_norm

ln = FusedLayerNorm(1024).cuda().half()            # same arguments as nn.LayerNorm
y = ln(torch.randn(8, 512, 1024, device="cuda", dtype=torch.half))

y = fused_layer_norm(x, (1024,), weight, bias, eps=1e-5)   # functional form
```

`nn.LayerNorm` state dicts load into `FusedLayerNorm` and vice versa, so swapping an existing model's
LayerNorms is a one-line change per module.

## Limitations

* No higher-order autograd; no `torch.compile` support for the custom op (it would need
  `torch.library.custom_op` registration and fake tensor kernels); no autocast awareness (inputs are used in the
  dtype given).
* Weight and bias must share the input dtype (no fp32 parameters with fp16 activations yet).
* CUDA only; element counts are limited to 2^31 - 1 rows and columns (offsets are 64-bit).
