<#
.SYNOPSIS
  Builds and installs the PyTorch extension in editable mode (Windows).
.DESCRIPTION
  Run from an "x64 Native Tools Command Prompt for VS 2022" (or a PowerShell that has loaded vcvars64),
  inside the Python environment that has PyTorch installed.
.EXAMPLE
  .\scripts\build_extension.ps1 -CudaArch 8.6
#>
param(
  [string]$CudaArch = "",   # e.g. "8.6"; empty = detect from the installed GPU
  [int]$MaxJobs = 4
)

$ErrorActionPreference = "Stop"
$env:DISTUTILS_USE_SDK = "1"
$env:MAX_JOBS = "$MaxJobs"
if ($CudaArch -ne "") { $env:TORCH_CUDA_ARCH_LIST = $CudaArch }

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  throw "cl.exe not found. Open an 'x64 Native Tools Command Prompt for VS 2022' first."
}

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
pip install -e . --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "extension build failed" }
python -c "import fused_layernorm as f; print('kernels:', f.forward_kernels())"
