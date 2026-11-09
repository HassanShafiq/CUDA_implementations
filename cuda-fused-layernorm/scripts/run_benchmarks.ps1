<#
.SYNOPSIS
  Runs the native benchmark and the PyTorch benchmark, storing timestamped output in results\.
.EXAMPLE
  .\scripts\run_benchmarks.ps1 -Rows 8192 -Cols "768,1024,4096" -Dtype fp16
#>
param(
  [string]$BuildDir = "out/build/windows-release",
  [int]$Rows        = 8192,
  [string]$Cols     = "768,1024,4096",
  [string]$Dtype    = "fp16",     # fp32 | fp16 | bf16 | fp64
  [int]$Iters       = 50,
  [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

$exe = Get-ChildItem -Path $BuildDir -Recurse -Filter "ln_native_bench.exe" | Select-Object -First 1
if (-not $exe) { throw "ln_native_bench.exe not found under '$BuildDir'. Build the project first." }
& $exe.FullName --rows $Rows --cols $Cols --dtype $Dtype --iters $Iters | Tee-Object -FilePath "results/native-$stamp.txt"
if ($LASTEXITCODE -ne 0) { throw "ln_native_bench failed with exit code $LASTEXITCODE" }

if (-not $SkipTorch) {
  $dt = $Dtype
  $colsList = $Cols -split ","
  python benchmarks/bench_layernorm.py --rows $Rows --cols $colsList --dtype $dt --iters $Iters --csv "results/torch-$stamp.csv"
  if ($LASTEXITCODE -ne 0) { throw "bench_layernorm.py failed with exit code $LASTEXITCODE" }
}
Write-Host "Results saved under results/ ($stamp)"
