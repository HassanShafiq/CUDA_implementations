<#
.SYNOPSIS
  Runs gemm_bench and stores a timestamped CSV in results\.
.EXAMPLE
  .\scripts\run_benchmarks.ps1 -Sizes "1024,2048,4096" -Iters 50
#>
param(
  [string]$BuildDir = "out/build/windows-release",
  [string]$Sizes    = "512,1024,2048,4096",
  [int]$Iters       = 50,
  [int]$Warmup      = 10
)

$ErrorActionPreference = "Stop"
$exe = Get-ChildItem -Path $BuildDir -Recurse -Filter "gemm_bench.exe" | Select-Object -First 1
if (-not $exe) { throw "gemm_bench.exe not found under '$BuildDir'. Build the project first." }

New-Item -ItemType Directory -Force -Path "results" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$csv = "results/bench-$stamp.csv"

& $exe.FullName --sizes $Sizes --iters $Iters --warmup $Warmup --csv $csv
if ($LASTEXITCODE -ne 0) { throw "gemm_bench failed with exit code $LASTEXITCODE" }
Write-Host "Saved $csv"
