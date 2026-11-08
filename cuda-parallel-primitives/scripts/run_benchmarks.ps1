<#
.SYNOPSIS
  Runs prim_bench and stores a timestamped CSV in results\.
.EXAMPLE
  .\scripts\run_benchmarks.ps1 -Op reduce -Sizes "1048576,16777216,67108864" -Iters 50
#>
param(
  [string]$BuildDir = "out/build/windows-release",
  [string]$Op       = "all",
  [string]$Sizes    = "1048576,16777216",
  [int]$Iters       = 50,
  [int]$Warmup      = 10
)

$ErrorActionPreference = "Stop"
$exe = Get-ChildItem -Path $BuildDir -Recurse -Filter "prim_bench.exe" | Select-Object -First 1
if (-not $exe) { throw "prim_bench.exe not found under '$BuildDir'. Build the project first." }

New-Item -ItemType Directory -Force -Path "results" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$csv = "results/bench-$stamp.csv"

& $exe.FullName --op $Op --sizes $Sizes --iters $Iters --warmup $Warmup --csv $csv
if ($LASTEXITCODE -ne 0) { throw "prim_bench failed with exit code $LASTEXITCODE" }
Write-Host "Saved $csv"
