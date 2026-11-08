# Profiling guide

The point of each kernel in this repo is a claim about hardware behavior. Profile to confirm the claim;
do not assume it.

## Prerequisites

* Install **Nsight Compute** (`ncu`) and **Nsight Systems** (`nsys`); both ship with the CUDA Toolkit
  or are available as separate downloads.
* Build in **Release**. The default `GEMM_ENABLE_LINEINFO=ON` embeds source line info so Nsight Compute
  can map metrics back to source lines.
* On Windows, if counters are unavailable, enable GPU performance counters for all users:
  NVIDIA Control Panel > Developer > Manage GPU Performance Counters (or run the shell as Administrator).

## Profile one kernel

`--no-baseline` skips cuBLAS (and verification) so the process launches only the selected kernel:

```powershell
ncu --set full --launch-count 1 -o results\ncu_kernel5 `
    .\out\build\windows-release\gemm_bench.exe --kernels 5 --sizes 2048 `
    --no-baseline --warmup 0 --iters 1
```

Open `results\ncu_kernel5.ncu-rep` in the Nsight Compute UI. Repeat with `--kernels 1`, `3`, `6`, etc.,
and use the **Baseline** feature in the UI to diff a kernel against the previous one.

## Reading the report: what to look at per step

| Question | Section / metric (names as of recent Nsight Compute; confirm with `ncu --query-metrics`) |
|----------|-------------------------------------------------------------------------------------------|
| Compute- or memory-bound? | *GPU Speed Of Light Throughput*, and the roofline chart (`--section SpeedOfLight_RooflineChart`) |
| Is global access coalesced? | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`; *Memory Workload Analysis* sectors/request |
| How much DRAM traffic? | `dram__throughput.avg.pct_of_peak_sustained_elapsed`, *Memory Chart* |
| Shared-memory bank conflicts? | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (and `_op_st`) |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` and *Occupancy* section (registers/thread, shared memory/block as limiters) |
| Why are warps not issuing? | *Warp State Statistics*: stall reasons (`barrier`, `short scoreboard`, `long scoreboard`, `math pipe throttle`) |
| Are the FMAs the limit? | *Compute Workload Analysis*: FMA pipe utilization |

Per-kernel expectations are listed in [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md).

## System-level view

```powershell
nsys profile -o results\nsys_run .\out\build\windows-release\gemm_bench.exe --sizes 2048
```

Useful for checking launch gaps, host/device copies and that benchmark timings are not dominated by
anything other than the kernel.

## Register and shared-memory usage at compile time

```powershell
cmake --preset windows-release -DGEMM_PTXAS_VERBOSE=ON
cmake --build --preset windows-release
```

`ptxas` then prints registers per thread, shared memory per block and any spills for each kernel
instantiation. Relate these to the occupancy limiters reported by Nsight Compute.

## Recording your findings

For each kernel, write down in the README (or a `results/NOTES.md`): the limiter you observed, the change
that addressed it, and the metric that moved. That causal chain is the deliverable of this project.
