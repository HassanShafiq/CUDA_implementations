# Profiling guide

Each kernel in this repo embodies a claim about hardware behavior. Profile to confirm the claim.

## Prerequisites

* Install **Nsight Compute** (`ncu`) and **Nsight Systems** (`nsys`) (bundled with the CUDA Toolkit or separate).
* Build in **Release**. `PRIM_ENABLE_LINEINFO=ON` (default) embeds source line info for source-level metrics.
* On Windows, if counters are unavailable, enable GPU performance counters for all users:
  NVIDIA Control Panel > Developer > Manage GPU Performance Counters (or run an elevated shell).

## Profile one kernel

`--no-baseline` skips CUB and the copy-bandwidth measurement; with `--no-verify`, `--warmup 0 --iters 1` the
process launches only the selected kernel once (reduction and scan launch several kernels per run: passes or phases).

```powershell
# Reduction variant 5 on 16M elements; profile only the first (largest) pass
ncu --set full --launch-count 1 -o results\ncu_reduce5 `
    .\out\build\windows-release\prim_bench.exe --op reduce --kernels 5 --sizes 16777216 `
    --no-baseline --no-verify --warmup 0 --iters 1

# Softmax variant 4, 8192 rows x 4096 columns
ncu --set full --launch-count 1 -o results\ncu_softmax4 `
    .\out\build\windows-release\prim_bench.exe --op softmax --kernels 4 --rows 8192 --cols 4096 `
    --no-baseline --no-verify --warmup 0 --iters 1
```

Repeat with other `--kernels` ids and use the **Baseline** feature of the Nsight Compute UI to diff a variant
against the previous step of its ladder. For scan, use `--launch-skip` / `--launch-count` (or the UI's kernel
filter) to pick the phase-1 tile-scan kernel versus `add_block_offsets`.

## Metrics worth checking (names as of recent Nsight Compute; confirm with `ncu --query-metrics`)

| Question | Section / metric |
|----------|------------------|
| Is the kernel memory-bound and how close to peak? | *GPU Speed Of Light Throughput*; `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| Divergence / idle lanes (reduce 1 vs 2) | `smsp__thread_inst_executed_per_inst_executed.ratio` (active threads per warp instruction) |
| Shared-memory bank conflicts (reduce 2 vs 3, scan 2 vs 3) | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and `..._op_st.sum` |
| Are global accesses well formed? | *Memory Workload Analysis*: sectors per request, `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` |
| Cost of synchronisation (scan, softmax) | *Warp State Statistics*: stall reasons `barrier`, `short scoreboard`, `long scoreboard` |
| Occupancy limiters | *Occupancy* section and `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Instruction count (vector vs scalar) | `smsp__inst_executed.sum`; SASS view for `LDG.E.128` / `LDS` / `SHFL` |

## System-level view

```powershell
nsys profile -o results\nsys_run .\out\build\windows-release\prim_bench.exe --op scan --sizes 16777216
```

Shows the number of kernel launches per run (passes and phases), gaps between them, and that timings are not
dominated by anything other than kernels.

## Register and shared-memory usage

```powershell
cmake --preset windows-release -DPRIM_PTXAS_VERBOSE=ON
cmake --build --preset windows-release
```

`ptxas` prints registers per thread and shared memory per block for each kernel.

## Recording your findings

For each ladder, record in the README (or `results/NOTES.md`): the limiter you observed, the change that
addressed it, and the metric that moved. That causal chain is the deliverable of this project.
