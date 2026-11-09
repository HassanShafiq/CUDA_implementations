# Profiling guide

Each kernel embodies a claim about hardware behavior ("one read and one write", "no spills", "no atomics").
Profile to confirm the claims.

## Prerequisites

* **Nsight Compute** (`ncu`) and **Nsight Systems** (`nsys`), bundled with the CUDA Toolkit or separate.
* Build in **Release**. `LN_ENABLE_LINEINFO=ON` (default) and the extension's `-lineinfo` embed source line
  info for source-level metrics.
* On Windows, if counters are unavailable: NVIDIA Control Panel > Developer > Manage GPU Performance Counters
  (allow all users), or use an elevated shell.

Kernel names to filter on: `ln_fwd_two_pass`, `ln_fwd_welford`, `ln_fwd_fused` (kernels 3 and 4),
`ln_bwd_fused`, `ln_bwd_reduce_params`.

## Profile one kernel (native build)

```powershell
# Forward kernels: first launch of the register-cached kernel on 8192 x 1024 fp16
ncu --set full --kernel-name regex:ln_fwd_fused --launch-count 1 -o results\ncu_fwd_fused `
    .\out\build\windows-release\ln_native_bench.exe --rows 8192 --cols 1024 --dtype fp16 `
    --no-backward --warmup 0 --iters 1

# Backward
ncu --set full --kernel-name regex:ln_bwd_fused --launch-count 1 -o results\ncu_bwd_fused `
    .\out\build\windows-release\ln_native_bench.exe --rows 8192 --cols 1024 --dtype fp16 --warmup 0 --iters 1
```

## Profile through PyTorch

```powershell
ncu --set full --kernel-name regex:ln_ --launch-count 4 -o results\ncu_torch `
    python benchmarks\bench_layernorm.py --rows 8192 --cols 1024 --dtype fp16 --iters 1 --warmup 0 --no-flush-l2
```

For an operator-level view (how much of a training step LayerNorm takes), use `torch.profiler`:

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    model(x).sum().backward()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

## What to check, per claim

| Claim | Where to look (metric names as of recent Nsight Compute; confirm with `ncu --query-metrics`) |
|-------|---------------------------------------------------------------------------------------------|
| Kernels 3 / 4 read and write each element once | *Memory Workload Analysis*: DRAM bytes read / written vs `rows * cols * sizeof(T)`; `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| Kernels are memory-bound and close to peak | *GPU Speed Of Light Throughput*; `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| 128-bit accesses are used | SASS view: `LDG.E.128` / `STG.E.128`; fewer global load instructions than kernel 2 |
| No register spills in the register-cached kernels | *Local memory* traffic: `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` and `..._op_st.sum` should be ~0; also `-DLN_PTXAS_VERBOSE=ON` and read "spill stores / loads" |
| Occupancy is adequate despite register caching | *Occupancy* section: registers per thread as the limiter; `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Warp-per-row kernel needs no block barrier | *Warp State Statistics*: `barrier` stall reason is absent for kernel 3 and present for 2 and 4 |
| Welford kernel does less memory work than the two-pass kernel | compare DRAM / L2 bytes of kernels 1 and 2 |
| Backward: dx reductions stay on chip, parameter sums are not a bottleneck | *Warp State Statistics* for `ln_bwd_fused`; duration of `ln_bwd_reduce_params` is small compared to `ln_bwd_fused` |
| Backward needs no atomics | no `RED` / `ATOM` instructions in the SASS view |

## System-level view

```powershell
nsys profile -o results\nsys_run python benchmarks\bench_layernorm.py --rows 8192 --cols 1024 --dtype fp16 --iters 5
```

Shows launch counts (compare `unfused` against `fused`: one launch instead of many), gaps between launches
and that timings are dominated by the kernels.

## Register and shared-memory usage at compile time

```powershell
cmake --preset windows-release -DLN_PTXAS_VERBOSE=ON
cmake --build --preset windows-release
```

`ptxas` prints registers per thread, shared memory per block and any spills for every kernel instantiation.

## Recording your findings

For each kernel, write down in the README (or `results/NOTES.md`): the limiter you observed, the change that
addressed it and the metric that moved (for example, DRAM bytes per element going from 3 reads to 1). That
causal chain is the deliverable of this project.
