# SPDX-License-Identifier: MIT
"""Benchmark the fused LayerNorm against PyTorch baselines.

Compared implementations
  aten      torch.nn.functional.layer_norm   (PyTorch's own fused CUDA kernel: the bar to clear)
  unfused   LayerNorm written as separate ops (what hand-written LayerNorm costs without fusion)
  fused[k]  every kernel of this repo's forward ladder that supports the shape, plus ``auto``

Methodology
  * Every implementation is verified against an fp32 / fp64 reference BEFORE it is timed.
  * CUDA events around each call after warm-up; the MEDIAN is reported. By default L2 is flushed before
    every timed call so small problems are measured from DRAM like real workloads (--no-flush-l2 to disable).
  * LayerNorm is memory-bound, so the metric is effective bandwidth on the MINIMUM traffic:
        forward      : read x + write y            = 2 * numel * itemsize
        fwd + bwd    : the above + read x, dy + write dx = 5 * numel * itemsize
    Implementations that move more data (unfused, multi-pass kernels) score lower by construction.

Examples
  python benchmarks/bench_layernorm.py
  python benchmarks/bench_layernorm.py --rows 8192 --cols 768 1024 4096 --dtype fp16 bf16 --csv results/bench.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from fused_layernorm import forward_kernels, fused_layer_norm, kernel_supported
from fused_layernorm.reference import layer_norm_unfused

DTYPES: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp64": torch.float64,
}
TOL = {torch.float32: (5e-5, 1e-4), torch.float16: (4e-3, 2e-3), torch.bfloat16: (3e-2, 1e-2), torch.float64: (1e-8, 1e-8)}


class L2Flusher:
    """Writes a buffer much larger than any GPU L2 so the next timed call starts cold."""

    def __init__(self, enabled: bool) -> None:
        self.buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda") if enabled else None

    def __call__(self) -> None:
        if self.buf is not None:
            self.buf.zero_()


def measure(fn: Callable[[], object], flush: L2Flusher, warmup: int, iters: int) -> Tuple[float, float]:
    """Returns (median_ms, min_ms) of ``fn`` using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(iters):
        flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples)


def copy_bandwidth_gbps(flush: L2Flusher, warmup: int, iters: int) -> float:
    nbytes = 128 * 1024 * 1024
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    median_ms, _ = measure(lambda: dst.copy_(src), flush, warmup, iters)
    return 2.0 * nbytes / (median_ms * 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=8192, help="rows (tokens), default 8192")
    ap.add_argument("--cols", type=int, nargs="+", default=[768, 1024, 4096], help="normalised sizes")
    ap.add_argument("--dtype", nargs="+", default=["fp16", "fp32"], choices=sorted(DTYPES), help="storage dtypes")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-backward", action="store_true", help="skip forward + backward timing")
    ap.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--csv", type=str, default=None, help="also write results to this CSV file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)

    props = torch.cuda.get_device_properties(0)
    print(f"Device 0: {props.name} | compute capability {props.major}.{props.minor} | {props.multi_processor_count} SMs | torch {torch.__version__}")
    flush = L2Flusher(args.flush_l2)
    copy_bw = copy_bandwidth_gbps(flush, args.warmup, args.iters)
    print(f"Device-to-device copy bandwidth (read + write): {copy_bw:.1f} GB/s")

    rows_out: List[List[object]] = []
    all_ok = True

    def report(phase: str, dtype_name: str, cols: int, name: str, median_ms: float, min_ms: float,
               traffic: float, aten_ms: Optional[float], verdict: str) -> None:
        gbps = traffic / (median_ms * 1e6)
        speedup = (aten_ms / median_ms) if aten_ms else float("nan")
        print(f"  {name:<14} {median_ms:10.4f} {gbps:9.1f} {100 * gbps / copy_bw:9.1f}% {speedup:9.2f}x  {verdict}")
        rows_out.append([phase, dtype_name, args.rows, cols, name, median_ms, min_ms, gbps, 100 * gbps / copy_bw, speedup, verdict])

    for dname in args.dtype:
        dtype = DTYPES[dname]
        itemsize = torch.empty((), dtype=dtype).element_size()
        for cols in args.cols:
            x = torch.randn(args.rows, cols, device="cuda", dtype=dtype) * 2
            w = torch.rand(cols, device="cuda", dtype=dtype) + 0.5
            b = torch.randn(cols, device="cuda", dtype=dtype)
            wide = torch.float64 if dtype == torch.float64 else torch.float32
            expected = F.layer_norm(x.to(wide), (cols,), w.to(wide), b.to(wide), 1e-5)
            atol, rtol = TOL[dtype]
            traffic = 2.0 * x.numel() * itemsize

            candidates: List[Tuple[str, Callable[[], torch.Tensor]]] = [
                ("aten", lambda: F.layer_norm(x, (cols,), w, b, 1e-5)),
                ("unfused", lambda: layer_norm_unfused(x, (cols,), w, b, 1e-5)),
            ]
            for kid, kname, _ in forward_kernels():
                if kernel_supported(kid, x, w, b):
                    candidates.append((f"fused[{kid}]{kname}", lambda kid=kid: fused_layer_norm(x, (cols,), w, b, 1e-5, kid)))
                else:
                    print(f"  (fused[{kid}]{kname} not supported for cols={cols} {dname}: skipped)")
            candidates.append(("fused[auto]", lambda: fused_layer_norm(x, (cols,), w, b, 1e-5, 0)))

            print(f"\n[forward] dtype={dname} rows={args.rows} cols={cols}")
            print(f"  {'implementation':<14} {'median [ms]':>10} {'GB/s':>9} {'% copy BW':>10} {'vs aten':>10}  verification")
            aten_ms: Optional[float] = None
            for name, fn in candidates:
                ok = True
                try:
                    torch.testing.assert_close(fn().to(wide), expected, atol=atol, rtol=rtol)
                except AssertionError:
                    ok = False
                all_ok &= ok
                median_ms, min_ms = measure(fn, flush, args.warmup, args.iters)
                if name == "aten":
                    aten_ms = median_ms
                report("forward", dname, cols, name, median_ms, min_ms, traffic, aten_ms, "PASS" if ok else "FAIL")

            if args.no_backward:
                continue

            # forward + backward through autograd
            dy = torch.randn_like(x)
            traffic_fb = 5.0 * x.numel() * itemsize

            def make_step(forward: Callable[..., torch.Tensor]) -> Callable[[], None]:
                xi = x.detach().clone().requires_grad_()
                wi = w.detach().clone().requires_grad_()
                bi = b.detach().clone().requires_grad_()

                def step() -> None:
                    xi.grad = wi.grad = bi.grad = None
                    forward(xi, wi, bi).backward(dy)

                return step

            print(f"\n[forward + backward] dtype={dname} rows={args.rows} cols={cols}")
            print(f"  {'implementation':<14} {'median [ms]':>10} {'GB/s':>9} {'% copy BW':>10} {'vs aten':>10}")
            aten_fb: Optional[float] = None
            for name, forward in (
                ("aten", lambda xi, wi, bi: F.layer_norm(xi, (cols,), wi, bi, 1e-5)),
                ("unfused", lambda xi, wi, bi: layer_norm_unfused(xi, (cols,), wi, bi, 1e-5)),
                ("fused[auto]", lambda xi, wi, bi: fused_layer_norm(xi, (cols,), wi, bi, 1e-5, 0)),
            ):
                median_ms, min_ms = measure(make_step(forward), flush, args.warmup, args.iters)
                if name == "aten":
                    aten_fb = median_ms
                report("fwd+bwd", dname, cols, name, median_ms, min_ms, traffic_fb, aten_fb, "")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "dtype", "rows", "cols", "implementation", "median_ms", "min_ms", "gb_per_s",
                             "pct_copy_bw", "speedup_vs_aten", "verification"])
            writer.writerows(rows_out)
        print(f"\nSaved {args.csv}")

    if not all_ok:
        print("\nERROR: at least one implementation failed verification.", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
