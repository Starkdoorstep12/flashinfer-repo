"""
Experiment 2: isolate pure kernel-launch overhead, independent of compute.

Launches a trivial (near-zero-work) Triton kernel N times back-to-back and
measures total latency vs N. The slope tells us the marginal cost of one
additional kernel launch, which we can then use to explain the v1 (2
launches) vs v2 (4 launches) gap quantitatively rather than just
qualitatively.
"""
import torch
import triton
import triton.language as tl

device = 'cuda'

@triton.jit
def noop_kernel(x_ptr):
    # Does the absolute minimum: one load, one store of a single element.
    # Real "work" here is negligible; total latency is dominated by launch
    # overhead (Python dispatch, CUDA driver call, Triton's own launch path).
    val = tl.load(x_ptr)
    tl.store(x_ptr, val)

x = torch.zeros(1, device=device, dtype=torch.float32)

def run_n_launches(n):
    for _ in range(n):
        noop_kernel[(1,)](x)

# Warmup (compile + cache the kernel)
for _ in range(5):
    run_n_launches(1)
torch.cuda.synchronize()

print(f"{'N launches':>12} {'Avg total (us)':>16} {'Per-launch (us)':>18}")
results = []
for n in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        run_n_launches(n)
    end.record()
    torch.cuda.synchronize()
    total_us = (start.elapsed_time(end) / 50) * 1000  # ms -> us
    per_launch_us = total_us / n
    results.append((n, total_us, per_launch_us))
    print(f"{n:>12} {total_us:>16.2f} {per_launch_us:>18.2f}")

# Simple linear fit: total_time ≈ intercept + slope * n
# slope = marginal cost per additional launch
import numpy as np
ns = np.array([r[0] for r in results])
totals = np.array([r[1] for r in results])
slope, intercept = np.polyfit(ns, totals, 1)
print(f"\nLinear fit: total_us ≈ {intercept:.2f} + {slope:.2f} * n_launches")
print(f"=> Marginal cost per additional kernel launch: ~{slope:.2f} us")
