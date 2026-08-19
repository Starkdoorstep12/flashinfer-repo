# Profiling: Nsight Systems / Nsight Compute

## Setup

`nsys` and `ncu` ship with the CUDA toolkit at `/usr/local/cuda/bin` (see
`SETUP.md`). No special permissions were needed for `ncu` hardware counter
access on this cluster — confirmed no `ERR_NVGPUCTRPERM`.

Profiling script: `profile_kernel.py`. It loads real workload shapes and
the real `sparse_indices` tensor from one passing trace
(`uuid=0c23b10c7b7645719517828c12eaa1d2`, a decode-style workload:
`num_tokens=1`, `num_pages=8462`), rather than synthetic toy tensors, so
the profile reflects production-representative data.

```bash
# Wall-clock timing
python profile_kernel.py

# Nsight Systems (timeline / kernel breakdown)
nsys profile -o ~/dsa_kernel_profile --stats=true python profile_kernel.py

# Nsight Compute (per-kernel deep metrics)
ncu --set basic -o ~/dsa_ncu_test python profile_kernel.py
ncu --import ~/dsa_ncu_test.ncu-rep --page details --kernel-name dsa_fwd_kernel --launch-count 1
```

Raw and extracted reports are in `profiling_results/`:
- `dsa_kernel_profile.nsys-rep` / `.sqlite` — Nsight Systems raw report
- `dsa_ncu_test.ncu-rep` — Nsight Compute raw report
- `nsys_kernel_summary.txt`, `nsys_api_summary.txt`, `nsys_memory_summary.txt` — extracted text summaries
- `ncu_dsa_fwd_kernel_details.txt` — extracted detailed metrics for `dsa_fwd_kernel`

**Note on `ncu` timing:** `ncu` replays each kernel launch multiple times
per metric group (visible as "N passes" in its output), which massively
inflates wall-clock time under profiling (e.g. reported ~764ms/call vs. the
real ~0.097ms/call). Never cite timing numbers captured under `ncu` — only
use its metrics (occupancy, throughput, stalls). Use `nsys` or plain
`torch.cuda.Event` timing for real latency numbers.

## Baseline result (pre-optimization)

Measured on `dsa_fwd_kernel` for the `uuid=0c23b10c...` workload
(`num_tokens=1`, `num_pages=8462`, `topk=2048`, `num_qo_heads=16`):

| Metric | Value |
|---|---|
| Latency (Nsight Systems avg, 50 iters) | 92.4 μs |
| Latency (`torch.cuda.Event` avg, 50 iters) | ~97 μs |
| Grid size | **1 block** |
| Block size | 128 threads |
| GPU | RTX 6000 Ada, 142 SMs |
| Compute (SM) Throughput | 0.23% |
| Memory Throughput | 0.29% |
| Achieved Occupancy | 8.33% |
| Theoretical Occupancy | 8.33% (shared-memory-limited) |
| Dynamic shared memory per block | 94.98 KB |
| Registers per thread | 203 |
| Nsight's own estimated speedup (fix grid) | 99.3% |
| Nsight's own estimated speedup (fix occupancy) | 91.67% |

### Interpretation

The kernel launches with **grid = (num_tokens,) = (1,)** for this
decode-style workload. With only 1 CUDA block launched, only 1 of the
GPU's 142 SMs is ever active — everything else in the profile (near-zero
compute and memory throughput) is a direct consequence of this single
fact, not a sign of being "memory-bound" or "compute-bound" in the usual
sense. This is a **parallelism-bound** kernel.

Nsight Compute's own launch-statistics optimizer flags this explicitly:
> "The grid for this launch is configured to execute only 1 block, which
> is less than the GPU's 142 multiprocessors... Est. Speedup: 99.3%"

Secondary constraint: even if the grid were larger, occupancy is further
capped by shared memory pressure — 94.98 KB of dynamic shared memory per
block leaves room for only 1 block per SM given a ~102.4 KB/SM budget,
limiting active warps to 4 out of a 12-warp-per-scheduler ceiling (8.33%
achieved occupancy).

### Root cause

`kernel.py`'s launch:
```python
grid = (num_tokens,)
```
parallelizes only across the token/batch dimension. For decode-style
inference (`num_tokens=1`, the common case for autoregressive generation),
there is no token-level parallelism to exploit, so the entire
`topk=2048`-length KV loop for all 16 heads runs serially inside one block
on one SM.

### Planned fix

Split the `topk` (KV) dimension across multiple thread blocks per token
(split-K / flash-decoding style): each block processes a chunk of the 2048
sparse indices and produces partial (max, sum, weighted-value) online-
softmax state, followed by a second reduction pass combining partial
results across blocks. This is the standard technique used by production
decode-phase attention kernels for exactly this SM-underutilization
problem.

## After optimization

Split-K implementation built and profiled — see `docs/SPLITK_OPTIMIZATION.md`
for full results and analysis. Summary: forward-kernel duration dropped
~6x (254.72us -> 42.30us at SPLIT_K=8) by fixing grid-size underutilization
(1 -> 8 blocks). A second, independent bottleneck (per-block shared memory
capping occupancy at ~8.3% regardless of grid size) remains and is not yet
addressed. Nsight estimates a further ~94% speedup available from pushing
SPLIT_K higher, separate from the shared-memory constraint.
