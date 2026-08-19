# Split-K Optimization: Results and Analysis

This documents the split-K redesign of `dsa_fwd_kernel`, built to address the
SM-underutilization bottleneck identified in `PROFILING.md`'s baseline
profiling (grid size of 1 block, 8.33% occupancy, 0.23% SM throughput).

## What changed

Original `kernel()`: launches `grid = (num_tokens,)`. For decode-style
workloads (`num_tokens=1`), this launches exactly 1 CUDA block, so only 1 of
the GPU's 142 SMs ever does work — the full `topk=2048` KV loop runs
serially inside that one block.

New `kernel_splitk()`: launches `grid = (num_tokens, SPLIT_K)` via
`dsa_fwd_kernel_splitk`, splitting the `topk` KV tokens into `SPLIT_K`
independent chunks, each processed by its own block. Each block computes a
*partial* online-softmax result (running max, running sum, weighted
accumulator) for its chunk only. A second, lightweight kernel
(`dsa_reduce_kernel`, one program per token) merges the `SPLIT_K` partial
results into the final `(output, lse)`, using the same rescale-by-alpha
online-softmax update already used inside the main loop — just applied
across blocks instead of across KV tiles within one block.

This is the standard "split-K" / "Flash-Decoding" technique used by
production decode-phase attention kernels to solve exactly this class of
problem (single query token → too few blocks to fill the GPU).

## Correctness

Verified `kernel_splitk()` against the already-verified `kernel()` directly
(not the slow golden reference, for speed) at `SPLIT_K = 4, 8, 16`, using
real workload data (`uuid=0c23b10c...`). Result: **exact match**
(`max_abs_err = 0.000000`) at all three `SPLIT_K` values — see
`test_splitk_correctness.py`.

Two bugs were found and fixed during development, both variants of the same
root cause: `-inf - (-inf) = NaN` when a split's KV chunk contains zero
valid (non-padding) sparse indices — something that cannot happen in the
original single-block kernel (which always sees the full, non-empty
`topk=2048` range) but can happen once that range is subdivided. Fixed by
guarding both the per-tile softmax update (inside `dsa_fwd_kernel_splitk`)
and the cross-split merge (inside `dsa_reduce_kernel`) with `t1.where` to
treat "no valid tokens seen" as a zero contribution instead of computing
`exp(NaN)`.

## Performance results (SPLIT_K = 8, RTX 6000 Ada)

Measured via Nsight Systems (`nsys`) and Nsight Compute (`ncu`) on the same
workload used for the baseline (`uuid=0c23b10c...`, `num_tokens=1`,
`num_pages=8462`, `topk=2048`, `num_qo_heads=16`).

| Metric | Before (1 block) | After (SPLIT_K=8) |
|---|---|---|
| Forward kernel duration (`ncu`, single launch) | 254.72 μs | 42.30 μs |
| Forward kernel avg (`nsys`, 50 iters) | 92.4 μs | 15.1 μs |
| Reduce kernel avg (`nsys`, 50 iters) | n/a | 8.5 μs |
| Grid size | 1 block | 8 blocks |
| Achieved Occupancy | 8.33% | 8.31% |
| Theoretical Occupancy | 8.33% | 8.33% |
| Compute (SM) Throughput | 0.23% | 1.68% |
| Memory Throughput | 0.29% | 2.06% |
| Block Limit (shared memory) | 1 block/SM | 1 block/SM |

**Forward-kernel-only speedup: ~6x** (254.72 μs → 42.30 μs, tracking closely
with the 8x increase in block count minus fixed per-launch overhead — a
clean, near-linear scaling result).

**End-to-end wall-clock speedup was smaller (~25-28%, ~97μs → ~70μs)**
because at these very short per-kernel durations, CPU-side kernel launch
dispatch overhead (`cudaLaunchKernel`, two launches per call now instead of
one) becomes a comparable-or-larger cost than the GPU compute time itself.
This is expected and is itself a useful profiling finding — see "Two
separate bottlenecks" below.

## Two separate bottlenecks — only one is fixed

**Bottleneck 1 (FIXED): too few blocks launched.** The original grid had
only 1 block, so only 1 of 142 SMs did any work at all. Split-K directly
fixes this by launching `SPLIT_K` independent blocks per token.

**Bottleneck 2 (NOT YET FIXED): per-block shared memory pressure.** Even
with 8 blocks now active, each block still requests 94.98 KB of dynamic
shared memory, and the GPU's ~102.4 KB/SM budget only allows **1 block per
SM** regardless of grid size. This caps achieved occupancy at 8.3% per
active SM — identical before and after the split-K change, since it's an
orthogonal constraint. Nsight Compute's own launch-statistics analysis
confirms this directly: with `SPLIT_K=8`, it estimates a further **94.37%
speedup** is available just from using more of the GPU's 142 SMs (i.e.
pushing `SPLIT_K` higher), separate from the shared-memory constraint.

These are two independent axes and need two independent fixes:
1. **Grid size vs. GPU SM count** — solved by increasing `SPLIT_K` (more
   blocks = more SMs used), up to a point where per-block work becomes too
   small to be worth the launch overhead.
2. **Per-block shared memory footprint** — requires reducing the 94.98 KB
   dynamic shared memory request itself (e.g. lower-precision intermediate
   accumulators, smaller per-block tile sizes) so more than 1 block can
   co-reside per SM.

## Open questions / next steps

- Sweep `SPLIT_K` beyond 16 (32, 64, up toward 142) to quantify how far
  bottleneck-1-only fixes can go before launch overhead or bottleneck 2
  dominates. Note: `SPLIT_K=32` currently fails to compile
  (`OutOfResources: shared memory`) at `BLOCK_N=64` because the loop body
  only iterates once when `chunk == BLOCK_N`, apparently changing Triton's
  pipelining/shared-memory allocation; likely fixable with a smaller
  `BLOCK_N` at high `SPLIT_K`.
- Reduce_kernel's own cost (8.5 μs) is now comparable to the (much smaller)
  forward kernel duration per block; worth checking whether it becomes the
  dominant cost at higher `SPLIT_K` and needs its own optimization pass.
- Address shared memory footprint directly (bottleneck 2) as a separate
  follow-up, independent of further `SPLIT_K` tuning.
- Once both bottlenecks are addressed, re-run the full 23-workload
  correctness + performance sweep (not just the one decode-style workload)
  to confirm the fix generalizes.
