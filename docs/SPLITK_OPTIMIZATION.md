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

## Phase 1 fix: adaptive BLOCK_N (unblocking high SPLIT_K)

The original `dsa_fwd_kernel_splitk` used a fixed `BLOCK_N=64` regardless of
`SPLIT_K`. At `SPLIT_K=32`, `chunk_size = topk // SPLIT_K = 64 = BLOCK_N`,
so the inner KV loop ran exactly once — this specific configuration
triggered a Triton shared-memory compilation failure
(`OutOfResources: shared memory, Required: 116736, Hardware limit:
101376`), unrelated to the two runtime bottlenecks described above.

**Fix:** made `BLOCK_N` adaptive —
`BLOCK_N = min(64, max(16, chunk_size // 2))` — keeping the inner loop at
≥2 iterations while respecting Triton's tensor-core minimum dot-product
dimension (16). Verified exact-match correctness (`max_abs_err=0.0`)
against `kernel()` across `SPLIT_K = 2, 4, 8, 16, 32, 64, 128`.

## Phase 2: SPLIT_K sweep — a third bottleneck found

With Phase 1's fix unblocking the full range, swept `SPLIT_K` from 2 to 128
on the same workload (`uuid=0c23b10c...`).

### Wall-clock latency (torch.cuda.Event, 50 iters, `profile_kernel_splitk.py`)

| SPLIT_K | Avg latency |
|---|---|
| 2 | 69.7 μs |
| 4 | 69.4 μs |
| 8 | 71.6 μs |
| 16 | 68.3 μs |
| 32 | 69.9 μs |
| 64 | 70.8 μs |
| 128 | **127.7 μs** |

Latency is essentially flat from `SPLIT_K=2` through `64` (~68-72 μs), then
nearly doubles at `128`. This was unexpected — if the only bottleneck were
grid-size underutilization (bottleneck 1) or per-block occupancy
(bottleneck 2), higher `SPLIT_K` should keep helping (or at least not
actively hurt) up to the GPU's SM count (142). Something else dominates at
high `SPLIT_K`.

### Kernel-level breakdown (nsys, forward vs. reduce kernel separately)

| SPLIT_K | `dsa_fwd_kernel_splitk` | `dsa_reduce_kernel` | Reduce/Forward ratio |
|---|---|---|---|
| 16 | 9,578 ns | 11,907 ns | 1.2x |
| 64 | 5,494 ns | 54,794 ns | **10x** |
| 128 | 4,285 ns | 111,994 ns | **26x** |

This isolates the cause precisely: **the forward kernel keeps improving as
SPLIT_K grows** (9.6 → 5.5 → 4.3 μs, more parallelism helping as expected),
but **`dsa_reduce_kernel`'s cost scales roughly linearly with SPLIT_K**
(12 → 55 → 112 μs — doubling from 64→128, matching SPLIT_K doubling). At
high SPLIT_K, the reduce kernel completely dominates and erases the
forward kernel's gains.

### Bottleneck 3 (NOT YET FIXED): sequential reduction

`dsa_reduce_kernel` merges `SPLIT_K` partial results using
`for split_id in t1.static_range(SPLIT_K):` — a compile-time-unrolled,
**strictly sequential** loop inside a single program per token. Its cost is
therefore directly proportional to `SPLIT_K`, with no parallelism at all
across splits. This is architecturally different from bottlenecks 1 and 2
(which are both about underutilizing the GPU's parallelism) — this one is
about an *intentionally* serial merge step that doesn't scale.

**Practical consequence:** the useful operating range for `SPLIT_K` on this
workload is roughly **8–16**, where forward-kernel gains and reduce-kernel
cost are still roughly balanced (reduce/forward ratio ~1-2x). Beyond that,
total latency gets worse, not better, despite the forward kernel itself
continuing to improve.

### Candidate fixes for bottleneck 3 (not yet implemented)

1. **Tree/hierarchical reduction** — merge splits pairwise across
   `log2(SPLIT_K)` sequential passes instead of one `SPLIT_K`-length linear
   scan, each pass parallel across pairs.
2. **Multi-threaded reduction within the block** — spread the `SPLIT_K`
   partial-result loads/merges across threads in `dsa_reduce_kernel`'s
   block instead of a single-threaded sequential loop.
3. **Cap SPLIT_K empirically** (8-16 for this workload shape) and accept
   the reduce kernel's cost at that scale as a simpler, "good enough"
   solution — avoids the added engineering complexity of options 1/2, at
   the cost of not fully solving bottlenecks 1/2 (some occupancy headroom
   at higher SPLIT_K left unused).

## Updated status: three bottlenecks identified, one fixed, two open

| # | Bottleneck | Status | Fix |
|---|---|---|---|
| 1 | Grid too small (1 block, 1/142 SMs) | **Fixed** | Split-K forward kernel |
| 2 | Per-block shared memory caps occupancy at ~8.3% | Open | Reduce shared-mem footprint (tiling, lower precision) |
| 3 | Reduce kernel scales linearly with SPLIT_K, caps useful SPLIT_K at ~16 | Open | Tree reduction or parallel merge |
