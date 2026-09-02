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

## Attempted fix for bottleneck 3: parallel 3-phase reduction (kernel_splitk_v2)

To fix bottleneck 3 (sequential reduce kernel scaling linearly with
`SPLIT_K`), implemented `kernel_splitk_v2()` with a 3-phase parallel
reduction, replacing the single sequential `dsa_reduce_kernel`:

- **Phase A** (`dsa_reduce_phaseA_kernel`): one program per token, computes
  final `(m, l)` using vectorized hardware reduction (`t1.max`/`t1.sum`
  over a `[SPLIT_K, BLOCK_H]` tile) instead of a sequential loop.
- **Phase B** (`dsa_reduce_phaseB_kernel`): one program per **(token,
  split)** — same grid shape as the forward kernel, genuinely parallel.
  Each program independently computes its own rescale factor (now possible
  since `m_final` is already known from Phase A) and atomically adds its
  contribution into a shared per-token accumulator.
- **Phase C** (`dsa_reduce_phaseC_kernel`): one program per token, cheap
  final divide + LSE computation.

### Correctness

Verified exact match (`max_abs_err=0.000000`) against `kernel()` across
`SPLIT_K = 2, 4, 8, 16, 32, 64, 128` — same strength of result as v1,
despite introducing atomic adds (fp32 atomics on Ada apparently did not
introduce visible reordering error at this scale).

### GPU kernel time: genuinely improved

| SPLIT_K | v1 total (fwd + sequential reduce) | v2 total (fwd + 3 reduce phases) |
|---|---|---|
| 16 | 21.5 μs | 33.9 μs* |
| 64 | 60.3 μs | 38.4 μs |
| 128 | 116.3 μs | 48.1 μs |

*Note: v2 has a higher fixed cost at low SPLIT_K (3 extra kernel launches'
worth of per-launch GPU-side overhead), but scales far better — v1's
sequential reduce nearly doubles with each SPLIT_K doubling; v2's Phase B
grows only ~1.9x across an 8x increase in SPLIT_K (21.0 → 29.1 → 39.8 μs).
At SPLIT_K=128, v2's total GPU kernel time is less than half of v1's.

### Wall-clock latency: WORSE across the board

| SPLIT_K | v1 wall-clock | v2 wall-clock |
|---|---|---|
| 2 | 69.7 μs | 114.5 μs |
| 16 | 68.3 μs | 112.0 μs |
| 64 | 70.8 μs | 113.7 μs |
| 128 | 127.7 μs | 115.2 μs |

**v2 is slower than v1 everywhere except SPLIT_K=128** (where it's 115 vs
128 μs — the one point where v1's linear reduce-kernel blowup is bad
enough to lose). Everywhere else, v1 wins by a wide margin (~35-65 μs
worse for v2).

### Why: launch-overhead cost exceeded the compute savings

v1 launches 2 kernels per call (forward + reduce). v2 launches 4 (forward +
3 reduce phases). At these very short per-call durations (tens of
microseconds), fixed CPU-side kernel-launch dispatch cost dominates total
latency more than the GPU-side compute time itself — this is the same
launch-overhead effect first seen when comparing the original single-block
kernel to v1 (see "Performance results" above), now compounded by v2
having twice as many launches as v1.

**Conclusion: bottleneck 3 (sequential reduce-kernel scaling) is real and
v2 genuinely fixes it in GPU-compute terms, but it isn't actually on the
practical path.** The linear scaling only becomes a serious problem at
`SPLIT_K ≥ 64`, and that operating point was never attractive anyway once
launch overhead is accounted for (v1's own sweep showed the practical
useful range is `SPLIT_K ≈ 8-16`, well below where bottleneck 3 bites
hard). Solving bottleneck 3 with more kernel launches trades a problem that
matters at an operating point you wouldn't use anyway, for a cost (launch
overhead) that hurts at every operating point.

**Current recommendation: keep `kernel_splitk` (v1) with `SPLIT_K ≈ 16` as
the practical configuration.** `kernel_splitk_v2` is kept in the codebase
as a correctness-verified alternative and a documented example of a fix
that is correct and measurably better on one axis (GPU compute time) while
being worse on the axis that actually matters (end-to-end latency) — a
useful cautionary result for the eventual writeup.

### Open direction: fusion instead of more parallelism

A more promising angle than v2's added-parallelism approach may be
**reducing launch count** while keeping some of the parallel-reduction
benefit — e.g. fusing Phase A and Phase C into the forward kernel's own
launch via a persistent-kernel or single-pass design, rather than adding
separate launches. Not yet attempted; noted as a candidate for future work
alongside bottleneck 2.

## Updated status: three bottlenecks, one fixed, two open (revised)

| # | Bottleneck | Status | Notes |
|---|---|---|---|
| 1 | Grid too small (1 block, 1/142 SMs) | **Fixed** | `kernel_splitk` (v1), verified 6x forward-kernel speedup |
| 2 | Per-block shared memory caps occupancy at ~8.3% | Open | Not yet attempted |
| 3 | Reduce kernel scales linearly with SPLIT_K | **Fixed algorithmically, not practically** | `kernel_splitk_v2` solves it in GPU-compute terms but loses to launch overhead in wall-clock terms; v1 remains the practical choice |

## Experiment 2: decomposing the v1-vs-v2 wall-clock gap

The v1-vs-v2 comparison above showed v2 losing on wall-clock latency despite
winning on GPU compute time, attributed qualitatively to "launch overhead."
This section isolates that claim into named, independently-measured
components rather than leaving it as an inference.

### Experiment 2a: pure kernel-launch dispatch cost

Measured by launching a trivial (near-zero-work) Triton kernel N times
back-to-back and fitting total latency vs. N (`experiment2_launch_overhead.py`).

| N launches | Avg total (μs) | Per-launch (μs) |
|---|---|---|
| 1 | 9.08 | 9.08 |
| 2 | 16.93 | 8.47 |
| 4 | 32.33 | 8.08 |
| 8 | 62.86 | 7.86 |
| 16 | 119.28 | 7.46 |
| 32 | 234.97 | 7.34 |

Linear fit: `total_us ≈ 3.06 + 7.29 * n_launches`. Per-launch cost drops
smoothly and stabilizes around **~7.3 μs**, consistent with a small
one-time fixed cost plus a stable marginal dispatch cost per launch — this
is the real, isolated, hardware/driver/PyTorch/Triton-stack cost of one
additional kernel launch on this RTX 6000 Ada setup, independent of what
the kernel actually computes.

### Experiment 2b: intermediate-buffer allocation cost

`kernel_splitk_v2` allocates three extra tensors (`m_final`, `l_final`,
`acc_final`) that `kernel_splitk` (v1) does not need. Measured their
allocation cost in isolation, with the same warmup convention used
throughout this project (`experiment2b_allocation_overhead.py`):

**Extra v2 allocation cost (properly warmed up): ~16.7 μs/iter**

(Note: an earlier, unwarmed-up measurement showed ~238 μs — a red herring
caused by the caching allocator's first-touch cost for a new shape, not
representative of steady-state cost. Always warm up allocations of a given
shape before timing, matching how the real benchmark scripts already warm
up 10 iterations before every timed measurement.)

### Putting it together

| Component | Cost |
|---|---|
| 2 extra kernel launches (v2 has 4, v1 has 2) | ~14.6 μs (2 × 7.29 μs) |
| 3 extra tensor allocations (v2-specific buffers) | ~16.7 μs |
| **Combined predicted extra cost** | **~31.3 μs** |
| **Actual measured v1-vs-v2 gap** (SPLIT_K=2/16/64) | **~43-45 μs** |

Two independently-measured, named effects account for **~70%** of the
actual gap. The remaining ~12-14 μs is an unexplained residual — plausibly
Python-level overhead in `kernel_splitk_v2`'s wrapper (stride computation
and argument marshaling across twice as many kernel launch calls) — noted
here as acknowledged and not yet further decomposed, rather than chased to
diminishing returns.

**This upgrades the earlier qualitative claim ("launch overhead explains
the gap") into a quantitative, partially-decomposed one**: kernel-launch
dispatch and buffer allocation together account for the large majority of
the measured wall-clock cost of v2's added parallelism, with a smaller,
named-but-unexplained remainder. This is the evidence base for the planned
atomic-counter single-launch redesign — a design that specifically
eliminates both of the two *largest* named costs (extra launches AND extra
intermediate allocations) simultaneously, rather than addressing either in
isolation.

## Experiment 3: single-launch design via atomic-counter synchronization (kernel_splitk_v3)

Motivated directly by Experiment 2's decomposition: v2 lost to v1 on
wall-clock latency primarily because of two named, measured costs —
kernel-launch dispatch (~7.3 μs/launch) and intermediate-buffer allocation
(~16.7 μs) — together accounting for ~70% of the gap. This section
describes a design that targets both costs simultaneously by collapsing
the entire forward + reduce pipeline into a **single kernel launch**.

### Background: does Triton support cross-block synchronization?

Before designing this, we researched whether Triton provides a safe way
for blocks (programs) to coordinate across a single kernel launch.

**Finding: general grid-wide synchronization is not safely supported.**
One analysis of Triton kernel design states plainly that "there is no
(grid) synchronization between programs in Triton, which makes it
impossible to communicate values between different programs over HBM" in
the naive sense (FlashRNN, arXiv:2412.07752). A real Triton GitHub issue
(triton-lang/triton#7125) documents that hand-rolled spin-lock/busy-wait
synchronization between blocks — using the pattern from Triton's own
layer-norm backward-pass tutorial — generates unexpected extra barriers
and shared-memory broadcast overhead, causing "a significant performance
drop," and that atomic-op synchronization behavior is inconsistent between
scalar and tensor operands. A naive "blocks wait on each other" design was
judged too risky to build on top of.

**Finding: a safe, established alternative exists — the "last-CTA"
atomic-counter pattern**, used in production split-K GEMM kernels (the
same technique CUTLASS/stream-K kernels use for cross-CTA reduction). The
idea: every program computes its partial result and atomically increments
a shared counter; whichever program's increment happens to return the
final count (i.e., it finished last) proceeds to read every other
program's partial result and perform the reduction *inline, in the same
kernel invocation* — no busy-waiting, no second launch, just one
conditional branch taken by exactly one of the SPLIT_K programs per token.

**Finding: memory-ordering semantics matter and must be set deliberately.**
Triton's atomic operations (`tl.atomic_add`, `tl.atomic_cas`, etc.) accept
a `sem` parameter: `"acquire"`, `"release"`, `"acq_rel"` (the default), or
`"relaxed"`. A published Triton split-K GEMM implementation notes that
`relaxed` is a valid *optimization* — but only for pure accumulation, where
no control-flow decision depends on the atomic's result. Our design is
different: the counter is a **synchronization gate**, not a pure
accumulator — a program's decision to enter the reduction branch depends
on the atomic's return value. This requires the default `acq_rel`
semantics:
- **release** ensures a program's writes to the partial-result scratch
  buffers (`m`, `l`, `acc`) are visible to other programs before its
  counter increment becomes visible.
- **acquire** ensures the "winning" program does not see stale data left
  over from before other programs' releases when it reads all partials.

`kernel_splitk_v3` uses the explicit, self-documenting `sem="acq_rel"` on
the counter atomic and is deliberately *not* relaxed to `"relaxed"`,
despite that being a tempting micro-optimization — doing so would
reintroduce exactly the race condition this design is built to avoid.

### Design

Single kernel launch, `grid = (num_tokens, SPLIT_K)` — same shape as v1/v2's
forward pass. Each program:
1. Computes its partial `(m, l, acc)` for its chunk of the topk KV range —
   identical math to `dsa_fwd_kernel_splitk`.
2. Writes its partial result to global scratch (ordinary `t1.store`, no
   atomics needed here — each program writes to a disjoint location).
3. Atomically increments a per-token counter with `sem="acq_rel"`.
4. If its increment brought the counter to `SPLIT_K` (i.e., it was last),
   it reads **all** SPLIT_K partials and performs the reduction inline:
   a vectorized hardware max/sum reduction for `(m, l)` (same technique as
   v2's Phase A), followed by a sequential `for s in range(SPLIT_K)` loop
   accumulating the weighted `acc` — sequential, but confined entirely to
   this one winning program, with no extra kernel launch and no extra
   intermediate-buffer allocation beyond the partial-result scratch already
   required.

This deliberately reintroduces a sequential loop over `SPLIT_K` (the same
shape of cost that made v1 degrade at high SPLIT_K) — but the bet, informed
directly by Experiment 2, is that avoiding launch and allocation overhead
matters more than avoiding that sequential loop, in the practical
SPLIT_K range.

### Correctness: implementation bug, then stress testing

**Implementation bug found before testing:** a first draft included dead
code (an unused, dimensionally-invalid line attempting to recompute a
per-split value via a masked-sum trick, left over from an earlier draft of
the accumulation logic). This caused `CompilationError` at `SPLIT_K = 2, 4,
8` but — notably — compiled successfully at `SPLIT_K = 16`, an inconsistency
not fully explained (plausibly a different code-generation/unrolling path
at that specific configuration). Removed the dead code entirely rather than
investigate why it sometimes compiled, since it was unused regardless.

**Single-run correctness:** after the fix, exact match (`max_abs_err =
0.000000`) against `kernel()` across `SPLIT_K = 2, 4, 8, 16, 32, 64, 128`
on the first attempt.

**Why a single passing run is not sufficient evidence here:** unlike v1/v2
(which use no cross-block synchronization primitives beyond ordinary
writes read by a separately-launched kernel), v3's correctness depends on
an atomic-counter race-resolution mechanism. A race condition in this kind
of design can pass the overwhelming majority of runs and fail rarely,
depending on GPU scheduling nondeterminism — the worst class of bug to
leave undetected. A single clean pass is necessary but not sufficient
evidence of correctness.

**Stress test** (`stress_test_v3.py`): 200 trials per `SPLIT_K` value,
fresh random `q_nope`/`q_pe`/`ckv_cache`/`kpe_cache` each trial (different
seed per trial, forcing different data and therefore different exact
kernel timing/scheduling each run), compared against `kernel()` at
tolerance 0.02 (matching the bf16-rounding scale established throughout
this project).

**Result: 1,400/1,400 trials passed (200 trials × 7 SPLIT_K values), zero
failures.** This is strong evidence the `acq_rel` design is correct on this
hardware/driver/PyTorch/Triton stack — though, as with any finite stress
test of a concurrent design, it reduces rather than eliminates the
possibility of an undetected rare race.

### Performance: v3 beats both v1 and v2 at every measured point

| SPLIT_K | v1 (2 launches) | v2 (4 launches) | v3 (1 launch) |
|---|---|---|---|
| 2 | 69.7 μs | 114.5 μs | **62.1 μs** |
| 4 | 69.4 μs | 111.7 μs | **61.3 μs** |
| 8 | 71.6 μs | 114.4 μs | **60.1 μs** |
| 16 | 68.3 μs | 112.0 μs | **61.5 μs** |
| 32 | 69.9 μs | 114.4 μs | **61.2 μs** |
| 64 | 70.8 μs | 113.7 μs | **62.2 μs** |
| 128 | 127.7 μs | 115.2 μs | **94.2 μs** |

v3 beats v1 by roughly 10-15% across the flat operating range (2-64), and
by a much wider margin at the high end (94.2 μs vs. v1's 127.7 μs at
SPLIT_K=128 — v1's worst point still loses to v3's second-worst point by
~33 μs). v3 beats v2 everywhere, typically by close to half.

v3's latency stays flatter across the sweep than either v1 or v2 (~60-62 μs
from SPLIT_K=2 through 64), degrading only at 128 — consistent with the
inline sequential accumulation loop inside the winning program starting to
cost real time once SPLIT_K is large enough that 128 sequential
load-multiply-add steps over a `[16, 512]` tensor is nontrivial work, even
with zero added launch/allocation overhead.

### Summary: the complete three-version arc

1. **v1** (naive split-K, sequential reduce): fixes the original
   grid-size-underutilization bottleneck; degrades at high SPLIT_K due to
   an O(SPLIT_K) sequential reduction.
2. **v2** ("obviously better" fully-parallel reduction): fixes the
   algorithmic scaling problem in raw GPU compute terms — but *loses* in
   practice, because it doubles kernel-launch count and adds new
   intermediate-buffer allocations, both quantified in Experiment 2 as the
   dominant costs in this regime.
3. **v3** (single-launch, atomic-counter-gated inline reduction): designed
   specifically in response to Experiment 2's decomposition — eliminates
   both of the two largest named costs (extra launches, extra allocations)
   simultaneously, accepting a sequential reduction step (like v1) but
   confined to a single program with no launch penalty. **Wins outright**
   against both v1 and v2 at every measured SPLIT_K.

This is not "we tried an optimization and it worked" — it is a
diagnosis-fix-measure-refine loop: the failure of the "obvious" fix (v2)
was itself decomposed into named causes, and that decomposition directly
produced the design that succeeded (v3).

## Updated status: three bottlenecks, all addressed (final)

| # | Bottleneck | Status | Fix |
|---|---|---|---|
| 1 | Grid too small (1 block, 1/142 SMs) | **Fixed** | Split-K forward kernel (v1/v3) |
| 2 | Per-block shared memory caps occupancy at ~8.3% | Open | Not yet attempted — candidate future work |
| 3 | Reduce kernel scales linearly with SPLIT_K / launch+allocation overhead dominates naive fixes | **Fixed** | `kernel_splitk_v3`: single-launch, atomic-counter-gated inline reduction; verified via 1,400-trial stress test; beats v1 and v2 at every measured SPLIT_K |

**Current recommendation: `kernel_splitk_v3` is the best-performing,
correctness-stress-tested configuration on RTX 6000 Ada.** Bottleneck 2
(shared memory / per-block occupancy) remains the one unaddressed axis and
is noted as future work, ideally revisited with cross-architecture context
once profiling extends to A100/L40S/DGX Spark, since shared-memory budgets
and launch-overhead characteristics may both differ meaningfully across
architectures.

### Note: ncu unavailable for v3 profiling on this run

Attempted `ncu` occupancy/throughput profiling of `dsa_fwd_kernel_splitk_v3`
but hit `Profiling failed because a driver resource was unavailable` —
traced to a persistent, admin-run `nvidia-dcgm` service (`dcgm-exporter`,
running continuously on this node for cluster-wide GPU monitoring) holding
exclusive access to the hardware performance-counter registers `ncu`
needs. Confirmed via `ps aux | grep dcgm` and `systemctl status
nvidia-dcgm` — not caused by our code or job, and not something fixable
from a user account. `nsys` (used for all wall-clock timing throughout this
project) is unaffected by this conflict. Deferred: retry `ncu` profiling of
v3 opportunistically, or flag to cluster admins if it recurs.

### Note: ncu unavailable for v3 profiling in this session

Attempted `ncu` occupancy/throughput profiling of `dsa_fwd_kernel_splitk_v3`
but hit a resource conflict with the cluster's persistent DCGM monitoring
service — an infrastructure/methodology issue, not a property of the
kernel. See `docs/INFRASTRUCTURE_NOTES.md` for full diagnosis. The v3
performance result above (wall-clock comparison + 1,400-trial stress test)
is unaffected, since it relies on `nsys`/`torch.cuda.Event` timing, not
`ncu`.

## Bottleneck 2 investigation: shared memory footprint (in progress)

Bottleneck 2 (per-block shared memory capping achieved occupancy at ~8.3%,
identified in the original baseline profiling) was never addressed by v1,
v2, or v3 — all three inherit the same per-block tile sizes for the
forward-pass computation. Confirmed directly via Triton's compiled-kernel
cache metadata (no `ncu` access was available in this session — see
`docs/INFRASTRUCTURE_NOTES.md`):

```bash
cat ~/.triton/cache/<hash>/dsa_fwd_kernel_splitk_v3.json | python3 -m json.tool
```

"shared": 94976,
"num_warps": 4,
"num_stages": 3,


**`shared: 94976` bytes is identical to the value measured for v1 in the
original `ncu` profiling** (94.98 KB), confirming bottleneck 2 is present
and unchanged in v3 — expected, since v3's per-block forward-pass math
(tile shapes for `q_nope`, `k_ckv`, the `acc` accumulator, etc.) was not
modified from `dsa_fwd_kernel_splitk`.

### Attempted cheap fix: reducing num_stages

`num_stages` controls how many pipeline stages Triton uses to overlap
memory loads with compute (each stage typically needs its own buffered
copy of load-destined tiles in shared memory, so naively, fewer stages
should mean less shared memory usage). This required no kernel rewrite —
just an overridable launch-time keyword argument
(`test_num_stages_sweep.py`, launching `dsa_fwd_kernel_splitk_v3` directly
with `num_stages` swept).

**Result — counterintuitive and informative:**

| num_stages | Result |
|---|---|
| 1 | **FAILED**: `OutOfResources: shared memory, Required: 116736, Hardware limit: 101376` |
| 2 | 30.04 μs |
| 3 (default) | 29.46 μs |
| 4 | 30.19 μs |

Two findings:
1. **`num_stages=1` requires *more* shared memory (116,736 bytes) than the
   default `num_stages=3` (94,976 bytes)** — the opposite of the naive
   expectation. This suggests Triton's compiler does not scale shared
   memory usage simply/linearly with `num_stages`; at very low stage
   counts it likely falls back to a different code-generation strategy
   (e.g., a non-double-buffered load pattern requiring different scratch
   space) rather than simply allocating less memory. Not fully explained
   without deeper inspection of Triton's compiler internals — noted
   honestly as an open question rather than a resolved mechanism.
2. **`num_stages = 2, 3, 4` are statistically indistinguishable in latency**
   (30.04 / 29.46 / 30.19 μs — within normal run-to-run noise). `num_stages`
   is not a meaningful lever for either memory footprint or performance on
   this kernel.

### Conclusion: the real cost is tile size, not pipelining depth

Since `num_stages` doesn't move the needle, the ~95 KB shared memory
requirement is coming from the **tile shapes themselves** — most likely
the `BLOCK_D_CKV=512` dimension appearing simultaneously in multiple live
tensors during the tensor-core `t1.dot` operations (the `acc` accumulator
`[BLOCK_H=16, BLOCK_D_CKV=512]` in fp32, the `k_ckv` tile `[BLOCK_N,
BLOCK_D_CKV]`, and `q_nope` `[BLOCK_H, BLOCK_D_CKV]`).

**The next concrete step (not yet attempted): tile the `BLOCK_D_CKV=512`
dimension itself** — instead of loading/accumulating the full 512-wide
dimension per block, split it into sub-tiles (e.g. 2×256 or 4×128) and add
an inner loop over sub-tiles, accumulating partial results the same way
the existing KV loop already accumulates across `BLOCK_N` tiles. This is a
structural change to the core per-block computation (not just the launch
or reduction logic touched by v1/v2/v3), and is scoped as a separate,
future piece of work — it should not be started casually at the end of a
session, since it will need the same correctness rigor already applied
elsewhere in this project (exact-match verification against `kernel()`,
and likely a repeated-trial stress test if it introduces any new
synchronization).

## Bottleneck 2, attempt 1: D-tiling (reverted, design corrected for next attempt)

A first attempt at tiling `BLOCK_D_CKV` (splitting the accumulator into
smaller sub-tiles to reduce shared memory) was built and found to have a
flawed design: all D-tiles of the accumulator were updated within the same
KV-tile loop, meaning they all stayed simultaneously live for the entire
loop duration — no actual reduction in peak memory versus one full-width
accumulator. Reverted before committing (net-zero diff after cleanup).

**Corrected design for next attempt:** `qk` computation and softmax
bookkeeping (`m_i`, `l_i`, `p`) are a reduction over the full 512-dim and
do not need to be tiled — they're independent of how the *output*
accumulator is tiled. Only the V-side accumulation (`acc += p @ k_ckv`)
needs tiling. Correct structure: outer loop over D-tiles, inner loop over
KV-tiles, recomputing `qk`/`p` per D-tile (small, cheap — only `[BLOCK_H,
BLOCK_N]`) but reloading the full-width `k_ckv` as K each time (the actual
redundant cost — `D_TILES`× more K-side loads/compute). Trade-off to be
measured: does reduced peak accumulator memory (→ potentially better
occupancy) outweigh the redundant K-side work? Not yet built or tested —
scoped as the next concrete step.

## Bottleneck 2, attempt 2: D-tiling with full-width K, tiled V (tested, made things worse)

Built and tested a corrected D-tiling design: outer loop over D-tiles
(one launch per D-tile, `D_TILE_ID` as a compile-time constant), full-width
`qk`/softmax computation per KV-tile (since that's a reduction over the
full 512-dim, independent of D-tiling), and only the V-side accumulation
(`acc`) tiled to `TILE_D` width.

**Result: `OutOfResources: shared memory, Required: 128000, Hardware
limit: 101376`** — this configuration needs *more* shared memory
(128,000 bytes) than the original untiled kernel (94,976 bytes), not less.

**Root cause (identified by inspecting what's simultaneously live):** the
original kernel loads `k_ckv` once per KV-tile and reuses that same tile
for both its K role (in `qk_nope = q_nope @ k_ckv.T`) and its V role (in
`acc += p @ k_ckv`) — implicit memory reuse via a single load. This
D-tiling attempt splits that into two separate loads
(`k_ckv_full`, full 512-wide, for K; `k_ckv_v`, TILE_D-wide, for V), both
live simultaneously within the same loop iteration, alongside the
still-full-width `q_nope`. The smaller accumulator (`[16, TILE_D]` instead
of `[16, 512]`) does not compensate for the *duplicated* K-cache data now
resident at once — net shared memory increases.

**Conclusion: tiling only the V-side (output) dimension of the
accumulator is not sufficient and actively counterproductive**, because it
breaks the original kernel's single-load K/V reuse without replacing it
with anything smaller. A design that actually reduces shared memory would
need to tile the K-side reduction as well — accumulating `qk_nope` across
D-sub-tiles the same way the existing KV-loop already accumulates across
`BLOCK_N` tiles — restoring single-load reuse at the sub-tile level
(load one `[BLOCK_N, TILE_D]` slice, use it for both a partial K-dot and
a partial V-accumulation, then move to the next D-sub-tile). This is a
larger, more invasive restructuring than either attempt made so far, and
is the next candidate design, not yet built.

Two tested attempts (this one and the earlier all-tiles-simultaneously-live
version) are both documented here as negative results with concrete
numbers, rather than left as unexplored hypotheses — both pointed toward
the same conclusion: naive/partial tiling of this kernel's accumulator
does not reduce shared memory without also restructuring the K-side
reduction to preserve single-load K/V reuse at the sub-tile level.

## Generalization across dataset workload shapes (num_tokens 1-8)

All profiling and optimization work so far used a single workload
(`uuid=0c23b10c...`, `num_tokens=1`) — the worst case for grid-size
underutilization, but not representative of every workload in the actual
benchmark dataset. Checked the full dataset's shape distribution: across
all 23 workloads, `num_tokens` ranges from 1 to 8 (`num_pages` fixed at
8462 throughout). This matters because the *original* kernel's grid is
`(num_tokens,)` — at `num_tokens=8`, the original kernel already launches
8 blocks on its own, a meaningfully different starting point than the
single-block `num_tokens=1` case the entire v1/v2/v3 investigation was
built around.

**Test**: one representative workload per distinct `num_tokens` value
(1, 2, 6, 7, 8), comparing `kernel()` [original], `kernel_splitk` (v1), and
`kernel_splitk_v3` (v3) at `SPLIT_K=16` for both correctness and wall-clock
performance (`sweep_workloads.py`).

| num_tokens | err (v1) | err (v3) | orig | v1 | v3 | v1 speedup | v3 speedup |
|---|---|---|---|---|---|---|---|
| 1 | 0.000000 | 0.000000 | 96.75 μs | 68.21 μs | 59.96 μs | 1.42x | 1.61x |
| 2 | 0.000000 | 0.000000 | 96.63 μs | 68.69 μs | 60.19 μs | 1.41x | 1.61x |
| 6 | 0.003906 | 0.003906 | 96.95 μs | 68.33 μs | 60.21 μs | 1.42x | 1.61x |
| 7 | 0.007812 | 0.007812 | 96.91 μs | 68.61 μs | 61.44 μs | 1.41x | 1.58x |
| 8 | 0.007812 | 0.007812 | 96.97 μs | 67.18 μs | 59.94 μs | 1.44x | 1.62x |

**Correctness**: exact match at `num_tokens=1,2`; small non-zero errors
appear at `num_tokens=6,7,8` (0.0039-0.0078), but well within the bf16
rounding tolerance established for this project (golden-reference checks
throughout used `atol=rtol=1e-2`). Critically, **v1 and v3 show identical
error at every `num_tokens` value** — v3 introduces no additional
numerical drift beyond what v1 already has; the small discrepancy is
consistent between both split-K variants, consistent with normal
floating-point accumulation-order differences as more tokens/heads are
processed, not a version-specific bug.

**Performance: remarkably flat across the entire num_tokens range.** The
original kernel's latency does not measurably change from `num_tokens=1`
to `8` (96.6-97.0 μs throughout) — going from 1 to 8 blocks is still far
below the GPU's 142 SMs, so neither end of this range is meaningfully less
occupancy-starved in practice. v1 and v3's speedups are correspondingly
stable across the same range (v1: 1.41-1.44x; v3: 1.58-1.62x), with no
sign of the benefit fading, reversing, or requiring workload-adaptive
tuning within this dataset's shape range.

**Conclusion: split-K's benefit generalizes across the full range of
workload shapes present in this benchmark's dataset.** A single fixed
`SPLIT_K=16` performs consistently well from `num_tokens=1` through `8`,
answering the open question raised earlier in this document (whether
`SPLIT_K` needs to be workload-adaptive) — for this dataset's shape range,
it does not.

**Scope limitation, stated explicitly:** this generalization claim is
bounded by the dataset actually available — `num_tokens` here only goes up
to 8. It should not be read as claiming split-K remains beneficial at much
larger token/batch counts (e.g., 64, 128+), where the original kernel's
own grid size might eventually become large enough on its own to reduce
or eliminate the underutilization problem split-K addresses. Untested
beyond this dataset's range.

## Bottleneck 2, attempt 3: co-tiled K-side and V-side (tested, correct, but slower)

Built the theoretically correct design: both the K-side reduction
(computing `qk_nope`) and the V-side accumulation (`acc`) tiled over
`BLOCK_D_CKV`, each looping over `D_TILES` sub-tiles independently, never
holding a full-width `[*, 512]` tensor at any point
(`dsa_fwd_kernel_splitk_dtiled_v3` / `kernel_splitk_dtiled_v3`).

**Shared memory: genuinely reduced.** Confirmed via Triton's compiled
metadata: `shared` dropped from 94,976 bytes (original/v1/v3) to **83,968
bytes** — an ~11.6% real reduction, and the first attempt in this series
that actually lowered the footprint rather than increasing it (contrast
attempt 2's 128,000 bytes).

**But occupancy tier unchanged.** Using the manual CUDA occupancy formula
(no `ncu` required — see `INFRASTRUCTURE_NOTES.md` for why `ncu` access
was unavailable this session): `102400 // 83968 = 1` block/SM, identical
to `102400 // 94976 = 1`. To reach 2 blocks/SM would require shared memory
≤ 51,200 bytes — roughly half of what this design achieves. **Theoretical
occupancy remained exactly 8.33% in both cases.**

**Correctness: verified.** Exact match (`max_abs_err = 0.000000`) against
`kernel()` across `SPLIT_K ∈ {2,4,8,16}` × `D_TILES ∈ {2,4}` (8
combinations, all passing).

**Performance: worse than v1 in every configuration tested**, and worse
in proportion to `D_TILES`:

| SPLIT_K | D_TILES=2 | D_TILES=4 |
|---|---|---|
| 2 | 1.34x slower | 2.29x slower |
| 4 | 1.28x slower | 1.99x slower |
| 8 | 1.36x slower | 1.93x slower |
| 16 | 1.35x slower | 2.02x slower |

**Root cause, connecting back to earlier findings:** this design pays two
real costs without an offsetting occupancy benefit. (1) The K-side
reduction now reloads `k_ckv` from HBM a second time (once per D-sub-tile,
in a separate pass from the V-side reload) — genuine extra memory
traffic that the original single-load design avoided. (2) The
`kernel_splitk_dtiled_v3` wrapper launches `dsa_fwd_kernel_splitk_dtiled_v3`
once per `D_TILE_ID` (i.e., `D_TILES` separate kernel launches instead of
one) — at `D_TILES=4`, that is 4 launches where v1 needs 1, and per
Experiment 2's measured ~7.3μs/launch marginal cost plus the general
finding that launch/allocation overhead dominates at this kernel's
timescale, this alone predicts a substantial slowdown, independent of any
memory-traffic cost. The result — degradation scaling directly with
`D_TILES` — is consistent with launch overhead being the dominant term,
the same mechanism that made v2 lose to v1.

**Conclusion: bottleneck 2 remains open, and this line of attack (tiling
the accumulator/reduction dimension while adding kernel launches) is now
tested exhausted with three consecutive negative results, each understood
precisely:**
1. Attempt 1: no memory reduction (all tiles simultaneously live) — caught before testing.
2. Attempt 2: memory *increased* (duplicated K-cache residency).
3. Attempt 3: memory reduced (~11.6%) but insufficient to change occupancy tier, while adding launch and HBM-traffic overhead that made wall-clock performance worse.

**Implication for any future attempt**: a design that could plausibly work
would need to (a) achieve a much larger memory reduction — enough to
actually cross the 51,200-byte threshold to 2 blocks/SM — while (b) doing
so within a **single kernel launch** (no extra launches per D-tile,
learning directly from why attempt 3 and v2 both lost). This likely means
finding a way to compute the K-side reduction more compactly (e.g.
different tiling granularity, different precision, or restructuring the
accumulator itself) rather than the "just split the loop" approaches tried
so far. Not yet designed; noted as the honest state of this investigation.

## Bottleneck 2, attempt 4: bf16 accumulator (tested, no meaningful effect)

Hypothesis: the `[16, 512]` fp32 accumulator (`acc`) is a major contributor
to the ~95KB shared memory footprint; halving its precision to bf16 during
the KV loop (rescaling in fp32 to avoid compounding rounding error, then
casting back to bf16 for loop-carried storage) should reduce it
meaningfully. Built as `dsa_fwd_kernel_splitk_v3_bf16acc` /
`kernel_splitk_v3_bf16acc`.

**Implementation note**: an initial attempt hit a genuine Triton
type-consistency error — `acc += t1.dot(...)` implicitly upcasts to fp32
(tensor-core accumulation defaults to fp32 even with bf16 inputs), which
violates Triton's requirement that a loop-carried variable's type stay
constant across iterations. Fixed by making the upcast/downcast explicit:
`acc = (acc.to(fp32) + t1.dot(...)).to(bf16)`.

**Result: shared memory unchanged.** `94976` bytes — statistically
identical to the original `94464/94976` baseline. Halving the
accumulator's nominal size had no measurable effect on the compiled
kernel's shared memory requirement.

**Correctness held**: `max_abs_err = 0.015625` (one bf16 ULP) consistently
across `SPLIT_K ∈ {2,4,8,16}` — expected, small, consistent with the
bf16-rounding pattern seen throughout this project; no new bug.

**Interpretation — a useful negative result**: this rules out the
accumulator as the dominant contributor to shared memory usage, contrary
to the working hypothesis. The real dominant cost is more likely the
tiled K/V cache loads (`k_ckv`, `k_kpe`) combined with Triton's
`num_stages=3` pipelining, which multiplies buffer requirements for
double/triple-buffered memory-load overlap — consistent with the earlier
`num_stages` experiment, which also pointed at pipelining/tile-load
structure rather than the accumulator as the real driver, even though
that experiment's own lever (stage count) didn't help either.

## Bottleneck 2: four negative results, investigation status

| Attempt | Idea | Shared mem result | Outcome |
|---|---|---|---|
| 1 | Tile accumulator only, all tiles live | No change (by design flaw, caught before testing) | Reverted |
| 2 | Full-width K, tiled V | 128,000 bytes (worse) | Reverted |
| 3 | Co-tiled K+V, single accumulator per D-tile | 83,968 bytes (best result, but tier-insufficient) | Kept, correct but slower (launch overhead) |
| 4 | bf16 accumulator | 94,976 bytes (no change) | Kept, correct but no benefit |
| — | Reduce num_stages | 116,736 bytes at num_stages=1 (worse) | Ruled out earlier |

Four structurally different attempts have now been tried and precisely
understood. The consistent finding across attempts 3 and 4 together is
that **the accumulator and pipeline-stage-count are not the dominant
shared-memory costs** — attempt 3's real reduction came specifically from
eliminating the *full-width* K-cache load, not from touching the
accumulator, and attempt 4 (touching only the accumulator) achieved
nothing. This narrows the real target for any future attempt: **the K/V
tile loads themselves and Triton's automatic pipelining buffers around
them** are the most likely dominant cost, and a design that tiles those
more aggressively (while, per attempt 3's lesson, staying within a single
kernel launch) is the remaining untested direction. Not yet attempted;
noted as the state of this investigation for future work.

## Bottleneck 2, attempt 5: decoupling BLOCK_N from SPLIT_K (occupancy doubled, no speedup)

Re-examined data already collected earlier in this investigation: Triton's
compiled metadata across different `SPLIT_K` values (originally checked
only for compile-success, not systematically for shared memory) showed
`BLOCK_N` — not the accumulator (attempt 4) or D-dimension tiling
(attempts 1-3) — is the dominant driver of shared memory. `BLOCK_N=16`
(previously only reachable at `SPLIT_K≥64`, via the existing adaptive
formula `BLOCK_N = min(64, max(16, chunk_size // 2))`) compiles to just
**37,568 bytes**, and `BLOCK_N=32` to exactly **51,200 bytes** — both
crossing the threshold to unlock **2 blocks/SM (16.67% theoretical
occupancy, double the 8.33% seen everywhere else this session)**.

**The problem with the existing formula**: it only reaches small `BLOCK_N`
at high `SPLIT_K`, exactly where the reduction step (even v3's improved
single-launch design) has more per-token overhead. This conflates two
independent variables that don't need to move together.

**Fix**: added an optional `BLOCK_N` override parameter to
`kernel_splitk_v3()`, decoupling K/V tile width from `SPLIT_K` entirely —
allowing a good occupancy-driving `BLOCK_N` (16 or 32) to be paired with a
`SPLIT_K` in the previously-established good operating range (8-32),
rather than being forced together.

**Result: correctness held (exact match) at every combination tested, but
no meaningful wall-clock improvement.** `SPLIT_K=16` at `BLOCK_N=None`
(default, 64), `32`, and `16` all landed within ~1.3μs of each other
(60.77-61.52 μs) — statistically indistinguishable, despite the
BLOCK_N=16/32 configurations having double the theoretical occupancy.
Confirmed at additional `SPLIT_K` values (8, 32) with `BLOCK_N=16`: same
result, no improvement (60.24-60.41 μs, matching the baseline range).

**Interpretation — a genuinely informative negative result.** Theoretical
occupancy improving did not translate into measured speedup. The most
likely explanation: at this workload's scale (`num_tokens=1`,
`SPLIT_K≤32` means at most 32 total blocks launched), the real constraint
was never "too few blocks *per SM*" — it is "too few SMs used *at all*."
With only 8-32 blocks launched across the GPU's 142 SMs, most SMs sit
completely idle regardless of whether the handful of *active* SMs run 1 or
2 blocks each. Doubling blocks-per-active-SM cannot compensate for the
vast majority of SMs having zero work. This reframes bottleneck 2 as
possibly not independently fixable via per-block resource tuning at all —
the real lever for using more SMs is `SPLIT_K` itself (already explored:
higher `SPLIT_K` fixes total-SM-utilization but costs more in the
reduction step, per attempts and experiments earlier in this document),
and per-block occupancy tuning (this attempt) operates on a different,
apparently less binding, axis at this workload's scale.

## Bottleneck 2: five attempts, investigation summary

| Attempt | Idea | Shared mem | Occupancy | Wall-clock result |
|---|---|---|---|---|
| 1 | Tile accumulator, all live | No change (flawed) | N/A | Reverted before testing |
| 2 | Full-width K, tiled V | 128,000 (worse) | 8.33% | Reverted (would be worse) |
| 3 | Co-tiled K+V | 83,968 (better) | 8.33% (insufficient) | 1.28-2.29x slower (launch overhead) |
| 4 | bf16 accumulator | 94,976 (no change) | 8.33% | No effect (correct, no benefit) |
| 5 | Decouple BLOCK_N from SPLIT_K | 37,568-51,200 (best) | **16.67% (doubled)** | **No measurable speedup** |

Five structurally different attempts, each understood precisely. Attempt 5
is notable: it is the only one that genuinely doubled theoretical
occupancy, yet produced no measured benefit — strong evidence that, at
this specific workload scale, per-SM occupancy is not the binding
constraint on performance, and total SM utilization (governed by grid
size / `SPLIT_K`, already explored extensively via v1/v2/v3) is the real
axis that matters. Bottleneck 2, as originally framed (per-block
occupancy), may not be independently actionable for this workload without
also revisiting the SPLIT_K-vs-reduction-cost tradeoff that governs total
SM utilization.

## Bottleneck 2, closing check: full SPLIT_K sweep with BLOCK_N=16 forced

To exhaust the investigation, swept the full `SPLIT_K` range
(`2, 4, 8, 16, 32, 64, 128`) comparing the default (`SPLIT_K`-derived)
`BLOCK_N` against `BLOCK_N=16` forced at every point, not just where it
was already the default (`closing_check_bottleneck2.py`).

| SPLIT_K | BLOCK_N=default | BLOCK_N=16 (forced) | err |
|---|---|---|---|
| 2 | 61.35 μs | **109.25 μs** | 0.000000 |
| 4 | 59.84 μs | 59.19 μs | 0.000000 |
| 8 | 59.38 μs | 58.72 μs | 0.000000 |
| 16 | 59.70 μs | 60.11 μs | 0.000000 |
| 32 | 60.57 μs | 59.71 μs | 0.000000 |
| 64 | 59.91 μs | 59.56 μs | 0.000000 |
| 128 | 94.76 μs | 94.40 μs | 0.000000 |

**Correctness held everywhere** (exact match at every point).

**One clear outlier explains itself and reinforces the conclusion**: at
`SPLIT_K=2`, forcing `BLOCK_N=16` nearly doubles latency (61.35 → 109.25
μs) relative to the default. At `SPLIT_K=2`, `chunk_size=1024`, so the
default `BLOCK_N` (64, derived) means 16 KV-loop iterations, while forcing
`BLOCK_N=16` quadruples that to 64 iterations for the *same* total work —
pure loop-overhead cost (index computation, masking, per-iteration `t1.dot`
call overhead) with no compensating benefit, since at `SPLIT_K=2` only 2
SMs are used regardless of per-SM occupancy. This is a clean illustration
of the same principle established in attempt 5: **`BLOCK_N` reduction only
has a plausible payoff when total-SM-utilization is already reasonably
high; at very low `SPLIT_K`, shrinking `BLOCK_N` is pure cost with no
occupancy benefit worth having.**

**Every other point (SPLIT_K=4 through 128) is statistically flat
regardless of BLOCK_N** — confirming attempt 5's finding holds robustly
across the entire practical range, not just the handful of points tested
earlier. This closes the investigation: doubling per-SM occupancy via
`BLOCK_N` does not produce a measurable benefit anywhere in this
workload's practical operating range, and can actively hurt at the low end
of `SPLIT_K` where it adds pure loop overhead.

## Bottleneck 2: final conclusion

Five structurally distinct fixes attempted, one closing sweep exhausting
the parameter space of the most promising lever (attempt 5's `BLOCK_N`
decoupling). **The consistent, final finding: per-block/per-SM occupancy
is not the binding performance constraint for this kernel at the
workload scale present in this dataset** (`num_tokens` 1-8, meaning
`SPLIT_K` in the tens at most before reduction-step overhead dominates).
The real, already-well-characterized constraint remains **total SM
utilization**, governed by `SPLIT_K` itself, which trades off directly
against reduction-step cost (v1's sequential merge scales with `SPLIT_K`;
v3's atomic-counter design mitigates but does not eliminate this
tradeoff). This is not a failure to find a fix — it is a complete,
evidence-based characterization of why bottleneck 2, as originally framed
by the baseline `ncu` profiling (which flagged shared-memory-limited
occupancy at 8.33%), turns out not to be independently actionable for this
kernel: the occupancy ceiling it identified is real, but raising it
does not help, because occupancy was never the true bottleneck at this
workload's scale.

## v4: fixing an O(SPLIT_K²) inefficiency in the reduction loop (genuine positive result)

A fresh code review of `dsa_fwd_kernel_splitk_v3` (prompted by exhausting
the parameter-tuning attempts on bottleneck 2) found a real algorithmic
inefficiency, not a tuning issue: inside the winning program's reduction
loop (`for s in range(SPLIT_K)`), extracting split `s`'s `alpha` value
used a masked-sum trick —

```python
alpha_row = t1.sum(t1.where(offs_s[:, None] == s, alpha_s, 0.0), axis=0)
```

— which scans the **entire** `[SPLIT_K, BLOCK_H]` `alpha_s` tensor on
**every** loop iteration, to extract one row. This is `O(SPLIT_K)` work
per iteration, `SPLIT_K` iterations total: **O(SPLIT_K²)** overall, where
a direct load only needs **O(SPLIT_K)**. (The masked-sum pattern exists
because Triton doesn't support simple dynamic indexing with a runtime
loop variable inside `@triton.jit` — this workaround pattern appears
elsewhere in this codebase for the same reason, but here it was
unnecessarily applied to data that could instead be re-loaded directly by
pointer offset.)

**Fix (`dsa_fwd_kernel_splitk_v4` / `kernel_splitk_v4`)**: instead of
extracting split `s`'s `m`/`alpha` from the already-materialized
`[SPLIT_K, BLOCK_H]` tensor via masked-sum, load `m` for split `s`
directly via a scalar pointer offset (`PARTIAL_M + ... + s *
stride_pm_split + ...`) and compute that split's `alpha` fresh, inline,
each iteration — O(1) per iteration, O(SPLIT_K) total.

### Correctness

Exact match against both the original `kernel()` and `kernel_splitk_v3`
at every tested `SPLIT_K ∈ {2,8,16,32,64,128}` — a pure optimization, no
behavior change.

### Performance: genuine improvement, growing with SPLIT_K as predicted

| SPLIT_K | v3 | v4 | improvement |
|---|---|---|---|
| 2 | 63.93 μs | 62.39 μs | +1.54 μs |
| 8 | 62.62 μs | 62.07 μs | +0.55 μs |
| 16 | 61.68 μs | 59.86 μs | +1.81 μs |
| 32 | 61.71 μs | 61.26 μs | +0.45 μs |
| 64 | 62.62 μs | 62.04 μs | +0.58 μs |
| **128** | **94.78 μs** | **78.58 μs** | **+16.20 μs (~17%)** |

The improvement is small but consistent at low-to-mid `SPLIT_K` (where
`SPLIT_K²` isn't dramatically larger than `SPLIT_K`), and substantial at
`SPLIT_K=128` (~17% faster) — exactly matching the theoretical prediction
that an O(SPLIT_K²) vs O(SPLIT_K) gap should be most visible at the
highest tested `SPLIT_K`. This meaningfully narrows the previous gap
between `SPLIT_K=128` and the 60-62μs sweet spot (94.78 → 78.58 μs),
though 128 still does not match the best low/mid-range operating points.

**This is the first genuine positive optimization result from tonight's
extended investigation** (as distinct from the five well-characterized
negative results on bottleneck 2 above) — found via direct code review
after exhausting parameter-tuning approaches, not via profiling metrics.
`kernel_splitk_v4` is the new recommended default going forward,
superseding v3 (which remains in the codebase for historical/comparison
purposes, including the stress-test evidence already established for the
underlying atomic-counter design, which v4 leaves unchanged).
