# RTX 6000 Ada Optimization: Summary

This document is the consolidated narrative for the DeepSeek-style sparse
attention (DSA) decode kernel optimization work on the RTX 6000 Ada. It
presents the story in the order a reader needs, not the order it was
discovered. Full experiment-by-experiment detail, including false starts
and negative results, lives in `SPLITK_OPTIMIZATION.md`,
`BUGS_FIXED.md`, `CORRECTNESS.md`, `PROFILING.md`, and
`INFRASTRUCTURE_NOTES.md` — this document links out to those rather than
repeating them.

## Starting point

A Triton implementation of DSA sparse attention (top-k=2048, 16 query
heads, decode-phase inference) was non-functional: it had six distinct
bugs preventing it from running at all (missing JIT decorator, wrong
language module alias, out-of-bounds pointer arithmetic, a stale
assertion, dead code, and a fundamental entry-point mismatch with the
evaluation harness). See `BUGS_FIXED.md` for the full list.

## Correctness baseline

After fixing all six bugs, the kernel was verified against the benchmark's
golden reference across all 23 workloads in the dataset, using
flashinfer-bench's evaluation harness (atol=rtol=1e-2, 100% of output
elements within tolerance required). See `CORRECTNESS.md` for the full
methodology, including why some elements show large *relative* error
without failing (a bf16-rounding artifact at near-zero reference values,
not a bug).

## Bottleneck diagnosis

Profiling the correct kernel with Nsight Systems and Nsight Compute
revealed the real performance problem: for decode-phase workloads
(single-digit `num_tokens`), the kernel's grid — sized by `num_tokens` —
launches far too few blocks to use the GPU's 142 SMs. The original kernel
achieved 8.33% occupancy and 0.23% SM compute throughput, using 1 SM out
of 142 at `num_tokens=1`. See `PROFILING.md` for the full baseline.

## The optimization arc: three kernel versions

**v1 (split-K forward pass).** Partitioned the top-k=2048 KV dimension
across `SPLIT_K` blocks per token instead of one block handling the whole
range serially, with a second kernel merging the partial results. Fixed
the grid-size problem directly: forward-kernel duration dropped ~6x
(254.72 μs → 42.30 μs at SPLIT_K=8, measured via Nsight Compute).
Sweeping `SPLIT_K` further exposed a second issue: the merge step, written
as a sequential loop, scales linearly with `SPLIT_K` and erases the
forward kernel's gains at high split counts.

**v2 (fully parallel reduction — negative result).** Replaced the
sequential merge with a three-phase, fully parallel reduction (vectorized
max/sum reduction, then parallel atomic accumulation, then finalize).
This is provably better in raw GPU compute time — under half of v1's total
kernel time at high `SPLIT_K` — but **worse in end-to-end wall-clock
latency almost everywhere**, because it doubles the kernel-launch count
and adds new intermediate-tensor allocations. A follow-up controlled
experiment isolated these two costs directly (a trivial no-op kernel
launched N times to measure marginal launch cost; a separate measurement
of the extra allocation cost) and found they explain roughly 70% of the
observed gap. This negative result — and the reason for it — is the more
interesting finding of this phase: **in this workload regime (single-token
decode, tens-of-microseconds kernels), increasing GPU-level parallelism
improved raw kernel execution time while increasing end-to-end latency,
because CPU-side kernel-launch and allocation overhead dominate at this
scale.**

**v3 (single-launch, atomic-counter-gated reduction).** Designed directly
in response to v2's quantified failure: collapse the entire pipeline into
one kernel launch using an atomic-counter synchronization pattern from
production split-K GEMM kernels (each block writes its partial result and
atomically increments a per-token counter; the block that happens to
finish last performs the reduction inline, in the same launch). This
required careful attention to memory-ordering semantics (the counter is a
synchronization gate, not a pure accumulator, so it needs `acq_rel`
atomics, not `relaxed`) and, given the concurrency risk, was validated
with a 1,400-trial stress test (not just a single passing run) before being
trusted. **v3 beats both v1 and v2 at every measured `SPLIT_K`** —
typically 10-15% faster than v1 in the normal operating range, and by a
much wider margin at high `SPLIT_K` where v1 degrades.

See `SPLITK_OPTIMIZATION.md` for full design details, the Triton
synchronization research behind v3, and the complete v1/v2/v3 performance
tables.

## Generalization

All of the above was developed and measured on a single workload shape
(`num_tokens=1`, the most extreme case for grid-size underutilization).
The actual benchmark dataset's 23 workloads span `num_tokens` from 1 to 8.
Retesting v1 and v3 across this full range (fixed `SPLIT_K=16`) showed the
speedup is stable throughout — v1 at 1.41-1.44x, v3 at 1.58-1.62x, with no
fading, reversal, or workload-dependence — answering, for this dataset,
whether `SPLIT_K` needs to be workload-adaptive: it does not. This claim
is explicitly scoped to the dataset's actual range (`num_tokens` ≤ 8); it
is not claimed to hold at much larger token counts, which this dataset
does not contain.

## Bottleneck 2: investigated to completion, found not independently actionable

**Bottleneck 2 (per-block shared memory, capping occupancy at ~8.3% in
the original baseline) was investigated exhaustively across five
structurally distinct attempts, plus a closing sweep, and found NOT to be
the true performance constraint at this workload's scale.**

1. Tiling only the accumulator: no memory reduction (design flaw, caught before testing).
2. Tiling only the V-side (output) dimension: shared memory *increased* (94,976 -> 128,000 bytes) by duplicating K-cache residency.
3. Co-tiling both K-side and V-side: shared memory genuinely reduced (83,968 bytes) but insufficient to change the occupancy tier, and the multi-launch design cost more in launch/HBM overhead than it saved.
4. bf16 accumulator: no effect on shared memory at all (94,976 bytes unchanged) — ruled out the accumulator as the dominant cost.
5. Decoupling `BLOCK_N` from `SPLIT_K`: found `BLOCK_N` (not the accumulator or D-tiling) is the real dominant driver of shared memory, and reducing it genuinely doubles theoretical occupancy (8.33% -> 16.67%, shared memory down to 37,568-51,200 bytes) — **but produced no measurable wall-clock speedup anywhere in the practical `SPLIT_K` range**, and actively hurt at very low `SPLIT_K` (pure loop-overhead cost with no offsetting benefit).

**Conclusion**: at this workload's scale (`num_tokens` 1-8, `SPLIT_K` in
the tens before reduction-step overhead dominates), per-block/per-SM
occupancy is not the binding constraint — most of the GPU's 142 SMs sit
idle regardless of how many blocks run per *active* SM. The real
constraint is **total SM utilization**, governed by `SPLIT_K` itself,
which trades directly against reduction-step cost — a tradeoff already
deeply characterized via the v1/v2/v3 arc above. This is a complete,
evidence-based finding, not an abandoned investigation: the original
`ncu`-flagged occupancy ceiling was real, but raising it does not help,
because occupancy was never the true bottleneck for this kernel at this
scale.

## Infrastructure limitation, noted for completeness

Nsight Compute (`ncu`), needed for real hardware-counter-based occupancy
confirmation, has been unreliable in later sessions due to a conflict with
the cluster's DCGM monitoring service — reproduced on multiple nodes, with
the officially documented workaround (`dcgmi profile --pause`/`--resume`)
not resolving it. This does not affect any of the wall-clock performance
results above (all measured via Nsight Systems / `torch.cuda.Event`,
unaffected by the conflict), but it does mean some supplementary
occupancy/throughput confirmations (e.g., for v3 specifically) rely on
static analysis of Triton's compiled-kernel metadata rather than live
`ncu` measurement. See `INFRASTRUCTURE_NOTES.md` for full diagnosis.

## Summary table

| Bottleneck | Status |
|---|---|
| 1. Grid-size underutilization | **Fixed** (v1/v3, ~1.4-1.6x end-to-end speedup, generalizes across dataset) |
| 2. Per-block shared memory / occupancy | **Investigated to completion** — 5 attempts + closing sweep establish this is not the true bottleneck at this workload's scale |
| 3. Reduction step launch/allocation overhead | **Fixed** (v3, stress-tested for correctness) |
