# Top-K Indexer Kernel: Investigation

This is a separate track from the sparse-attention kernel work documented
in `SPLITK_OPTIMIZATION.md` — a different MLSys contest definition
(`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`, vs. the attention track's
`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`). Investigated after
completing the attention-kernel work, motivated by extending the same
methodology to the full two-stage DSA pipeline (top-k selection + sparse
attention) for the paper, rather than one stage in isolation.

Code: `solution/triton/indexer_kernel.py` (`indexer_kernel`, `topk_kernel`,
`run_indexer_and_topk`).

## Finding 1: real correctness bug — batch_size must be a power of 2

`indexer_kernel`'s batch-id resolution logic used
`t1.arange(0, batch_size)` directly, but Triton requires `arange` ranges
to be powers of 2. Checked the actual dataset
(`~/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl`):
batch sizes present are `[1, 2, 3, 4, 6, 7, 8, 11, 12, 14, 15, 16, 25, 26,
27, 29, 30, 31]` — **13 of 18 (72%) are not powers of 2**. This means the
kernel as originally written would fail on the large majority of the
track's real workloads, not an edge case.

**Fix**: pad `batch_size` to the next power of 2 for the `arange` call,
masking the padded lanes with a sentinel value (`2**30`) so they never
satisfy `pid >= tile_offsets_padded[i]` and cannot corrupt the real
`batch_id` computation.

```python
batch_size_padded: t1.constexpr = triton.next_power_of_2(batch_size)
offs_b = t1.arange(0, batch_size_padded)
b_mask = offs_b < batch_size
tile_offsets_padded = t1.load(tile_offsets_ptr + offs_b, mask=b_mask, other=2**30)
batch_id = t1.sum(t1.cast(pid >= tile_offsets_padded, t1.int32))
```

**Verified in isolation** (`test_isolated_batch_id.py`) against a small
hand-checked case (`tile_offsets=[5,10,15]`, `batch_size=3`) — produced the
exactly correct batch assignment for every `pid` tested.

## Finding 2: real-spec parameters expose a large, previously-untested compile cost

Earlier testing of this pipeline (weeks prior, during initial setup) used
a toy `num_index_heads=8`, not the real dataset spec of
**`num_index_heads=64`** (confirmed via the definition JSON:
`~/mlsys26-contest/definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json`).
Testing at the real spec value for the first time (this session) surfaced
something the toy test never could have shown.

**Symptom**: `run_indexer_and_topk` appeared to hang indefinitely on a
real dataset workload. Diagnosed with `timeout` + `ps aux` (checking the
actual child process's CPU state, not the wrapping `timeout` process) —
confirmed the process was genuinely computing (`69.2% CPU`, `STAT: Sl`),
not deadlocked.

**Measured**: with a cold Triton compile cache
(`rm -rf ~/.triton/cache`), the *first* call to `indexer_kernel` at the
real spec (`num_index_heads=64`, padded `batch_size`) took **62.03
seconds** — all compile time (`user` time ≈ `real` time, confirming
CPU-bound compilation, not I/O or GPU waiting). Once cached, subsequent
calls with the same shape took **0.150 ms** — a normal, fast runtime.

**Why this matters**: `batch_size` is a Triton `constexpr` (baked into the
compiled kernel), so **every distinct `batch_size` value triggers its own
independent ~60-second compile** the first time it's seen. The real
dataset has 18 distinct `batch_size` values. A cold-cache evaluation run
across the full dataset could plausibly incur on the order of **15-20
minutes of pure compilation overhead** before the fast, cached runtime
ever kicks in — a real practical risk against per-workload evaluation
timeouts (the attention track's `EVALUATION.md` specifies `--timeout 300`
per workload; if this track uses a similar budget, a 60+ second cold
compile could meaningfully eat into it, or exceed it if combined with
other overhead).

**Status**: documented as a discovered, real cost — not yet mitigated.
Candidate future directions (not yet attempted): making `batch_size` a
runtime value instead of a `constexpr` (trading compile-time specialization
for eliminating per-batch-size recompilation, at a potential runtime
performance cost that would need to be measured); or reducing the size of
whatever part of the compiled kernel scales expensively with
`num_index_heads=64`.

## Finding 3: `topk_kernel`'s serial top-K selection (real dataset scale estimated, not yet directly measured)

`topk_kernel` selects the top-`K` (2048) scored tokens per sequence using
a "replace-the-minimum" selection scan:

```python
for start in t1.static_range(0, MAX_SEQ_LEN, BLOCK):    # outer loop
    ...
    for i in range(BLOCK):                               # inner loop, fully serial
        ...
        min_val = t1.min(top_scores, axis=0)              # O(K) scan of the full top-K buffer
        ...                                                # replace-if-smaller, O(K) per step
```

This is O(K) work per token, applied one token at a time
(`MAX_SEQ_LEN` total tokens) — **O(MAX_SEQ_LEN × K)** overall, with no
parallelism across tokens within a sequence (`grid = (batch_size,)`, one
program per batch element).

**Real dataset scale** (from actual `seq_lens` loaded from the dataset,
e.g. one real batch: `[88, 91, 91, 96, 1721, 153, 114, 89, 157, 107, 91,
230, 5761, 97, 1]`): `MAX_SEQ_LEN` up to ~5761 in the observed data
(bounded by `max_num_pages × page_size`, with `max_num_pages` up to 91 in
the dataset). At `K=2048`, `BLOCK=16`: this is on the order of
**~12 million O(K)-scale vectorized operations** for the largest observed
sequence — real, but the scale of harm to actual latency was not
distinctly separated from the compile-cost effect (Finding 2) in this
session's testing, since both were entangled in the same hanging-then-slow
observations. Needs isolated timing (with a warm compile cache, comparing
short vs. long sequences within the same `batch_size`/compiled kernel) to
separate the two effects cleanly — not yet done.

**Status**: flagged as a credible, real-data-grounded performance concern
consistent with (though not yet cleanly isolated from) the compile-cost
finding above; a parallel top-K algorithm (e.g. bitonic top-K or
radix-select) is the natural candidate fix but has not been designed or
attempted.

## Summary and next steps

| Finding | Status |
|---|---|
| 1. batch_size power-of-2 correctness bug | **Fixed and verified** in isolation |
| 2. ~60s per-batch_size compile cost at real spec | **Discovered and measured**; not yet mitigated |
| 3. topk_kernel's O(seq_len × K) serial scan | **Scale estimated from real data**; not yet isolated from Finding 2 or fixed |

Next: (a) run the full `run_indexer_and_topk` pipeline end-to-end with the
correctness fix, across multiple real dataset workloads, verifying output
correctness (no golden reference comparison has been done yet for this
track — unlike the attention track, which was checked against
`flashinfer-bench`'s harness throughout); (b) isolate Finding 3's timing
cleanly from Finding 2's compile cost using a warm cache; (c) if Finding 3
proves to be a real, separable bottleneck, design and test a parallel
top-K replacement.

## Finding 2/3 update: compile time scales with MAX_SEQ_LEN, not just batch_size

Isolated the compile-time cost specifically as a function of `MAX_SEQ_LEN`
(the `constexpr` controlling `topk_kernel`'s `t1.static_range` unroll
depth), holding `batch_size=4` fixed across both tests so any difference
is attributable to sequence length alone, not batch size.

| Case | seq_len | Cold compile time | Warm runtime |
|---|---|---|---|
| SHORT | 128 | 60.87s | 0.284 ms |
| LONG | 5824 (matches real dataset max) | **167.20s** | not yet measured |

**Compile time scales sub-linearly but substantially with `MAX_SEQ_LEN`**:
a ~45x increase in sequence length (128 → 5824) produced only a ~2.75x
increase in compile time (61s → 167s) — not exponential blowup, but a
real, substantial absolute cost. Combined with Finding 2 (compile cost
also scales independently with `batch_size` due to `constexpr`
specialization), the two effects compound: a workload with both a new
`batch_size` *and* a new `MAX_SEQ_LEN` not previously seen could plausibly
approach or exceed **~3-4 minutes of cold-compile latency** before any
actual computation happens.

**Framing this finding correctly**: this is a compile-time cost, not a
runtime cost — importantly different from a kernel simply being "slow."
Once compiled and cached, this kernel's warm runtime is fast (0.284ms for
the SHORT case, consistent with typical GPU kernel latencies). In a real
serving system, this cost would normally be paid once per unique shape
via ahead-of-time warmup, not per-request — the same principle
`flashinfer-bench`'s own evaluation methodology applies (separate warmup
and timed phases). **The actual finding is not "this kernel is slow" but
"this kernel's compile-time specialization strategy (baking `batch_size`
and `MAX_SEQ_LEN` in as Triton `constexpr`s) does not scale gracefully to
the shape diversity seen in this track's realistic dataset** (18 distinct
batch sizes, sequence lengths spanning at least 1 to 5824+ tokens observed)
— a real engineering/deployment concern distinct from, and arguably more
interesting than, a simple runtime-speed bottleneck.

**Candidate fix directions** (not yet attempted): making `MAX_SEQ_LEN`
and/or `batch_size` runtime values rather than compile-time constants
(likely requires restructuring `t1.static_range` loops to regular runtime
loops, at a possible cost to the aggressive unroll-based optimization
Triton currently applies); or capping/bucketing shapes into a small number
of pre-compiled size classes (padding sequences up to the nearest bucket)
to bound the number of distinct compiles needed, a common technique in
production LLM serving systems for exactly this class of problem.
