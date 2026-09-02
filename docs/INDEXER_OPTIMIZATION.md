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
