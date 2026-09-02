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

## Finding 4 (critical): indexer_kernel is numerically INCORRECT against the golden reference

Built a proper correctness test (`correctness_test_indexer.py`) using
**real FP8-formatted data** matching the golden reference's exact
expectations — `torch.float8_e4m3fn` queries, and a KV cache packed in
the reference's documented per-page layout
(`[fp8_data: page_size×128 bytes][scale_data: page_size×4 bytes]`).
This is the first time this kernel has been tested against real-format
data and a real golden reference — all prior testing tonight (and in
earlier sessions) used loose `torch.randint(..., dtype=torch.int8)` random
data that never matched the actual FP8 semantics or memory layout.

**Result: severe mismatch.** Golden reference correctly selected 50/80
valid top-k tokens (matching real `seq_len`) with proper `-1` padding for
the remainder. Our kernel selected token index `0` for **all 2048**
output slots on both tested batch elements — a degenerate, clearly wrong
result (overlap with golden: 1/50 and 0/80 selected tokens respectively).

**Root cause (suspected, not yet fully confirmed)**: the dequantization
layout mismatch identified when first reading the golden reference (see
above) — `indexer_kernel` assumes each token's FP8 values are immediately
followed by that token's scale (`[fp8_128, scale_4]` interleaved per
token), while the golden reference's documented layout is
`[fp8_data for ALL tokens in page][scale_data for ALL tokens in page]`
(blocked, not interleaved) — plus a different numeric interpretation
(`k_tile.to(t1.float16)` naive cast vs. the reference's proper
`float8_e4m3fn` bit-level decode). Corrupted/degenerate K values from
this mismatch plausibly explain the "everything selects index 0" pattern:
`topk_kernel`'s replace-the-minimum selection algorithm can degenerate to
always picking the same (first-seen or tied) index when fed garbage or
constant scores.

**Status: NOT fixed.** This is a substantial, separate fix — correcting
`indexer_kernel`'s pointer arithmetic and dequantization logic to match
the golden reference's exact packed-block layout and proper FP8 decode —
deserving its own careful design and testing pass, not attempted in this
session. Flagged as the highest-priority next step for this track: **all
prior findings in this document (compile-time scaling, the batch_size
correctness fix) apply to a kernel that does not currently produce correct
output**, so none of the performance numbers gathered so far can be
trusted as representative of a working, correct system until this is
fixed.

## Updated summary

| Finding | Status |
|---|---|
| 1. batch_size power-of-2 bug | Fixed and verified in isolation |
| 2/3. Compile-time scaling (batch_size and MAX_SEQ_LEN) | Quantified |
| **4. Dequantization layout mismatch — kernel produces wrong output** | **Confirmed broken, NOT fixed — highest priority** |

**This reframes the whole indexer investigation**: findings 1-3 remain
real and valid observations about this kernel's behavior, but Finding 4
means the kernel is currently non-functional for its actual purpose. Any
future work on this track should fix Finding 4 first, then re-verify
findings 1-3 still hold (or re-measure them) against a kernel that
actually produces correct output.

## Finding 5 (fixed): topk_kernel had two separate bugs, both now fixed

After fixing Finding 4's dequantization (both K-side layout/decode and a
previously-unnoticed identical bug on the Q-side, which had no FP8 cast
at all), scores matched the golden reference to floating-point precision.
The remaining mismatch was isolated entirely to `topk_kernel`'s selection
logic, which had two independent bugs:

**Bug 5a: tie-breaking in the replace-the-minimum selection.**
`top_scores` initializes all `K` slots to the identical sentinel `-1e9`.
The original `is_min = top_scores == min_val` matches **every** tied slot
simultaneously, not just one — on the very first real token processed,
every one of the `K=2048` slots (all still at the initial sentinel) gets
overwritten with that single token's score in one step, corrupting the
entire top-k buffer immediately. Fixed by selecting only the
lowest-index slot among ties (`t1.min(t1.where(is_min, offs_k, MAX_K))`)
before constructing the replace mask, guaranteeing exactly one slot is
ever updated per token.

**Bug 5b: physical vs. logical index-space mismatch.** The golden
reference's output indices are **physical KV-cache page addresses**
(`page_idx * page_size + offset_in_page`, from `dequant_fp8_kv_cache`'s
paged layout). `indexer_kernel` internally indexes `acc_ptr` (and
therefore `topk_kernel`'s `top_indices`) by **logical, sequence-relative
position** (`seq_start + offset_token` — position within the concatenated
batch of sequences). These are different address spaces that only
coincidentally agree for a single-page sequence whose page happens to be
page 0. Fixed by converting each selected logical position back to a
physical page address inside `topk_kernel` before storing, using
`block_table` (passed in as a new parameter): recover the sequence-local
offset, derive `page_id`/`offset_in_page`, look up the physical page via
`block_table`, and reconstruct `physical_page * page_size + offset_in_page`.

**Verification**: `correctness_test_indexer.py`, using real
`float8_e4m3fn`-formatted data matching the golden reference's exact
input format — **both test batches now match exactly**: batch 0
(single-page, `seq_len=50`): 50/50 overlap; batch 1 (multi-page,
`seq_len=80`, spanning 2 pages): 80/80 overlap.

## Final summary: all four findings now resolved for correctness

| Finding | Status |
|---|---|
| 1. batch_size power-of-2 bug | **Fixed** |
| 2/3. Compile-time scaling (batch_size, MAX_SEQ_LEN) | Quantified (not a correctness issue; a deployment/engineering cost) |
| 4. Dequantization mismatch (K-side layout + Q-side missing cast) | **Fixed** |
| 5. topk_kernel: tie-breaking bug + physical/logical index mismatch | **Fixed** |

The indexer kernel (`run_indexer_and_topk`) is now verified numerically
correct against the golden reference for both single-page and
multi-page sequences. Compile-time scaling (Findings 2/3) remains an
open engineering concern, distinct from correctness, worth revisiting
for production deployment but not blocking correctness verification.

**Next**: run a broader stress test across more batch sizes and sequence
length combinations (including non-power-of-2 batch sizes, per Finding 1)
to confirm the fix generalizes, then re-time the corrected kernel's
compile and runtime cost (Findings 2/3's numbers were gathered before
these correctness fixes and should be re-confirmed still hold, since
correctness fixes could in principle change performance characteristics).
