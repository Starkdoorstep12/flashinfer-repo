# Bugs Found and Fixed

Log of every functional bug found in the starting solution, in the order
they were hit. Each blocked progress until fixed — this is roughly the
critical path from "won't even import" to "passes correctness."

## 1. Missing `@triton.jit` decorator (`solution/triton/kernel.py`)

`dsa_fwd_kernel` had no `@triton.jit` decorator, but `kernel()` called it as
`dsa_fwd_kernel[grid](...)` — that subscript syntax only works on a
`triton.jit`-wrapped `JITFunction`. Fails with
`TypeError: 'function' object is not subscriptable`.

**Fix:** add `@triton.jit` immediately above `def dsa_fwd_kernel(`.

## 2. Wrong Triton language alias (`solution/triton/kernel.py`)

The file imported `triton.language as tl`, but the entire kernel body used
`t1.program_id`, `t1.load`, `t1.dot`, etc. (`t1`, not `tl` — a leftover from
copying patterns out of `indexer_kernel.py`, which does use `t1`). Fails
with `NameError: name 't1' is not defined` the moment the kernel compiles.

**Fix:** `import triton.language as t1`.

## 3. Out-of-bounds pointer arithmetic (`solution/triton/indexer_kernel.py`)

```python
k_page_ptr = k_index_cache_fp8 + (
    page_index * page_size * head_dim_with_scale * kv_cache_num_heads
)
```

`k_index_cache_fp8`'s actual layout (per its own docstring) is
`[num_pages * page_size * head_dim_with_scale]` — flat, no per-head
dimension. The stray `* kv_cache_num_heads` factor pushed the pointer up to
~16x past the buffer's real extent for any page index beyond the first,
causing `RuntimeError: Triton Error [CUDA]: an illegal memory access was
encountered` (reported on the *next* CUDA call, since illegal-access errors
are sticky — the traceback pointed at an unrelated later kernel launch).

**Fix:** drop the extra factor:
```python
k_page_ptr = k_index_cache_fp8 + (page_index * page_size * head_dim_with_scale)
```

## 4. Head-count mismatch: golden reference vs. dataset ground truth

`indexer_kernel.py`'s embedded golden reference `run()` asserted
`num_qo_heads == 64`. `kernel.py`'s actual kernel asserted `== 16`. The
dataset's authoritative definition
(`~/mlsys26-contest/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json`)
confirms **16** is correct ("Number of query heads after tensor parallel
split (128/8=16)"), and the dataset's own embedded reference function also
asserts 16. `indexer_kernel.py`'s copy was simply stale/wrong.

**Fix:** `assert num_qo_heads == 16` in `indexer_kernel.py`.

## 5. `wrapper.py` calling a function with a kwarg it doesn't accept

`wrapper.py`'s local copy of `run_sparse_attention_pipeline` called
`run_indexer_and_topk(..., num_pages=num_pages, ...)`, but
`run_indexer_and_topk`'s real signature in `indexer_kernel.py` has no
`num_pages` parameter. `TypeError: unexpected keyword argument 'num_pages'`
on any run of `wrapper.py`'s own `__main__` smoke test.

**Fix:** deleted `wrapper.py` — it was a stale duplicate of the pipeline
logic that already existed correctly in `kernel.py`, and `config.toml`'s
entry point never pointed at it anyway.

## 6. Wrong entry point / wrong abstraction level entirely

The original `config.toml` pointed at
`kernel.py::run_sparse_attention_pipeline`, a function that runs *both* the
top-k indexer *and* the sparse attention kernel, taking ~22 parameters
(indexer config knobs + attention inputs). But the `sparse_attention` track
definition's inputs are `(q_nope, q_pe, ckv_cache, kpe_cache,
sparse_indices, sm_scale)` — 6 inputs. `sparse_indices` is an *input* to
this track, meaning top-k selection is assumed already done upstream (it's
a separate track: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`).

flashinfer-bench's builder validates the entry point's signature against the
destination-passing-style (DPS) parameter count derived from the
definition, and failed with:
BuildError: Destination-passing style callable: expected 8 parameters, but got 22
**Fix:**
- Pointed `entry_point` at `kernel.py::kernel` instead — it already has
  exactly the right signature and matches the golden reference.
- Set `destination_passing_style = false` in `config.toml`, since `kernel()`
  returns `(output, lse)` rather than writing into pre-allocated output
  tensors passed as arguments.
- Also corrected `definition = "sparse_attention"` (a track shorthand, not a
  real definition key) to the full dataset key
  `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`, which was silently
  causing flashinfer-bench to report "No matching solutions" rather than any
  clearer error.

After all six fixes, the solution passed correctness on all 23 workloads in
the `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` definition. See
`CORRECTNESS.md` for the tolerance methodology and result interpretation.
