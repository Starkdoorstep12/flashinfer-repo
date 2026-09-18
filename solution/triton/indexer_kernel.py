import torch
import triton
import triton.language as t1
@triton.jit


def indexer_kernel(
   q_index_fp8,           # [batch_size, num_index_heads, index_head_dim] fp16
   k_index_cache_fp8,     # [num_pages * page_size * head_dim_with_scale] int8
   weights,               # [batch_size, num_index_heads] float32
   seq_lens,              # [batch_size] int32
   block_table,           # [batch_size, max_num_pages] int32
   seq_offsets,           # [batch_size] int32, cumulative sum of seq_lens
   tile_offsets_ptr,      # cumulative tile offsets (pid ’ batch mapping)


   acc_ptr,
   batch_size: t1.constexpr, # Declared as constexpr
   num_index_heads: t1.constexpr,
   index_head_dim: t1.constexpr,
   page_size: t1.constexpr,
   kv_cache_num_heads: t1.constexpr,
   head_dim_with_scale: t1.constexpr,
   max_num_pages: t1.constexpr,


   BLOCK_TOKENS: t1.constexpr,
   BLOCK_HEADS: t1.constexpr
):


   # Each program processes a tile of tokens for a given sequence (batch element),
   # instead of splitting work across heads.


   # We divide the KV cache into pages and further into token tiles.
   # For each program:
   #   - it owns a BLOCK_TOKENS chunk of tokens for one sequence
   #   - it computes the final score for those tokens across ALL heads


   # Execution flow:
   # 1. Map program_id ’ (batch_id, token_tile_id) using tile_offsets
   # 2. For the given sequence:
   #    - fetch its KV cache pages using block_table
   # 3. Load a tile of K (for the current token tile)
   #    - dequantize FP8 values using per-token scales
   # 4. For each head:
   #    - load corresponding Q vector
   #    - compute dot(Q, K_tile) ’ scores per token
   #    - apply activation (ReLU) and head-specific weights
   #    - accumulate into token_scores


   # Key design decisions:
   # - Parallelism is across tokens (not heads)
   # - Each token is processed by exact1y one program ’ Earlier, each program was handling some BLOCK_HEADS and writing result per token in a register
   #   This writing was atomic and was causing serialisation of programs, breaking parallelism
   # - K tiles are loaded once per page and reused across all heads (improves memory efficiency)
   # - Q is reloaded per head (acceptable since Q is small compared to K)


   # Total programs launched:
   #   sum_over_batch ceil(seq_len / BLOCK_TOKENS)


   # This design:
   #   - eliminates contention from atomic adds (against earlier implementation)
   #   - improves effective memory bandwidth usage
   #   - maintains good reuse of K across heads within a program


   # Conceptual layout:
   #
   # For each sequence:
   #   tokens ’ split across programs
   #   heads  ’ processed inside each program (reduction dimension)
   #
   #        head_id
   #      0   1   2
   # token
   # 0    h0  h1  h2
   # 1    h3  h4  h5
   # 2    h6  h7  h8
   #
   # Each program handles a vertical slice (tokens),
   # and reduces across all heads locally.


   # -------------------------------------------------------
   # PROGRAM MAPPING
   # program = (batch_id, token_tile)
   # -------------------------------------------------------
   pid = t1.program_id(0)


   # -------------------------------------------------------
   # FIND batch_id USING tile_offsets
   # -------------------------------------------------------
   # -------------------------------------------------------
   # PROGRAM MAPPING: pid ’ (batch_id, token_tile)
   # -------------------------------------------------------


   # Determine batch_id without using 'break'
   # Find the first batch where pid is less than its tile_offset
   # This effectively means finding the index 'b' where tile_offsets_ptr[b-1] <= pid < tile_offsets_ptr[b]
   # batch_size may not be a power of 2 (confirmed with real dataset
   # workloads: e.g. batch_size=15 is common), but t1.arange requires a
   # power-of-2 range. Pad up to the next power of 2 and mask out the
   # padded entries so they don't corrupt the batch_id computation.
   batch_size_padded: t1.constexpr = triton.next_power_of_2(batch_size)
   offs_b = t1.arange(0, batch_size_padded)
   b_mask = offs_b < batch_size
   tile_offsets_padded = t1.load(tile_offsets_ptr + offs_b, mask=b_mask, other=2**30)
   batch_id = t1.sum(t1.cast(pid >= tile_offsets_padded, t1.int32))


   prev_offset = t1.where(
       batch_id > 0,
       t1.load(tile_offsets_ptr + batch_id - 1),
       0
   )


   token_tile_id = pid - prev_offset


   # -------------------------------------------------------
   # LOAD SEQUENCE METADATA
   # -------------------------------------------------------
   seq_len = t1.load(seq_lens + batch_id)
   seq_start = t1.load(seq_offsets + batch_id)


   # -------------------------------------------------------
   # PAGE-ALIGNED TOKEN TILE
   # -------------------------------------------------------
   token_start = token_tile_id * BLOCK_TOKENS


   # compute page id for this tile
   page_id = token_start // page_size
   offset_in_page = token_start % page_size


   # clamp so tile does not cross page
   tokens_left_in_page = page_size - offset_in_page
   effective_tokens = t1.minimum(BLOCK_TOKENS, tokens_left_in_page)


   offs_t = t1.arange(0, BLOCK_TOKENS)
   offset_token = token_start + offs_t


   token_mask = offset_token < (token_start + effective_tokens)
   token_mask &= offset_token < seq_len


   offs_d = t1.arange(0, index_head_dim)


   # -------------------------------------------------------
   # FETCH PAGE POINTER (ONLY ONE PAGE)
   # -------------------------------------------------------
   page_index = t1.load(
       block_table + batch_id * max_num_pages + page_id
   )


   k_page_ptr = k_index_cache_fp8 + (
       page_index * page_size * head_dim_with_scale
   )


   # -------------------------------------------------------
   # LOAD K TILE
   # -------------------------------------------------------
   # CORRECTED LAYOUT (per the golden reference's dequant_fp8_kv_cache):
   # each page is packed as [fp8_data: page_size*index_head_dim bytes]
   # followed by [scale_data: page_size*4 bytes] -- FP8 values for ALL
   # tokens come first (stride = index_head_dim per token, NOT
   # head_dim_with_scale), then all per-token float32 scales in a
   # separate trailing block. The original code assumed an interleaved
   # per-token [fp8_128, scale_4] layout, which does not match the real
   # deep_gemm packed format and produced numerically wrong K values
   # (confirmed via correctness_test_indexer.py against the golden
   # reference -- see docs/INDEXER_OPTIMIZATION.md, Finding 4).
   k_ptrs = (
       k_page_ptr
       + (offset_in_page + offs_t)[:, None] * index_head_dim
       + offs_d[None, :]
   )


   # Raw bytes are int8 (reinterpreted as uint8 per the reference); each
   # byte IS one float8_e4m3fn value bit-for-bit, so we bitcast rather
   # than numerically convert.
   k_tile_raw = t1.load(
       k_ptrs,
       mask=token_mask[:, None],
       other=0
   )
   k_tile_fp8 = t1.cast(k_tile_raw, t1.float8e4nv, bitcast=True)
   k_tile = k_tile_fp8.to(t1.float32)


   # Scale block starts after all FP8 data in the page; one float32
   # (4 bytes) per token, reinterpreted via bitcast from the 4 raw bytes
   # rather than numerically converted. Loaded as 4 separate 1D loads
   # (Triton does not support indexing a single column out of a 2D
   # tensor the way scale_bytes[:, i] would require).
   scale_block_start = page_size * index_head_dim
   scale_base_ptrs = k_page_ptr + scale_block_start + (offset_in_page + offs_t) * 4

   sb0 = t1.load(scale_base_ptrs + 0, mask=token_mask, other=0)
   sb1 = t1.load(scale_base_ptrs + 1, mask=token_mask, other=0)
   sb2 = t1.load(scale_base_ptrs + 2, mask=token_mask, other=0)
   sb3 = t1.load(scale_base_ptrs + 3, mask=token_mask, other=0)

   b0 = t1.cast(sb0, t1.uint8).to(t1.int32)
   b1 = t1.cast(sb1, t1.uint8).to(t1.int32)
   b2 = t1.cast(sb2, t1.uint8).to(t1.int32)
   b3 = t1.cast(sb3, t1.uint8).to(t1.int32)
   scale_int32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
   scale_vals = t1.cast(scale_int32, t1.float32, bitcast=True)


   # Dequantize FP8 values using per-token scales
   k_vals = k_tile * scale_vals[:, None]


   # -------------------------------------------------------
   # ACCUMULATE ACROSS HEADS (REDUCTION INSIDE PROGRAM)
   # -------------------------------------------------------
   token_scores = t1.zeros([BLOCK_TOKENS], t1.float32)


   for h_block in t1.static_range(num_index_heads // BLOCK_HEADS):


   # ---------------------------------------------
   # LOAD Q BLOCK [BLOCK_HEADS, head_dim]
   # ---------------------------------------------
       offs_h = t1.arange(0, BLOCK_HEADS)
       h_ids = h_block * BLOCK_HEADS + offs_h


       q_ptrs = (
           q_index_fp8
           + batch_id * num_index_heads * index_head_dim
           + offs_d[:, None]                        # dim is now primary
           + h_ids[None, :] * index_head_dim
       )


       q_block_raw = t1.load(
           q_ptrs,
           mask=(offs_d[:, None] < index_head_dim) & (h_ids[None, :] < num_index_heads),
           other=0
       )
       # q_index_fp8 is real float8_e4m3fn data (raw int8 bytes); must bitcast
       # rather than numerically convert, same fix as applied to the K-side
       # dequantization above. This was previously loaded with other=0.0 and
       # no cast at all, implicitly treating raw FP8 bit patterns as if they
       # were already meaningful float values -- confirmed via
       # debug_indexer_scores.py as a second, independent source of the score
       # mismatch found in Finding 4 (docs/INDEXER_OPTIMIZATION.md).
       q_block_fp8 = t1.cast(q_block_raw, t1.float8e4nv, bitcast=True)
       q_block = q_block_fp8.to(t1.float32)
       q_block = t1.trans(q_block) # this is just for correcting math
   # ---------------------------------------------
   # LOAD WEIGHTS [BLOCK_HEADS]
   # ---------------------------------------------
       w_ptrs = weights + batch_id * num_index_heads + h_ids


       w_block = t1.load(
           w_ptrs,
           mask=h_ids < num_index_heads,
           other=0.0
       )


   # ---------------------------------------------
   # COMPUTE: [tokens, dim]   [heads, dim]
   # ---------------------------------------------
   # k_vals: [BLOCK_TOKENS, dim]
   # q_block: [BLOCK_HEADS, dim]


   # broadcast multiply ’ [BLOCK_HEADS, BLOCK_TOKENS]
       # q_block is now [dim, heads]


       scores = t1.sum(
           q_block[None, :, :] * k_vals[:, None, :],
           axis=2
       )


   # activation
       scores = t1.maximum(scores, 0.0)


   # apply weights
       # scores = scores * w_block[:, None]
       scores = scores * w_block[None, :]
       # please check which one works better and is better


   # reduce across heads
       token_scores += t1.sum(scores, axis=1)


   # global token indices
   global_token_ids = seq_start + offset_token


   # store results
   t1.store(
       acc_ptr + global_token_ids,
       token_scores,
       mask=token_mask
   )

@triton.jit
def topk_kernel(
   acc_ptr,
   seq_offsets,
   seq_lens,
   block_table,
   topk_indices_ptr,
   K: t1.constexpr,
   MAX_K: t1.constexpr,
   BLOCK: t1.constexpr,
   MAX_SEQ_LEN: t1.constexpr,
   page_size: t1.constexpr,
   max_num_pages: t1.constexpr
):
    batch_id = t1.program_id(0)

    seq_start = t1.load(seq_offsets + batch_id)
    seq_len   = t1.load(seq_lens + batch_id)

    offs_block = t1.arange(0, BLOCK)
    offs_k     = t1.arange(0, K)

    # init
    top_scores  = t1.full((K,), -1e9, dtype=t1.float32)
    top_indices = t1.full((K,), -1,   dtype=t1.int32)

    for start in t1.static_range(0, MAX_SEQ_LEN, BLOCK):
        offs = start + offs_block
        mask = offs < seq_len

        vals = t1.load(acc_ptr + seq_start + offs, mask=mask, other=-1e9)
        ids  = (seq_start + offs).to(t1.int32)

        # process BLOCK elements one-by-one (mask extraction)
        for i in range(BLOCK):
            lane = offs_block == i

            v = t1.sum(t1.where(lane, vals, 0.0), axis=0)
            idx = t1.sum(t1.where(lane, ids, 0), axis=0)

            valid = t1.sum(t1.where(lane, mask.to(t1.int32), 0), axis=0) > 0

            # ---- core trick ----
            # find smallest in top-k
            min_val = t1.min(top_scores, axis=0)

            # check if new value should enter top-k
            cond = valid & (v > min_val)

            # BUG FIX: top_scores can have MULTIPLE slots tied at the
            # current minimum (e.g. all K slots start at -1e9, so on the
            # very first real token EVERY slot matches is_min and would
            # get overwritten simultaneously, corrupting the whole top-k
            # buffer in one step -- confirmed via correctness_test_indexer.py
            # against the golden reference, see docs/INDEXER_OPTIMIZATION.md
            # Finding 5). Break ties by only replacing the FIRST (lowest
            # array-index) slot that matches the minimum, using an argmin
            # via a one-hot mask on the position of the first True in
            # is_min, rather than replacing every tied slot at once.
            is_min = top_scores == min_val
            # position of first True in is_min (lowest index among ties)
            first_min_pos = t1.min(t1.where(is_min, offs_k, MAX_K), axis=0)
            replace_mask = cond & (offs_k == first_min_pos)

            top_scores = t1.where(replace_mask, v, top_scores)
            top_indices = t1.where(replace_mask, idx, top_indices)

    # BUG FIX (Finding 5, docs/INDEXER_OPTIMIZATION.md): top_indices holds
    # LOGICAL sequence-relative positions (from acc_ptr's indexing, i.e.
    # seq_start + offset_token), but the golden reference expects PHYSICAL
    # page addresses (page_idx * page_size + offset_in_page). These are
    # different index spaces that only coincidentally agree for
    # single-page sequences allocated at page 0. Convert here using
    # block_table before storing.
    is_valid_idx = top_indices != -1
    local_offset = t1.where(is_valid_idx, top_indices - seq_start, 0)
    page_id = local_offset // page_size
    offset_in_page = local_offset % page_size
    physical_page = t1.load(
        block_table + batch_id * max_num_pages + page_id,
        mask=is_valid_idx,
        other=0,
    )
    physical_addr = physical_page * page_size + offset_in_page
    top_indices_physical = t1.where(is_valid_idx, physical_addr, -1)

    # store result
    out_offs = batch_id * K + t1.arange(0, K)
    t1.store(topk_indices_ptr + out_offs, top_indices_physical)

def run_indexer_and_topk(
    q_index_fp8,
    k_index_cache_fp8,
    weights,
    seq_lens,
    block_table,
    seq_offsets,
    batch_size,
    num_index_heads,
    index_head_dim,
    page_size,
    kv_cache_num_heads,
    head_dim_with_scale,
    max_num_pages,
    topk,
    BLOCK_TOKENS=32,
    BLOCK_HEADS=8,
    device='cuda'
):

    # -------------------------------------------------------
    # STEP 0: compute tile_offsets (pid → batch mapping)
    # -------------------------------------------------------
    tiles_per_seq = (seq_lens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    tile_offsets = torch.cumsum(tiles_per_seq, dim=0)

    total_tiles = tile_offsets[-1].item()

    # -------------------------------------------------------
    # STEP 1: allocate per-token scores (NO atomics)
    # -------------------------------------------------------
    total_tokens = seq_offsets[-1] + seq_lens[-1]

    acc = torch.zeros(total_tokens, device=device, dtype=torch.float32)

    # -------------------------------------------------------
    # STEP 2: launch indexer kernel
    # -------------------------------------------------------
    grid = (total_tiles,)
    MAX_SEQ_LEN:t1.constexpr = int(seq_lens.max().item())

    indexer_kernel[grid](
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        seq_offsets=seq_offsets,
        tile_offsets_ptr=tile_offsets,
        acc_ptr=acc,

        batch_size=batch_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        page_size=page_size,
        kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages,

        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HEADS=BLOCK_HEADS
    )

    # -------------------------------------------------------
    # STEP 3: allocate topk output
    # -------------------------------------------------------
    topk_indices = torch.zeros(
        (batch_size, topk),
        device=device,
        dtype=torch.int32
    )

    # -------------------------------------------------------
    # STEP 4: launch topk kernel
    # -------------------------------------------------------
    topk_grid = (batch_size,)

    topk_kernel[topk_grid](
        acc_ptr=acc,
        seq_offsets=seq_offsets,
        seq_lens=seq_lens,
        block_table=block_table,
        topk_indices_ptr=topk_indices,
        K=topk,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        BLOCK=16,
        MAX_K=topk,   # static upper bound
        page_size=page_size,
        max_num_pages=max_num_pages,
    )

    return topk_indices

# the below mentioned function provides the golden function against which the kernel is to be compared
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        # For page_size=64, indices encode (page_idx * 64 + offset)
        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]
        qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        # Compute attention logits
        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        # Compute attention output
        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[t] = out.to(torch.bfloat16)

    return output, lse


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def run_indexer_and_topk_bucketed(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    seq_offsets: torch.Tensor,
    batch_size: int,
    num_index_heads: int,
    index_head_dim: int,
    page_size: int,
    kv_cache_num_heads: int,
    head_dim_with_scale: int,
    max_num_pages: int,
    topk: int,
    BLOCK_TOKENS: int = 32,
    BLOCK_HEADS: int = 8,
):
    """
    Shape-bucketed wrapper around run_indexer_and_topk. batch_size and
    max_num_pages are rounded up to the next power of 2 (padding inputs
    with masked-out dummy entries), so that many real workloads sharing
    the same bucket reuse a single compiled kernel instead of each
    triggering its own independent compile. See
    docs/INDEXER_OPTIMIZATION.md, "Bucketing" section: on the real
    dataset (128 workloads), this reduces distinct compiles needed from
    128 (one per workload) to 28 -- a 4.6x reduction.

    MAX_SEQ_LEN (topk_kernel's outer-loop bound) is derived from the
    BUCKETED max_num_pages, not the real seq_lens.max(), so the compiled
    shape depends only on the bucket, not the exact real sequence length.
    Positions beyond each sequence's real length are already masked out
    by the existing seq_lens-based validity checks in both kernels, so
    this only costs extra (masked, non-contributing) loop iterations, not
    incorrect results -- this is the padding-waste side of the tradeoff.
    """
    device = q_index_fp8.device

    bucketed_batch_size = _next_pow2(batch_size)
    bucketed_max_num_pages = _next_pow2(max_num_pages)

    # Pad batch dimension
    if bucketed_batch_size > batch_size:
        pad_b = bucketed_batch_size - batch_size
        seq_lens_p = torch.cat([seq_lens, torch.zeros(pad_b, dtype=seq_lens.dtype, device=device)])
        q_index_fp8_p = torch.cat([q_index_fp8, torch.zeros((pad_b,) + tuple(q_index_fp8.shape[1:]), dtype=q_index_fp8.dtype, device=device)])
        weights_p = torch.cat([weights, torch.zeros((pad_b,) + tuple(weights.shape[1:]), dtype=weights.dtype, device=device)])
        block_table_p = torch.cat([block_table, torch.zeros((pad_b, block_table.shape[1]), dtype=block_table.dtype, device=device)])
    else:
        seq_lens_p = seq_lens
        q_index_fp8_p = q_index_fp8
        weights_p = weights
        block_table_p = block_table

    # Pad page dimension
    if bucketed_max_num_pages > block_table_p.shape[1]:
        pad_p = bucketed_max_num_pages - block_table_p.shape[1]
        block_table_p = torch.cat(
            [block_table_p, torch.zeros((block_table_p.shape[0], pad_p), dtype=block_table_p.dtype, device=device)],
            dim=1,
        )

    seq_offsets_p = torch.cat([torch.tensor([0], device=device), seq_lens_p.cumsum(0)[:-1]]).to(torch.int32)

    padded_out = run_indexer_and_topk(
        q_index_fp8=q_index_fp8_p,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights_p,
        seq_lens=seq_lens_p,
        block_table=block_table_p,
        seq_offsets=seq_offsets_p,
        batch_size=bucketed_batch_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        page_size=page_size,
        kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=bucketed_max_num_pages,
        topk=topk,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HEADS=BLOCK_HEADS,
    )

    # Slice back down to the real batch_size, discarding padded rows.
    return padded_out[:batch_size]


# ═══════════════════════════════════════════════════════════════════════════
# Qrita-inspired top-k: ternary pivot search instead of replace-the-minimum
# scan, adapted from vLLM's production Triton implementation
# (vllm/v1/sample/ops/topk_topp_triton.py, based on Park et al.,
# "Qrita: High-performance Top-k and Top-p using Pivot-based Truncation
# and Selection", arXiv:2602.01518).
#
# Key adaptations from the original (which masks a dense [batch, vocab]
# logits tensor in-place):
#   - Outputs INDICES (topk_indices[batch, topk]), not masked values,
#     since the sparse attention kernel needs actual positions.
#   - Variable per-row seq_len (via seq_offsets), not a fixed VOCAB_SIZE,
#     since sequences have different lengths (paged KV cache).
#   - Fixed topk=2048 (compile-time constant), not a per-row runtime k.
#   - Grid-strided across batch (for row_id in range(pid, batch_size,
#     num_programs)) using min(num_sm, batch_size) programs, adapted
#     directly from the vLLM source -- this is the actual grid-
#     parallelism fix motivating this port (analogous to the
#     grid-underutilization fix, "bottleneck 1", in the sparse attention
#     kernel work).
#   - This first version omits Qrita's Gaussian sigma-truncation
#     optimization (searches the full row directly via ternary search
#     rather than first narrowing to a small outlier buffer) -- a
#     simplification to verify core correctness first; the truncation
#     optimization can be layered in afterward if this baseline is
#     correct and its performance profile justifies it.
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """Update running (min, count) of values above a pivot across tiles.
    Adapted from vLLM's topk_topp_triton.py (Qrita reference implementation).
    Tracks the smallest value strictly above a pivot and how many times
    it occurs, merged across tiles."""
    tile_min = t1.min(t1.where(above_mask, data, sentinel))
    tile_eq = above_mask & (t1.abs(data - tile_min) < 1e-9)
    tile_cnt = t1.sum(tile_eq.to(t1.int32))
    is_new = tile_min < min_larger
    is_same = t1.abs(tile_min - min_larger) < 1e-9
    num_min_larger = t1.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = t1.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def topk_kernel_qrita(
    ACC,             # [total_tokens] float32, flat concatenated scores
    SEQ_OFFSETS,     # [batch_size] int32, cumulative offset per sequence
    SEQ_LENS,        # [batch_size] int32
    BLOCK_TABLE,     # [batch_size, max_num_pages] int32
    TOPK_INDICES,    # [batch_size, K] int32 output
    BATCH_SIZE,      # runtime int, number of sequences
    K: t1.constexpr,
    BLOCK_SIZE: t1.constexpr,
    page_size: t1.constexpr,
    max_num_pages: t1.constexpr,
):
    pid = t1.program_id(0)
    num_programs = t1.num_programs(0)

    for row_id in range(pid, BATCH_SIZE, num_programs):
        seq_start = t1.load(SEQ_OFFSETS + row_id)
        seq_len = t1.load(SEQ_LENS + row_id)
        ACC_ROW = ACC + seq_start

        NUM_TILES = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Fast path: if seq_len <= K, every token is selected -- no
        # search needed. This is exactly the degenerate case TensorRT-LLM
        # identified as warranting a dedicated bypass (seq_len <= topk).
        if seq_len <= K:
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + t1.arange(0, BLOCK_SIZE)
                mask_n = offs_n < seq_len
                page_id = offs_n // page_size
                offset_in_page = offs_n % page_size
                physical_page = t1.load(
                    BLOCK_TABLE + row_id * max_num_pages + page_id,
                    mask=mask_n, other=0,
                )
                physical_addr = physical_page * page_size + offset_in_page
                out_ptrs = TOPK_INDICES + row_id * K + offs_n
                t1.store(out_ptrs, physical_addr.to(t1.int32), mask=mask_n)
            # Remaining K - seq_len slots are left as -1, via the
            # Python wrapper pre-filling TOPK_INDICES with -1 before launch.
        else:
            # General path: ternary search for the k-th largest value (pivot).
            max_val = -float("inf")
            min_val = float("inf")
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + t1.arange(0, BLOCK_SIZE)
                mask_n = offs_n < seq_len
                vals = t1.load(ACC_ROW + offs_n, mask=mask_n, other=-float("inf"))
                max_val = t1.maximum(max_val, t1.max(vals))
                vals_for_min = t1.load(ACC_ROW + offs_n, mask=mask_n, other=float("inf"))
                min_val = t1.minimum(min_val, t1.min(vals_for_min))

            min_range = min_val
            max_range = max_val
            pivot = min_range
            num_iters = 0
            found = 0
            min_larger = float("inf")
            num_min_larger = t1.zeros((), dtype=t1.int32)
            k_count = t1.zeros((), dtype=t1.int32)

            while found == 0:
                pivot_0 = (max_range - min_range) / 3.0 + min_range
                pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                cnt_0 = t1.zeros((), dtype=t1.int32)
                cnt_1 = t1.zeros((), dtype=t1.int32)
                ml_0 = float("inf")
                nml_0 = t1.zeros((), dtype=t1.int32)
                ml_1 = float("inf")
                nml_1 = t1.zeros((), dtype=t1.int32)

                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + t1.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < seq_len
                    vals = t1.load(ACC_ROW + offs_n, mask=mask_n, other=-float("inf"))
                    above_0 = (vals > pivot_0) & mask_n
                    above_1 = (vals > pivot_1) & mask_n
                    cnt_0 += t1.sum(above_0.to(t1.int32))
                    cnt_1 += t1.sum(above_1.to(t1.int32))
                    ml_0, nml_0 = _update_min_larger_stats(vals, above_0, ml_0, nml_0, float("inf"))
                    ml_1, nml_1 = _update_min_larger_stats(vals, above_1, ml_1, nml_1, float("inf"))

                if (cnt_0 >= K) and (cnt_0 - nml_0 < K):
                    pivot = pivot_0
                    k_count = cnt_0
                    min_larger = ml_0
                    num_min_larger = nml_0
                    found = 1
                if (cnt_1 >= K) and (cnt_1 - nml_1 < K):
                    pivot = pivot_1
                    k_count = cnt_1
                    min_larger = ml_1
                    num_min_larger = nml_1
                    found = 1

                if cnt_1 > K:
                    min_range = pivot_1
                elif cnt_0 > K:
                    min_range = pivot_0
                if cnt_0 < K:
                    max_range = pivot_0
                elif cnt_1 < K:
                    max_range = pivot_1

                num_iters += 1
                if (num_iters >= 30) or (t1.abs(max_range - min_range) < 1e-9):
                    pivot = (max_range + min_range) / 2.0
                    min_larger = ml_0
                    num_min_larger = nml_0
                    found = 1

            # BUG FIX (verified by hand against instrumented debug output,
            # not guessed): min_larger/num_min_larger/k_count from the
            # ternary loop are only trustworthy when they came from the
            # SAME accepted branch (cnt_0/cnt_1 satisfying the termination
            # condition). When the loop instead exits via the fallback
            # (num_iters>=30 or range-collapse), pivot is freshly computed
            # as a midpoint, but min_larger/num_min_larger/k_count are
            # stale leftovers from the previous iteration's REJECTED probe
            # -- inconsistent with the actual final pivot. Always recompute
            # fresh, relative to the actual final pivot, via one more scan
            # (using the same _update_min_larger_stats helper already used
            # in the ternary loop above) -- correct regardless of which
            # branch produced pivot, and a strict superset of the original
            # (already-correct) trim logic for the normal case.
            cnt_above_final = t1.zeros((), dtype=t1.int32)
            cnt_equal_final = t1.zeros((), dtype=t1.int32)
            min_larger_final = float("inf")
            num_min_larger_final = t1.zeros((), dtype=t1.int32)
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + t1.arange(0, BLOCK_SIZE)
                mask_n = offs_n < seq_len
                vals = t1.load(ACC_ROW + offs_n, mask=mask_n, other=-float("inf"))
                above_final = (vals > pivot) & mask_n
                cnt_above_final += t1.sum(above_final.to(t1.int32))
                cnt_equal_final += t1.sum(((t1.abs(vals - pivot) < 1e-9) & mask_n).to(t1.int32))
                min_larger_final, num_min_larger_final = _update_min_larger_stats(
                    vals, above_final, min_larger_final, num_min_larger_final, float("inf")
                )

            # Two exhaustive, mutually-exclusive cases based on where the
            # true K-th-largest value sits relative to pivot:
            #   cnt_above_final >= K: true boundary is a value STRICTLY
            #     ABOVE pivot (the tied-at-min_larger_final group) -- trim
            #     the excess from that group. [This is the original,
            #     already-correct logic for the normal ternary-search
            #     accept case -- unaffected by this fix.]
            #   cnt_above_final < K: true boundary IS pivot itself (or the
            #     search landed on a real, common data value) -- add ties
            #     AT pivot (cnt_equal_final group) to make up the shortfall.
            #     [This is the previously-missing case: Finding 6 testing
            #     never exercised heavy ties, so this path was never hit.]
            use_trim_path = cnt_above_final >= K
            num_keep = t1.where(use_trim_path, num_min_larger_final - (cnt_above_final - K), 0)
            num_keep_equal = t1.where(use_trim_path, 0, t1.maximum(t1.minimum(K - cnt_above_final, cnt_equal_final), 0))
            min_larger = min_larger_final
            # BUG FIX: num_min_larger must be zeroed in the add-case
            # (use_trim_path=False), otherwise the OLD trim block below
            # ("if num_keep < num_min_larger") spuriously fires even when
            # not in the trim case -- since num_keep=0 there and
            # num_min_larger_final is generically nonzero (something is
            # always the smallest value strictly above pivot), the trim
            # block's dup_keep = (dup_cumsum <= 0) silently REMOVES the
            # entire min_larger-tied group from the already-correct
            # "> pivot" selection, undercounting even cnt_above_final's
            # legitimately-selected values. Found via per-tile device_print
            # tracing showing keep_mask_sum totaling far below cnt_above_final.
            num_min_larger = t1.where(use_trim_path, num_min_larger_final, 0)
            num_kept = t1.zeros((), dtype=t1.int32)
            write_pos = t1.zeros((), dtype=t1.int32)

            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + t1.arange(0, BLOCK_SIZE)
                mask_n = offs_n < seq_len
                vals = t1.load(ACC_ROW + offs_n, mask=mask_n, other=-float("inf"))
                keep_mask = (vals > pivot) & mask_n

                if num_keep < num_min_larger:
                    dup_mask = (t1.abs(vals - min_larger) < 1e-9) & mask_n
                    dup_cumsum = t1.cumsum(dup_mask.to(t1.int32)) + num_kept
                    dup_keep = (dup_cumsum <= num_keep) & dup_mask
                    dup_remove = dup_mask & (~dup_keep)
                    num_kept += t1.sum(dup_keep.to(t1.int32))
                    keep_mask = keep_mask & (~dup_remove)

                if num_keep_equal > 0:
                    eq_mask = (t1.abs(vals - pivot) < 1e-9) & mask_n
                    eq_cumsum = t1.cumsum(eq_mask.to(t1.int32)) + num_kept
                    eq_keep = (eq_cumsum <= num_keep_equal) & eq_mask
                    num_kept += t1.sum(eq_keep.to(t1.int32))
                    keep_mask = keep_mask | eq_keep

                page_id = offs_n // page_size
                offset_in_page = offs_n % page_size
                physical_page = t1.load(
                    BLOCK_TABLE + row_id * max_num_pages + page_id,
                    mask=keep_mask, other=0,
                )
                physical_addr = physical_page * page_size + offset_in_page

                cumulative_pos = t1.cumsum(keep_mask.to(t1.int32)) - 1 + write_pos
                out_ptrs = TOPK_INDICES + row_id * K + cumulative_pos
                t1.store(out_ptrs, physical_addr.to(t1.int32), mask=keep_mask)
                write_pos += t1.sum(keep_mask.to(t1.int32))


def compute_topk_qrita(acc, seq_offsets, seq_lens, block_table, page_size, max_num_pages,
                        batch_size, topk=2048, BLOCK_SIZE=1024):
    """
    Qrita-inspired pivot-search top-k, as an alternative to the existing
    sequential replace-the-minimum topk_kernel. See module comment above
    topk_kernel_qrita for design notes and adaptations from the original.

    Like topk_kernel, output indices are PHYSICAL page addresses
    (page_idx * page_size + offset_in_page), not local sequence-relative
    offsets -- converted via block_table, matching topk_kernel's Finding 5
    fix so outputs from both kernels are directly comparable.
    """
    device = acc.device
    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)

    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    num_programs = min(num_sm, batch_size)

    topk_kernel_qrita[(num_programs,)](
        acc, seq_offsets, seq_lens, block_table, topk_indices, batch_size,
        K=topk, BLOCK_SIZE=BLOCK_SIZE, page_size=page_size, max_num_pages=max_num_pages,
    )
    return topk_indices
