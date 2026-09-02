import triton
import triton.language as t1
import torch 
import math

# ── Local kernel imports ──────────────────────────────────────────────────────
from .indexer_kernel import run_indexer_and_topk   # TopK indexer + selector

@triton.jit
def dsa_fwd_kernel(
    Q_NOPE, Q_PE, CKV_CACHE, KPE_CACHE, SPARSE_INDICES, OUTPUT, LSE,
    sm_scale,
    stride_qt_tok, stride_qt_h, stride_qt_d,
    stride_qpe_tok, stride_qpe_h, stride_qpe_d,
    stride_ckv_page, stride_ckv_tok, stride_ckv_d,
    stride_kpe_page, stride_kpe_tok, stride_kpe_d,
    stride_idx_tok, stride_idx_k,
    stride_out_tok, stride_out_h, stride_out_d,
    stride_lse_tok, stride_lse_h,
    page_size: t1.constexpr, topk: t1.constexpr,
    BLOCK_N: t1.constexpr,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_D_KPE: t1.constexpr,
    BLOCK_H: t1.constexpr
):
    #each program processes one token's full sparse attention computation
    #we launch num_tokens programs, each streaming over the topk selected K tokens
    #all BLOCK_H heads are processed in parallel within one program
    #NOTE: since all heads share the same sparse indices, we avoid redundant index loads by grouping heads together
    tok_id = t1.program_id(0)

    #offset vectors for indexing into head and dimension axes
    offs_h = t1.arange(0, BLOCK_H)
    #compressed KV dimension offsets, shape: BLOCK_D_CKV
    offs_d_ckv = t1.arange(0, BLOCK_D_CKV)
    #positional encoding dimension offsets, shape: BLOCK_D_KPE
    offs_d_kpe = t1.arange(0, BLOCK_D_KPE)

    #load Q_NOPE tile for this token across all heads
    #shape: [BLOCK_H, BLOCK_D_CKV] — the non-positional component of query
    q_nope_ptrs = Q_NOPE + tok_id * stride_qt_tok + offs_h[:, None] * stride_qt_h + offs_d_ckv[None, :] * stride_qt_d
    q_nope = t1.load(q_nope_ptrs)

    #load Q_PE tile for this token across all heads
    #shape: [BLOCK_H, BLOCK_D_KPE] — the positional encoding component of query
    q_pe_ptrs = Q_PE + tok_id * stride_qpe_tok + offs_h[:, None] * stride_qpe_h + offs_d_kpe[None, :] * stride_qpe_d
    q_pe = t1.load(q_pe_ptrs)

    #online softmax accumulators, maintained per head
    #m_i tracks the running maximum logit for numerical stability
    m_i = t1.full([BLOCK_H], -float("inf"), dtype=t1.float32)
    #l_i tracks the running sum of exponentiated scores (softmax denominator)
    l_i = t1.zeros([BLOCK_H], dtype=t1.float32)
    #acc accumulates the weighted value sum, shape: [BLOCK_H, BLOCK_D_CKV]
    acc = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.float32)

    #stream over sparse K tokens in tiles of BLOCK_N
    #each iteration processes BLOCK_N of the topk selected tokens
    for n_start in range(0, topk, BLOCK_N):
        #compute token offsets within the sparse index list for this tile
        offs_n = n_start + t1.arange(0, BLOCK_N)
        idx_ptrs = SPARSE_INDICES + tok_id * stride_idx_tok + offs_n * stride_idx_k
        
        #mask to handle the last tile where offs_n may exceed topk
        mask_n = offs_n < topk
        
        #load sparse indices — these are global token positions in the paged KV cache
        indices = t1.load(idx_ptrs, mask=mask_n, other=-1)
        #broadcasting masks for 2D operations: col-wise for K loads, row-wise for score masking
        valid_mask_col = indices[:, None] != -1
        valid_mask_row = indices[None, :] != -1
        
        #convert global token index to page_idx and token_offset within that page
        page_idx = indices // page_size
        tok_offset = indices % page_size
        
        #load K positional encoding tile from paged cache
        #shape: [BLOCK_N, BLOCK_D_KPE]
        k_kpe_ptrs = KPE_CACHE + page_idx[:, None] * stride_kpe_page + tok_offset[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = t1.load(k_kpe_ptrs, mask=valid_mask_col, other=0.0)

        #load compressed KV tile from paged cache (serves as both K_nope and V)
        #shape: [BLOCK_N, BLOCK_D_CKV]
        k_ckv_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d
        k_ckv = t1.load(k_ckv_ptrs, mask=valid_mask_col, other=0.0)

        #compute attention scores: split into non-positional and positional components
        #qk_nope = Q_nope @ K_ckv^T, shape: [BLOCK_H, BLOCK_N]
        qk_nope = t1.dot(q_nope, k_ckv.T)
        #qk_pe = Q_pe @ K_pe^T, shape: [BLOCK_H, BLOCK_N]
        qk_pe = t1.dot(q_pe, k_kpe.T)
        
        #combined attention logits scaled by sm_scale (1/sqrt(d))
        qk = (qk_nope + qk_pe) * sm_scale
        
        #mask out invalid positions (padding from last tile)
        qk = t1.where(valid_mask_row, qk, -float("inf"))

        #online softmax: update running max, rescale previous accumulator, and add new contributions
        #this avoids materialising the full attention matrix across all topk tokens
        m_ij = t1.maximum(m_i, t1.max(qk, axis=1))
        p = t1.exp(qk - m_ij[:, None])
        
        #rescaling factor for previously accumulated values when max changes
        alpha = t1.exp(m_i - m_ij)
        
        #update softmax denominator with rescaled old sum + new sum
        l_i = l_i * alpha + t1.sum(p, axis=1)
        m_i = m_ij
        
        #rescale previous accumulator and add new weighted values
        #NOTE: k_ckv is reused here as V since DeepSeek MLA shares compressed KV
        acc = acc * alpha[:, None]
        acc += t1.dot(p.to(t1.bfloat16), k_ckv)

    #finalise output: normalise accumulated values by softmax denominator
    #handle edge case where a head saw no valid tokens (m_i stays -inf)
    m_i_finite = m_i != -float("inf")
    m_i_finite_col = m_i[:, None] != -float("inf")
    
    #divide by l_i to complete the softmax-weighted average; zero out heads with no valid tokens
    acc_out = t1.where(m_i_finite_col, acc / l_i[:, None], 0.0)
    
    #store the attention output, shape: [BLOCK_H, BLOCK_D_CKV]
    out_ptrs = OUTPUT + tok_id * stride_out_tok + offs_h[:, None] * stride_out_h + offs_d_ckv[None, :] * stride_out_d
    t1.store(out_ptrs, acc_out.to(t1.bfloat16))
    
    #compute log-sum-exp in base-2 for numerical stability export
    #LSE = (m + log(l)) / log(2), used downstream for multi-split merging
    math_log2 = 0.6931471805599453
    lse = (m_i + t1.log(l_i)) / math_log2
    lse_out = t1.where(m_i_finite, lse, -float("inf"))
    
    #store per-head LSE values for this token
    lse_ptrs = LSE + tok_id * stride_lse_tok + offs_h * stride_lse_h
    t1.store(lse_ptrs, lse_out)

def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    #extract tensor dimensions for kernel configuration
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]
    
    #static assertions matching the DeepSeek MLA configuration
    #these allow Triton to use compile-time constants for optimal register allocation
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    
    device = q_nope.device
    
    #allocate output tensors: attention result and log-sum-exp per head
    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)
    
    #launch one program per token — each program handles all heads for that token
    grid = (num_tokens,)
    
    #BLOCK_N defines how many sparse K tokens are processed per iteration of the inner loop
    BLOCK_N = 64
    
    dsa_fwd_kernel[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        page_size=page_size, topk=topk,
        BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_D_KPE=head_dim_kpe,
        BLOCK_H=num_qo_heads,
    )
    
    return output, lse    

def run_sparse_attention_pipeline(
    # ── Indexer inputs ────────────────────────────────────────────────────────
    q_index_fp8: torch.Tensor,        # [batch, num_index_heads, index_head_dim] fp8
    k_index_cache_fp8: torch.Tensor,  # [num_pages * page_size * head_dim_with_scale] fp8
    weights: torch.Tensor,            # [batch, num_index_heads] float32  — per-head importance weights
    seq_lens: torch.Tensor,           # [batch] int32  — actual sequence length per sample
    block_table: torch.Tensor,        # [batch, max_num_pages] int32  — page-table for paged KV cache
    seq_offsets: torch.Tensor,        # [batch] int32  — cumulative sum of seq_lens (token start offsets)
    # ── Indexer config ────────────────────────────────────────────────────────
    num_index_heads: int,
    index_head_dim: int,
    num_pages: int,
    page_size: int,
    kv_cache_num_heads: int,
    head_dim_with_scale: int,         # index_head_dim + 1 scale element per token
    max_num_pages: int,
    topk: int,                        # number of top-K tokens to select per sequence
    # ── Sparse-attention inputs ───────────────────────────────────────────────
    q_nope: torch.Tensor,             # [num_tokens, num_qo_heads, head_dim_ckv] bfloat16
    q_pe: torch.Tensor,               # [num_tokens, num_qo_heads, head_dim_kpe] bfloat16
    ckv_cache: torch.Tensor,          # [num_pages, page_size, head_dim_ckv]  bfloat16
    kpe_cache: torch.Tensor,          # [num_pages, page_size, head_dim_kpe]  bfloat16
    sm_scale: float,                  # softmax scale, typically 1 / sqrt(head_dim)
    # ── Tuning knobs (indexer) ────────────────────────────────────────────────
    BLOCK_TOKENS: int = 32,
    BLOCK_HEADS: int = 8,
    device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full pipeline: TopK indexer → DSA sparse attention.

    Parameters
    ----------
    q_index_fp8 : Tensor [batch, num_index_heads, index_head_dim]
        FP8-quantised query used by the indexer to score KV tokens.
    k_index_cache_fp8 : Tensor [num_pages * page_size * head_dim_with_scale]
        Flat paged KV cache for the indexer (FP8 + per-token scale appended).
    weights : Tensor [batch, num_index_heads]
        Per-head importance weights applied after dot-product scoring.
    seq_lens : Tensor [batch]
        True sequence lengths (in tokens) for each batch entry.
    block_table : Tensor [batch, max_num_pages]
        Page table mapping (batch, page_id) → physical page index.
    seq_offsets : Tensor [batch]
        Cumulative token offsets; seq_offsets[i] = sum(seq_lens[:i]).
    num_index_heads : int
        Number of index heads used for scoring (e.g. 8).
    index_head_dim : int
        Dimension of each index head (e.g. 128).
    num_pages : int
        Total number of physical pages in the KV cache.
    page_size : int
        Tokens per physical page (e.g. 64).
    kv_cache_num_heads : int
        Number of heads in the main KV cache (for stride calculation).
    head_dim_with_scale : int
        index_head_dim + 1 (one trailing scalar scale per token per head).
    max_num_pages : int
        Maximum pages a single sequence can occupy (for block-table stride).
    topk : int
        How many tokens to select per sequence for sparse attention.
    q_nope : Tensor [num_tokens, num_qo_heads, head_dim_ckv]
        Non-positional query component for DSA forward.
    q_pe : Tensor [num_tokens, num_qo_heads, head_dim_kpe]
        Positional-encoding query component for DSA forward.
    ckv_cache : Tensor [num_pages, page_size, head_dim_ckv]
        Compressed KV cache (K_nope == V in DeepSeek MLA).
    kpe_cache : Tensor [num_pages, page_size, head_dim_kpe]
cale : float
        Softmax temperature scale, usually 1/sqrt(head_dim_ckv + head_dim_kpe).
    BLOCK_TOKENS : int
        Triton tile width along the token dimension (indexer + topk).
    BLOCK_HEADS : int
        Number of query heads processed in one Triton program (indexer).
    device : str
        Target CUDA device string (passed to the indexer allocator).

    Returns
    -------
    output : Tensor [num_tokens, num_qo_heads, head_dim_ckv]  bfloat16
        Sparse-attention output.
    lse : Tensor [num_tokens, num_qo_heads]  float32
        Log-sum-exp (base-2) per head, useful for multi-split merging.
    """

    batch_size = seq_lens.shape[0]

    # ── Step 1: TopK Indexer ──────────────────────────────────────────────────
    # Scores every KV token with a lightweight FP8 dot-product, then picks the
    # top-K global token indices per batch entry.
    #
    # Returns sparse_indices : [batch_size, topk]  int32
    sparse_indices = run_indexer_and_topk(
        q_index_fp8=q_index_fp8,
        k_index_cache_fp8=k_index_cache_fp8,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        seq_offsets=seq_offsets,
        batch_size=batch_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        page_size=page_size,
        kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages,
        topk=topk,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HEADS=BLOCK_HEADS,
        device=device,
    )
    # sparse_indices shape: [batch_size, topk]
    # Each row lists the global token positions (across the paged cache) that
    # scored highest for that sequence — these become the sparse K/V set.

    # ── Step 2: Broadcast sparse indices to match query token dimension ───────
    # `q_nope` is laid out as [num_tokens, …] where num_tokens == sum(seq_lens).
    # The DSA kernel expects sparse_indices shaped [num_tokens, topk], with each
    # token row containing its batch's selected indices.
    num_tokens = q_nope.shape[0]

    # Expand per-batch indices to per-token indices using seq_lens as a repeat count.
    #   seq_lens[i] tells how many query tokens belong to batch i.
    #   torch.repeat_interleave replicates row i of sparse_indices seq_lens[i] times.
    sparse_indices_per_token = torch.repeat_interleave(
        sparse_indices, seq_lens.to(torch.long), dim=0

    )  # [num_tokens, topk]

    assert sparse_indices_per_token.shape == (num_tokens, topk), (
        f"Shape mismatch after broadcast: expected ({num_tokens}, {topk}), "
        f"got {sparse_indices_per_token.shape}"
    )

    # ── Step 3: DSA Sparse Attention ──────────────────────────────────────────
    # Runs the full DeepSeek-Style Sparse Attention forward pass over the
    # top-K tokens identified by the indexer.
    output, lse = kernel(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        sparse_indices=sparse_indices_per_token,
        sm_scale=sm_scale,
    )
    # output : [num_tokens, num_qo_heads, head_dim_ckv]  bfloat16
    # lse    : [num_tokens, num_qo_heads]                float32

    return output, lse


# ═══════════════════════════════════════════════════════════════════════════
# Split-K variant: parallelizes the topk (KV) dimension across multiple
# thread blocks per token, instead of one block handling all topk=2048
# entries serially. Fixes SM underutilization for decode-style workloads
# (num_tokens=1) where the original per-token grid launches too few blocks
# to fill the GPU. See docs/PROFILING.md for the baseline diagnosis.
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def dsa_fwd_kernel_splitk(
    Q_NOPE, Q_PE, CKV_CACHE, KPE_CACHE, SPARSE_INDICES,
    PARTIAL_ACC, PARTIAL_M, PARTIAL_L,
    sm_scale,
    stride_qt_tok, stride_qt_h, stride_qt_d,
    stride_qpe_tok, stride_qpe_h, stride_qpe_d,
    stride_ckv_page, stride_ckv_tok, stride_ckv_d,
    stride_kpe_page, stride_kpe_tok, stride_kpe_d,
    stride_idx_tok, stride_idx_k,
    stride_pacc_tok, stride_pacc_split, stride_pacc_h, stride_pacc_d,
    stride_pm_tok, stride_pm_split, stride_pm_h,
    stride_pl_tok, stride_pl_split, stride_pl_h,
    page_size: t1.constexpr, topk: t1.constexpr,
    SPLIT_K: t1.constexpr,
    BLOCK_N: t1.constexpr,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_D_KPE: t1.constexpr,
    BLOCK_H: t1.constexpr,
):
    # Each program now handles ONE (token, split) pair, instead of one
    # program owning the entire topk loop for a token. This multiplies the
    # grid size by SPLIT_K, giving the GPU SPLIT_K times more independent
    # blocks to schedule across SMs.
    tok_id = t1.program_id(0)
    split_id = t1.program_id(1)

    # chunk = how many of the topk sparse indices this program is responsible for
    chunk = topk // SPLIT_K
    chunk_start = split_id * chunk

    offs_h = t1.arange(0, BLOCK_H)
    offs_d_ckv = t1.arange(0, BLOCK_D_CKV)
    offs_d_kpe = t1.arange(0, BLOCK_D_KPE)

    q_nope_ptrs = Q_NOPE + tok_id * stride_qt_tok + offs_h[:, None] * stride_qt_h + offs_d_ckv[None, :] * stride_qt_d
    q_nope = t1.load(q_nope_ptrs)

    q_pe_ptrs = Q_PE + tok_id * stride_qpe_tok + offs_h[:, None] * stride_qpe_h + offs_d_kpe[None, :] * stride_qpe_d
    q_pe = t1.load(q_pe_ptrs)

    # Same online-softmax accumulators as the original kernel, but scoped
    # only to this program's chunk of the topk KV tokens — this is a
    # PARTIAL result, not the final answer.
    m_i = t1.full([BLOCK_H], -float("inf"), dtype=t1.float32)
    l_i = t1.zeros([BLOCK_H], dtype=t1.float32)
    acc = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.float32)

    for n_start in range(0, chunk, BLOCK_N):
        offs_n = chunk_start + n_start + t1.arange(0, BLOCK_N)
        idx_ptrs = SPARSE_INDICES + tok_id * stride_idx_tok + offs_n * stride_idx_k

        mask_n = offs_n < (chunk_start + chunk)
        mask_n &= offs_n < topk

        indices = t1.load(idx_ptrs, mask=mask_n, other=-1)
        valid_mask_col = indices[:, None] != -1
        valid_mask_row = indices[None, :] != -1

        page_idx = indices // page_size
        tok_offset = indices % page_size

        k_kpe_ptrs = KPE_CACHE + page_idx[:, None] * stride_kpe_page + tok_offset[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = t1.load(k_kpe_ptrs, mask=valid_mask_col, other=0.0)

        k_ckv_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d
        k_ckv = t1.load(k_ckv_ptrs, mask=valid_mask_col, other=0.0)

        qk_nope = t1.dot(q_nope, k_ckv.T)
        qk_pe = t1.dot(q_pe, k_kpe.T)
        qk = (qk_nope + qk_pe) * sm_scale
        qk = t1.where(valid_mask_row, qk, -float("inf"))
        m_ij = t1.maximum(m_i, t1.max(qk, axis=1))
        m_ij_finite_col = m_ij[:, None] != -float("inf")
        p = t1.where(m_ij_finite_col, t1.exp(qk - m_ij[:, None]), 0.0)
        alpha = t1.where(m_i == -float("inf"), 0.0, t1.exp(m_i - m_ij))
        l_i = l_i * alpha + t1.sum(p, axis=1)
        m_i = m_ij
        acc = acc * alpha[:, None]
        acc += t1.dot(p.to(t1.bfloat16), k_ckv)

    # Write PARTIAL state (not normalized, not final) to scratch buffers.
    # The reduce kernel combines these across SPLIT_K per token.
    pacc_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + split_id * stride_pacc_split
                 + offs_h[:, None] * stride_pacc_h + offs_d_ckv[None, :] * stride_pacc_d)
    t1.store(pacc_ptrs, acc)

    pm_ptrs = PARTIAL_M + tok_id * stride_pm_tok + split_id * stride_pm_split + offs_h * stride_pm_h
    t1.store(pm_ptrs, m_i)

    pl_ptrs = PARTIAL_L + tok_id * stride_pl_tok + split_id * stride_pl_split + offs_h * stride_pl_h
    t1.store(pl_ptrs, l_i)


@triton.jit
def dsa_reduce_kernel(
    PARTIAL_ACC, PARTIAL_M, PARTIAL_L, OUTPUT, LSE,
    stride_pacc_tok, stride_pacc_split, stride_pacc_h, stride_pacc_d,
    stride_pm_tok, stride_pm_split, stride_pm_h,
    stride_pl_tok, stride_pl_split, stride_pl_h,
    stride_out_tok, stride_out_h, stride_out_d,
    stride_lse_tok, stride_lse_h,
    SPLIT_K: t1.constexpr,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_H: t1.constexpr,
):
    # One program per token. Cheap — just combines SPLIT_K partial results
    # using the same rescale-by-alpha trick already used inside the main
    # loop, just applied across blocks instead of across KV tiles.
    tok_id = t1.program_id(0)

    offs_h = t1.arange(0, BLOCK_H)
    offs_d_ckv = t1.arange(0, BLOCK_D_CKV)

    m_i = t1.full([BLOCK_H], -float("inf"), dtype=t1.float32)
    l_i = t1.zeros([BLOCK_H], dtype=t1.float32)
    acc = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.float32)

    for split_id in t1.static_range(SPLIT_K):
        pm_ptrs = PARTIAL_M + tok_id * stride_pm_tok + split_id * stride_pm_split + offs_h * stride_pm_h
        m_split = t1.load(pm_ptrs)

        pl_ptrs = PARTIAL_L + tok_id * stride_pl_tok + split_id * stride_pl_split + offs_h * stride_pl_h
        l_split = t1.load(pl_ptrs)

        pacc_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + split_id * stride_pacc_split
                     + offs_h[:, None] * stride_pacc_h + offs_d_ckv[None, :] * stride_pacc_d)
        acc_split = t1.load(pacc_ptrs)

        m_new = t1.maximum(m_i, m_split)
        # Guard against -inf - (-inf) = NaN when neither this split nor the
        # accumulated state so far has seen any valid token yet.
        alpha_old = t1.where(m_i == -float("inf"), 0.0, t1.exp(m_i - m_new))
        alpha_split = t1.where(m_split == -float("inf"), 0.0, t1.exp(m_split - m_new))

        l_i = l_i * alpha_old + l_split * alpha_split
        acc = acc * alpha_old[:, None] + acc_split * alpha_split[:, None]
        m_i = m_new

    m_i_finite = m_i != -float("inf")
    m_i_finite_col = m_i[:, None] != -float("inf")
    acc_out = t1.where(m_i_finite_col, acc / l_i[:, None], 0.0)

    out_ptrs = OUTPUT + tok_id * stride_out_tok + offs_h[:, None] * stride_out_h + offs_d_ckv[None, :] * stride_out_d
    t1.store(out_ptrs, acc_out.to(t1.bfloat16))

    math_log2 = 0.6931471805599453
    lse = (m_i + t1.log(l_i)) / math_log2
    lse_out = t1.where(m_i_finite, lse, -float("inf"))
    lse_ptrs = LSE + tok_id * stride_lse_tok + offs_h * stride_lse_h
    t1.store(lse_ptrs, lse_out)


def kernel_splitk(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=16):
    """
    Drop-in alternative to kernel() that parallelizes the topk KV dimension
    across SPLIT_K blocks per token, instead of one block per token handling
    the full topk=2048 loop serially. Same signature/output as kernel().
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert topk % SPLIT_K == 0, f"topk={topk} must be divisible by SPLIT_K={SPLIT_K}"

    device = q_nope.device

    # Chunk size per split. At high SPLIT_K, chunk gets small; if BLOCK_N
    # equals chunk exactly, the inner loop runs exactly once, which has been
    # observed to blow past Triton's shared-memory pipelining budget on this
    # kernel (OutOfResources at SPLIT_K=32 with a fixed BLOCK_N=64). Keeping
    # BLOCK_N <= chunk // 2 (loop runs >=2 iterations) avoids this.

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)

    # Scratch buffers for partial results, one slot per (token, split, head)
    partial_acc = torch.zeros((num_tokens, SPLIT_K, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    partial_m = torch.full((num_tokens, SPLIT_K, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    partial_l = torch.zeros((num_tokens, SPLIT_K, num_qo_heads), dtype=torch.float32, device=device)

    chunk_size = topk // SPLIT_K
    BLOCK_N = min(64, max(16, chunk_size // 2))

    grid_splitk = (num_tokens, SPLIT_K)
    dsa_fwd_kernel_splitk[grid_splitk](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        partial_acc, partial_m, partial_l,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        page_size=page_size, topk=topk,
        SPLIT_K=SPLIT_K,
        BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_D_KPE=head_dim_kpe,
        BLOCK_H=num_qo_heads,
    )

    grid_reduce = (num_tokens,)
    dsa_reduce_kernel[grid_reduce](
        partial_acc, partial_m, partial_l, output, lse,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        SPLIT_K=SPLIT_K,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_H=num_qo_heads,
    )

    return output, lse


# ═══════════════════════════════════════════════════════════════════════════
# Split-K v2: fixes bottleneck 3 (dsa_reduce_kernel scaling linearly with
# SPLIT_K due to a sequential t1.static_range loop). Splits the reduction
# into three phases:
#   Phase A: compute final (m, l) per token — cheap, vectorized hardware
#            reduction over the small [SPLIT_K, BLOCK_H] tile.
#   Phase B: once m_final/l_final are known, each split's rescale factor
#            is independent — accumulate the (large) weighted acc tensor
#            in PARALLEL across splits via atomic adds, instead of a
#            sequential per-split loop in one program.
#   Phase C: finalize (divide by l_final) — cheap, one program per token.
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def dsa_reduce_phaseA_kernel(
    PARTIAL_M, PARTIAL_L, M_FINAL, L_FINAL,
    stride_pm_tok, stride_pm_split, stride_pm_h,
    stride_pl_tok, stride_pl_split, stride_pl_h,
    stride_mf_tok, stride_mf_h,
    stride_lf_tok, stride_lf_h,
    SPLIT_K: t1.constexpr,
    BLOCK_H: t1.constexpr,
):
    # One program per token. Loads the whole [SPLIT_K, BLOCK_H] tile of
    # partial (m, l) at once and reduces with hardware max/sum instead of
    # a sequential Python-level loop — SPLIT_K only affects tile size here,
    # not instruction count, since t1.max/t1.sum are real parallel reductions.
    tok_id = t1.program_id(0)

    offs_s = t1.arange(0, SPLIT_K)
    offs_h = t1.arange(0, BLOCK_H)

    pm_ptrs = PARTIAL_M + tok_id * stride_pm_tok + offs_s[:, None] * stride_pm_split + offs_h[None, :] * stride_pm_h
    m_split = t1.load(pm_ptrs)  # [SPLIT_K, BLOCK_H]

    pl_ptrs = PARTIAL_L + tok_id * stride_pl_tok + offs_s[:, None] * stride_pl_split + offs_h[None, :] * stride_pl_h
    l_split = t1.load(pl_ptrs)  # [SPLIT_K, BLOCK_H]

    m_final = t1.max(m_split, axis=0)  # [BLOCK_H]

    m_final_row = m_final[None, :]
    finite_mask = m_final_row != -float("inf")
    alpha = t1.where(finite_mask, t1.exp(m_split - m_final_row), 0.0)  # [SPLIT_K, BLOCK_H]

    l_final = t1.sum(alpha * l_split, axis=0)  # [BLOCK_H]

    mf_ptrs = M_FINAL + tok_id * stride_mf_tok + offs_h * stride_mf_h
    t1.store(mf_ptrs, m_final)

    lf_ptrs = L_FINAL + tok_id * stride_lf_tok + offs_h * stride_lf_h
    t1.store(lf_ptrs, l_final)


@triton.jit
def dsa_reduce_phaseB_kernel(
    PARTIAL_ACC, PARTIAL_M, M_FINAL, ACC_FINAL,
    stride_pacc_tok, stride_pacc_split, stride_pacc_h, stride_pacc_d,
    stride_pm_tok, stride_pm_split, stride_pm_h,
    stride_mf_tok, stride_mf_h,
    stride_af_tok, stride_af_h, stride_af_d,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_H: t1.constexpr,
):
    # One program per (token, split) — same grid shape as the forward
    # kernel, genuinely parallel across splits. Each program independently
    # rescales its own partial acc by alpha_split (now computable directly
    # since m_final is already known from Phase A) and atomically adds it
    # into a shared per-token accumulator.
    tok_id = t1.program_id(0)
    split_id = t1.program_id(1)

    offs_h = t1.arange(0, BLOCK_H)
    offs_d = t1.arange(0, BLOCK_D_CKV)

    pm_ptrs = PARTIAL_M + tok_id * stride_pm_tok + split_id * stride_pm_split + offs_h * stride_pm_h
    m_split = t1.load(pm_ptrs)  # [BLOCK_H]

    mf_ptrs = M_FINAL + tok_id * stride_mf_tok + offs_h * stride_mf_h
    m_final = t1.load(mf_ptrs)  # [BLOCK_H]

    finite_mask = m_final != -float("inf")
    alpha = t1.where(finite_mask, t1.exp(m_split - m_final), 0.0)  # [BLOCK_H]

    pacc_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + split_id * stride_pacc_split
                 + offs_h[:, None] * stride_pacc_h + offs_d[None, :] * stride_pacc_d)
    acc_split = t1.load(pacc_ptrs)  # [BLOCK_H, BLOCK_D_CKV]

    contribution = alpha[:, None] * acc_split

    af_ptrs = (ACC_FINAL + tok_id * stride_af_tok
               + offs_h[:, None] * stride_af_h + offs_d[None, :] * stride_af_d)
    t1.atomic_add(af_ptrs, contribution)


@triton.jit
def dsa_reduce_phaseC_kernel(
    ACC_FINAL, M_FINAL, L_FINAL, OUTPUT, LSE,
    stride_af_tok, stride_af_h, stride_af_d,
    stride_mf_tok, stride_mf_h,
    stride_lf_tok, stride_lf_h,
    stride_out_tok, stride_out_h, stride_out_d,
    stride_lse_tok, stride_lse_h,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_H: t1.constexpr,
):
    # One program per token. Cheap final divide + LSE computation, same as
    # the tail end of the original dsa_fwd_kernel / dsa_reduce_kernel.
    tok_id = t1.program_id(0)

    offs_h = t1.arange(0, BLOCK_H)
    offs_d = t1.arange(0, BLOCK_D_CKV)

    mf_ptrs = M_FINAL + tok_id * stride_mf_tok + offs_h * stride_mf_h
    m_final = t1.load(mf_ptrs)

    lf_ptrs = L_FINAL + tok_id * stride_lf_tok + offs_h * stride_lf_h
    l_final = t1.load(lf_ptrs)

    af_ptrs = (ACC_FINAL + tok_id * stride_af_tok
               + offs_h[:, None] * stride_af_h + offs_d[None, :] * stride_af_d)
    acc_final = t1.load(af_ptrs)

    m_finite = m_final != -float("inf")
    m_finite_col = m_final[:, None] != -float("inf")

    acc_out = t1.where(m_finite_col, acc_final / l_final[:, None], 0.0)

    out_ptrs = OUTPUT + tok_id * stride_out_tok + offs_h[:, None] * stride_out_h + offs_d[None, :] * stride_out_d
    t1.store(out_ptrs, acc_out.to(t1.bfloat16))

    math_log2 = 0.6931471805599453
    lse = (m_final + t1.log(l_final)) / math_log2
    lse_out = t1.where(m_finite, lse, -float("inf"))
    lse_ptrs = LSE + tok_id * stride_lse_tok + offs_h * stride_lse_h
    t1.store(lse_ptrs, lse_out)


def kernel_splitk_v2(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=16):
    """
    Same as kernel_splitk() but with a 3-phase parallel reduction instead of
    the sequential t1.static_range(SPLIT_K) merge, fixing the linear-in-
    SPLIT_K scaling of the reduce step (see docs/SPLITK_OPTIMIZATION.md,
    bottleneck 3).
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert topk % SPLIT_K == 0, f"topk={topk} must be divisible by SPLIT_K={SPLIT_K}"

    device = q_nope.device

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)

    partial_acc = torch.zeros((num_tokens, SPLIT_K, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    partial_m = torch.full((num_tokens, SPLIT_K, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    partial_l = torch.zeros((num_tokens, SPLIT_K, num_qo_heads), dtype=torch.float32, device=device)

    chunk_size = topk // SPLIT_K
    BLOCK_N = min(64, max(16, chunk_size // 2))

    # ── Forward pass: same kernel as kernel_splitk() ──────────────────────
    grid_splitk = (num_tokens, SPLIT_K)
    dsa_fwd_kernel_splitk[grid_splitk](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        partial_acc, partial_m, partial_l,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        page_size=page_size, topk=topk,
        SPLIT_K=SPLIT_K,
        BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_D_KPE=head_dim_kpe,
        BLOCK_H=num_qo_heads,
    )

    # ── Phase A: compute final (m, l) per token ────────────────────────────
    m_final = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    l_final = torch.zeros((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

    dsa_reduce_phaseA_kernel[(num_tokens,)](
        partial_m, partial_l, m_final, l_final,
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        m_final.stride(0), m_final.stride(1),
        l_final.stride(0), l_final.stride(1),
        SPLIT_K=SPLIT_K,
        BLOCK_H=num_qo_heads,
    )

    # ── Phase B: parallel weighted accumulation via atomics ───────────────
    acc_final = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

    dsa_reduce_phaseB_kernel[(num_tokens, SPLIT_K)](
        partial_acc, partial_m, m_final, acc_final,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        m_final.stride(0), m_final.stride(1),
        acc_final.stride(0), acc_final.stride(1), acc_final.stride(2),
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_H=num_qo_heads,
    )

    # ── Phase C: finalize ───────────────────────────────────────────────────
    dsa_reduce_phaseC_kernel[(num_tokens,)](
        acc_final, m_final, l_final, output, lse,
        acc_final.stride(0), acc_final.stride(1), acc_final.stride(2),
        m_final.stride(0), m_final.stride(1),
        l_final.stride(0), l_final.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_H=num_qo_heads,
    )

    return output, lse


# ═══════════════════════════════════════════════════════════════════════════
# Split-K v3: single-launch design using an atomic counter as a
# synchronization gate ("last CTA does the reduction"), informed by
# Experiment 2's finding that kernel-launch dispatch (~7.3us/launch) and
# intermediate-buffer allocation (~16.7us) — not reduction algorithm
# complexity — dominate the wall-clock cost in the practical SPLIT_K range.
#
# Each program computes its partial result exactly as dsa_fwd_kernel_splitk
# does, then atomically increments a per-token counter. The one program
# whose atomic_add returns SPLIT_K-1 (i.e. it was the last to finish) then
# reads all SPLIT_K partials and performs the reduction inline — no second
# kernel launch, no separate intermediate-buffer allocations beyond the
# scratch already required for partial results.
#
# CORRECTNESS-CRITICAL: the counter atomic MUST use acq_rel (or an
# explicit release+acquire pair), NOT relaxed. This is not a pure
# accumulator (where relaxed would be safe) — it is a synchronization gate.
# release ensures this program's writes to the partial-result scratch
# buffers are visible to other programs before the counter update is
# visible; acquire ensures the "last" program does not see stale data left
# over from before other programs' releases. Do not "optimize" this to
# relaxed — see docs/SPLITK_OPTIMIZATION.md, Experiment 3 for the reasoning
# and the Triton/CUDA split-K precedent this follows.
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def dsa_fwd_kernel_splitk_v3(
    Q_NOPE, Q_PE, CKV_CACHE, KPE_CACHE, SPARSE_INDICES,
    PARTIAL_ACC, PARTIAL_M, PARTIAL_L, COUNTER, OUTPUT, LSE,
    sm_scale,
    stride_qt_tok, stride_qt_h, stride_qt_d,
    stride_qpe_tok, stride_qpe_h, stride_qpe_d,
    stride_ckv_page, stride_ckv_tok, stride_ckv_d,
    stride_kpe_page, stride_kpe_tok, stride_kpe_d,
    stride_idx_tok, stride_idx_k,
    stride_pacc_tok, stride_pacc_split, stride_pacc_h, stride_pacc_d,
    stride_pm_tok, stride_pm_split, stride_pm_h,
    stride_pl_tok, stride_pl_split, stride_pl_h,
    stride_out_tok, stride_out_h, stride_out_d,
    stride_lse_tok, stride_lse_h,
    page_size: t1.constexpr, topk: t1.constexpr,
    SPLIT_K: t1.constexpr,
    BLOCK_N: t1.constexpr,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_D_KPE: t1.constexpr,
    BLOCK_H: t1.constexpr,
):
    tok_id = t1.program_id(0)
    split_id = t1.program_id(1)

    offs_h = t1.arange(0, BLOCK_H)
    offs_d_ckv = t1.arange(0, BLOCK_D_CKV)
    offs_d_kpe = t1.arange(0, BLOCK_D_KPE)

    # ── Forward pass: identical to dsa_fwd_kernel_splitk ──────────────────
    q_nope_ptrs = Q_NOPE + tok_id * stride_qt_tok + offs_h[:, None] * stride_qt_h + offs_d_ckv[None, :] * stride_qt_d
    q_nope = t1.load(q_nope_ptrs)

    q_pe_ptrs = Q_PE + tok_id * stride_qpe_tok + offs_h[:, None] * stride_qpe_h + offs_d_kpe[None, :] * stride_qpe_d
    q_pe = t1.load(q_pe_ptrs)

    m_i = t1.full([BLOCK_H], -float("inf"), dtype=t1.float32)
    l_i = t1.zeros([BLOCK_H], dtype=t1.float32)
    acc = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.float32)

    chunk = topk // SPLIT_K
    chunk_start = split_id * chunk

    for n_start in range(0, chunk, BLOCK_N):
        offs_n = chunk_start + n_start + t1.arange(0, BLOCK_N)
        idx_ptrs = SPARSE_INDICES + tok_id * stride_idx_tok + offs_n * stride_idx_k

        mask_n = offs_n < (chunk_start + chunk)
        mask_n &= offs_n < topk

        indices = t1.load(idx_ptrs, mask=mask_n, other=-1)
        valid_mask_col = indices[:, None] != -1
        valid_mask_row = indices[None, :] != -1

        page_idx = indices // page_size
        tok_offset = indices % page_size

        k_kpe_ptrs = KPE_CACHE + page_idx[:, None] * stride_kpe_page + tok_offset[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = t1.load(k_kpe_ptrs, mask=valid_mask_col, other=0.0)

        k_ckv_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d
        k_ckv = t1.load(k_ckv_ptrs, mask=valid_mask_col, other=0.0)

        qk_nope = t1.dot(q_nope, k_ckv.T)
        qk_pe = t1.dot(q_pe, k_kpe.T)
        qk = (qk_nope + qk_pe) * sm_scale
        qk = t1.where(valid_mask_row, qk, -float("inf"))

        m_ij = t1.maximum(m_i, t1.max(qk, axis=1))
        m_ij_finite_col = m_ij[:, None] != -float("inf")
        p = t1.where(m_ij_finite_col, t1.exp(qk - m_ij[:, None]), 0.0)
        alpha = t1.where(m_i == -float("inf"), 0.0, t1.exp(m_i - m_ij))
        l_i = l_i * alpha + t1.sum(p, axis=1)
        m_i = m_ij
        acc = acc * alpha[:, None]
        acc += t1.dot(p.to(t1.bfloat16), k_ckv)

    # ── Write partial result to scratch (visible to all programs via HBM) ─
    pacc_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + split_id * stride_pacc_split
                 + offs_h[:, None] * stride_pacc_h + offs_d_ckv[None, :] * stride_pacc_d)
    t1.store(pacc_ptrs, acc)

    pm_ptrs = PARTIAL_M + tok_id * stride_pm_tok + split_id * stride_pm_split + offs_h * stride_pm_h
    t1.store(pm_ptrs, m_i)

    pl_ptrs = PARTIAL_L + tok_id * stride_pl_tok + split_id * stride_pl_split + offs_h * stride_pl_h
    t1.store(pl_ptrs, l_i)

    # ── Synchronization gate: atomic counter with acq_rel semantics ───────
    # release: ensures the writes above are visible before this increment
    #          becomes visible to other programs.
    # acquire (on the "winning" program's subsequent reads): ensures it
    #          sees all other programs' writes, not stale data.
    # DO NOT change sem to "relaxed" — see module docstring above.
    count_ptr = COUNTER + tok_id
    old_count = t1.atomic_add(count_ptr, 1, sem="acq_rel")
    is_last = (old_count + 1) == SPLIT_K

    if is_last:
        # ── Inline reduction: this program only, no extra kernel launch ──
        offs_s = t1.arange(0, SPLIT_K)

        pm_all_ptrs = PARTIAL_M + tok_id * stride_pm_tok + offs_s[:, None] * stride_pm_split + offs_h[None, :] * stride_pm_h
        m_split = t1.load(pm_all_ptrs)  # [SPLIT_K, BLOCK_H]

        pl_all_ptrs = PARTIAL_L + tok_id * stride_pl_tok + offs_s[:, None] * stride_pl_split + offs_h[None, :] * stride_pl_h
        l_split = t1.load(pl_all_ptrs)  # [SPLIT_K, BLOCK_H]

        m_final = t1.max(m_split, axis=0)  # [BLOCK_H], vectorized hardware reduction
        m_final_row = m_final[None, :]
        finite_mask = m_final_row != -float("inf")
        alpha_s = t1.where(finite_mask, t1.exp(m_split - m_final_row), 0.0)  # [SPLIT_K, BLOCK_H]
        l_final = t1.sum(alpha_s * l_split, axis=0)  # [BLOCK_H]

        # acc accumulation: sequential over SPLIT_K, but only in this ONE
        # winning program, with no separate kernel launch or extra
        # allocation — the tradeoff Experiment 2's decomposition motivates.
        acc_final = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.float32)
        for s in range(SPLIT_K):
            pacc_s_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + s * stride_pacc_split
                           + offs_h[:, None] * stride_pacc_h + offs_d_ckv[None, :] * stride_pacc_d)
            acc_s = t1.load(pacc_s_ptrs)

            alpha_row = t1.sum(t1.where(offs_s[:, None] == s, alpha_s, 0.0), axis=0)  # [BLOCK_H]
            acc_final += alpha_row[:, None] * acc_s

        m_finite = m_final != -float("inf")
        m_finite_col = m_final[:, None] != -float("inf")
        acc_out = t1.where(m_finite_col, acc_final / l_final[:, None], 0.0)

        out_ptrs = OUTPUT + tok_id * stride_out_tok + offs_h[:, None] * stride_out_h + offs_d_ckv[None, :] * stride_out_d
        t1.store(out_ptrs, acc_out.to(t1.bfloat16))

        math_log2 = 0.6931471805599453
        lse = (m_final + t1.log(l_final)) / math_log2
        lse_out = t1.where(m_finite, lse, -float("inf"))
        lse_ptrs = LSE + tok_id * stride_lse_tok + offs_h * stride_lse_h
        t1.store(lse_ptrs, lse_out)



@triton.jit
def dsa_fwd_kernel_splitk_v3_bf16acc(
    Q_NOPE, Q_PE, CKV_CACHE, KPE_CACHE, SPARSE_INDICES,
    PARTIAL_ACC, PARTIAL_M, PARTIAL_L, COUNTER, OUTPUT, LSE,
    sm_scale,
    stride_qt_tok, stride_qt_h, stride_qt_d,
    stride_qpe_tok, stride_qpe_h, stride_qpe_d,
    stride_ckv_page, stride_ckv_tok, stride_ckv_d,
    stride_kpe_page, stride_kpe_tok, stride_kpe_d,
    stride_idx_tok, stride_idx_k,
    stride_pacc_tok, stride_pacc_split, stride_pacc_h, stride_pacc_d,
    stride_pm_tok, stride_pm_split, stride_pm_h,
    stride_pl_tok, stride_pl_split, stride_pl_h,
    stride_out_tok, stride_out_h, stride_out_d,
    stride_lse_tok, stride_lse_h,
    page_size: t1.constexpr, topk: t1.constexpr,
    SPLIT_K: t1.constexpr,
    BLOCK_N: t1.constexpr,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_D_KPE: t1.constexpr,
    BLOCK_H: t1.constexpr,
):
    tok_id = t1.program_id(0)
    split_id = t1.program_id(1)

    offs_h = t1.arange(0, BLOCK_H)
    offs_d_ckv = t1.arange(0, BLOCK_D_CKV)
    offs_d_kpe = t1.arange(0, BLOCK_D_KPE)

    # ── Forward pass: identical to dsa_fwd_kernel_splitk ──────────────────
    q_nope_ptrs = Q_NOPE + tok_id * stride_qt_tok + offs_h[:, None] * stride_qt_h + offs_d_ckv[None, :] * stride_qt_d
    q_nope = t1.load(q_nope_ptrs)

    q_pe_ptrs = Q_PE + tok_id * stride_qpe_tok + offs_h[:, None] * stride_qpe_h + offs_d_kpe[None, :] * stride_qpe_d
    q_pe = t1.load(q_pe_ptrs)

    m_i = t1.full([BLOCK_H], -float("inf"), dtype=t1.float32)
    l_i = t1.zeros([BLOCK_H], dtype=t1.float32)
    acc = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.bfloat16)

    chunk = topk // SPLIT_K
    chunk_start = split_id * chunk

    for n_start in range(0, chunk, BLOCK_N):
        offs_n = chunk_start + n_start + t1.arange(0, BLOCK_N)
        idx_ptrs = SPARSE_INDICES + tok_id * stride_idx_tok + offs_n * stride_idx_k

        mask_n = offs_n < (chunk_start + chunk)
        mask_n &= offs_n < topk

        indices = t1.load(idx_ptrs, mask=mask_n, other=-1)
        valid_mask_col = indices[:, None] != -1
        valid_mask_row = indices[None, :] != -1

        page_idx = indices // page_size
        tok_offset = indices % page_size

        k_kpe_ptrs = KPE_CACHE + page_idx[:, None] * stride_kpe_page + tok_offset[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = t1.load(k_kpe_ptrs, mask=valid_mask_col, other=0.0)

        k_ckv_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + offs_d_ckv[None, :] * stride_ckv_d
        k_ckv = t1.load(k_ckv_ptrs, mask=valid_mask_col, other=0.0)

        qk_nope = t1.dot(q_nope, k_ckv.T)
        qk_pe = t1.dot(q_pe, k_kpe.T)
        qk = (qk_nope + qk_pe) * sm_scale
        qk = t1.where(valid_mask_row, qk, -float("inf"))

        m_ij = t1.maximum(m_i, t1.max(qk, axis=1))
        m_ij_finite_col = m_ij[:, None] != -float("inf")
        p = t1.where(m_ij_finite_col, t1.exp(qk - m_ij[:, None]), 0.0)
        alpha = t1.where(m_i == -float("inf"), 0.0, t1.exp(m_i - m_ij))
        l_i = l_i * alpha + t1.sum(p, axis=1)
        m_i = m_ij
        acc = (acc.to(t1.float32) * alpha[:, None]).to(t1.bfloat16)
        acc = (acc.to(t1.float32) + t1.dot(p.to(t1.bfloat16), k_ckv)).to(t1.bfloat16)

    # ── Write partial result to scratch (visible to all programs via HBM) ─
    pacc_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + split_id * stride_pacc_split
                 + offs_h[:, None] * stride_pacc_h + offs_d_ckv[None, :] * stride_pacc_d)
    t1.store(pacc_ptrs, acc.to(t1.float32))

    pm_ptrs = PARTIAL_M + tok_id * stride_pm_tok + split_id * stride_pm_split + offs_h * stride_pm_h
    t1.store(pm_ptrs, m_i)

    pl_ptrs = PARTIAL_L + tok_id * stride_pl_tok + split_id * stride_pl_split + offs_h * stride_pl_h
    t1.store(pl_ptrs, l_i)

    # ── Synchronization gate: atomic counter with acq_rel semantics ───────
    # release: ensures the writes above are visible before this increment
    #          becomes visible to other programs.
    # acquire (on the "winning" program's subsequent reads): ensures it
    #          sees all other programs' writes, not stale data.
    # DO NOT change sem to "relaxed" — see module docstring above.
    count_ptr = COUNTER + tok_id
    old_count = t1.atomic_add(count_ptr, 1, sem="acq_rel")
    is_last = (old_count + 1) == SPLIT_K

    if is_last:
        # ── Inline reduction: this program only, no extra kernel launch ──
        offs_s = t1.arange(0, SPLIT_K)

        pm_all_ptrs = PARTIAL_M + tok_id * stride_pm_tok + offs_s[:, None] * stride_pm_split + offs_h[None, :] * stride_pm_h
        m_split = t1.load(pm_all_ptrs)  # [SPLIT_K, BLOCK_H]

        pl_all_ptrs = PARTIAL_L + tok_id * stride_pl_tok + offs_s[:, None] * stride_pl_split + offs_h[None, :] * stride_pl_h
        l_split = t1.load(pl_all_ptrs)  # [SPLIT_K, BLOCK_H]

        m_final = t1.max(m_split, axis=0)  # [BLOCK_H], vectorized hardware reduction
        m_final_row = m_final[None, :]
        finite_mask = m_final_row != -float("inf")
        alpha_s = t1.where(finite_mask, t1.exp(m_split - m_final_row), 0.0)  # [SPLIT_K, BLOCK_H]
        l_final = t1.sum(alpha_s * l_split, axis=0)  # [BLOCK_H]

        # acc accumulation: sequential over SPLIT_K, but only in this ONE
        # winning program, with no separate kernel launch or extra
        # allocation — the tradeoff Experiment 2's decomposition motivates.
        acc_final = t1.zeros([BLOCK_H, BLOCK_D_CKV], dtype=t1.float32)
        for s in range(SPLIT_K):
            pacc_s_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + s * stride_pacc_split
                           + offs_h[:, None] * stride_pacc_h + offs_d_ckv[None, :] * stride_pacc_d)
            acc_s = t1.load(pacc_s_ptrs)

            alpha_row = t1.sum(t1.where(offs_s[:, None] == s, alpha_s, 0.0), axis=0)  # [BLOCK_H]
            acc_final += alpha_row[:, None] * acc_s

        m_finite = m_final != -float("inf")
        m_finite_col = m_final[:, None] != -float("inf")
        acc_out = t1.where(m_finite_col, acc_final / l_final[:, None], 0.0)

        out_ptrs = OUTPUT + tok_id * stride_out_tok + offs_h[:, None] * stride_out_h + offs_d_ckv[None, :] * stride_out_d
        t1.store(out_ptrs, acc_out.to(t1.bfloat16))

        math_log2 = 0.6931471805599453
        lse = (m_final + t1.log(l_final)) / math_log2
        lse_out = t1.where(m_finite, lse, -float("inf"))
        lse_ptrs = LSE + tok_id * stride_lse_tok + offs_h * stride_lse_h
        t1.store(lse_ptrs, lse_out)


def kernel_splitk_v3(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=16):
    """
    Single-launch split-K variant using an atomic-counter synchronization
    gate ("last CTA does the reduction"). See module docstring above and
    docs/SPLITK_OPTIMIZATION.md, Experiment 3.
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert topk % SPLIT_K == 0, f"topk={topk} must be divisible by SPLIT_K={SPLIT_K}"

    device = q_nope.device

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)

    partial_acc = torch.zeros((num_tokens, SPLIT_K, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    partial_m = torch.full((num_tokens, SPLIT_K, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    partial_l = torch.zeros((num_tokens, SPLIT_K, num_qo_heads), dtype=torch.float32, device=device)
    counter = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

    chunk_size = topk // SPLIT_K
    BLOCK_N = min(64, max(16, chunk_size // 2))

    grid = (num_tokens, SPLIT_K)
    dsa_fwd_kernel_splitk_v3[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        partial_acc, partial_m, partial_l, counter, output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        page_size=page_size, topk=topk,
        SPLIT_K=SPLIT_K,
        BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_D_KPE=head_dim_kpe,
        BLOCK_H=num_qo_heads,
    )

    return output, lse


@triton.jit
def dsa_fwd_kernel_splitk_dtiled_v3(
    Q_NOPE, Q_PE, CKV_CACHE, KPE_CACHE, SPARSE_INDICES,
    PARTIAL_ACC, PARTIAL_M, PARTIAL_L,
    sm_scale,
    stride_qt_tok, stride_qt_h, stride_qt_d,
    stride_qpe_tok, stride_qpe_h, stride_qpe_d,
    stride_ckv_page, stride_ckv_tok, stride_ckv_d,
    stride_kpe_page, stride_kpe_tok, stride_kpe_d,
    stride_idx_tok, stride_idx_k,
    stride_pacc_tok, stride_pacc_split, stride_pacc_h, stride_pacc_d,
    stride_pm_tok, stride_pm_split, stride_pm_h,
    stride_pl_tok, stride_pl_split, stride_pl_h,
    page_size: t1.constexpr, topk: t1.constexpr,
    SPLIT_K: t1.constexpr,
    BLOCK_N: t1.constexpr,
    BLOCK_D_CKV: t1.constexpr,
    BLOCK_D_KPE: t1.constexpr,
    BLOCK_H: t1.constexpr,
    D_TILE_ID: t1.constexpr,
    D_TILES: t1.constexpr,
):
    # Both the K-side reduction (computing qk_nope) AND the V-side
    # accumulation are tiled over BLOCK_D_CKV, each as its own D-sub-tile
    # loop. Never holds a full-width [*, BLOCK_D_CKV] tensor -- only ever
    # a [*, TILE_D] slice at a time, reloaded per sub-tile in each pass.
    # This accepts a real cost (K-cache read twice: once per D-sub-tile in
    # each pass, so 2x total D-sub-tile loads vs the original's 1x
    # full-width load) as the price for bounded peak memory. Launched once
    # per (token, split, D_TILE_ID); D_TILE_ID only affects which slice of
    # the OUTPUT accumulator this launch is responsible for -- the qk/
    # softmax computation (needing the full 512-dim) is still redone in
    # every D_TILE_ID's launch, same redundancy as splitk_dtiled_v2.
    tok_id = t1.program_id(0)
    split_id = t1.program_id(1)

    TILE_D: t1.constexpr = BLOCK_D_CKV // D_TILES
    d_offset = D_TILE_ID * TILE_D

    offs_h = t1.arange(0, BLOCK_H)
    offs_d_kpe = t1.arange(0, BLOCK_D_KPE)
    offs_td = t1.arange(0, TILE_D)

    q_pe_ptrs = Q_PE + tok_id * stride_qpe_tok + offs_h[:, None] * stride_qpe_h + offs_d_kpe[None, :] * stride_qpe_d
    q_pe = t1.load(q_pe_ptrs)

    m_i = t1.full([BLOCK_H], -float("inf"), dtype=t1.float32)
    l_i = t1.zeros([BLOCK_H], dtype=t1.float32)
    acc = t1.zeros([BLOCK_H, TILE_D], dtype=t1.float32)

    chunk = topk // SPLIT_K
    chunk_start = split_id * chunk

    for n_start in range(0, chunk, BLOCK_N):
        offs_n = chunk_start + n_start + t1.arange(0, BLOCK_N)
        idx_ptrs = SPARSE_INDICES + tok_id * stride_idx_tok + offs_n * stride_idx_k

        mask_n = offs_n < (chunk_start + chunk)
        mask_n &= offs_n < topk

        indices = t1.load(idx_ptrs, mask=mask_n, other=-1)
        valid_mask_col = indices[:, None] != -1
        valid_mask_row = indices[None, :] != -1

        page_idx = indices // page_size
        tok_offset = indices % page_size

        k_kpe_ptrs = KPE_CACHE + page_idx[:, None] * stride_kpe_page + tok_offset[:, None] * stride_kpe_tok + offs_d_kpe[None, :] * stride_kpe_d
        k_kpe = t1.load(k_kpe_ptrs, mask=valid_mask_col, other=0.0)
        qk_pe = t1.dot(q_pe, k_kpe.T)

        # ── Pass 1 (K-side, tiled): accumulate qk_nope across D-sub-tiles.
        # Only ONE [BLOCK_N, TILE_D] k_ckv slice + [BLOCK_H, TILE_D] q_nope
        # slice ever resident at a time.
        qk_nope = t1.zeros([BLOCK_H, BLOCK_N], dtype=t1.float32)
        for d in range(D_TILES):
            sub_offset = d * TILE_D
            q_nope_sub_ptrs = Q_NOPE + tok_id * stride_qt_tok + offs_h[:, None] * stride_qt_h + (sub_offset + offs_td)[None, :] * stride_qt_d
            q_nope_sub = t1.load(q_nope_sub_ptrs)

            k_ckv_sub_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + (sub_offset + offs_td)[None, :] * stride_ckv_d
            k_ckv_sub = t1.load(k_ckv_sub_ptrs, mask=valid_mask_col, other=0.0)

            qk_nope += t1.dot(q_nope_sub, k_ckv_sub.T)

        qk = (qk_nope + qk_pe) * sm_scale
        qk = t1.where(valid_mask_row, qk, -float("inf"))

        m_ij = t1.maximum(m_i, t1.max(qk, axis=1))
        m_ij_finite_col = m_ij[:, None] != -float("inf")
        p = t1.where(m_ij_finite_col, t1.exp(qk - m_ij[:, None]), 0.0)
        alpha = t1.where(m_i == -float("inf"), 0.0, t1.exp(m_i - m_ij))
        l_i = l_i * alpha + t1.sum(p, axis=1)
        m_i = m_ij

        # ── Pass 2 (V-side, tiled): only THIS launch's D_TILE_ID slice.
        k_ckv_v_ptrs = CKV_CACHE + page_idx[:, None] * stride_ckv_page + tok_offset[:, None] * stride_ckv_tok + (d_offset + offs_td)[None, :] * stride_ckv_d
        k_ckv_v = t1.load(k_ckv_v_ptrs, mask=valid_mask_col, other=0.0)

        acc = acc * alpha[:, None]
        acc += t1.dot(p.to(t1.bfloat16), k_ckv_v)

    m_i_finite_col = m_i[:, None] != -float("inf")
    acc_out = t1.where(m_i_finite_col, acc, 0.0)

    pacc_ptrs = (PARTIAL_ACC + tok_id * stride_pacc_tok + split_id * stride_pacc_split
                 + offs_h[:, None] * stride_pacc_h + (d_offset + offs_td)[None, :] * stride_pacc_d)
    t1.store(pacc_ptrs, acc_out)

    if D_TILE_ID == 0:
        pm_ptrs = PARTIAL_M + tok_id * stride_pm_tok + split_id * stride_pm_split + offs_h * stride_pm_h
        t1.store(pm_ptrs, m_i)
        pl_ptrs = PARTIAL_L + tok_id * stride_pl_tok + split_id * stride_pl_split + offs_h * stride_pl_h
        t1.store(pl_ptrs, l_i)


def kernel_splitk_dtiled_v3(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=16, D_TILES=2):
    """
    Bottleneck-2 attempt 3: split-K forward pass with co-tiled K-side
    reduction and V-side accumulation (dsa_fwd_kernel_splitk_dtiled_v3),
    reducing peak shared memory from 94976 to 83968 bytes. Uses the
    existing (sequential) reduce kernel from v1 for the SPLIT_K merge,
    since this attempt is isolating the D-tiling effect specifically, not
    combining it with v3's atomic-counter reduction yet.
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert topk % SPLIT_K == 0
    assert head_dim_ckv % D_TILES == 0

    device = q_nope.device

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)

    partial_acc = torch.zeros((num_tokens, SPLIT_K, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    partial_m = torch.full((num_tokens, SPLIT_K, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    partial_l = torch.zeros((num_tokens, SPLIT_K, num_qo_heads), dtype=torch.float32, device=device)

    chunk_size = topk // SPLIT_K
    BLOCK_N = min(64, max(16, chunk_size // 2))
    grid = (num_tokens, SPLIT_K)

    for d_tile_id in range(D_TILES):
        dsa_fwd_kernel_splitk_dtiled_v3[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
            partial_acc, partial_m, partial_l,
            sm_scale,
            q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
            q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
            ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
            kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
            sparse_indices.stride(0), sparse_indices.stride(1),
            partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            page_size=page_size, topk=topk, SPLIT_K=SPLIT_K, BLOCK_N=BLOCK_N,
            BLOCK_D_CKV=head_dim_ckv, BLOCK_D_KPE=head_dim_kpe, BLOCK_H=num_qo_heads,
            D_TILE_ID=d_tile_id, D_TILES=D_TILES,
        )

    grid_reduce = (num_tokens,)
    dsa_reduce_kernel[grid_reduce](
        partial_acc, partial_m, partial_l, output, lse,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        SPLIT_K=SPLIT_K,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_H=num_qo_heads,
    )

    return output, lse


def kernel_splitk_v3_bf16acc(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=16):
    """
    Bottleneck-2 attempt 4: same as kernel_splitk_v3, but the in-kernel
    working accumulator (acc) is kept in bfloat16 during the KV loop
    instead of float32, halving its footprint. Rescaling (acc * alpha) is
    done in float32 to avoid compounding rounding error across iterations,
    then cast back to bf16 for storage between iterations. The scratch
    PARTIAL_ACC buffer remains float32 (only the in-loop working tensor's
    dtype changes) to isolate this specific change from the reduction
    logic. See docs/SPLITK_OPTIMIZATION.md.
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert topk % SPLIT_K == 0

    device = q_nope.device

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)

    partial_acc = torch.zeros((num_tokens, SPLIT_K, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    partial_m = torch.full((num_tokens, SPLIT_K, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    partial_l = torch.zeros((num_tokens, SPLIT_K, num_qo_heads), dtype=torch.float32, device=device)
    counter = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

    chunk_size = topk // SPLIT_K
    BLOCK_N = min(64, max(16, chunk_size // 2))

    grid = (num_tokens, SPLIT_K)
    dsa_fwd_kernel_splitk_v3_bf16acc[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        partial_acc, partial_m, partial_l, counter, output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        page_size=page_size, topk=topk,
        SPLIT_K=SPLIT_K,
        BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv,
        BLOCK_D_KPE=head_dim_kpe,
        BLOCK_H=num_qo_heads,
    )

    return output, lse
