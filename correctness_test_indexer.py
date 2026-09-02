"""
Correctness test for run_indexer_and_topk against the golden reference
from the dataset definition (dsa_topk_indexer_fp8_h64_d128_topk2048_ps64).

Builds REAL FP8-formatted input data matching the golden reference's exact
expectations (torch.float8_e4m3fn queries, packed [fp8_bytes][scale_bytes]
per-page KV cache layout) -- not the loose random-int8 approximations used
in earlier timing-only tests tonight, which never validated correctness.
"""
import torch
import sys

sys.path.insert(0, '/tmp')
from golden_indexer_reference import run as golden_run

from solution.triton.indexer_kernel import run_indexer_and_topk

device = 'cuda'
torch.manual_seed(0)

# Small, fast-to-compile shapes for a first correctness pass
batch_size = 2
num_index_heads = 64
index_head_dim = 128
page_size = 64
seq_lens_list = [50, 80]
max_seq_len = max(seq_lens_list)
max_num_pages = (max_seq_len + page_size - 1) // page_size
num_pages = max_num_pages * batch_size + 2
head_dim_with_scale = 132  # 128 + 4 scale bytes
topk = 2048

seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)
block_table = torch.zeros((batch_size, max_num_pages), dtype=torch.int32, device=device)
for b in range(batch_size):
    npages = (seq_lens_list[b] + page_size - 1) // page_size
    block_table[b, :npages] = torch.arange(b * max_num_pages, b * max_num_pages + npages, device=device)

# ---- Real FP8 query tensor ----
q_index_fp32 = torch.randn(batch_size, num_index_heads, index_head_dim, device=device) * 2.0
q_index_fp8 = q_index_fp32.to(torch.float8_e4m3fn)

# ---- Real packed KV cache: [num_pages, page_size, 1, head_dim_with_scale] uint8 ----
# Per-page layout: [fp8_data (page_size*128 bytes)][scale_data (page_size*4 bytes)]
k_fp32 = torch.randn(num_pages, page_size, index_head_dim, device=device) * 2.0
k_fp8 = k_fp32.to(torch.float8_e4m3fn)
scales = torch.rand(num_pages, page_size, device=device) * 0.5 + 0.5  # scales in [0.5, 1.0]

k_index_cache_fp8 = torch.zeros((num_pages, page_size, 1, head_dim_with_scale), dtype=torch.uint8, device=device)
# Write FP8 bytes (raw bit pattern of float8_e4m3fn is 1 byte each)
k_index_cache_fp8_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_with_scale)
fp8_bytes = k_fp8.view(torch.uint8).view(num_pages, page_size * index_head_dim)
k_index_cache_fp8_flat[:, :page_size * index_head_dim] = fp8_bytes
scale_bytes = scales.to(torch.float32).view(torch.uint8).view(num_pages, page_size * 4)
k_index_cache_fp8_flat[:, page_size * index_head_dim:] = scale_bytes

weights = torch.rand(batch_size, num_index_heads, device=device)

# ---- Run golden reference ----
print("Running golden reference...")
golden_topk_indices, = golden_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
print("Golden output shape:", golden_topk_indices.shape)
print("Golden sample (batch 0, first 10):", golden_topk_indices[0, :10])
print("Golden sample (batch 0, valid count):", (golden_topk_indices[0] != -1).sum().item())

# ---- Run our kernel ----
# Our kernel expects k_index_cache_fp8 as a flat 1D int8 tensor
k_index_cache_fp8_kernel = k_index_cache_fp8.view(torch.int8).reshape(-1)

print("\nRunning our kernel...")
try:
    our_topk_indices = run_indexer_and_topk(
        q_index_fp8=q_index_fp8.view(torch.int8),  # kernel loads raw bytes, cast happens inside
        k_index_cache_fp8=k_index_cache_fp8_kernel,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        seq_offsets=seq_offsets,
        batch_size=batch_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        page_size=page_size,
        kv_cache_num_heads=1,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages,
        topk=topk,
        BLOCK_TOKENS=32,
        BLOCK_HEADS=8,
    )
    print("Our output shape:", our_topk_indices.shape)
    print("Our sample (batch 0, first 10):", our_topk_indices[0, :10])
    print("Our sample (batch 0, valid count):", (our_topk_indices[0] != -1).sum().item())

    # Compare as SETS per batch (order within top-k may legitimately differ)
    for b in range(batch_size):
        golden_set = set(golden_topk_indices[b][golden_topk_indices[b] != -1].tolist())
        our_set = set(our_topk_indices[b][our_topk_indices[b] != -1].tolist())
        overlap = len(golden_set & our_set)
        print(f"Batch {b}: golden={len(golden_set)} selected, ours={len(our_set)} selected, overlap={overlap}")
        if golden_set != our_set:
            print(f"  MISMATCH: golden-only={len(golden_set - our_set)}, ours-only={len(our_set - golden_set)}")
except Exception as e:
    print(f"KERNEL FAILED: {type(e).__name__}: {e}")
