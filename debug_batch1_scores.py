import torch
import sys
sys.path.insert(0, '/tmp')
from golden_indexer_reference import dequant_fp8_kv_cache

from solution.triton.indexer_kernel import indexer_kernel

device = 'cuda'
torch.manual_seed(0)

batch_size = 2
num_index_heads = 64
index_head_dim = 128
page_size = 64
seq_lens_list = [50, 80]
max_seq_len = max(seq_lens_list)
max_num_pages = (max_seq_len + page_size - 1) // page_size
num_pages = max_num_pages * batch_size + 2
head_dim_with_scale = 132
BLOCK_TOKENS = 32
BLOCK_HEADS = 8

seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)
block_table = torch.zeros((batch_size, max_num_pages), dtype=torch.int32, device=device)
for b in range(batch_size):
    npages = (seq_lens_list[b] + page_size - 1) // page_size
    block_table[b, :npages] = torch.arange(b * max_num_pages, b * max_num_pages + npages, device=device)

print("block_table:", block_table)

q_index_fp32 = torch.randn(batch_size, num_index_heads, index_head_dim, device=device) * 2.0
q_index_fp8 = q_index_fp32.to(torch.float8_e4m3fn)

k_fp32 = torch.randn(num_pages, page_size, index_head_dim, device=device) * 2.0
k_fp8 = k_fp32.to(torch.float8_e4m3fn)
scales = torch.rand(num_pages, page_size, device=device) * 0.5 + 0.5

k_index_cache_fp8 = torch.zeros((num_pages, page_size, 1, head_dim_with_scale), dtype=torch.uint8, device=device)
k_index_cache_fp8_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_with_scale)
fp8_bytes = k_fp8.view(torch.uint8).view(num_pages, page_size * index_head_dim)
k_index_cache_fp8_flat[:, :page_size * index_head_dim] = fp8_bytes
scale_bytes = scales.to(torch.float32).view(torch.uint8).view(num_pages, page_size * 4)
k_index_cache_fp8_flat[:, page_size * index_head_dim:] = scale_bytes

weights = torch.rand(batch_size, num_index_heads, device=device)

# ---- Reference scores for batch 1 ----
K_all_ref = dequant_fp8_kv_cache(k_index_cache_fp8)
q_ref = q_index_fp8.to(torch.float32)

b = 1
seq_len = seq_lens_list[b]
num_pages_for_seq = (seq_len + page_size - 1) // page_size
page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
print(f"Batch {b}: seq_len={seq_len}, num_pages_for_seq={num_pages_for_seq}, page_indices={page_indices}")
K_paged = K_all_ref[page_indices]
K = K_paged.reshape(-1, index_head_dim)[:seq_len]
q_b = q_ref[b]
scores = q_b @ K.T
scores_relu = torch.relu(scores)
w = weights[b]
weighted_scores = scores_relu * w[:, None]
final_scores_ref = weighted_scores.sum(dim=0)
print("Reference scores batch 1, tokens 60-70 (crossing page boundary at 64):", final_scores_ref[60:70])

# ---- Our kernel scores ----
k_index_cache_fp8_kernel = k_index_cache_fp8.view(torch.int8).reshape(-1)
tiles_per_seq = (seq_lens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
tile_offsets = torch.cumsum(tiles_per_seq, dim=0)
total_tiles = tile_offsets[-1].item()
total_tokens = (seq_offsets[-1] + seq_lens[-1]).item()
acc = torch.zeros(total_tokens, device=device, dtype=torch.float32)

grid = (total_tiles,)
indexer_kernel[grid](
    q_index_fp8=q_index_fp8.view(torch.int8),
    k_index_cache_fp8=k_index_cache_fp8_kernel,
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
    kv_cache_num_heads=1,
    head_dim_with_scale=head_dim_with_scale,
    max_num_pages=max_num_pages,
    BLOCK_TOKENS=BLOCK_TOKENS,
    BLOCK_HEADS=BLOCK_HEADS,
)
torch.cuda.synchronize()

our_scores_b1 = acc[seq_offsets[1]:seq_offsets[1]+seq_lens_list[1]]
print("Our scores batch 1, tokens 60-70:", our_scores_b1[60:70])
print("Max abs diff (full batch 1):", (final_scores_ref - our_scores_b1).abs().max().item())

print("\n--- Checking actual top-k selection for batch 1 ---")
from solution.triton.indexer_kernel import topk_kernel
import math

K_val = 2048
MAX_K = 2048
BLOCK = 16
MAX_SEQ_LEN = int(seq_lens.max().item())

topk_indices = torch.zeros((batch_size, K_val), dtype=torch.int32, device=device)
topk_grid = (batch_size,)
topk_kernel[topk_grid](
    acc_ptr=acc,
    seq_offsets=seq_offsets,
    seq_lens=seq_lens,
    topk_indices_ptr=topk_indices,
    K=K_val, MAX_SEQ_LEN=MAX_SEQ_LEN, BLOCK=BLOCK, MAX_K=MAX_K,
)
torch.cuda.synchronize()

our_b1_indices = topk_indices[1]
valid_our = our_b1_indices[our_b1_indices != -1]
print("Our batch 1 selected (sorted):", torch.sort(valid_our)[0][:20], "...")
print("Our batch 1 valid count:", len(valid_our))

# What SHOULD be selected: top 80 scores out of 80 tokens = ALL of them (since seq_len=80=topk cap for this batch... wait topk=2048 > 80, so ALL 80 tokens should be selected)
print("\nSince seq_len=80 < topk cap, ALL 80 tokens (0-79) should be selected.")
print("Are all 0-79 present in our selection?", set(range(seq_offsets[1].item(), seq_offsets[1].item()+80)) == set(valid_our.tolist()))
