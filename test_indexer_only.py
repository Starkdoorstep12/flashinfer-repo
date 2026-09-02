import torch, time
from safetensors.torch import load_file
from solution.triton.indexer_kernel import indexer_kernel

device = 'cuda'
torch.manual_seed(0)

seq_lens = torch.tensor([88, 91, 91, 96, 91], dtype=torch.int32, device=device)
batch_size = seq_lens.shape[0]
seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)

num_index_heads = 64
index_head_dim = 128
page_size = 64
kv_cache_num_heads = 1
head_dim_with_scale = 132
max_num_pages = 2
num_pages = 100
BLOCK_TOKENS = 32
BLOCK_HEADS = 8

block_table = torch.randint(0, num_pages, (batch_size, max_num_pages), dtype=torch.int32, device=device)
q_index_fp8 = torch.randint(-128, 127, (batch_size, num_index_heads, index_head_dim), dtype=torch.int8, device=device)
k_index_cache_fp8 = torch.randint(-128, 127, (num_pages * page_size * head_dim_with_scale,), dtype=torch.int8, device=device)
weights = torch.rand(batch_size, num_index_heads, device=device)

tiles_per_seq = (seq_lens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
tile_offsets = torch.cumsum(tiles_per_seq, dim=0)
total_tiles = tile_offsets[-1].item()
total_tokens = (seq_offsets[-1] + seq_lens[-1]).item()
acc = torch.zeros(total_tokens, device=device, dtype=torch.float32)

print("Launching indexer_kernel only...")
t0 = time.time()
grid = (total_tiles,)
indexer_kernel[grid](
    q_index_fp8=q_index_fp8, k_index_cache_fp8=k_index_cache_fp8, weights=weights,
    seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
    tile_offsets_ptr=tile_offsets, acc_ptr=acc,
    batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
    page_size=page_size, kv_cache_num_heads=kv_cache_num_heads,
    head_dim_with_scale=head_dim_with_scale, max_num_pages=max_num_pages,
    BLOCK_TOKENS=BLOCK_TOKENS, BLOCK_HEADS=BLOCK_HEADS,
)
torch.cuda.synchronize()
print(f"indexer_kernel done in {time.time()-t0:.2f}s")
