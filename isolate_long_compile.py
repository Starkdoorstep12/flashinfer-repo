import torch, time
from solution.triton.indexer_kernel import run_indexer_and_topk

device = 'cuda'
torch.manual_seed(0)
batch_size = 4
seq_len_value = 5824
seq_lens = torch.full((batch_size,), seq_len_value, dtype=torch.int32, device=device)
seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)
page_size = 64
max_num_pages = (seq_len_value + page_size - 1) // page_size
num_pages = max_num_pages * batch_size + 10
num_index_heads = 64
index_head_dim = 128
kv_cache_num_heads = 1
head_dim_with_scale = 132
topk = 2048

block_table = torch.randint(0, num_pages, (batch_size, max_num_pages), dtype=torch.int32, device=device)
q_index_fp8 = torch.randint(-128, 127, (batch_size, num_index_heads, index_head_dim), dtype=torch.int8, device=device)
k_index_cache_fp8 = torch.randint(-128, 127, (num_pages * page_size * head_dim_with_scale,), dtype=torch.int8, device=device)
weights = torch.rand(batch_size, num_index_heads, device=device)

print('Compiling LONG case (seq_len=5824)...', flush=True)
t0 = time.time()
out = run_indexer_and_topk(
    q_index_fp8=q_index_fp8, k_index_cache_fp8=k_index_cache_fp8, weights=weights,
    seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
    batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
    page_size=page_size, kv_cache_num_heads=kv_cache_num_heads,
    head_dim_with_scale=head_dim_with_scale, max_num_pages=max_num_pages, topk=topk,
    BLOCK_TOKENS=32, BLOCK_HEADS=8,
)
torch.cuda.synchronize()
print(f'Done in {time.time()-t0:.2f}s', flush=True)
