import torch, math
from safetensors.torch import load_file
from solution.triton.indexer_kernel import run_indexer_and_topk

device = 'cuda'
torch.manual_seed(0)

path = "/home/vedant.tejas/mlsys26-contest/blob/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64_3eab2c37-dbd9-4f7a-890e-0643cc9ca2ee.safetensors"
tensors = load_file(path)
seq_lens = tensors['seq_lens'].to(device)
block_table = tensors['block_table'].to(device)

print("seq_lens:", seq_lens)
print("block_table shape:", block_table.shape)

batch_size = seq_lens.shape[0]
num_index_heads = 64
index_head_dim = 128
page_size = 64
kv_cache_num_heads = 1
head_dim_with_scale = 132
max_num_pages = block_table.shape[1]
num_pages = 11923
topk = 2048

seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)

q_index_fp8 = torch.randint(-128, 127, (batch_size, num_index_heads, index_head_dim), dtype=torch.int8, device=device)
k_index_cache_fp8 = torch.randint(-128, 127, (num_pages * page_size * head_dim_with_scale,), dtype=torch.int8, device=device)
weights = torch.rand(batch_size, num_index_heads, device=device)

def run():
    return run_indexer_and_topk(
        q_index_fp8=q_index_fp8, k_index_cache_fp8=k_index_cache_fp8, weights=weights,
        seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
        batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
        page_size=page_size, kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale, max_num_pages=max_num_pages, topk=topk,
    )

for _ in range(5):
    out = run()
torch.cuda.synchronize()

s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(20):
    out = run()
e.record(); torch.cuda.synchronize()
print(f"Avg latency: {s.elapsed_time(e)/20:.3f} ms")
print("Output shape:", out.shape)
