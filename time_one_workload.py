"""
Times ONE workload (unbucketed or bucketed) in a fresh, isolated process.
Called repeatedly via subprocess from measure_bucketing_isolated.py so
each workload gets a completely fresh CUDA context -- avoiding whatever
rare cross-shape state issue causes the intermittent crash when many
distinct shapes run in one long-lived process.
"""
import torch, json, time, sys
from safetensors.torch import load_file
from solution.triton.indexer_kernel import run_indexer_and_topk, run_indexer_and_topk_bucketed

device = 'cuda'
num_index_heads = 64
index_head_dim = 128
page_size = 64
head_dim_with_scale = 132
topk = 2048

uuid_prefix = sys.argv[1]
mode = sys.argv[2]  # "unbucketed" or "bucketed"

with open('/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl') as f:
    workloads = [json.loads(line)['workload'] for line in f]
wl = next(w for w in workloads if w['uuid'].startswith(uuid_prefix))

batch_size = wl['axes']['batch_size']
max_num_pages = wl['axes']['max_num_pages']
path = f"/home/vedant.tejas/mlsys26-contest/{wl['inputs']['seq_lens']['path']}"
tensors = load_file(path)
seq_lens = tensors['seq_lens'].to(device)
block_table = tensors['block_table'].to(device)
num_pages = int(block_table.max().item()) + 5
seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)

torch.manual_seed(hash(wl['uuid']) % (2**31))
q_index_fp32 = torch.randn(batch_size, num_index_heads, index_head_dim, device=device) * 2.0
q_index_fp8 = q_index_fp32.to(torch.float8_e4m3fn)
k_fp32 = torch.randn(num_pages, page_size, index_head_dim, device=device) * 2.0
k_fp8 = k_fp32.to(torch.float8_e4m3fn)
scales = torch.rand(num_pages, page_size, device=device) * 0.5 + 0.5
k_index_cache_fp8 = torch.zeros((num_pages, page_size, 1, head_dim_with_scale), dtype=torch.uint8, device=device)
k_index_cache_fp8_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_with_scale)
fp8_bytes = k_fp8.view(torch.uint8).view(num_pages, page_size * index_head_dim)
k_index_cache_fp8_flat[:, :page_size * index_head_dim] = fp8_bytes
scale_bytes_t = scales.to(torch.float32).view(torch.uint8).view(num_pages, page_size * 4)
k_index_cache_fp8_flat[:, page_size * index_head_dim:] = scale_bytes_t
weights = torch.rand(batch_size, num_index_heads, device=device)
k_index_cache_fp8_kernel = k_index_cache_fp8.view(torch.int8).reshape(-1)

fn = run_indexer_and_topk if mode == "unbucketed" else run_indexer_and_topk_bucketed

t0 = time.time()
out = fn(
    q_index_fp8=q_index_fp8.view(torch.int8), k_index_cache_fp8=k_index_cache_fp8_kernel, weights=weights,
    seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
    batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
    page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
    max_num_pages=max_num_pages, topk=topk,
)
torch.cuda.synchronize()
print(f"TIME={time.time()-t0:.4f}")
