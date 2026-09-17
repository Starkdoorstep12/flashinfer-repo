"""
Reproduces the exact sequence that failed: workload 1 (batch=1, pages=1)
followed immediately by workload 2 (batch=1, pages=2), in the same
process, to catch the suspected context-corruption/carryover bug.
"""
import torch, json, time
from safetensors.torch import load_file
from solution.triton.indexer_kernel import run_indexer_and_topk

device = 'cuda'
num_index_heads = 64
index_head_dim = 128
page_size = 64
head_dim_with_scale = 132
topk = 2048

with open('/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl') as f:
    workloads = [json.loads(line)['workload'] for line in f]

def run_one(wl, label):
    uuid = wl['uuid']
    batch_size = wl['axes']['batch_size']
    max_num_pages = wl['axes']['max_num_pages']
    path = f"/home/vedant.tejas/mlsys26-contest/{wl['inputs']['seq_lens']['path']}"
    tensors = load_file(path)
    seq_lens = tensors['seq_lens'].to(device)
    block_table = tensors['block_table'].to(device)
    print(f"[{label}] uuid={uuid} batch_size={batch_size} max_num_pages={max_num_pages} seq_lens={seq_lens} block_table={block_table}")
    num_pages = int(block_table.max().item()) + 5
    seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)

    torch.manual_seed(hash(uuid) % (2**31))
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

    t0 = time.time()
    out = run_indexer_and_topk(
        q_index_fp8=q_index_fp8.view(torch.int8), k_index_cache_fp8=k_index_cache_fp8_kernel, weights=weights,
        seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
        batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
        page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages, topk=topk,
    )
    torch.cuda.synchronize()
    print(f"[{label}] SUCCESS in {time.time()-t0:.2f}s, output shape:", out.shape)
    print(f"[{label}] output sample (first row, first 10):", out[0][:10])
    print(f"[{label}] valid count (batch 0):", (out[0] != -1).sum().item())
    return out

print("=== Running workload 1 ===")
run_one(workloads[0], "WL1")
torch.cuda.synchronize()
print("\n=== Running workload 2 (immediately after, same process) ===")
run_one(workloads[1], "WL2")
torch.cuda.synchronize()
print("BOTH SYNCED SUCCESSFULLY")
