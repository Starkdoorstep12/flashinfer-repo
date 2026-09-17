"""
Baseline performance profile of the now-correct indexer kernel
(run_indexer_and_topk), across real dataset workloads spanning different
batch_size and seq_len combinations. First performance measurement since
all four correctness bugs (Findings 1, 4, 5a, 5b) were fixed.
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

# Same diverse set used in the correctness stress test -- known-good shapes
target_uuids_prefixes = ["30cecff1", "44ddaa65", "b2098949", "4279d75e", "83cb81c5", "1ece7fb3", "70d53807"]

trace_path = "/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = []
with open(trace_path) as f:
    for line in f:
        d = json.loads(line)
        wl = d['workload']
        if any(wl['uuid'].startswith(p) for p in target_uuids_prefixes):
            workloads.append(wl)

print(f"Profiling {len(workloads)} real dataset workloads...\n")
print(f"{'uuid':14s} {'batch':>6s} {'max_seq_len':>12s} {'compile(s)':>11s} {'warm(ms)':>10s}")

for wl in workloads:
    uuid = wl['uuid']
    batch_size = wl['axes']['batch_size']
    max_num_pages = wl['axes']['max_num_pages']
    path = f"/home/vedant.tejas/mlsys26-contest/{wl['inputs']['seq_lens']['path']}"
    tensors = load_file(path)
    seq_lens = tensors['seq_lens'].to(device)
    block_table = tensors['block_table'].to(device)
    num_pages = int(block_table.max().item()) + 5
    max_seq_len = int(seq_lens.max().item())

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

    def run():
        return run_indexer_and_topk(
            q_index_fp8=q_index_fp8.view(torch.int8),
            k_index_cache_fp8=k_index_cache_fp8_kernel,
            weights=weights,
            seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
            batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
            page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
            max_num_pages=max_num_pages, topk=topk,
        )

    t0 = time.time()
    out = run()
    torch.cuda.synchronize()
    t_compile = time.time() - t0

    t0 = time.time()
    for _ in range(10):
        out = run()
    torch.cuda.synchronize()
    t_warm = (time.time() - t0) / 10 * 1000

    print(f"{uuid[:12]:14s} {batch_size:6d} {max_seq_len:12d} {t_compile:11.2f} {t_warm:10.3f}")
