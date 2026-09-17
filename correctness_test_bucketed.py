"""
Correctness test for run_indexer_and_topk_bucketed against the unbucketed
run_indexer_and_topk, across the same 7 real dataset workloads used in
stress_test_indexer_dataset.py. Bucketing pads shapes internally but
should produce IDENTICAL output to the unbucketed version, since padded
entries are masked out.
"""
import torch, json
from safetensors.torch import load_file
from solution.triton.indexer_kernel import run_indexer_and_topk, run_indexer_and_topk_bucketed

device = 'cuda'
num_index_heads = 64
index_head_dim = 128
page_size = 64
head_dim_with_scale = 132
topk = 2048

target_uuids_prefixes = ["30cecff1", "44ddaa65", "b2098949", "4279d75e", "83cb81c5", "1ece7fb3", "70d53807"]

trace_path = "/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = []
with open(trace_path) as f:
    for line in f:
        d = json.loads(line)
        wl = d['workload']
        if any(wl['uuid'].startswith(p) for p in target_uuids_prefixes):
            workloads.append(wl)

print(f"Testing {len(workloads)} workloads: bucketed vs unbucketed\n")

all_pass = True
for wl in workloads:
    uuid = wl['uuid']
    batch_size = wl['axes']['batch_size']
    max_num_pages = wl['axes']['max_num_pages']
    path = f"/home/vedant.tejas/mlsys26-contest/{wl['inputs']['seq_lens']['path']}"
    tensors = load_file(path)
    seq_lens = tensors['seq_lens'].to(device)
    block_table = tensors['block_table'].to(device)
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

    try:
        unbucketed = run_indexer_and_topk(
            q_index_fp8=q_index_fp8.view(torch.int8), k_index_cache_fp8=k_index_cache_fp8_kernel, weights=weights,
            seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
            batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
            page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
            max_num_pages=max_num_pages, topk=topk,
        )

        bucketed = run_indexer_and_topk_bucketed(
            q_index_fp8=q_index_fp8.view(torch.int8), k_index_cache_fp8=k_index_cache_fp8_kernel, weights=weights,
            seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
            batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
            page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
            max_num_pages=max_num_pages, topk=topk,
        )

        match = True
        for b in range(batch_size):
            uset = set(unbucketed[b][unbucketed[b] != -1].tolist())
            bset = set(bucketed[b][bucketed[b] != -1].tolist())
            if uset != bset:
                match = False

        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"[{status}] uuid={uuid[:12]} batch_size={batch_size:3d} max_num_pages={max_num_pages:3d}")
    except Exception as e:
        all_pass = False
        print(f"[ERROR] uuid={uuid[:12]}: {type(e).__name__}: {str(e)[:200]}")

    # Explicit cleanup between workloads to rule out memory
    # pressure/fragmentation as a cause of intermittent failures.
    del q_index_fp32, q_index_fp8, k_fp32, k_fp8, k_index_cache_fp8, k_index_cache_fp8_kernel, weights, seq_lens, block_table, seq_offsets
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
