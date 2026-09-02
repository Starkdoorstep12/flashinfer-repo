"""
Multi-workload correctness stress test for the corrected indexer kernel,
using REAL dataset shapes (batch_size, seq_lens, block_table) from
several actual workloads, not just the two hand-built test cases in
correctness_test_indexer.py. Covers a range of batch_size values
including non-power-of-2 ones (Finding 1) and varying sequence lengths
(Finding 4/5's dequantization and index-space fixes).
"""
import torch, json, sys
sys.path.insert(0, '/tmp')
from golden_indexer_reference import run as golden_run
from safetensors.torch import load_file
from solution.triton.indexer_kernel import run_indexer_and_topk

device = 'cuda'
num_index_heads = 64
index_head_dim = 128
page_size = 64
head_dim_with_scale = 132
topk = 2048

# Pick a diverse set of real workloads: small/large batch_size, including non-power-of-2
target_uuids_prefixes = ["30cecff1", "44ddaa65", "b2098949", "4279d75e", "83cb81c5", "1ece7fb3", "70d53807"]

trace_path = "/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = []
with open(trace_path) as f:
    for line in f:
        d = json.loads(line)
        wl = d['workload']
        if any(wl['uuid'].startswith(p) for p in target_uuids_prefixes):
            workloads.append(wl)

print(f"Testing {len(workloads)} real dataset workloads...\n")

results = []
for wl in workloads:
    uuid = wl['uuid']
    batch_size = wl['axes']['batch_size']
    max_num_pages = wl['axes']['max_num_pages']
    path = f"/home/vedant.tejas/mlsys26-contest/{wl['inputs']['seq_lens']['path']}"
    tensors = load_file(path)
    seq_lens = tensors['seq_lens'].to(device)
    block_table = tensors['block_table'].to(device)
    num_pages = int(block_table.max().item()) + 5  # ensure enough pages allocated

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

    try:
        golden_topk, = golden_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)

        k_index_cache_fp8_kernel = k_index_cache_fp8.view(torch.int8).reshape(-1)
        our_topk = run_indexer_and_topk(
            q_index_fp8=q_index_fp8.view(torch.int8),
            k_index_cache_fp8=k_index_cache_fp8_kernel,
            weights=weights,
            seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
            batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
            page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
            max_num_pages=max_num_pages, topk=topk,
        )

        total_overlap, total_golden, total_ours = 0, 0, 0
        for b in range(batch_size):
            gset = set(golden_topk[b][golden_topk[b] != -1].tolist())
            oset = set(our_topk[b][our_topk[b] != -1].tolist())
            total_overlap += len(gset & oset)
            total_golden += len(gset)
            total_ours += len(oset)

        match = (total_overlap == total_golden == total_ours)
        status = "PASS" if match else "FAIL"
        print(f"[{status}] uuid={uuid[:12]} batch_size={batch_size:3d} (pow2={batch_size&(batch_size-1)==0})  "
              f"overlap={total_overlap}/{total_golden} (ours={total_ours})")
        results.append((uuid, batch_size, match))
    except Exception as e:
        print(f"[ERROR] uuid={uuid[:12]} batch_size={batch_size:3d}: {type(e).__name__}: {str(e)[:150]}")
        results.append((uuid, batch_size, False))

print(f"\n{sum(1 for _,_,m in results if m)}/{len(results)} workloads passed")
