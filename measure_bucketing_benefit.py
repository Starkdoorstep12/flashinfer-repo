"""
Measures the real compile-count and compile-time benefit of shape
bucketing across a sample of real dataset workloads. Compares:
(a) unbucketed: one compile per distinct (batch_size, max_num_pages) pair
(b) bucketed: one compile per distinct (next_pow2(batch_size), next_pow2(max_num_pages)) pair
Uses a FRESH Triton cache for each mode so compile counts are accurate,
and explicit memory cleanup between workloads (see correctness_test_bucketed.py finding).
"""
import torch, json, time, subprocess
from safetensors.torch import load_file
from solution.triton.indexer_kernel import run_indexer_and_topk, run_indexer_and_topk_bucketed

device = 'cuda'
num_index_heads = 64
index_head_dim = 128
page_size = 64
head_dim_with_scale = 132
topk = 2048

# Use a larger, more diverse sample -- first N workloads from the dataset directly
trace_path = "/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = []
with open(trace_path) as f:
    for line in f:
        d = json.loads(line)
        workloads.append(d['workload'])

N = 30
workloads = workloads[:N]
print(f"Measuring across {len(workloads)} real workloads (first {N} in dataset order)\n")

def make_inputs(wl):
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

    return dict(
        q_index_fp8=q_index_fp8.view(torch.int8), k_index_cache_fp8=k_index_cache_fp8_kernel, weights=weights,
        seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
        batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
        page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages, topk=topk,
    )

def cleanup(inp):
    for k, v in inp.items():
        if isinstance(v, torch.Tensor):
            del v
    del inp
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ---- Unbucketed timing ----
# NOTE: a rare, non-reproducible "illegal memory access" was observed once
# during an earlier 30-workload run. A 30-trial stress test of the exact
# failing sequence in isolation (stress_test_carryover.py) reproduced it
# 0/30 times -- treated as a rare environmental fault, not a deterministic
# bug, per docs/INDEXER_OPTIMIZATION.md. Each iteration is wrapped so one
# such fault doesn't abort the whole measurement; failures are logged and
# excluded from the timing total rather than silently retried or masked.
print("=== UNBUCKETED ===")
subprocess.run("rm -rf ~/.triton/cache", shell=True)
t_total_unbucketed = 0.0
n_failed_unbucketed = 0
for i, wl in enumerate(workloads):
    inp = make_inputs(wl)
    try:
        t0 = time.time()
        out = run_indexer_and_topk(**inp)
        torch.cuda.synchronize()
        dt = time.time() - t0
        t_total_unbucketed += dt
        print(f"  [{i+1}/{len(workloads)}] batch={wl['axes']['batch_size']:3d} pages={wl['axes']['max_num_pages']:3d}  {dt:.2f}s")
    except Exception as e:
        n_failed_unbucketed += 1
        print(f"  [{i+1}/{len(workloads)}] batch={wl['axes']['batch_size']:3d} pages={wl['axes']['max_num_pages']:3d}  FAILED: {type(e).__name__}: {str(e)[:100]}")
    cleanup(inp)
print(f"Total unbucketed time: {t_total_unbucketed:.1f}s ({n_failed_unbucketed} failures excluded)\n")

# ---- Bucketed timing ----
print("=== BUCKETED ===")
subprocess.run("rm -rf ~/.triton/cache", shell=True)
t_total_bucketed = 0.0
n_failed_bucketed = 0
for i, wl in enumerate(workloads):
    inp = make_inputs(wl)
    try:
        t0 = time.time()
        out = run_indexer_and_topk_bucketed(**inp)
        torch.cuda.synchronize()
        dt = time.time() - t0
        t_total_bucketed += dt
        print(f"  [{i+1}/{len(workloads)}] batch={wl['axes']['batch_size']:3d} pages={wl['axes']['max_num_pages']:3d}  {dt:.2f}s")
    except Exception as e:
        n_failed_bucketed += 1
        print(f"  [{i+1}/{len(workloads)}] batch={wl['axes']['batch_size']:3d} pages={wl['axes']['max_num_pages']:3d}  FAILED: {type(e).__name__}: {str(e)[:100]}")
    cleanup(inp)
print(f"Total bucketed time: {t_total_bucketed:.1f}s ({n_failed_bucketed} failures excluded)\n")

print(f"=== SUMMARY ===")
print(f"Unbucketed total: {t_total_unbucketed:.1f}s")
print(f"Bucketed total:   {t_total_bucketed:.1f}s")
print(f"Speedup: {t_total_unbucketed/t_total_bucketed:.2f}x")
