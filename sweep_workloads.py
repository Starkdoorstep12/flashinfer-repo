"""
Generalization sweep: test kernel() [original], kernel_splitk (v1), and
kernel_splitk_v3 across representative workloads spanning the full
num_tokens range seen in the dataset (1, 2, 6, 7, 8), not just the single
num_tokens=1 workload used throughout prior profiling. This directly tests
whether split-K's benefit holds, shrinks, or reverses as num_tokens grows
and the ORIGINAL kernel's grid (= num_tokens blocks) becomes less
severely underutilized on its own.
"""
import torch, math, json
from safetensors.torch import load_file
from solution.triton.kernel import kernel, kernel_splitk, kernel_splitk_v3

device = 'cuda'
num_qo_heads, head_dim_ckv, head_dim_kpe = 16, 512, 64
page_size = 64
SPLIT_K = 16

# One representative workload per distinct num_tokens value seen in the dataset
representative_uuids = {
    1: "0c23b10c7b7645719517828c12eaa1d2",
    2: "9d4a5f21268e484ea05a2f2af91d9fa7",
    6: "ddfa9e340b264f76abe7418692faa876",
    7: "3838996164a94d728710f913477feba8",
    8: "385742b2717e4f02b918c7349dde23d8",
}

trace_path = "/home/vedant.tejas/mlsys26-contest/traces/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
workload_info = {}
with open(trace_path) as f:
    for line in f:
        d = json.loads(line)
        wl = d.get("workload", {})
        uuid = wl.get("uuid")
        if uuid in representative_uuids.values():
            workload_info[uuid] = wl

results = []
for num_tokens, uuid in representative_uuids.items():
    wl = workload_info[uuid]
    num_pages = wl["axes"]["num_pages"]
    sm_scale = wl["inputs"]["sm_scale"]["value"]
    sparse_indices_path = f"/home/vedant.tejas/mlsys26-contest/{wl['inputs']['sparse_indices']['path'][2:]}"
    sparse_indices = load_file(sparse_indices_path)["sparse_indices"].to(device)

    torch.manual_seed(0)
    q_nope = torch.randn(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Correctness: v1 and v3 vs original kernel()
    out_orig, lse_orig = kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    out_v1, lse_v1 = kernel_splitk(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K)
    out_v3, lse_v3 = kernel_splitk_v3(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K)

    err_v1 = (out_orig.float() - out_v1.float()).abs().max().item()
    err_v3 = (out_orig.float() - out_v3.float()).abs().max().item()

    # Performance: wall-clock for each
    def time_fn(fn, *args, **kwargs):
        for _ in range(10):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 50 * 1000  # us

    t_orig = time_fn(kernel, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    t_v1 = time_fn(kernel_splitk, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K)
    t_v3 = time_fn(kernel_splitk_v3, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K)

    results.append((num_tokens, err_v1, err_v3, t_orig, t_v1, t_v3))
    print(f"num_tokens={num_tokens}  err_v1={err_v1:.6f}  err_v3={err_v3:.6f}  "
          f"orig={t_orig:.2f}us  v1={t_v1:.2f}us  v3={t_v3:.2f}us  "
          f"v1_speedup={t_orig/t_v1:.2f}x  v3_speedup={t_orig/t_v3:.2f}x")

print("\n=== Summary ===")
for r in results:
    print(r)
