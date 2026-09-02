"""
Sweep num_warps on kernel_splitk_v4 (the final Ada kernel) -- untried
parameter, never varied from the default (4) throughout this
investigation. Tests correctness and wall-clock performance at each value.
"""
import torch
from safetensors.torch import load_file
from solution.triton.kernel import kernel, dsa_fwd_kernel_splitk_v4

device = 'cuda'
torch.manual_seed(0)
num_tokens, num_pages, page_size = 1, 8462, 64
num_qo_heads, head_dim_ckv, head_dim_kpe = 16, 512, 64
sm_scale = 0.1352337788608801
SPLIT_K = 16
topk = 2048

path = "/home/vedant.tejas/mlsys26-contest/blob/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_0c23b10c7b7645719517828c12eaa1d2.safetensors"
sparse_indices = load_file(path)["sparse_indices"].to(device)

q_nope = torch.randn(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

out_orig, lse_orig = kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

chunk_size = topk // SPLIT_K
BLOCK_N = min(64, max(16, chunk_size // 2))
grid = (num_tokens, SPLIT_K)

def run_with_num_warps(nw):
    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    partial_acc = torch.zeros((num_tokens, SPLIT_K, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    partial_m = torch.full((num_tokens, SPLIT_K, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
    partial_l = torch.zeros((num_tokens, SPLIT_K, num_qo_heads), dtype=torch.float32, device=device)
    counter = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

    dsa_fwd_kernel_splitk_v4[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        partial_acc, partial_m, partial_l, counter, output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(1), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(1), kpe_cache.stride(2),
        sparse_indices.stride(0), sparse_indices.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        page_size=page_size, topk=topk, SPLIT_K=SPLIT_K, BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv, BLOCK_D_KPE=head_dim_kpe, BLOCK_H=num_qo_heads,
        num_warps=nw,
    )
    return output, lse

def time_fn(nw):
    for _ in range(10): run_with_num_warps(nw)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50): run_with_num_warps(nw)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/50*1000

for nw in [1, 2, 4, 8, 16]:
    try:
        out, lse = run_with_num_warps(nw)
        err = (out_orig.float() - out.float()).abs().max().item()
        t = time_fn(nw)
        print(f"num_warps={nw:3d}  err={err:.6f}  latency={t:.2f}us")
    except Exception as ex:
        print(f"num_warps={nw:3d}  FAILED: {type(ex).__name__}: {str(ex)[:150]}")
