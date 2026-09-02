"""
Closing check on bottleneck 2: does forcing BLOCK_N=16 (best occupancy,
37568 bytes shared mem) across the FULL SPLIT_K range change the
total-SM-utilization picture, now that BLOCK_N isn't forced to shrink
further at high SPLIT_K (it's already at its floor)? Tests whether the
SPLIT_K-vs-reduction-cost tradeoff looks any different with occupancy
maximized throughout, versus the default (SPLIT_K-derived) BLOCK_N.
"""
import torch
from safetensors.torch import load_file
from solution.triton.kernel import kernel, kernel_splitk_v3

device = 'cuda'
torch.manual_seed(0)
num_tokens, num_pages, page_size = 1, 8462, 64
num_qo_heads, head_dim_ckv, head_dim_kpe = 16, 512, 64
sm_scale = 0.1352337788608801

path = "/home/vedant.tejas/mlsys26-contest/blob/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_0c23b10c7b7645719517828c12eaa1d2.safetensors"
sparse_indices = load_file(path)["sparse_indices"].to(device)

q_nope = torch.randn(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

out_orig, lse_orig = kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

def time_fn(fn, *a, **kw):
    for _ in range(10): fn(*a, **kw)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50): fn(*a, **kw)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/50*1000

print(f"{'SPLIT_K':>8} {'BLOCK_N=default':>16} {'BLOCK_N=16 (forced)':>20} {'err (forced)':>14}")
for sk in [2, 4, 8, 16, 32, 64, 128]:
    out_def, _ = kernel_splitk_v3(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=sk, BLOCK_N=None)
    t_def = time_fn(kernel_splitk_v3, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=sk, BLOCK_N=None)

    out_16, _ = kernel_splitk_v3(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=sk, BLOCK_N=16)
    t_16 = time_fn(kernel_splitk_v3, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=sk, BLOCK_N=16)

    err = (out_orig.float() - out_16.float()).abs().max().item()
    print(f"{sk:>8} {t_def:>15.2f}us {t_16:>19.2f}us {err:>14.6f}")
