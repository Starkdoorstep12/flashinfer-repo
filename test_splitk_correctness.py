import torch, math
from safetensors.torch import load_file
from solution.triton.kernel import kernel, kernel_splitk

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

for split_k in [4, 8, 16]:
    out_split, lse_split = kernel_splitk(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=split_k)
    abs_err = (out_orig.float() - out_split.float()).abs().max().item()
    lse_err = (lse_orig - lse_split).abs().max().item()
    print(f"SPLIT_K={split_k:3d}  max_abs_err(output)={abs_err:.6f}  max_abs_err(lse)={lse_err:.6f}")
