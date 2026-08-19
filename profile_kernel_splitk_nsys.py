import torch, math
from safetensors.torch import load_file
from solution.triton.kernel import kernel_splitk

device = 'cuda'
torch.manual_seed(0)
num_tokens, num_pages, page_size = 1, 8462, 64
num_qo_heads, head_dim_ckv, head_dim_kpe = 16, 512, 64
sm_scale = 0.1352337788608801
SPLIT_K = 8

path = "/home/vedant.tejas/mlsys26-contest/blob/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_0c23b10c7b7645719517828c12eaa1d2.safetensors"
sparse_indices = load_file(path)["sparse_indices"].to(device)

q_nope = torch.randn(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

for _ in range(10):
    output, lse = kernel_splitk(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    output, lse = kernel_splitk(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K)
end.record()
torch.cuda.synchronize()
print(f"Avg latency: {start.elapsed_time(end)/50:.4f} ms")
