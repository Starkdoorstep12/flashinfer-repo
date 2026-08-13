import torch, math
from safetensors.torch import load_file
from solution.triton.kernel import kernel as dsa_kernel

device = 'cuda'
torch.manual_seed(0)

# Real workload shapes from trace uuid=0c23b10c...
num_tokens = 1
num_pages = 8462
page_size = 64
num_qo_heads = 16
head_dim_ckv = 512
head_dim_kpe = 64
sm_scale = 0.1352337788608801

# Load real sparse_indices from the dataset
sparse_indices_path = "/home/vedant.tejas/mlsys26-contest/blob/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_0c23b10c7b7645719517828c12eaa1d2.safetensors"
tensors = load_file(sparse_indices_path)
print("Keys in safetensors:", list(tensors.keys()))
sparse_indices = tensors["sparse_indices"].to(device)
print("sparse_indices shape:", sparse_indices.shape, sparse_indices.dtype)

q_nope = torch.randn(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

# Warmup
for _ in range(10):
    output, lse = dsa_kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
torch.cuda.synchronize()

# Timed section (this is what nsys will capture)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    output, lse = dsa_kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
end.record()
torch.cuda.synchronize()

print(f"Avg latency: {start.elapsed_time(end)/50:.4f} ms")
print("output:", output.shape, output.dtype)
print("lse:", lse.shape, lse.dtype)
