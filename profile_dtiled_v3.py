import torch, sys
from safetensors.torch import load_file
from solution.triton.kernel import kernel_splitk_dtiled_v3, kernel_splitk

device = 'cuda'
torch.manual_seed(0)
num_tokens, num_pages, page_size = 1, 8462, 64
num_qo_heads, head_dim_ckv, head_dim_kpe = 16, 512, 64
sm_scale = 0.1352337788608801
SPLIT_K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
D_TILES = int(sys.argv[2]) if len(sys.argv) > 2 else 2

path = "/home/vedant.tejas/mlsys26-contest/blob/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_0c23b10c7b7645719517828c12eaa1d2.safetensors"
sparse_indices = load_file(path)["sparse_indices"].to(device)

q_nope = torch.randn(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

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
    return start.elapsed_time(end) / 50 * 1000

t_v1 = time_fn(kernel_splitk, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K)
t_dtiled = time_fn(kernel_splitk_dtiled_v3, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, SPLIT_K=SPLIT_K, D_TILES=D_TILES)

print(f"SPLIT_K={SPLIT_K} D_TILES={D_TILES}  v1={t_v1:.2f}us  dtiled_v3={t_dtiled:.2f}us  ratio={t_dtiled/t_v1:.2f}x")
