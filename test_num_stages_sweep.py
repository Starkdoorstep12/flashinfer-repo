"""
Sweep num_stages on dsa_fwd_kernel_splitk_v3 to see if reducing pipeline
buffering lowers shared memory usage (currently 94,976 bytes, unchanged
from the original bottleneck-2 diagnosis), and whether that helps or hurts
wall-clock latency. Uses direct kernel launch (bypassing kernel_splitk_v3's
wrapper) so we can pass num_stages explicitly.
"""
import torch
from safetensors.torch import load_file
from solution.triton.kernel import dsa_fwd_kernel_splitk_v3, dsa_reduce_phaseA_kernel

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

output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
lse = torch.full((num_tokens, num_qo_heads), fill_value=-float("inf"), dtype=torch.float32, device=device)
partial_acc = torch.zeros((num_tokens, SPLIT_K, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
partial_m = torch.full((num_tokens, SPLIT_K, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
partial_l = torch.zeros((num_tokens, SPLIT_K, num_qo_heads), dtype=torch.float32, device=device)
counter = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

chunk_size = topk // SPLIT_K
BLOCK_N = min(64, max(16, chunk_size // 2))
grid = (num_tokens, SPLIT_K)

def launch(num_stages):
    counter.zero_()
    dsa_fwd_kernel_splitk_v3[grid](
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
        page_size=page_size, topk=topk,
        SPLIT_K=SPLIT_K, BLOCK_N=BLOCK_N,
        BLOCK_D_CKV=head_dim_ckv, BLOCK_D_KPE=head_dim_kpe, BLOCK_H=num_qo_heads,
        num_stages=num_stages,
    )

for ns in [1, 2, 3, 4]:
    try:
        for _ in range(10):
            launch(ns)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            launch(ns)
        end.record()
        torch.cuda.synchronize()
        print(f"num_stages={ns}  Avg latency: {start.elapsed_time(end)/50*1000:.2f} us")
    except Exception as e:
        print(f"num_stages={ns}  FAILED: {type(e).__name__}: {str(e)[:150]}")
