"""
Stress test for kernel_splitk_v3's atomic-counter synchronization gate.
A single passing run is not sufficient evidence of correctness for a
design using cross-block atomics as a synchronization primitive — races
can pass most runs and fail rarely due to GPU scheduling nondeterminism.
This runs many repeated trials per SPLIT_K, with fresh random inputs and
real sparse_indices data each time, and reports any failures.
"""
import torch
from safetensors.torch import load_file
from solution.triton.kernel import kernel, kernel_splitk_v3

device = 'cuda'
num_tokens, num_pages, page_size = 1, 8462, 64
num_qo_heads, head_dim_ckv, head_dim_kpe = 16, 512, 64
sm_scale = 0.1352337788608801

path = "/home/vedant.tejas/mlsys26-contest/blob/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64_0c23b10c7b7645719517828c12eaa1d2.safetensors"
sparse_indices_base = load_file(path)["sparse_indices"].to(device)

N_TRIALS = 200
TOLERANCE = 0.02  # matches bf16 rounding scale seen throughout this project

split_ks = [2, 4, 8, 16, 32, 64, 128]
failures = {sk: [] for sk in split_ks}

for trial in range(N_TRIALS):
    torch.manual_seed(trial)  # different data each trial

    q_nope = torch.randn(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    out_orig, lse_orig = kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices_base, sm_scale)

    for sk in split_ks:
        out_v3, lse_v3 = kernel_splitk_v3(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices_base, sm_scale, SPLIT_K=sk)
        abs_err = (out_orig.float() - out_v3.float()).abs().max().item()
        if abs_err > TOLERANCE:
            failures[sk].append((trial, abs_err))

    if (trial + 1) % 20 == 0:
        print(f"Completed {trial + 1}/{N_TRIALS} trials...")

print("\n=== Stress test results ===")
total_failures = 0
for sk in split_ks:
    n_fail = len(failures[sk])
    total_failures += n_fail
    status = "PASS" if n_fail == 0 else f"FAIL ({n_fail} failures)"
    print(f"SPLIT_K={sk:4d}: {status}")
    if n_fail > 0:
        for trial, err in failures[sk][:5]:
            print(f"    trial={trial}  max_abs_err={err:.6f}")

print(f"\nTotal trials: {N_TRIALS * len(split_ks)}, Total failures: {total_failures}")
if total_failures == 0:
    print("All trials passed. No race condition detected across", N_TRIALS, "trials per SPLIT_K.")
else:
    print("RACE CONDITION LIKELY PRESENT — do not trust kernel_splitk_v3 until fixed.")
