"""
Driver script: builds the SAME test case as flashinfer-repo/correctness_test_indexer.py
(same seed, same shapes: batch_size=2, seq_lens=[50, 80]) so results are directly comparable
to the RTX 6000 correctness run -- just on CPU (no CUDA on this machine) and pointing at
C:\golden_ref instead of /tmp.

Does not modify correctness_test_indexer.py or golden_indexer_reference.py.
"""

import sys
import torch

sys.path.insert(0, r'C:\golden_ref')
sys.path.insert(0, r'C:\flashinfer-repo\npu')

from onnx_export_indexer import validate_against_golden, export_to_onnx  # noqa: E402

device = 'cpu'
torch.manual_seed(0)

batch_size = 2
num_index_heads = 64
index_head_dim = 128
page_size = 64
seq_lens_list = [50, 80]
max_seq_len = max(seq_lens_list)
max_num_pages = (max_seq_len + page_size - 1) // page_size
num_pages = max_num_pages * batch_size + 2
head_dim_with_scale = 132

seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
block_table = torch.zeros((batch_size, max_num_pages), dtype=torch.int32, device=device)
for b in range(batch_size):
    npages = (seq_lens_list[b] + page_size - 1) // page_size
    block_table[b, :npages] = torch.arange(b * max_num_pages, b * max_num_pages + npages, device=device)

q_index_fp32 = torch.randn(batch_size, num_index_heads, index_head_dim, device=device) * 2.0
q_index_fp8 = q_index_fp32.to(torch.float8_e4m3fn)

k_fp32 = torch.randn(num_pages, page_size, index_head_dim, device=device) * 2.0
k_fp8 = k_fp32.to(torch.float8_e4m3fn)
scales = torch.rand(num_pages, page_size, device=device) * 0.5 + 0.5

k_index_cache_fp8 = torch.zeros((num_pages, page_size, 1, head_dim_with_scale), dtype=torch.uint8, device=device)
k_index_cache_fp8_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_with_scale)
fp8_bytes = k_fp8.view(torch.uint8).view(num_pages, page_size * index_head_dim)
k_index_cache_fp8_flat[:, :page_size * index_head_dim] = fp8_bytes
scale_bytes = scales.to(torch.float32).view(torch.uint8).view(num_pages, page_size * 4)
k_index_cache_fp8_flat[:, page_size * index_head_dim:] = scale_bytes

weights = torch.rand(batch_size, num_index_heads, device=device)

print("=== Step 1: validate static-shape rewrite against golden_run() ===")
q, K_padded, w, valid_mask, token_ids, model = validate_against_golden(
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, max_seq_len=max(max_seq_len, 2048)
)
print("Validation PASSED -- static-shape module matches golden reference exactly.")

print("")
print("=== Step 2: export to ONNX ===")
export_to_onnx(model, q, K_padded, w, valid_mask, token_ids, out_path=r"C:\flashinfer-repo\npu\indexer.onnx")
