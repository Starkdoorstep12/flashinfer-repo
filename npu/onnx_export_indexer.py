"""
ONNX-exportable, static-shape version of the top-k indexer for NPU benchmarking.

Scope and safety notes:
- Does NOT modify golden_indexer_reference.py or any file in solution/.
- Reuses dequant_fp8_kv_cache() from the golden reference UNCHANGED (imported, not reimplemented) --
  the verified FP8 unpacking math is untouched.
- Only the outer orchestration (per-batch Python loop, data-dependent top-k) is rewritten into a
  static-shape, masked/vectorized form, because ONNX export requires fixed shapes and cannot trace
  Python-level control flow that varies with runtime values (same class of constraint as the
  Triton kernel's power-of-2 padding fix for batch_size).
- Scope: exports the scoring + top-k selection step (the compute-heavy, benchmark-relevant part),
  matching what the Triton kernel actually does. The page-gather/dequantization step is done once
  as plain-PyTorch data prep before tracing, since it is fixed-shape per test case and not itself
  the part under NPU benchmarking.
- Validated against the ORIGINAL golden_run() on identical input before export (see
  validate_against_golden() below) -- if this check fails, STOP, do not proceed to export.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, r'C:\golden_ref')
from golden_indexer_reference import run as golden_run, dequant_fp8_kv_cache  # noqa: E402

TOPK = 2048
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64


class StaticShapeIndexer(nn.Module):
    def __init__(self, topk: int = TOPK):
        super().__init__()
        self.topk = topk

    def forward(self, q, K_padded, weights, valid_mask, token_ids):
        scores = torch.matmul(q, K_padded.transpose(1, 2))
        scores = torch.relu(scores)

        weighted = scores * weights.unsqueeze(-1)
        final_scores = weighted.sum(dim=1)

        neg_inf = torch.finfo(final_scores.dtype).min
        final_scores = torch.where(valid_mask, final_scores, torch.full_like(final_scores, neg_inf))

        topk_scores, topk_idx = torch.topk(final_scores, self.topk, dim=1)

        topk_tokens = torch.gather(token_ids, 1, topk_idx)

        was_padding = topk_scores <= (neg_inf / 2)
        topk_tokens = torch.where(was_padding, torch.full_like(topk_tokens, -1), topk_tokens)

        return topk_tokens.to(torch.int32)


def build_static_inputs_from_golden_style(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table,
                                           max_seq_len: int):
    batch_size, num_heads, head_dim = q_index_fp8.shape
    device = q_index_fp8.device
    page_size = k_index_cache_fp8.shape[1]

    q = q_index_fp8.to(torch.float32)
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)

    K_padded = torch.zeros((batch_size, max_seq_len, head_dim), dtype=torch.float32, device=device)
    token_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.int32, device=device)
    valid_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]
        K_seq = K_paged.reshape(-1, head_dim)[:seq_len]

        offsets = torch.arange(seq_len, device=device)
        page_idx_per_token = offsets // page_size
        offset_per_token = offsets % page_size
        global_page_idx = page_indices[page_idx_per_token]
        tok_ids = (global_page_idx * page_size + offset_per_token).to(torch.int32)

        K_padded[b, :seq_len] = K_seq
        token_ids[b, :seq_len] = tok_ids
        valid_mask[b, :seq_len] = True

    return q, K_padded, weights, valid_mask, token_ids


@torch.no_grad()
def validate_against_golden(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, max_seq_len):
    golden_topk_indices, = golden_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)

    q, K_padded, w, valid_mask, token_ids = build_static_inputs_from_golden_style(
        q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, max_seq_len
    )
    model = StaticShapeIndexer(topk=TOPK)
    ours_topk_indices = model(q, K_padded, w, valid_mask, token_ids)

    batch_size = q_index_fp8.shape[0]
    for b in range(batch_size):
        golden_set = set(golden_topk_indices[b][golden_topk_indices[b] != -1].tolist())
        our_set = set(ours_topk_indices[b][ours_topk_indices[b] != -1].tolist())
        assert golden_set == our_set, (
            f"Batch {b}: MISMATCH -- golden-only={len(golden_set - our_set)}, "
            f"ours-only={len(our_set - golden_set)}"
        )
        print(f"Batch {b}: OK ({len(golden_set)} tokens match exactly)")

    return q, K_padded, w, valid_mask, token_ids, model


def export_to_onnx(model, q, K_padded, weights, valid_mask, token_ids, out_path):
    torch.onnx.export(
        model,
        (q, K_padded, weights, valid_mask, token_ids),
        out_path,
        input_names=["q", "K_padded", "weights", "valid_mask", "token_ids"],
        output_names=["topk_indices"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported to {out_path}")


if __name__ == "__main__":
    print("Import this module and call validate_against_golden(...) then export_to_onnx(...).")
    print("See correctness_test_indexer.py for how it builds q_index_fp8/k_index_cache_fp8/etc.")
