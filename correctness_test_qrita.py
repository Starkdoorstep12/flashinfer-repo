"""
Correctness test: compare topk_kernel_qrita's selection against the
existing, already golden-reference-verified topk_kernel, using real
indexer scores (via indexer_kernel) on synthetic-but-realistic data.
Top-k SETS should match exactly (order doesn't matter).
"""
import torch
from solution.triton.indexer_kernel import indexer_kernel, topk_kernel, compute_topk_qrita

device = 'cuda'
torch.manual_seed(0)

def run_comparison(seq_lens_list, topk, label):
    batch_size = len(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)
    total_tokens = seq_lens.sum().item()

    # Random but realistic-scale scores (not actually running the real
    # indexer_kernel here -- just testing topk selection logic directly
    # on a random score array, same as debug_indexer_scores.py did)
    acc = torch.randn(total_tokens, device=device, dtype=torch.float32) * 10

    # Existing (verified correct) topk_kernel -- needs block_table,
    # page_size, max_num_pages since Finding 5's physical/logical index
    # fix. Not testing real page mapping here (pure top-k selection logic
    # comparison), so use a trivial 1:1 block_table (page i -> physical
    # page i) so physical == logical for this comparison.
    page_size = 64
    max_num_pages = (int(seq_lens.max().item()) + page_size - 1) // page_size + 1
    block_table = torch.arange(batch_size * max_num_pages, dtype=torch.int32, device=device).reshape(batch_size, max_num_pages)

    topk_indices_existing = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    topk_grid = (batch_size,)
    topk_kernel[topk_grid](
        acc_ptr=acc, seq_offsets=seq_offsets, seq_lens=seq_lens,
        block_table=block_table, topk_indices_ptr=topk_indices_existing,
        K=topk, MAX_SEQ_LEN=int(seq_lens.max().item()), BLOCK=16, MAX_K=topk,
        page_size=page_size, max_num_pages=max_num_pages,
    )
    torch.cuda.synchronize()

    # New Qrita-inspired kernel
    topk_indices_qrita = compute_topk_qrita(
        acc, seq_offsets, seq_lens, block_table, page_size, max_num_pages,
        batch_size, topk=topk, BLOCK_SIZE=1024,
    )
    torch.cuda.synchronize()

    all_match = True
    for b in range(batch_size):
        existing_set = set(topk_indices_existing[b][topk_indices_existing[b] != -1].tolist())
        qrita_set = set(topk_indices_qrita[b][topk_indices_qrita[b] != -1].tolist())
        match = existing_set == qrita_set
        if not match:
            all_match = False
        print(f"[{label}] batch={b} seq_len={seq_lens_list[b]:5d} existing={len(existing_set)} qrita={len(qrita_set)} overlap={len(existing_set & qrita_set)} match={match}")

        if not match:
            # Check if it's a near-miss (values very close to the pivot boundary, could be float precision)
            only_existing = existing_set - qrita_set
            only_qrita = qrita_set - existing_set
            print(f"    only in existing: {list(only_existing)[:5]}")
            print(f"    only in qrita:    {list(only_qrita)[:5]}")
    return all_match

print("=== Test 1: small batch, seq_len > topk (general path) ===")
r1 = run_comparison([100, 150, 500], topk=64, label="T1")

print("\n=== Test 2: seq_len <= topk (fast path) ===")
r2 = run_comparison([10, 5, 2], topk=64, label="T2")

print("\n=== Test 3: mixed (some fast path, some general path) ===")
r3 = run_comparison([10, 500, 2048, 3000], topk=2048, label="T3")

print("\n=== Test 4: real dataset scale ===")
r4 = run_comparison([2, 65, 5824, 91], topk=2048, label="T4")

print(f"\nAll tests passed: {r1 and r2 and r3 and r4}")
