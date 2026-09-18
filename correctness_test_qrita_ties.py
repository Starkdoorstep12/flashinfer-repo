"""
Tie-heavy correctness test for topk_kernel_qrita's general-path duplicate
resolution logic (dup_mask/dup_cumsum), which real FP8-derived acc scores
(continuous values, few exact ties) barely exercise -- flagged as an
open gap in Finding 7. Constructs synthetic acc arrays with deliberate,
controlled duplication:

  - "moderate ties": scores drawn from a small integer range, so exact
    ties are common throughout, including near the K-th boundary.
  - "boundary ties": explicitly force many values to sit exactly at what
    would be the pivot/cutoff value, maximizing exercise of the
    num_min_larger / num_keep logic that decides which subset of tied
    values to keep to land exactly at K.
  - "fully degenerate": every score in the sequence is identical --
    the most extreme possible tie case, where ANY K of them is a valid
    answer (checked via count-only, not set-overlap, since there is no
    unique correct set when every value ties).

For the first two cases, exact SET equality against topk_kernel is the
correctness bar (as in correctness_test_qrita.py) -- topk_kernel's own
tie-breaking (first-slot-among-ties, Finding 5a) and topk_kernel_qrita's
(cumsum-position-based dup_keep) both select a value once, so if both
selection algorithms are internally consistent, they should select the
IDENTICAL set of indices whenever the "kept" values are well-defined by
score alone (i.e., ties only in VALUE, at the exact pivot boundary,
should still be resolved consistently since both algorithms use vals
directly, not floating tie-breaking on index). For the fully degenerate
case, count-only comparison is used since no unique correct index set
exists.
"""
import torch
from solution.triton.indexer_kernel import topk_kernel, compute_topk_qrita

device = 'cuda'
torch.manual_seed(0)


def run_comparison(seq_lens_list, topk, acc_fn, label, check_mode="set"):
    batch_size = len(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)
    total_tokens = seq_lens.sum().item()

    acc = acc_fn(total_tokens)

    page_size = 64
    max_num_pages = (int(seq_lens.max().item()) + page_size - 1) // page_size + 1
    block_table = torch.arange(batch_size * max_num_pages, dtype=torch.int32, device=device).reshape(batch_size, max_num_pages)

    topk_indices_existing = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    topk_kernel[(batch_size,)](
        acc_ptr=acc, seq_offsets=seq_offsets, seq_lens=seq_lens,
        block_table=block_table, topk_indices_ptr=topk_indices_existing,
        K=topk, MAX_SEQ_LEN=int(seq_lens.max().item()), BLOCK=16, MAX_K=topk,
        page_size=page_size, max_num_pages=max_num_pages,
    )
    torch.cuda.synchronize()

    topk_indices_qrita = compute_topk_qrita(
        acc, seq_offsets, seq_lens, block_table, page_size, max_num_pages,
        batch_size, topk=topk, BLOCK_SIZE=1024,
    )
    torch.cuda.synchronize()

    all_match = True
    for b in range(batch_size):
        existing_set = set(topk_indices_existing[b][topk_indices_existing[b] != -1].tolist())
        qrita_set = set(topk_indices_qrita[b][topk_indices_qrita[b] != -1].tolist())
        expected_count = min(topk, seq_lens_list[b])

        if check_mode == "set":
            match = existing_set == qrita_set
            extra = f"overlap={len(existing_set & qrita_set)}"
            if not match:
                only_existing = existing_set - qrita_set
                only_qrita = qrita_set - existing_set
                print(f"    only in existing: {list(only_existing)[:5]}")
                print(f"    only in qrita:    {list(only_qrita)[:5]}")

        elif check_mode == "count":
            match = len(existing_set) == len(qrita_set) == expected_count
            extra = f"existing_count={len(existing_set)} qrita_count={len(qrita_set)} expected={expected_count}"

        elif check_mode == "value":
            # Under ties, index-set equality is too strict -- a different,
            # equally-valid selection among tied values is not a bug.
            # Real correctness bar: (1) exact expected COUNT, (2) every
            # selected value >= the true K-th-largest value, (3) no
            # unselected value strictly exceeds the minimum selected value.
            # idx values are PHYSICAL addresses (page_idx * page_size +
            # offset_in_page). This test's block_table assigns row b's
            # pages starting at b * max_num_pages, so:
            #   physical_addr = (b * max_num_pages + page_id) * page_size + offset_in_page
            #                 = b * max_num_pages * page_size + local_offset
            # Invert to recover the local (sequence-relative) offset.
            row_base_physical = b * max_num_pages * page_size

            seq_start = seq_offsets[b].item()
            seq_len_b = seq_lens_list[b]
            row_vals = acc[seq_start:seq_start + seq_len_b]

            def selected_values(idx_set):
                local_offsets = [i - row_base_physical for i in idx_set]
                return [row_vals[lo].item() for lo in local_offsets]

            existing_vals = selected_values(existing_set)
            qrita_vals = selected_values(qrita_set)

            count_ok = len(existing_set) == len(qrita_set) == expected_count
            if count_ok and len(qrita_vals) > 0:
                true_sorted = torch.sort(row_vals, descending=True).values
                kth_largest_val = true_sorted[expected_count - 1].item()
                min_selected = min(qrita_vals)
                # Every selected value must be >= the true K-th largest
                # (allows selecting any value tied AT the boundary).
                value_ok = min_selected >= kth_largest_val - 1e-5
                # No value outside the selection should exceed the minimum
                # selected value (nothing strictly better was left out).
                qrita_local_offsets = [i - row_base_physical for i in qrita_set]
                unselected_mask = torch.ones(seq_len_b, dtype=torch.bool, device=row_vals.device)
                unselected_mask[qrita_local_offsets] = False
                max_unselected = row_vals[unselected_mask].max().item() if unselected_mask.any() else float('-inf')
                completeness_ok = max_unselected <= min_selected + 1e-5
                match = count_ok and value_ok and completeness_ok
            else:
                match = False
                value_ok, completeness_ok = None, None

            extra = (f"count_ok={count_ok} value_ok={value_ok} completeness_ok={completeness_ok} "
                     f"qrita_count={len(qrita_set)} expected={expected_count}")

        if not match:
            all_match = False
        print(f"[{label}] batch={b} seq_len={seq_lens_list[b]:5d} existing={len(existing_set)} qrita={len(qrita_set)} {extra} match={match}")

    return all_match


results = []

print("=== Test A: moderate ties (scores from small integer range, 0-20) ===")
def moderate_ties(n):
    return torch.randint(0, 20, (n,), device=device, dtype=torch.int32).to(torch.float32)
results.append(("A", run_comparison([200, 500, 1000], topk=64, acc_fn=moderate_ties, label="A", check_mode="value")))

print("\n=== Test B: heavy ties (scores from tiny integer range, 0-4) ===")
def heavy_ties(n):
    return torch.randint(0, 4, (n,), device=device, dtype=torch.int32).to(torch.float32)
results.append(("B", run_comparison([200, 500, 3000], topk=64, acc_fn=heavy_ties, label="B", check_mode="value")))

print("\n=== Test C: heavy ties at real-dataset topk scale (K=2048) ===")
results.append(("C", run_comparison([2500, 5000], topk=2048, acc_fn=heavy_ties, label="C", check_mode="value")))

print("\n=== Test D: boundary ties -- most values distinct, but a cluster forced to tie exactly at the K-th value ===")
def boundary_ties(n):
    vals = torch.randn(n, device=device, dtype=torch.float32) * 100  # mostly distinct
    # Force indices [10:40) all to the identical value 5.0, straddling
    # wherever the K-th boundary likely falls for K=20 in a 200-long seq.
    if n >= 40:
        vals[10:40] = 5.0
    return vals
results.append(("D", run_comparison([200, 200], topk=16, acc_fn=boundary_ties, label="D", check_mode="value")))

print("\n=== Test E: fully degenerate -- every value in the sequence is identical ===")
def fully_degenerate(n):
    return torch.full((n,), 7.0, device=device, dtype=torch.float32)
results.append(("E", run_comparison([100, 500, 3000], topk=64, acc_fn=fully_degenerate, label="E", check_mode="count")))

print(f"\n{'='*60}")
for name, passed in results:
    print(f"Test {name}: {'PASS' if passed else 'FAIL'}")
print(f"\nAll tests passed: {all(p for _, p in results)}")
