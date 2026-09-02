"""
Isolate Finding 3 (topk_kernel's O(seq_len x K) serial scan) from Finding 2
(compile-time cost). Uses the SAME batch_size across short and long
sequence tests so the compile happens once (shared, warm after first
call) and only the effect of seq_len/MAX_SEQ_LEN on runtime is measured.
"""
import torch, time
from solution.triton.indexer_kernel import run_indexer_and_topk

device = 'cuda'
torch.manual_seed(0)

num_index_heads = 64
index_head_dim = 128
page_size = 64
kv_cache_num_heads = 1
head_dim_with_scale = 132
topk = 2048
BLOCK_TOKENS = 32
BLOCK_HEADS = 8

def run_case(seq_len_value, label):
    batch_size = 4  # fixed across both cases
    seq_lens = torch.full((batch_size,), seq_len_value, dtype=torch.int32, device=device)
    seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)

    max_num_pages = (seq_len_value + page_size - 1) // page_size
    num_pages = max_num_pages * batch_size + 10

    block_table = torch.randint(0, num_pages, (batch_size, max_num_pages), dtype=torch.int32, device=device)
    q_index_fp8 = torch.randint(-128, 127, (batch_size, num_index_heads, index_head_dim), dtype=torch.int8, device=device)
    k_index_cache_fp8 = torch.randint(-128, 127, (num_pages * page_size * head_dim_with_scale,), dtype=torch.int8, device=device)
    weights = torch.rand(batch_size, num_index_heads, device=device)

    def run():
        return run_indexer_and_topk(
            q_index_fp8=q_index_fp8, k_index_cache_fp8=k_index_cache_fp8, weights=weights,
            seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
            batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
            page_size=page_size, kv_cache_num_heads=kv_cache_num_heads,
            head_dim_with_scale=head_dim_with_scale, max_num_pages=max_num_pages, topk=min(topk, seq_len_value),
            BLOCK_TOKENS=BLOCK_TOKENS, BLOCK_HEADS=BLOCK_HEADS,
        )

    print(f"[{label}] seq_len={seq_len_value}, first call (compiles)...")
    t0 = time.time()
    out = run()
    torch.cuda.synchronize()
    t_first = time.time() - t0
    print(f"[{label}]   first call: {t_first:.3f}s")

    print(f"[{label}] warmed-up timing (20 calls)...")
    t0 = time.time()
    for _ in range(20):
        out = run()
    torch.cuda.synchronize()
    t_avg = (time.time() - t0) / 20 * 1000
    print(f"[{label}]   avg per call (warm): {t_avg:.3f} ms")
    return t_first, t_avg

print("=" * 60)
t_first_short, t_avg_short = run_case(128, "SHORT (seq_len=128)")
print("=" * 60)
t_first_long, t_avg_long = run_case(5824, "LONG (seq_len=5824, matches real dataset max)")
print("=" * 60)
print(f"\nSummary:")
print(f"  SHORT: first={t_first_short:.3f}s  warm_avg={t_avg_short:.3f}ms")
print(f"  LONG:  first={t_first_long:.3f}s  warm_avg={t_avg_long:.3f}ms")
print(f"  Warm-runtime ratio (long/short): {t_avg_long/t_avg_short:.2f}x")
