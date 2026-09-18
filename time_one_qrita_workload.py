"""
Runs correctness + timing for ONE workload in an isolated subprocess,
following measure_bucketing_isolated.py's architecture (avoids the
documented intermittent CUDA context corruption from multiple distinct
compiled shapes coexisting in one process).
"""
import torch, json, sys
from safetensors.torch import load_file
from solution.triton.indexer_kernel import indexer_kernel, topk_kernel, compute_topk_qrita

device = 'cuda'
num_index_heads = 64
index_head_dim = 128
page_size = 64
head_dim_with_scale = 132
topk = 2048

uuid_prefix = sys.argv[1]

trace_path = "/home/vedant.tejas/mlsys26-contest/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
with open(trace_path) as f:
    workloads = [json.loads(line)['workload'] for line in f]
wl = next(w for w in workloads if w['uuid'].startswith(uuid_prefix))

uuid = wl['uuid']
batch_size = wl['axes']['batch_size']
max_num_pages = wl['axes']['max_num_pages']
path = f"/home/vedant.tejas/mlsys26-contest/{wl['inputs']['seq_lens']['path']}"
tensors = load_file(path)
seq_lens = tensors['seq_lens'].to(device)
block_table = tensors['block_table'].to(device)
num_pages = int(block_table.max().item()) + 5

seq_offsets = torch.cat([torch.tensor([0], device=device), seq_lens.cumsum(0)[:-1]]).to(torch.int32)
total_tokens = (seq_offsets[-1] + seq_lens[-1]).item()

torch.manual_seed(hash(uuid) % (2**31))
q_index_fp32 = torch.randn(batch_size, num_index_heads, index_head_dim, device=device) * 2.0
q_index_fp8 = q_index_fp32.to(torch.float8_e4m3fn)
k_fp32 = torch.randn(num_pages, page_size, index_head_dim, device=device) * 2.0
k_fp8 = k_fp32.to(torch.float8_e4m3fn)
scales = torch.rand(num_pages, page_size, device=device) * 0.5 + 0.5
k_index_cache_fp8 = torch.zeros((num_pages, page_size, 1, head_dim_with_scale), dtype=torch.uint8, device=device)
k_index_cache_fp8_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_with_scale)
fp8_bytes = k_fp8.view(torch.uint8).view(num_pages, page_size * index_head_dim)
k_index_cache_fp8_flat[:, :page_size * index_head_dim] = fp8_bytes
scale_bytes_t = scales.to(torch.float32).view(torch.uint8).view(num_pages, page_size * 4)
k_index_cache_fp8_flat[:, page_size * index_head_dim:] = scale_bytes_t
weights = torch.rand(batch_size, num_index_heads, device=device)
k_index_cache_fp8_kernel = k_index_cache_fp8.view(torch.int8).reshape(-1)

tiles_per_seq = (seq_lens + 32 - 1) // 32
tile_offsets = torch.cumsum(tiles_per_seq, dim=0)
total_tiles = tile_offsets[-1].item()
acc = torch.zeros(total_tokens, device=device, dtype=torch.float32)
indexer_kernel[(total_tiles,)](
    q_index_fp8=q_index_fp8.view(torch.int8), k_index_cache_fp8=k_index_cache_fp8_kernel,
    weights=weights, seq_lens=seq_lens, block_table=block_table, seq_offsets=seq_offsets,
    tile_offsets_ptr=tile_offsets, acc_ptr=acc,
    batch_size=batch_size, num_index_heads=num_index_heads, index_head_dim=index_head_dim,
    page_size=page_size, kv_cache_num_heads=1, head_dim_with_scale=head_dim_with_scale,
    max_num_pages=max_num_pages, BLOCK_TOKENS=32, BLOCK_HEADS=8,
)
torch.cuda.synchronize()

MAX_SEQ_LEN = int(seq_lens.max().item())

def run_existing():
    out = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    topk_kernel[(batch_size,)](
        acc_ptr=acc, seq_offsets=seq_offsets, seq_lens=seq_lens,
        block_table=block_table, topk_indices_ptr=out,
        K=topk, MAX_SEQ_LEN=MAX_SEQ_LEN, BLOCK=16, MAX_K=topk,
        page_size=page_size, max_num_pages=max_num_pages,
    )
    return out

def run_qrita():
    return compute_topk_qrita(
        acc, seq_offsets, seq_lens, block_table, page_size, max_num_pages,
        batch_size, topk=topk, BLOCK_SIZE=1024,
    )

existing_out = run_existing()
qrita_out = run_qrita()
torch.cuda.synchronize()

total_overlap, total_existing, total_qrita = 0, 0, 0
for b in range(batch_size):
    eset = set(existing_out[b][existing_out[b] != -1].tolist())
    qset = set(qrita_out[b][qrita_out[b] != -1].tolist())
    total_overlap += len(eset & qset)
    total_existing += len(eset)
    total_qrita += len(qset)
match = (total_overlap == total_existing == total_qrita)

for _ in range(5):
    run_existing()
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(20):
    run_existing()
e.record(); torch.cuda.synchronize()
existing_ms = s.elapsed_time(e) / 20

for _ in range(5):
    run_qrita()
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(20):
    run_qrita()
e.record(); torch.cuda.synchronize()
qrita_ms = s.elapsed_time(e) / 20

print(f"RESULT uuid={uuid[:12]} batch={batch_size} max_seq_len={MAX_SEQ_LEN} match={match} "
      f"overlap={total_overlap}/{total_existing} existing_ms={existing_ms:.4f} qrita_ms={qrita_ms:.4f}")
