"""
Experiment 2b: isolate tensor-allocation overhead specifically attributable
to kernel_splitk_v2's extra intermediate buffers (m_final, l_final,
acc_final) that kernel_splitk (v1) does not need.

Companion to experiment2_launch_overhead.py — together they decompose the
v1-vs-v2 wall-clock gap into named, independently-measured components.
"""
import torch

device = 'cuda'

# Warmup: let the caching allocator settle for these exact shapes first.
# Without this, the first few allocations of a new size may trigger real
# cudaMalloc calls (much more expensive than cached-allocator reuse),
# which would not reflect the shape's real steady-state cost inside the
# actual benchmark loop (which always warms up 10 iters before timing).
for _ in range(20):
    m_final = torch.full((1, 16), -float("inf"), dtype=torch.float32, device=device)
    l_final = torch.zeros((1, 16), dtype=torch.float32, device=device)
    acc_final = torch.zeros((1, 16, 512), dtype=torch.float32, device=device)
torch.cuda.synchronize()

n_iters = 50
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(n_iters):
    m_final = torch.full((1, 16), -float("inf"), dtype=torch.float32, device=device)
    l_final = torch.zeros((1, 16), dtype=torch.float32, device=device)
    acc_final = torch.zeros((1, 16, 512), dtype=torch.float32, device=device)
end.record()
torch.cuda.synchronize()
print(f"Extra v2 allocations (warmed up): {start.elapsed_time(end)/n_iters*1000:.2f} us/iter")
