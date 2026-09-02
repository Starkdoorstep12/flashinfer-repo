import torch
import triton
import triton.language as t1

@triton.jit
def test_kernel(tile_offsets_ptr, out_ptr, batch_size: t1.constexpr):
    pid = t1.program_id(0)
    batch_size_padded: t1.constexpr = triton.next_power_of_2(batch_size)
    offs_b = t1.arange(0, batch_size_padded)
    b_mask = offs_b < batch_size
    tile_offsets_padded = t1.load(tile_offsets_ptr + offs_b, mask=b_mask, other=2**30)
    batch_id = t1.sum(t1.cast(pid >= tile_offsets_padded, t1.int32))
    t1.store(out_ptr + pid, batch_id)

device = 'cuda'
tile_offsets = torch.tensor([5, 10, 15], dtype=torch.int64, device=device)
out = torch.zeros(20, dtype=torch.int32, device=device)
print('Launching isolated test kernel...')
test_kernel[(20,)](tile_offsets, out, batch_size=3)
torch.cuda.synchronize()
print('Result:', out)
