import torch

x = torch.randn(3)

print(x.is_pinned()) # False, as expected

q_int = torch.randint(0, 100, [1, 2, 3], device="cuda", dtype=torch.int)

q = torch._make_per_tensor_quantized_tensor(q_int, scale=0., zero_point=0)

x = torch.randn(3)

print(x.is_pinned()) #True, all host allocations after this point are in pinned memory