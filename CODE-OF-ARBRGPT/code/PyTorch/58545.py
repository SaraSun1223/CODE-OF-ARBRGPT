import torch

rng = torch.Generator(device="cuda")

torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.randint(low=0, high=10, size=(1,), generator=rng) # Works well

idx_pts = torch.randperm(n=50, generator=rng) # Fail and raise error