import torch

x = torch.randn(3, 3, requires_grad=True)

z1 = torch.einsum("ij,jk->ik", (x, torch.randn(3, 3)))

z2 = torch.einsum("ij,jk->ik", (x, torch.randn(3, 3)))

z1.sum().backward()