import torch

A = torch.ones(5, 2, dtype=torch.double)
b = torch.rand(5)

print(b)

A[:, 1] = b
print(A)

A[:, [1]] = b.unsqueeze(-1)
print(A)

A[:, [1]] = b.double().unsqueeze(-1)
print(A)