import torch

a = torch.tensor([0, -1, 1, -2, 2]).cuda()[::2]

print(a)


result = a.kthvalue(2)
print(result)
