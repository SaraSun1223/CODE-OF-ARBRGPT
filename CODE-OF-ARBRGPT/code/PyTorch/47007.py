import torch

a = torch.tensor([True, False]).cuda()

result = True * a
print(result)
