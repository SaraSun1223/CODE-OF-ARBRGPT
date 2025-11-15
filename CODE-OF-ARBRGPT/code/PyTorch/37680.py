import torch

from torch import nn



t = torch.rand(10, requires_grad=True)

bad = torch.argmax(t)



res = bad + 2



res.sum().backward()