import torch
from torch.nn import functional as F
x = torch.rand(2500, 100, 4).permute(0, 2, 1)
F.max_pool1d(x, 1)