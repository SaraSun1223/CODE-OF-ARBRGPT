import torch

x = torch.randn(3, 3)
x.align_to('N', 'C')