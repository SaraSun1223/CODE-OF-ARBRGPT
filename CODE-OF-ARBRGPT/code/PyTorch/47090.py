import torch

class _T(torch.Tensor): ...

x = torch.tensor([1,2]).as_subclass(_T)

torch.max(x, dim=0)