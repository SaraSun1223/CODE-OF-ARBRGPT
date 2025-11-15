import torch

torch.where(torch.tensor([1], dtype=torch.uint8).cuda(), torch.zeros(1).cpu(), torch.zeros(1).cpu())