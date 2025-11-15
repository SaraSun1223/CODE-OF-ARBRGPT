import torch

torch.cosine_similarity(torch.ones((10, 2, 3, 4)), torch.ones((2, 3, 4)), dim=-1)