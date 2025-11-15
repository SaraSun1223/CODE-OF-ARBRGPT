
import torch

embedding_matrix = torch.randn(10, 5)
indices = torch.LongTensor([0, 1, 2, 1])
offsets = torch.LongTensor([0, 2, 4])
max_norm = 1.0
norm_type = 2.0
scale_grad_by_freq = False
mode = 'mean'
sparse = False

output = torch.nn.functional.embedding_bag(embedding_matrix, indices, offsets, max_norm, norm_type, scale_grad_by_freq, mode, sparse)
