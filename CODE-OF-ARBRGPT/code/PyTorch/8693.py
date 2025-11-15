import torch

# Create a 5D tensor and indices
array_5d = torch.randn(1, 1, 10, 10, 10)
indices = torch.randn(1, 1, 1, 1, 3)

# Call grid_sample with an invalid mode
torch.nn.functional.grid_sample(array_5d, indices, mode='anything')