import torch

# Initialize a tensor with requires_grad=True
x = torch.zeros(2, requires_grad=True)

# Expand the tensor to a new shape
xx = x.expand(3, 2)

# Generate random tensor for computation
z = torch.randn(3, 2)

# First gradient computation using expanded tensor
grad1 = torch.autograd.grad((xx * z).mean(), x)[0]
print("First gradient:", grad1)

# Second gradient computation using as_strided
grad2 = torch.autograd.grad((xx.as_strided([3, 2], xx.stride()) * z).mean(), x)[0]
print("Second gradient:", grad2)