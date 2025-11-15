import torch

print(torch.__version__)

import numpy as np

# Test case 1
x = torch.tensor(0. - 1.0000e+20j)
print("x.sqrt():", x.sqrt())

# Convert to NumPy and compute sqrt
print("np.sqrt(x.numpy()):", np.sqrt(x.numpy()))

# Test case with complex128 dtype
x = torch.tensor(-1.0000e+20 - 4988429.2000j, dtype=torch.complex128)
print("x.sqrt():", x.sqrt())

# NumPy comparison
print("np.sqrt(x.numpy()):", np.sqrt(x.numpy()))

# GPU computation and comparison
is_equal = x.cuda().sqrt().cpu().numpy() == np.sqrt(x.numpy())
print("GPU vs NumPy sqrt equal:", is_equal)
