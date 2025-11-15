import torch
import torch.nn as nn

# Initialize Conv2d layers with different padding modes
circular = nn.Conv2d(6, 1, (3, 3), padding=(0, 1), padding_mode='circular')
normal = nn.Conv2d(6, 1, (3, 3), padding=(0, 1))

# Prepare input tensor
input_tensor = torch.zeros(1, 6, 20, 10)

# Execute models
output_circular = circular(input_tensor)
output_normal = normal(input_tensor)

# Check output shapes
print("Circular padding output shape:", output_circular.shape)
print("Normal padding output shape:", output_normal.shape)