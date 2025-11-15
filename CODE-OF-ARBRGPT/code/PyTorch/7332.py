import torch
import torch.nn as nn

# Define a simple conv2D layer
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=[1, 1])

# Prepare a 3D tensor input (incorrect input shape)
input_tensor = torch.randn(1, 28, 28)  # Shape: [batch_size, height, width]

# Execute conv2D operation
output = conv_layer(input_tensor)