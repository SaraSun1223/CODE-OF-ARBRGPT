import torch

import torch.nn as nn



class Compute(nn.Module):

    def forward(self, x):

        return torch.sum(x**2)



x = torch.rand(10, 3)

torch.onnx.export(Compute(), x, "test.onnx", verbose=True, input_names=['positions'], output_names=['energy'])