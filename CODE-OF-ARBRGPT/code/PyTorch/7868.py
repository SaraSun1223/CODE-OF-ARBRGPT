
import torch

class Test(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2).transpose(0, 1)

torch.onnx.export(Test(), torch.rand(2, 3, 5), 'test.onnx', verbose=True)
