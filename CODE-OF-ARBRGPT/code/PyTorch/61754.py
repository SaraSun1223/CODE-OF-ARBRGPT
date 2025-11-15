import torch





class Zeros(torch.nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, x):

        y = torch.zeros(())

        y += x

        return y





x = torch.tensor(42.)

torch.onnx.export(Zeros(), x, 'zeros.onnx')