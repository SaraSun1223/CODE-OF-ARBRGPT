import torch

print(torch.__version__)


a = torch.randn(1,1,2,2).cuda().half()

print(a)

print(a.argmax())
