import torch

def foo(x: torch.Tensor) -> float: 

    return x[-3]

bar=torch.jit.script(foo)

print(bar(torch.rand((10,))))