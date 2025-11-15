import torch

import torch.nn as nn

from torch.fx import symbolic_trace



class TestModule(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, a: torch.Tensor) -> torch.Size:

        return getattr(a, "nonexistent_attr", torch.Size([1,2]))



m = TestModule()

traced = symbolic_trace(m)



# WRONG: AttributeError: 'Tensor' object has no attribute 'nonexistent_attr'

# traced(torch.rand(3, 4))



print(traced.graph)

"""

graph():

    %a : torch.Tensor [#users=1] = placeholder[target=a]

    %getattr_1 : [#users=1] = call_function[target=builtins.getattr](args = (%a, nonexistent_attr), kwargs = {})

    return getattr_1

"""