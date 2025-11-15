import torch

import torch.nn



print(f"Torch version: {torch.__version__}")





class MyModule(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = torch.nn.Linear(100, 50).to('cuda:0')

        self.layer2 = torch.nn.Linear(50, 20).to('cuda:1')



    def forward(self, x):

        a = self.layer1(x)

        b = self.layer2(a.to('cuda:1'))

        return b





m = MyModule()

x = torch.randn(32, 100).to('cuda:0')

loss = m(x).sum()

loss.backward()

print(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))