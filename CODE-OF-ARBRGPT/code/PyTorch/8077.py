import torch



class M(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.B = torch.nn.Parameter(torch.Tensor())



class A(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.m = M()



a = A()

print(type(a.m.B))

torch.save(a, 'test')

a = torch.load('test')

print(type(a.m.B))