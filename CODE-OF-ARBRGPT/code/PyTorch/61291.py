import torch

torch.set_default_tensor_type('torch.cuda.DoubleTensor')



net = torch.nn.Linear(2, 1)

print('a', net(torch.zeros(1, 1)).item(), net.state_dict())  

print('b', net(torch.zeros(1, 1)).item(), net.state_dict())

print('c', net(torch.zeros(1, 1)).item(), net.state_dict())