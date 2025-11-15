import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from torch import nn
from torch.nn.utils import weight_norm

device = torch.device('cuda')
model = weight_norm(nn.Linear(20, 30), dim=None)
model = nn.DataParallel(model).to(device)

x = torch.rand(40, 20).to(device)
y = model(x)
loss = y.mean()
loss.backward()