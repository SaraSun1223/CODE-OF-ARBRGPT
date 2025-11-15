import torch

import cuda_if



#CPU Tensor, works as expected

a = torch.Tensor(1).fill_(4.0)

print(cuda_if.lshift(a, 1)) #tensor([ 8.]) as expected

print(cuda_if.rshift(a, 1)) #tensor([ 2.]) as expected



#CUDA Tensor, **BUG in rshift(!!!)**

a = torch.Tensor(1).fill_(4.0).cuda()

print(cuda_if.lshift(a, 1)) #tensor([ 8.], device='cuda:0') as expected

print(cuda_if.rshift(a, 1)) #tensor([ 8.], device='cuda:0') (**BUG!!!** expected 2)