import mxnet as mx

from mxnet.gluon import nn



N = 1

C = 3

H = W = 2

block = nn.BatchNorm() 

block.collect_params().initialize()

block.collect_params().setattr('grad_req', 'add')



x = mx.nd.arange(N*C*H*W).reshape((N, C, H, W))

x.attach_grad()

for i in range(3):

    with mx.autograd.record():

        y = block(x)

        loss = (y * y).sum() 

    loss.backward()

print(x.grad, block.gamma.grad(), block.beta.grad())