import logging
import os
from mxnet import init
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.nn import BatchNorm

import mxnet as mx

class BuggyModel(HybridBlock):

    def __init__(
        self,
        channels,
        norm_layer=BatchNorm,
        norm_kwargs=None,
        in_channels=3,
        **kwargs
    ):
        super(BuggyModel, self).__init__(**kwargs)
        self.in_channels = in_channels
        with self.name_scope():
            self.conv1 = nn.Conv3D(
                    in_channels=self.in_channels,
                    channels=channels,
                    kernel_size=(1, 7, 7),
                    strides=(1, 2, 2),
                    padding=(0, 3, 3),
                    use_bias=False,
                    )
            self.bn1 = norm_layer(in_channels=channels, **({} if norm_kwargs is None else norm_kwargs))

    def hybrid_forward(self, F, x):
        """Hybrid forward of R2+1D net"""
        x = self.conv1(x)
        x = self.bn1(x)
        return x


print(f"input channel of 45")
net = BuggyModel(channels=45)
net.initialize(init=init.Constant(1))
input_data = mx.nd.zeros((1, 3, 8, 160, 160), ctx=mx.cpu())
out = net(input_data).asnumpy()
print(f"input channel of 45, {out.shape}")

print(f"input channel of 64")
net = BuggyModel(channels=64)
net.initialize(init=init.Constant(1))
input_data = mx.nd.zeros((1, 3, 8, 160, 160), ctx=mx.cpu())
out = net(input_data).asnumpy()
print(f"input channel of 64, {out.shape}")