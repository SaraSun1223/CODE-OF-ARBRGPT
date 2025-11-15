from mxnet.util import use_np
from mxnet.gluon import nn, HybridBlock
import mxnet as mx
import numpy as np

attr = {'sg_mkldnn_conv_bn_0': {'with_bn': 'true'}}
data = mx.symbol.Variable('data', shape=(64, 4, 10, 10), dtype='float32')
data2 = mx.symbol.Variable('data2', shape=(64, 64, 10, 10), dtype='float32')
weight1 = mx.symbol.Variable('conv1_weight', dtype='float32')
weight2 = mx.symbol.Variable('conv2_weight', dtype='float32', shape=(64, 64, 1, 1))
conv1 = mx.symbol.Convolution(data=data, weight=weight1, name='conv1', num_filter=64,
                              kernel=(1, 1), stride=(1, 1), no_bias=True)
bn1 = mx.symbol.BatchNorm(data=conv1, name="bn1")
conv2 = mx.symbol.Convolution(data=bn1, weight=weight2, name='conv2', num_filter=64,
                              kernel=(1, 1), stride=(1, 1), no_bias=True)
bn2 = mx.symbol.BatchNorm(data=conv2, name="bn2")
sum = bn2 + data2
inputs = mx.sym.var('data', dtype='float32')
sym_block = mx.gluon.SymbolBlock(sum, [inputs])
for k, v in sym_block.collect_params().items():
    v.initialize()
mm = sym_block(mx.nd.zeros((64, 4, 10, 10)))
sym_block.optimize_for(mx.nd.zeros((64, 4, 10, 10)), backend='MKLDNN_QUANTIZE')
