import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose
from mxnet.gluon import HybridBlock
mx.npx.set_np()

class Foo(HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(Foo, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, valid_length):
        mask = (F.np.ones((10,)) < valid_length).astype(np.float32)
        mask2 = (F.np.ones((10,)) < valid_length).astype(np.float32)
        mask = mask * F.np.expand_dims(mask2, axis=-1)
        return mask

foo = Foo()
foo.hybridize()
out = foo(mx.np.ones((10,), ctx=mx.gpu()))