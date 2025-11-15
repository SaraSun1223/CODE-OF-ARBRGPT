import mxnet as mx

from mxnet.gluon import HybridBlock,nn

mx.npx.set_np()



class Foobar(HybridBlock):

    def __init__(self, units, prefix=None, params=None):

        super(Foobar, self).__init__(prefix=prefix, params=params)

        self.dense = nn.Dense(1, flatten=False)

        self.layernorm = nn.LayerNorm(epsilon=1e-12, in_channels=768)

    def hybrid_forward(self, F, x):

        out = self.layernorm(x)

        return out



class Foo(HybridBlock):

    def __init__(self, units, prefix=None, params=None):

        super(Foo, self).__init__(prefix=prefix, params=params)

        self.dense = nn.Dense(1, flatten=False)

        self.layernorm = nn.LayerNorm(epsilon=1e-12, in_channels=768)

    def hybrid_forward(self, F, x):

        out = self.layernorm(x)

        out = self.dense(out)

        return out



foo_0 = Foobar(units=1024)

foo_0.initialize(ctx=mx.gpu())

foo_0.hybridize()

out = foo_0(mx.np.random.normal(0,1,size=(10,1024), ctx=mx.gpu()))



foo_1 = Foo(units=1024)

foo_1.initialize(ctx=mx.gpu())

out = foo_1(mx.np.random.normal(0,1,size=(10,1024), ctx=mx.gpu()))



foo_2 = Foo(units=1024)

foo_2.initialize(ctx=mx.gpu())

foo_2.hybridize()

out = foo_2(mx.np.random.normal(0,1,size=(10,1024), ctx=mx.gpu()))