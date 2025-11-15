import mxnet as mx

mx.npx.set_np()



class IntegerIndexing(mx.gluon.HybridBlock):

        def __init__(self, a, prefix=None, params=None):

            super().__init__(prefix=prefix, params=params)

            self._a = a

        def hybrid_forward(self, F, x):

            return x[self._a]



net = IntegerIndexing(-1)

net.hybridize()

net(mx.np.ones(3,))