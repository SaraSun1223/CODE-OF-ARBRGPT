import mxnet as mx

a = mx.nd.random.randn(2, 20)
mx.nd.unravel_index(a, shape=(-1, 10))
