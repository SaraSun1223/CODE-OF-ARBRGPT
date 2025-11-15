import mxnet as mx

mx.npx.set_np()

net = mx.gluon.nn.Dense(16, in_units=16)

net.cast("float16")

net.initialize(ctx=mx.gpu())

net.hybridize()

net(mx.np.random.normal(0, 1, (16, 16), dtype=mx.np.float16, ctx=mx.gpu()))