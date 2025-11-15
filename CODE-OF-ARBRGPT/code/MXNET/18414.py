import mxnet as mx

mx.npx.set_np()

a = mx.np.ones((10, 10), dtype=mx.np.int32)

b = -a

print(b.dtype)