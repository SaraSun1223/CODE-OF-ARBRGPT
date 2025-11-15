import mxnet as mx

from mxnet import npx

npx.set_np()

a = mx.np.array([]).reshape(2, 1, 0)

b = mx.npx.softmax(a)

b.wait_to_read()
