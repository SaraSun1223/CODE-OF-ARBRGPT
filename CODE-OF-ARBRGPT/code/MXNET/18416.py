import mxnet as mx

mx.npx.set_np()

condition = mx.np.array([1, 0, 1, 0, 1])

x = mx.np.ones_like(condition)

y = mx.np.zeros_like(condition)

mx.np.where(condition, x, y)

mx.np.where(condition, 1, 0)