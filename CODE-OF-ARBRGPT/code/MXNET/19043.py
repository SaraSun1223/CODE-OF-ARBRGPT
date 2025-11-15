import mxnet as mx
import numpy as np

mx.npx.set_np()

ctx = mx.gpu()
a = mx.np.ones((3, 3, 3), ctx=ctx)
mult = np.random.normal(0, 1, (3, 3, 3))
a.attach_grad()

with mx.autograd.record():
    b = mx.np.pad(a[:, 1:], ((0, 0), (0, 1), (0, 0))) * mx.np.array(mult, ctx=ctx)
    b = b.sum()
b.backward()

print(a.grad)