import mxnet as mx

mx.npx.set_np()



a = mx.np.array([1, 0, 1])

b = mx.np.array([2, 3, 4])



b.attach_grad()



with mx.autograd.record():

    c = mx.np.where(a, b, -1)

    c.backward()

print(b.grad)