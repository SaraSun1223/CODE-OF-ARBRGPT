import numpy as np

import mxnet as mx

from numpy.testing import assert_allclose

mx.npx.set_np()

mx.npx.random.seed(123)



ctx = mx.cpu()



A = mx.np.random.normal(0, 1, (1, 1, 5, 3), ctx=ctx)

B = mx.np.random.normal(0, 1, (1, 1, 3, 2), ctx=ctx)

out_grad = mx.np.random.normal(0, 1, (1, 1, 5, 2), ctx=ctx)



A.attach_grad()

B.attach_grad()



with mx.autograd.record():

    out = mx.np.einsum('bnij,bnjc->bnic', A, B)

    out.backward(out_grad)



out_gt = A.asnumpy()[0, 0].dot(B.asnumpy()[0, 0])

A_gt_grad = out_grad.asnumpy()[0, 0].dot(B.asnumpy()[0, 0].T)

B_gt_grad = A.asnumpy()[0, 0].T.dot(out_grad.asnumpy()[0, 0])

A_einsum_grad = A.grad.asnumpy()

B_einsum_grad = B.grad.asnumpy()



A.grad[:] = 0

B.grad[:] = 0

with mx.autograd.record():

    out = mx.np.matmul(A, B)

    out.backward(out_grad)

A_matmul_grad = A.grad.asnumpy()

B_matmul_grad = B.grad.asnumpy()





assert_allclose(A_gt_grad, A_matmul_grad[0, 0], 1E-5, 1E-5)

assert_allclose(B_gt_grad, B_matmul_grad[0, 0], 1E-5, 1E-5)

assert_allclose(A_gt_grad, A_einsum_grad[0, 0], 1E-5, 1E-5)

assert_allclose(B_gt_grad, B_einsum_grad[0, 0], 1E-5, 1E-5)