import numpy as _np

import mxnet as mx

from mxnet import np, npx

from mxnet.gluon import HybridBlock

from mxnet.base import MXNetError

from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray

from mxnet.test_utils import check_numeric_gradient, use_np, collapse_sum_like, effective_dtype


@use_np
def test_activation():
    def np_log_sigmoid(a):
        return _np.log(_np.divide(1.0, _np.add(1.0, _np.exp(-a))))

    def np_log_sigmoid_grad(a):
        return _np.divide(1.0, _np.add(1.0, _np.exp(a)))

    class TestLogSigmoid(HybridBlock):

        def __init__(self):
            super(TestLogSigmoid, self).__init__()

        def hybrid_forward(self, F, a):
            return F.npx.activation(a, act_type='log_sigmoid')

    shape = ()

    # shape = (1,)

    test_log_sigmoid = TestLogSigmoid()

    x = mx.np.random.uniform(low=-1.0, high=1.0, size=shape)

    x.attach_grad()

    np_out = np_log_sigmoid(x.asnumpy())

    with mx.autograd.record():
        mx_out = test_log_sigmoid(x)

    assert mx_out.shape == np_out.shape

    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    mx_out.backward()

    np_backward = np_log_sigmoid_grad(x.asnumpy())

    print(np_backward)

    print(x.grad.asnumpy())

    assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

    mx_out = npx.activation(x, act_type='log_sigmoid')

    np_out = np_log_sigmoid(x.asnumpy())

    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_activation_gpu():
    def np_log_sigmoid(a):
        return _np.log(_np.divide(1.0, _np.add(1.0, _np.exp(-a))))

    def np_log_sigmoid_grad(a):
        return _np.divide(1.0, _np.add(1.0, _np.exp(a)))

    class TestLogSigmoid(HybridBlock):

        def __init__(self):
            super(TestLogSigmoid, self).__init__()

        def hybrid_forward(self, F, a):
            return F.npx.activation(a, act_type='log_sigmoid')

    # shape = ()

    shape = (1,)

    test_log_sigmoid = TestLogSigmoid()

    x = mx.np.random.uniform(low=-1.0, high=1.0, size=shape, ctx=mx.gpu())

    x.attach_grad()

    np_out = np_log_sigmoid(x.asnumpy())

    with mx.autograd.record():
        mx_out = test_log_sigmoid(x)

    assert mx_out.shape == np_out.shape

    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    mx_out.backward()

    np_backward = np_log_sigmoid_grad(x.asnumpy())

    print(np_backward)

    print(x.grad.asnumpy())

    assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

    mx_out = npx.activation(x, act_type='log_sigmoid')

    np_out = np_log_sigmoid(x.asnumpy())

    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
