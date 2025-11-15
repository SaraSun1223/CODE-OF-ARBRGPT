import numpy as np
import mxnet as mx
from mxnet.gluon.nn import HybridBlock, Conv1D, HybridSequential, HybridLambda, Dense
from mxnet import autograd, nd
from mxnet.gluon.loss import L2Loss


def print_grads(model, ctx=mx.cpu()):
    pd = model.collect_params()
    total_grad_l2 = 0
    total_grad_l1 = 0
    total_grad_linf = 0
    for p in pd:
        try:
            g = pd[p].grad(ctx) / N
            g2 = (g**2).sum().as_in_context(mx.cpu()).asscalar()
            g1 = g.abs().sum().as_in_context(mx.cpu()).asscalar()
            ginf = g.max().as_in_context(mx.cpu()).asscalar()
            total_grad_linf = max(total_grad_linf, ginf)
            total_grad_l2 += g2
            total_grad_l1 += g1
            print(f"||g_param||_2: {g2**0.5:.2E} | Param: {p}")
        except Exception:
            pass
    grad_info = f"""
    -------------------------------------------
    -------  Grad Info
    *  ||g||_2: {total_grad_l2**0.5:.2E}
    *  ||g||_1: {total_grad_l1:.2E}
    *  ||g||_inf: {total_grad_linf:.2E}

    """
    print(grad_info)


def run_model(model, loss, X, Y, num_iters=1):
    for i in range(num_iters):
        with autograd.record():
            Y_hat = model(X)
            ll = loss(Y_hat, Y)
            ll = ll.sum()
            ll.backward()
            print_grads(model)
    return Y_hat


def conv_layer(atrous_rates, num_channels):
    convs = HybridSequential()
    convs.add(HybridLambda(lambda F, x: F.transpose(x, (0, 2, 1))))
    for rate in atrous_rates:
        convs.add(Conv1D(num_channels, 3, padding=rate, dilation=rate, activation='tanh'))
    convs.add(HybridLambda(lambda F, x: F.transpose(x, (0, 2, 1))))
    return convs


class Model(HybridBlock):
    """
    Model takes tensors of shape N x T x C and produces predictions with shape N x T
    """

    def __init__(self, conv_units, atrous_rates, use_take=False, **kwargs):
        super().__init__(prefix=kwargs.get('prefix', None), params=kwargs.get('params', None))
        self.use_take = use_take
        with self.name_scope():
            self.convs = conv_layer(atrous_rates, conv_units)
            self.dense_out = Dense(1, flatten=False, activation='tanh')

    def hybrid_forward(self, F, X):
        X1 = X
        X2 = self.convs(X1)
        if self.use_take:
            X3 = F.take(X2, nd.array([1, 2, 3]), axis=-1)
        else:
            X3 = F.slice_axis(X2, begin=1, end=4, axis=-1)
        X4 = self.dense_out(X3)
        X4 = F.squeeze(X4, axis=-1)
        return X4


if __name__ == "__main__":
    N = 30
    T = 20
    C = 8
    conv_units = 5
    atrous_rates = [1, 2, 4]
    np.random.seed(1234)

    X = np.random.normal(size=(N, T, C))
    Y = np.random.normal(size=(N, T))
    X, Y = nd.array(X), nd.array(Y)

    # Using F.take
    mx.random.seed(12354)
    model = Model(conv_units, atrous_rates, use_take=True)
    model.initialize()
    loss = L2Loss()
    Y_hat1 = run_model(model, loss, X, Y)

    # Using F.slice_axis
    mx.random.seed(12354)
    model2 = Model(conv_units, atrous_rates, use_take=False)
    model2.initialize()
    loss2 = L2Loss()
    Y_hat2 = run_model(model2, loss2, X, Y)

    delta = nd.abs(Y_hat1-Y_hat2).sum().asscalar()
    print("==== Same outputs?")
    print(f"Y_hat1 - Yhat2 = {delta:.4f}")