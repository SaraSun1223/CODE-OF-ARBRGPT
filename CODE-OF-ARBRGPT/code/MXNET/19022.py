import mxnet as mx
from mxnet import np, npx

def test_rnn():
    INT_OVERFLOW = 2**10

    def batch_check(x, modes, params):
        for m, p in zip(modes, params):
            state = np.random.normal(0, 1, (1, 4, 1))
            x.attach_grad()
            state.attach_grad()
            p.attach_grad()

            with mx.autograd.record():
                y = npx.rnn(data=x, parameters=p, mode=m, \
                    state=state, state_size=1, num_layers=1)
            assert y.shape == (INT_OVERFLOW, 4, 1)
            assert type(y[0]).__name__ == 'ndarray'
            y.backward()
            print(state.grad)

    data = np.random.normal(0, 1, (INT_OVERFLOW, 4, 4))
    modes = ['rnn_relu', 'rnn_tanh', 'gru']
    params = [np.random.normal(0, 1, (7,)), \
        np.random.normal(0, 1, (7,)), \
        np.random.normal(0, 1, (21,))]
    batch_check(data, modes, params)

test_rnn()