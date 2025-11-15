import mxnet as mx
from mxnet import util

@util.use_np
def func():
    # argmax issue
    d = mx.np.array([0, 1])
    print(mx.np.argmax(d))
    print(mx.np.argmax(d).dtype)

    # repeat issue
    x = mx.np.array([[1,2],[3,4]])
    print(mx.np.repeat(x, [1, 2], axis=0))

    # softmax issue
    x_empty = mx.np.array([], ctx=mx.gpu(0)).reshape(2, 0, 0)
    print(mx.npx.softmax(x_empty))

    # mean issue
    print(mx.np.mean(x_empty))

    # prod issue
    print(mx.np.prod(x_empty))

func()