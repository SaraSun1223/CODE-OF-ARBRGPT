import mxnet as mx

import mxnet._deferred_compute as dc

from mxnet import np, npx

npx.set_np()

with dc.context():

    a = np.ones((2, 2))

    b = np.tril(a, 1)

    c = np.tril(a, -1)



sym = dc.get_symbol([b, c], sym_cls=mx.sym.np._Symbol)

res = sym.bind(mx.context.current_context(), args={}).forward()

res