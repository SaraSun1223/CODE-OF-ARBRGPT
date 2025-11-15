import mxnet as mx
import copy

v = mx.sym.Variable('a').as_np_ndarray()
b = copy.copy(v)
b._set_attr(name='b')

print(v)
print(b)