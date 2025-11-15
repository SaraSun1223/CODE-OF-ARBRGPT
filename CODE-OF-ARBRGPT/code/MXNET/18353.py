import mxnet as mx

import numpy as np

mx.npx.set_np()

numerator = mx.np.array(5.0).astype(np.float64)

denominator = mx.np.array(2.0)

result =numerator/denominator
print(result)

