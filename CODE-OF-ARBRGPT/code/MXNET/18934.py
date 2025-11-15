import mxnet

import numpy as np



input = mxnet.nd.array(np.random.rand(0,1,1))

mxnet.ndarray.LeakyReLU(input)