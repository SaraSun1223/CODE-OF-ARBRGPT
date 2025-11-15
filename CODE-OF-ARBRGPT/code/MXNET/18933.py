import mxnet

import numpy as np



input = mxnet.nd.array(np.random.rand(0,1,1)) # batch = 0

gamma = mxnet.nd.array(np.random.rand(1))

beta = mxnet.nd.array(np.random.rand(1))

mxnet.ndarray.InstanceNorm(input, gamma, beta)