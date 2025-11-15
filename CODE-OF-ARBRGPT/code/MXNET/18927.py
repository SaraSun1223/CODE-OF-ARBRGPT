import mxnet as mx

import numpy as np



input = mx.nd.array(np.random.rand(0,2))

mx.ndarray.contrib.bipartite_matching(input, threshold=0.1)