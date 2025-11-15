import mxnet

import numpy as np

sample = mxnet.nd.array(np.random.rand(4,0))

alpha = mxnet.nd.array(np.random.rand(1))

mxnet.ndarray.op.random_pdf_dirichlet(sample=sample, alpha=alpha)