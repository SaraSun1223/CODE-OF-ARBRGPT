import scipy.sparse as sps

import mxnet.ndarray.sparse as mxsps

import mxnet as mx

x = mxsps.csr_matrix(sps.coo_matrix(([2.0], ([99], [999]))).tocsr(), ctx=mx.gpu(0))

y = mx.gluon.utils.split_and_load(x, (mx.gpu(0), mx.gpu(0)))

print(x)

print(y)