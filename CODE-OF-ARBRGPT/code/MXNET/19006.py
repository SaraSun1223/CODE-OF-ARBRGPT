import mxnet

from mxnet import np, npx



INT_OVERFLOW = 2**31



A = np.array([INT_OVERFLOW], dtype='int64')

assert A == INT_OVERFLOW