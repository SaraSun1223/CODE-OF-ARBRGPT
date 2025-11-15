import mxnet as mx
from mxnet import nd


def run_test():

  # large tensor, only works on int 64 BLAS
  A=mx.nd.ones(shape=(1, 2**31))
  nd.linalg.syrk(A)
  nd.waitall()

if __name__ == '__main__':
    run_test()